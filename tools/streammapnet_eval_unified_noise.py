#!/usr/bin/env python3
"""
Unified StreamMapNet Evaluation Script with NOISE in Camera Parameters
======================================================================
Runs StreamMapNet inference with configurable camera inputs AND evaluates with camera-specific FOV clipping.

**KEY MODIFICATION**: Adds Gaussian NOISE to camera extrinsics to test robustness.

This script adds controlled noise to camera2ego parameters to test how StreamMapNet performs 
with imperfect camera calibration, following the robustness experiments from MapTR paper:
- Translation noise: Gaussian noise with std σ₁ (meters) added to [Δx, Δy, Δz]
- Rotation noise: Gaussian noise with std σ₂ (radians) added to [θx, θy, θz]

Usage Examples:
---------------
# Translation noise σ = 0.1m
python tools/streammapnet_eval_unified_noise.py --cameras CAM_FRONT --noise-type translation --noise-std 0.1

# Rotation noise σ = 0.01 rad
python tools/streammapnet_eval_unified_noise.py --cameras CAM_FRONT --noise-type rotation --noise-std 0.01

# No noise (baseline)
python tools/streammapnet_eval_unified_noise.py --cameras CAM_FRONT --noise-std 0
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import shapely

# Add StreamMapNet project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# StreamMapNet imports
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import get_root_logger
import os.path as osp

# NuScenes and geometry utilities
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString, Point, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

# Import shared camera FOV utilities
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
    extract_gt_with_fov_clipping,
    process_predictions_with_fov_clipping
)

# Add StreamMapNet project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== CAMERA CONFIGURATION ====================
CAMERA_MAP = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_FRONT_LEFT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_BACK_RIGHT': 5
}

def parse_camera_config(camera_args: List[str]) -> List[int]:
    """
    Parse camera configuration from command line arguments.
    Handles both space-separated and comma-separated camera names.
    
    Args:
        camera_args: List of camera names or 'all'
                    Examples: ['CAM_FRONT', 'CAM_BACK'] or ['CAM_FRONT,CAM_BACK']
    
    Returns:
        List of camera indices (0-5)
    """
    if not camera_args or camera_args[0] == 'all':
        return list(range(6))  # All cameras
    
    # Handle comma-separated camera names
    camera_names_flat = []
    for arg in camera_args:
        if ',' in arg:
            camera_names_flat.extend([name.strip() for name in arg.split(',')])
        else:
            camera_names_flat.append(arg)
    
    camera_indices = []
    for cam_name in camera_names_flat:
        if cam_name in CAMERA_MAP:
            camera_indices.append(CAMERA_MAP[cam_name])
        else:
            print(f"Warning: Unknown camera name '{cam_name}'. Valid names: {list(CAMERA_MAP.keys())}")
    
    if not camera_indices:
        print("Warning: No valid cameras specified. Using all cameras.")
        return list(range(6))
    
    return camera_indices


def add_noise_to_camera_extrinsics(
    img_metas_list: List[Dict], 
    noise_trans_std: float = 0.0,
    noise_rot_std: float = 0.0,
    active_camera_indices: List[int] = None,
    seed: int = None,
    logger=None
) -> List[Dict]:
    """
    Add Gaussian noise to camera extrinsics to test robustness (following MapTR paper).
    Can apply translation and rotation noise simultaneously.
    
    Noise Types:
    - Translation: Add N(0, σ²) noise to camera translation [Δx, Δy, Δz]
    - Rotation: Add N(0, σ²) noise to rotation angles [θx, θy, θz] in radians
    
    Args:
        img_metas_list: List of img_metas dicts, one per timestamp in sequence
        noise_trans_std: Standard deviation of Gaussian noise for translation (meters)
        noise_rot_std: Standard deviation of Gaussian noise for rotation (radians)
        active_camera_indices: Cameras to add noise to (default: all 6)
        seed: Random seed for reproducibility (default: None)
        logger: Optional logger
    
    Returns:
        Modified img_metas_list with noisy camera extrinsics
    """
    if noise_trans_std == 0 and noise_rot_std == 0:
        if logger:
            logger.info("Noise stds are 0, skipping noise addition")
        return img_metas_list
    
    if active_camera_indices is None:
        active_camera_indices = list(range(6))
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    if logger:
        logger.info("\n" + "="*80)
        logger.info(f"Adding noise to camera extrinsics")
        if noise_trans_std > 0:
            logger.info(f"Translation Noise std (σ): {noise_trans_std} meters")
        if noise_rot_std > 0:
            logger.info(f"Rotation Noise std (σ): {noise_rot_std} radians")
        logger.info(f"Active cameras: {[list(CAMERA_MAP.keys())[i] for i in active_camera_indices]}")
        logger.info("="*80)
    
    noise_applied = False
    for t_idx, img_meta in enumerate(img_metas_list):
        # StreamMapNet / NuscDataset typically has 'cam_extrinsics' (ego2cam) directly
        # Format: list of 4x4 arrays
        
        # Check available keys
        if 'cam_extrinsics' in img_meta:
            # This is EGO2CAM
            ego2cam_list = img_meta['cam_extrinsics']
            
            # Convert to camera2ego (inverse) for noise application logic consistency
            # My noise logic works on camera2ego (cam->ego)
            # cam_extrinsics in StreamMapNet seems to be ego2cam (based on name usually, but let's verification)
            # In NuscDataset.get_sample:
            # extrinsic = np.eye(4); extrinsic[:3, :3] = ego2cam_r; ...
            # cam_extrinsics.append(extrinsic.tolist())
            # So yes, it is EGO2CAM.
            
            # We need camera2ego for the noise logic (noise is applied to cam->ego usually)
            camera2ego_list = []
            for e2c in ego2cam_list:
                e2c_mat = np.array(e2c)
                camera2ego_list.append(np.linalg.inv(e2c_mat))
        
        elif 'camera2ego' in img_meta:
            camera2ego_list = img_meta['camera2ego']
        
        else:
            if logger and t_idx == 0:
                logger.warning(f"Neither 'cam_extrinsics' nor 'camera2ego' found. Keys: {list(img_meta.keys())}")
            continue
            
        # Ensure we have camera2ego list
        if not isinstance(camera2ego_list, list) or len(camera2ego_list) != 6:
            if logger and t_idx == 0:
                logger.warning(f"Expected 6 camera2ego matrices, got {len(camera2ego_list) if isinstance(camera2ego_list, list) else 'non-list'}")
            continue
        
        cam_names = list(CAMERA_MAP.keys())
        
        # Add noise to each active camera
        for cam_idx in active_camera_indices:
            # =========================================================
            # ROBUST INTRINSIC RECOVERY STRATEGY
            # =========================================================
            
            # Get original Extrinsics
            cam2ego_orig = camera2ego_list[cam_idx]
            if isinstance(cam2ego_orig, np.ndarray):
                cam2ego_np = cam2ego_orig.copy()
            elif torch.is_tensor(cam2ego_orig):
                cam2ego_np = cam2ego_orig.cpu().numpy().copy()
            else:
                cam2ego_np = np.array(cam2ego_orig).copy()
            
            # For StreamMapNet, we might not have lidar2img/lidar2ego directly handy or consistent
            # But wait - StreamMapNet uses 'ego2img' which is K @ ego2cam
            # If we noise ego2cam, we must update ego2img.
            
            # 1. Get original K (intrinsic)
            if 'cam_intrinsics' in img_meta:
                intrinsic_list = img_meta['cam_intrinsics']
                K = np.array(intrinsic_list[cam_idx])
            elif 'camera_intrinsics' in img_meta:
                 intrinsic_list = img_meta['camera_intrinsics']
                 K = np.array(intrinsic_list[cam_idx])
            else:
                if logger and t_idx == 0:
                    logger.warning("No intrinsics found, skipping noise")
                continue
                
            # Pad K to 4x4
            if K.shape == (3, 3):
                K_4x4 = np.eye(4)
                K_4x4[:3, :3] = K
            else:
                K_4x4 = K

            # =========================================================
            # 2. Apply Noise to Extrinsics (Camera -> Ego)
            # =========================================================
            
            cam2ego_noisy = cam2ego_np.copy()
            local_noise_applied = False
            
            if noise_trans_std > 0:
                noise = np.random.normal(0, noise_trans_std, size=3)
                if np.abs(noise).max() > 1e-8:
                    cam2ego_noisy[:3, 3] += noise
                    local_noise_applied = True
            
            if noise_rot_std > 0:
                delta_angles = np.random.normal(0, noise_rot_std, size=3)
                if np.abs(delta_angles).max() > 1e-8:
                    rx = np.array([
                        [1, 0, 0],
                        [0, np.cos(delta_angles[0]), -np.sin(delta_angles[0])],
                        [0, np.sin(delta_angles[0]), np.cos(delta_angles[0])]
                    ])
                    ry = np.array([
                        [np.cos(delta_angles[1]), 0, np.sin(delta_angles[1])],
                        [0, 1, 0],
                        [-np.sin(delta_angles[1]), 0, np.cos(delta_angles[1])]
                    ])
                    rz = np.array([
                        [np.cos(delta_angles[2]), -np.sin(delta_angles[2]), 0],
                        [np.sin(delta_angles[2]), np.cos(delta_angles[2]), 0],
                        [0, 0, 1]
                    ])
                    rotation_perturbation = rz @ ry @ rx
                    cam2ego_noisy[:3, :3] = rotation_perturbation @ cam2ego_noisy[:3, :3]
                    local_noise_applied = True
            
            if local_noise_applied:
                noise_applied = True
            
            # Update lists
            if isinstance(camera2ego_list[cam_idx], np.ndarray):
                camera2ego_list[cam_idx] = cam2ego_noisy
            else:
                camera2ego_list[cam_idx] = cam2ego_noisy

            # =========================================================
            # 3. Update Dependent Matrices (ego2cam, ego2img)
            # =========================================================
            if local_noise_applied:
                # Update ego2cam (inverse of cam2ego)
                ego2cam_noisy = np.linalg.inv(cam2ego_noisy)
                
                # Update 'cam_extrinsics' if it existed
                if 'cam_extrinsics' in img_meta:
                    if isinstance(img_meta['cam_extrinsics'][cam_idx], list):
                        img_meta['cam_extrinsics'][cam_idx] = ego2cam_noisy.tolist()
                    else:
                        img_meta['cam_extrinsics'][cam_idx] = ego2cam_noisy
                
                # Update 'ego2img' (= K @ ego2cam)
                # StreamMapNet uses 'ego2img' extensively
                if 'ego2img' in img_meta:
                    ego2img_new = K_4x4 @ ego2cam_noisy
                    if isinstance(img_meta['ego2img'][cam_idx], list):
                        img_meta['ego2img'][cam_idx] = ego2img_new.tolist()
                    else:
                        img_meta['ego2img'][cam_idx] = ego2img_new
                        
                # Update 'lidar2img' if it exists (K @ ego2cam @ lidar2ego)
                if 'lidar2img' in img_meta and 'lidar2ego_rotation' in img_meta:
                     # Reconstruct lidar2ego keys are often rotation(quaternion) + translation
                     l2e_r = Quaternion(img_meta['lidar2ego_rotation']).rotation_matrix
                     l2e_t = np.array(img_meta['lidar2ego_translation'])
                     l2e_mat = np.eye(4)
                     l2e_mat[:3, :3] = l2e_r
                     l2e_mat[:3, 3] = l2e_t
                     
                     lidar2img_new = K_4x4 @ ego2cam_noisy @ l2e_mat
                     if isinstance(img_meta['lidar2img'][cam_idx], list):
                         img_meta['lidar2img'][cam_idx] = lidar2img_new.tolist()
                     elif torch.is_tensor(img_meta['lidar2img'][cam_idx]):
                         img_meta['lidar2img'][cam_idx] = torch.from_numpy(lidar2img_new).to(img_meta['lidar2img'][cam_idx].device)
                     else:
                         img_meta['lidar2img'][cam_idx] = lidar2img_new

    if logger:
        if noise_applied:
            logger.info("\n✓ Noise addition complete (cam_extrinsics + ego2img updated)")
        else:
            logger.info("\n✓ Noise was below precision threshold, skipped")
        logger.info("="*80 + "\n")
    
    return img_metas_list


# ==================== DATASET PATCHING ====================
def patch_nusc_dataset(cfg, logger):
    """
    Patch NuscDataset to handle annotation format variations.
    Copied from save_streammapnet_predictions.py
    """
    try:
        from plugin.datasets.nusc_dataset import NuscDataset
        from pyquaternion import Quaternion
        import numpy as np
        
        original_load_annotations = NuscDataset.load_annotations
        
        def patched_load_annotations(self, ann_file):
            """Patched version that handles both list and dict formats."""
            ann = mmcv.load(ann_file)
            
            # Handle dict format: {'infos': [...]} or {'samples': [...]}
            if isinstance(ann, dict):
                if 'infos' in ann:
                    ann = ann['infos']
                elif 'samples' in ann:
                    ann = ann['samples']
                else:
                    # Take first list value from dict
                    for key, value in ann.items():
                        if isinstance(value, list):
                            ann = value
                            break
            
            # Apply interval slicing and set self.samples
            self.samples = ann[::self.interval]
        
        NuscDataset.load_annotations = patched_load_annotations
        
        # Completely replace get_sample to handle all key name variations
        def patched_get_sample(self, idx):
            """Full replacement to handle different annotation formats."""
            sample = self.samples[idx]
            
            # Extract location
            location = sample.get('location') or sample.get('map_location')
            if location is None:
                raise KeyError(f"Sample missing both 'location' and 'map_location'. Keys: {list(sample.keys())}")
            
            # Extract ego2global translation/rotation
            e2g_translation = sample.get('e2g_translation') or sample.get('ego2global_translation')
            e2g_rotation = sample.get('e2g_rotation') or sample.get('ego2global_rotation')
            if e2g_translation is None or e2g_rotation is None:
                raise KeyError(f"Sample missing ego2global keys. Keys: {list(sample.keys())}")
            
            # Extract sample index
            sample_idx = sample.get('sample_idx') or sample.get('frame_idx', idx)
            scene_name = sample.get('scene_name') or sample.get('scene_token', '')
            
            # Get map geometry
            map_geoms = self.map_extractor.get_map_geom(location, e2g_translation, e2g_rotation)
            map_label2geom = {}
            for k, v in map_geoms.items():
                if k in self.cat2id.keys():
                    map_label2geom[self.cat2id[k]] = v
            
            # Process camera data
            ego2img_rts = []
            img_filenames = []
            cam_intrinsics = []
            cam_extrinsics = []
            
            for cam_name, c in sample['cams'].items():
                # Get extrinsics (ego2cam)
                if 'extrinsics' in c:
                    extrinsic = np.array(c['extrinsics'])
                elif 'ego2cam' in c:
                    extrinsic = np.array(c['ego2cam'])
                elif 'sensor2ego_translation' in c and 'sensor2ego_rotation' in c:
                    # Build ego2cam from sensor2ego
                    sensor2ego_r = Quaternion(c['sensor2ego_rotation']).rotation_matrix
                    sensor2ego_t = np.array(c['sensor2ego_translation'])
                    ego2cam_r = sensor2ego_r.T
                    ego2cam_t = -ego2cam_r @ sensor2ego_t
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = ego2cam_r
                    extrinsic[:3, 3] = ego2cam_t
                else:
                    raise KeyError(f"Camera '{cam_name}' missing extrinsics. Keys: {list(c.keys())}")
                
                # Get intrinsics
                if 'intrinsics' in c:
                    intrinsic = np.array(c['intrinsics'])
                elif 'camera_intrinsic' in c:
                    intrinsic = np.array(c['camera_intrinsic'])
                elif 'cam_intrinsic' in c:
                    intrinsic = np.array(c['cam_intrinsic'])
                else:
                    raise KeyError(f"Camera '{cam_name}' missing intrinsics. Keys: {list(c.keys())}")
                
                # Ensure intrinsics is 3x3
                if intrinsic.shape == (3, 4):
                    intrinsic = intrinsic[:, :3]
                elif intrinsic.shape == (4, 4):
                    intrinsic = intrinsic[:3, :3]
                elif intrinsic.shape != (3, 3) and intrinsic.size == 9:
                    intrinsic = intrinsic.reshape(3, 3)
                
                # Get image path
                img_path = c.get('img_fpath') or c.get('data_path') or c.get('img_path', '')
                
                # Build ego2img transform
                ego2cam_rt = extrinsic.copy()
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2cam_rt = viewpad @ ego2cam_rt
                ego2img_rts.append(ego2cam_rt)
                
                img_filenames.append(img_path)
                cam_intrinsics.append(intrinsic.tolist())
                cam_extrinsics.append(extrinsic.tolist())
            
            input_dict = {
                'location': location,
                'token': sample['token'],
                'img_filenames': img_filenames,
                'cam_intrinsics': cam_intrinsics,
                'cam_extrinsics': cam_extrinsics,
                'ego2img': ego2img_rts,
                'map_geoms': map_label2geom,
                'ego2global_translation': e2g_translation,
                'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
                'sample_idx': sample_idx,
                'scene_name': scene_name,
                # Add lidar2ego info for robust noise eval if available
                'lidar2ego_translation': sample.get('lidar2ego_translation', [0,0,0]),
                'lidar2ego_rotation': sample.get('lidar2ego_rotation', [1,0,0,0]),
            }
            
            return input_dict
        
        NuscDataset.get_sample = patched_get_sample
        logger.info('Applied comprehensive dataset patching for StreamMapNet')
        
        return True
    except Exception as e:
        logger.warning(f'Could not patch NuscDataset: {e}')
        return False


# ==================== TEMPORAL/STREAMING HELPERS ====================
def is_streaming_model(model):
    """
    Check if model uses temporal/streaming BEV features.
    StreamMapNet maintains historical BEV features across frames.
    """
    if hasattr(model, 'module'):
        model = model.module
    return hasattr(model, 'streaming_bev') and model.streaming_bev

def reset_bev_memory(model, logger=None):
    if logger:
        logger.info('  ✓ BEV memory will auto-reset on scene change')
    return True

def group_samples_by_scene(dataset, logger=None):
    from collections import defaultdict
    scenes = defaultdict(list)
    for idx in range(len(dataset)):
        sample_info = dataset.samples[idx]
        scene_token = sample_info.get('scene_name') or sample_info.get('scene_token', 'default_scene')
        timestamp = sample_info.get('sample_idx', sample_info.get('frame_idx', idx))
        scenes[scene_token].append((timestamp, idx, sample_info))
    
    for scene_token in scenes:
        scenes[scene_token].sort(key=lambda x: x[0])
    
    if logger:
        logger.info(f'Grouped {len(dataset)} samples into {len(scenes)} scenes')
    return scenes


# ==================== INFERENCE CODE ====================
def run_streammapnet_inference(
    config_path: str,
    checkpoint_path: str,
    output_pkl: str,
    camera_indices: List[int],
    score_thresh: float = 0.0,
    samples_pkl: str = None,
    noise_trans_std: float = 0.0,
    noise_rot_std: float = 0.0,
    noise_seed: int = None
) -> str:
    print("\n" + "="*80)
    print("STEP 1: Running StreamMapNet Inference")
    print("="*80)
    
    cfg = Config.fromfile(config_path)
    
    if samples_pkl is not None:
        cfg.data.test.ann_file = samples_pkl
        print(f"Overriding dataset annotation file to: {samples_pkl}")
    
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
            
        # Patch pipeline to ensure we have extrinsics/intrinsics for noise injection
        for transform in cfg.data.test.pipeline:
            if transform['type'] == 'Collect3D':
                if 'meta_keys' in transform:
                    meta_keys = list(transform['meta_keys'])
                    for key in ['cam_extrinsics', 'cam_intrinsics', 'lidar2img', 'lidar2ego_translation', 'lidar2ego_rotation']:
                        if key not in meta_keys:
                            meta_keys.append(key)
                    transform['meta_keys'] = tuple(meta_keys)
                    print(f"DEBUG: Patched Collect3D meta_keys: {transform['meta_keys']}")
    
    logger = get_root_logger()
    logger.info('Building dataset...')
    
    patch_nusc_dataset(cfg, logger)
    dataset = build_dataset(cfg.data.test)
    
    # Path fixing logic
    data_root = cfg.data.test.get('data_root', '')
    if not data_root:
        if hasattr(dataset, 'data_root'):
            data_root = dataset.data_root
        elif hasattr(dataset, 'nusc') and hasattr(dataset.nusc, 'dataroot'):
            data_root = dataset.nusc.dataroot
            
    if data_root:
        from plugin.datasets.nusc_dataset import NuscDataset
        original_get_sample = NuscDataset.get_sample
        def patched_get_sample_with_data_root(self, idx):
            input_dict = original_get_sample(self, idx)
            fixed_img_filenames = []
            for img_path in input_dict['img_filenames']:
                if img_path:
                    img_path_abs = os.path.abspath(img_path) if not os.path.isabs(img_path) else img_path
                    if not os.path.exists(img_path_abs):
                        img_path_clean = img_path.lstrip('./')
                        if 'nuscenes/' in img_path_clean:
                            parts = img_path_clean.split('nuscenes/', 1)
                            if len(parts) > 1:
                                fixed_path = os.path.join(data_root, parts[1])
                                if os.path.exists(fixed_path):
                                    fixed_img_filenames.append(fixed_path)
                                    continue
                        if 'samples/' in img_path_clean:
                            parts = img_path_clean.split('samples/', 1)
                            if len(parts) > 1:
                                fixed_path = os.path.join(data_root, 'samples', parts[1])
                                if os.path.exists(fixed_path):
                                    fixed_img_filenames.append(fixed_path)
                                    continue
                        fixed_path = os.path.join(data_root, img_path_clean)
                        if os.path.exists(fixed_path):
                            fixed_img_filenames.append(fixed_path)
                        else:
                            fixed_img_filenames.append(img_path)
                    else:
                        fixed_img_filenames.append(img_path_abs)
                else:
                    fixed_img_filenames.append(img_path)
            input_dict['img_filenames'] = fixed_img_filenames
            return input_dict
        NuscDataset.get_sample = patched_get_sample_with_data_root
        logger.info(f'Patched NuscDataset.get_sample to fix image paths with data_root: {data_root}')

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info(f'Built dataset with {len(dataset)} samples')
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    logger.info(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    logger.info('Model loaded and ready')
    
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    logger.info(f'\nCamera configuration:')
    logger.info(f'  Active cameras ({len(camera_indices)}/6): {", ".join(camera_names)}')
    
    predictions = {}
    use_streaming = is_streaming_model(model)
    
    if use_streaming:
        logger.info('\n' + '='*80)
        logger.info('⚠️  STREAMING MODEL DETECTED')
        logger.info('='*80)
        scenes = group_samples_by_scene(dataset, logger)
        total_samples = sum(len(samples) for samples in scenes.values())
        prog_bar = mmcv.ProgressBar(total_samples)
        
        global_frame_idx = 0
        
        for scene_idx, (scene_token, scene_samples) in enumerate(scenes.items()):
            logger.info(f'\nScene {scene_idx+1}/{len(scenes)}: {scene_token} ({len(scene_samples)} samples)')
            reset_bev_memory(model, logger)
            
            for timestamp, dataset_idx, sample_info in scene_samples:
                # Manual data handling for streaming
                data = dataset[dataset_idx]
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].unsqueeze(0).cuda()
                    elif hasattr(data[key], 'data'):
                        if isinstance(data[key].data, list):
                            data[key].data = [d.unsqueeze(0).cuda() if isinstance(d, torch.Tensor) else d for d in data[key].data]
                        elif isinstance(data[key].data, torch.Tensor):
                            data[key].data = data[key].data.unsqueeze(0).cuda()
                
                try:
                    if hasattr(data['img_metas'], 'data'):
                        img_metas = data['img_metas'].data[0]
                    else:
                        img_metas = [data['img_metas']]

                    # ADD NOISE (Streaming loop)
                    if noise_trans_std > 0 or noise_rot_std > 0:
                        # Unwrap unwrapped img_metas if needed
                        # In this streaming loop, img_metas is usually a list of 1 dict because we fetched data[dataset_idx] (single sample)
                        # and then wrapped in list.
                        
                        img_metas = add_noise_to_camera_extrinsics(
                            img_metas,
                            noise_trans_std=noise_trans_std,
                            noise_rot_std=noise_rot_std,
                            active_camera_indices=camera_indices,
                            seed=noise_seed + global_frame_idx if noise_seed is not None else None,
                            logger=logger if global_frame_idx == 0 else None
                        )
                        
                        # CRITICAL: Update the tensors in 'data' because the model uses them directly!
                        # Update img_metas in data container
                        if hasattr(data['img_metas'], 'data'):
                             data['img_metas'].data[0] = img_metas
                        
                        # Update lidar2img tensor if present
                        if 'lidar2img' in data:
                            l2i_list = [m['lidar2img'] for m in img_metas]
                            # l2i_list is list of lists of 4x4 arrays.
                            # data['lidar2img'] is likely [1, N_views, 4, 4]
                            
                            new_l2i_tensor = torch.tensor(l2i_list).to(data['lidar2img'].device)
                            if len(new_l2i_tensor.shape) == 3: # [N_views, 4, 4]
                                new_l2i_tensor = new_l2i_tensor.unsqueeze(0)
                            
                            data['lidar2img'] = new_l2i_tensor
                            
                        # Update ego2img tensor if present (StreamMapNet often uses this)
                        if 'ego2img' in data:
                            e2i_list = [m['ego2img'] for m in img_metas]
                            new_e2i_tensor = torch.tensor(e2i_list).to(data['ego2img'].device)
                            if len(new_e2i_tensor.shape) == 3:
                                new_e2i_tensor = new_e2i_tensor.unsqueeze(0)
                            data['ego2img'] = new_e2i_tensor
                    
                    if 'token' in img_metas[0]:
                        sample_token = img_metas[0]['token']
                    elif 'sample_idx' in img_metas[0]:
                        sample_token = str(img_metas[0]['sample_idx'])
                    else:
                        sample_token = sample_info.get('token', f'sample_{dataset_idx}')
                    
                    # Zero out inactive cameras
                    if len(camera_indices) < 6 and 'img' in data:
                        if hasattr(data['img'], 'data'):
                            imgs = data['img'].data[0]
                        else:
                            imgs = data['img']
                        
                        if len(imgs.shape) == 5:
                            for view_idx in range(imgs.shape[1]):
                                if view_idx not in camera_indices:
                                    imgs[:, view_idx, :, :, :] = 0
                        elif len(imgs.shape) == 4:
                            for view_idx in range(imgs.shape[0]):
                                if view_idx not in camera_indices:
                                    imgs[view_idx, :, :, :] = 0
                                    
                    with torch.no_grad():
                        result = model(return_loss=False, rescale=True, **data)
                    
                    # Extract predictions
                    result_item = result[0]
                    if isinstance(result_item, dict):
                        if 'vectors' in result_item:
                            pred_vectors = result_item['vectors']
                            pred_scores = result_item['scores']
                            pred_labels = result_item['labels']
                        elif 'pts_bbox' in result_item:
                            result_dic = result_item['pts_bbox']
                            pred_vectors = result_dic.get('pts_3d')
                            pred_scores = result_dic.get('scores_3d')
                            pred_labels = result_dic.get('labels_3d')
                            if torch.is_tensor(pred_vectors):
                                pred_vectors = pred_vectors.cpu().numpy()
                            if torch.is_tensor(pred_scores):
                                pred_scores = pred_scores.cpu().numpy()
                            if torch.is_tensor(pred_labels):
                                pred_labels = pred_labels.cpu().numpy()
                    
                    # Denormalize
                    if len(pred_scores) > 0:
                        keep = pred_scores > score_thresh
                        pred_vectors = pred_vectors[keep]
                        pred_labels = pred_labels[keep]
                        pred_scores = pred_scores[keep]
                    else:
                        pred_vectors = np.array([]).reshape(0, 20, 2) if len(pred_vectors.shape) == 3 else np.array([])
                        pred_labels = np.array([])
                        pred_scores = np.array([])

                    if len(pred_vectors) > 0 and pred_vectors.shape[-1] == 2:
                        # Assuming roi_size for now if pc_range not explicit
                        roi_size = (60.0, 30.0)
                        if hasattr(cfg, 'roi_size'): roi_size = cfg.roi_size
                        
                        pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
                        x_min, y_min = pc_range[0], pc_range[1]
                        x_max, y_max = pc_range[3], pc_range[4]
                        
                        pred_vectors = pred_vectors.copy()
                        pred_vectors[..., 0] = pred_vectors[..., 0] * (x_max - x_min) + x_min
                        pred_vectors[..., 1] = pred_vectors[..., 1] * (y_max - y_min) + y_min
                        
                        # Coordinate transform: Ego -> Lidar
                        lidar2ego_rot = sample_info.get('lidar2ego_rotation')
                        lidar2ego_trans = sample_info.get('lidar2ego_translation')
                        
                        if lidar2ego_rot is not None:
                            l2e_r = Quaternion(lidar2ego_rot).rotation_matrix
                            l2e_t = np.array(lidar2ego_trans)
                            l2e_mat = np.eye(4)
                            l2e_mat[:3, :3] = l2e_r
                            l2e_mat[:3, 3] = l2e_t
                            e2l_mat = np.linalg.inv(l2e_mat)
                            
                            original_shape = pred_vectors.shape
                            preds_flat = pred_vectors.reshape(-1, 2)
                            preds_homo = np.zeros((preds_flat.shape[0], 4))
                            preds_homo[:, 0] = preds_flat[:, 0]
                            preds_homo[:, 1] = preds_flat[:, 1]
                            preds_homo[:, 2] = 0.0; preds_homo[:, 3] = 1.0
                            preds_lidar = (e2l_mat @ preds_homo.T).T
                            pred_vectors = preds_lidar[:, :2].reshape(original_shape)
                        else:
                            # Fallback rotation
                            pred_vectors_rotated = pred_vectors.copy()
                            pred_vectors_rotated[..., 0] = -pred_vectors[..., 1]
                            pred_vectors_rotated[..., 1] = pred_vectors[..., 0]
                            pred_vectors = pred_vectors_rotated
                    
                    predictions[sample_token] = {
                        'vectors': pred_vectors,
                        'labels': pred_labels,
                        'scores': pred_scores
                    }
                    
                except Exception as e:
                    logger.warning(f'Error processing sample {dataset_idx}: {str(e)}')
                
                prog_bar.update()
                global_frame_idx += 1
    else:
        # Standard loop
        logger.info('\nRunning inference (frame-independent)...')
        prog_bar = mmcv.ProgressBar(len(dataset))
        for i, data in enumerate(data_loader):
            try:
                img_metas = data['img_metas'].data[0]
                
                # Extract sample token
                # Extract sample token
                if 'token' in img_metas[0]:
                    sample_token = img_metas[0]['token']
                elif 'sample_idx' in img_metas[0]:
                    sample_token = str(img_metas[0]['sample_idx'])
                else:
                    if hasattr(dataset, 'samples') and i < len(dataset.samples):
                        sample_token = dataset.samples[i].get('token', f'sample_{i}')
                    else:
                        sample_token = f'sample_{i}'
                
                # ADD NOISE
                if noise_trans_std > 0 or noise_rot_std > 0:
                     img_metas = add_noise_to_camera_extrinsics(
                        img_metas if isinstance(img_metas, list) else [img_metas],
                        noise_trans_std=noise_trans_std,
                        noise_rot_std=noise_rot_std,
                        active_camera_indices=camera_indices,
                        seed=noise_seed + i if noise_seed is not None else None,
                        logger=logger if i == 0 else None
                    )
                     data['img_metas'].data[0] = img_metas
                     
                     # CRITICAL: Update the tensors in 'data' dictionary as well
                     # Update lidar2img tensor if present
                     if 'lidar2img' in data:
                        l2i_list = [m['lidar2img'] for m in img_metas]
                        new_l2i_tensor = torch.tensor(l2i_list).to(data['lidar2img'].data[0].device if hasattr(data['lidar2img'], 'data') else data['lidar2img'].device)
                        if len(new_l2i_tensor.shape) == 3: 
                            new_l2i_tensor = new_l2i_tensor.unsqueeze(0)
                        
                        if hasattr(data['lidar2img'], 'data'):
                             data['lidar2img'].data[0] = new_l2i_tensor
                        else:
                             data['lidar2img'] = new_l2i_tensor

                     # Update ego2img tensor if present
                     if 'ego2img' in data:
                        e2i_list = [m['ego2img'] for m in img_metas]
                        new_e2i_tensor = torch.tensor(e2i_list).to(data['ego2img'].data[0].device if hasattr(data['ego2img'], 'data') else data['ego2img'].device)
                        if len(new_e2i_tensor.shape) == 3:
                            new_e2i_tensor = new_e2i_tensor.unsqueeze(0)
                            
                        if hasattr(data['ego2img'], 'data'):
                             data['ego2img'].data[0] = new_e2i_tensor
                        else:
                             data['ego2img'] = new_e2i_tensor
                
                # Run inference
                # Zero out inactive cameras
                if len(camera_indices) < 6 and 'img' in data:
                    if hasattr(data['img'], 'data'):
                        imgs = data['img'].data[0]
                    else:
                         imgs = data['img'][0].data[0]
                    
                    if len(imgs.shape) == 5:
                        for view_idx in range(imgs.shape[1]):
                            if view_idx not in camera_indices:
                                imgs[:, view_idx, :, :, :] = 0
                    elif len(imgs.shape) == 4:
                        for view_idx in range(imgs.shape[0]):
                            if view_idx not in camera_indices:
                                imgs[view_idx, :, :, :] = 0

                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)

                # Extract predictions
                result_item = result[0]
                if isinstance(result_item, dict):
                    if 'vectors' in result_item:
                        pred_vectors = result_item['vectors']
                        pred_scores = result_item['scores']
                        pred_labels = result_item['labels']
                    elif 'pts_bbox' in result_item:
                        result_dic = result_item['pts_bbox']
                        pred_vectors = result_dic.get('pts_3d')
                        pred_scores = result_dic.get('scores_3d')
                        pred_labels = result_dic.get('labels_3d')
                        if torch.is_tensor(pred_vectors):
                            pred_vectors = pred_vectors.cpu().numpy()
                        if torch.is_tensor(pred_scores):
                            pred_scores = pred_scores.cpu().numpy()
                        if torch.is_tensor(pred_labels):
                            pred_labels = pred_labels.cpu().numpy()
                
                # Denormalize
                if len(pred_scores) > 0:
                    keep = pred_scores > score_thresh
                    pred_vectors = pred_vectors[keep]
                    pred_labels = pred_labels[keep]
                    pred_scores = pred_scores[keep]
                else:
                    pred_vectors = np.array([]).reshape(0, 20, 2) if len(pred_vectors.shape) == 3 else np.array([])
                    pred_labels = np.array([])
                    pred_scores = np.array([])

                if len(pred_vectors) > 0 and pred_vectors.shape[-1] == 2:
                    # Assuming roi_size for now if pc_range not explicit
                    roi_size = (60.0, 30.0)
                    if hasattr(cfg, 'roi_size'): roi_size = cfg.roi_size
                    
                    pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
                    x_min, y_min = pc_range[0], pc_range[1]
                    x_max, y_max = pc_range[3], pc_range[4]
                    
                    pred_vectors = pred_vectors.copy()
                    pred_vectors[..., 0] = pred_vectors[..., 0] * (x_max - x_min) + x_min
                    pred_vectors[..., 1] = pred_vectors[..., 1] * (y_max - y_min) + y_min
                    
                    # Coordinate transform: Ego -> Lidar
                    # Try to find sample info in dataset
                    sample_info = None
                    if hasattr(dataset, 'samples'):
                        # Optimization: Use direct lookup if possible, or iterative
                        # Since i corresponds to dataset index here in standard loop
                        if i < len(dataset.samples) and dataset.samples[i].get('token') == sample_token:
                             sample_info = dataset.samples[i]
                        else:
                            for s in dataset.samples:
                                if s.get('token') == sample_token:
                                    sample_info = s
                                    break
                    
                    lidar2ego_rot = sample_info.get('lidar2ego_rotation') if sample_info else None
                    lidar2ego_trans = sample_info.get('lidar2ego_translation') if sample_info else None
                    
                    if lidar2ego_rot is not None:
                        l2e_r = Quaternion(lidar2ego_rot).rotation_matrix
                        l2e_t = np.array(lidar2ego_trans)
                        l2e_mat = np.eye(4)
                        l2e_mat[:3, :3] = l2e_r
                        l2e_mat[:3, 3] = l2e_t
                        e2l_mat = np.linalg.inv(l2e_mat)
                        
                        original_shape = pred_vectors.shape
                        preds_flat = pred_vectors.reshape(-1, 2)
                        preds_homo = np.zeros((preds_flat.shape[0], 4))
                        preds_homo[:, 0] = preds_flat[:, 0]
                        preds_homo[:, 1] = preds_flat[:, 1]
                        preds_homo[:, 2] = 0.0; preds_homo[:, 3] = 1.0
                        preds_lidar = (e2l_mat @ preds_homo.T).T
                        pred_vectors = preds_lidar[:, :2].reshape(original_shape)
                    else:
                        # Fallback rotation
                        pred_vectors_rotated = pred_vectors.copy()
                        pred_vectors_rotated[..., 0] = -pred_vectors[..., 1]
                        pred_vectors_rotated[..., 1] = pred_vectors[..., 0]
                        pred_vectors = pred_vectors_rotated
                
                predictions[sample_token] = {
                    'vectors': pred_vectors,
                    'labels': pred_labels,
                    'scores': pred_scores
                }
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                import traceback
                traceback.print_exc()
            prog_bar.update()
            
    logger.info(f'\nSaving {len(predictions)} predictions to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    logger.info('✓ Inference complete!')
    return output_pkl

# ==================== EVALUATION CODE ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    EXACT COPY from evaluate_with_fov_clipping_standalone.py
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 100,
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None,
        num_workers: int = 1
    ):
        """
        Args:
            nuscenes_data_path: Path to NuScenes dataset
            pc_range: BEV range [-x, -y, -z, x, y, z]
            num_sample_pts: Number of points to resample vectors to (MUST match training: 100)
            thresholds_chamfer: Chamfer distance thresholds (MapTR uses [0.5, 1.0, 1.5])
            camera_names: List of camera names to evaluate
        """
        self.nuscenes_data_path = nuscenes_data_path
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.num_sample_pts: int = num_sample_pts
        self.thresholds_chamfer = thresholds_chamfer or [0.5, 1.0, 1.5]
        self.camera_names = camera_names or ['CAM_FRONT']
        self.num_workers = num_workers # NEW: Store num_workers
        
        # Calculate patch size from pc_range
        self.patch_size = (self.pc_range[4] - self.pc_range[1], self.pc_range[3] - self.pc_range[0])
        
        # Accumulators
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.predictions_per_camera = {cam: [] for cam in self.camera_names}
        self.ground_truths_per_camera = {cam: [] for cam in self.camera_names}
        self.num_samples_processed = 0
    
    def resample_vector_linestring(self, vector: np.ndarray, num_sample: int) -> np.ndarray:
        """
        Resample a vector to fixed number of points using LineString interpolation.
        EXACT match to MapTR's implementation.
        """
        if len(vector) < 2:
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        
        return sampled_points
    
    def process_gt_with_fov_clipping(
        self,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and process GT vectors for a specific camera.
        Uses shared extract_gt_with_fov_clipping() for 100% identical logic.
        """
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        if len(vectors) == 0:
            return np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(gt_labels)
    
    def process_predictions_with_fov_clipping_and_rotation(
        self,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply optional FOV clipping AND camera-centric rotation to predictions.
        Uses shared process_predictions_with_fov_clipping() for 100% identical logic.
        """
        if len(pred_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        vectors, labels, scores = process_predictions_with_fov_clipping(
            pred_vectors=pred_vectors,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            apply_clipping=apply_clipping
        )
        
        if len(vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(labels), np.array(scores)
    
    def compute_chamfer_distance_matrix_maptr_official(self,
                                                        pred_vectors: np.ndarray,
                                                        gt_vectors: np.ndarray,
                                                        linewidth: float = 2.0,
                                                        pred_geometries: List = None,
                                                        gt_geometries: List = None) -> np.ndarray:
        """
        Compute Chamfer Distance matrix using EXACT MapTR official method.
        EXACT copy from MapTR's tpfp_chamfer.py:custom_polyline_score()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        
        Returns NEGATIVE CD values (higher = better match).
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        if num_preds == 0 or num_gts == 0:
            return np.full((num_preds, num_gts), -100.0)
        
        # Use pre-computed geometries if provided, otherwise create them
        if pred_geometries is None:
            pred_lines_shapely = [
                LineString(pred_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_preds)
            ]
        else:
            pred_lines_shapely = pred_geometries
        
        if gt_geometries is None:
            gt_lines_shapely = [
                LineString(gt_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_gts)
            ]
        else:
            gt_lines_shapely = gt_geometries
        
        # STRtree spatial indexing
        tree = STRtree(pred_lines_shapely)
        
        # Initialize with -100.0 for non-intersecting pairs
        cd_matrix = np.full((num_preds, num_gts), -100.0)
        
        # Compute CD only for intersecting buffered geometries
        for i, gt_line in enumerate(gt_lines_shapely):
            query_result = tree.query(gt_line)
            
            # Handle both Shapely 1.x and 2.x
            if len(query_result) > 0 and isinstance(query_result[0], (int, np.integer)):
                # Shapely 2.x: returns indices
                for pred_idx in query_result:
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
            else:
                # Shapely 1.x: returns geometries
                for pred_idx in range(num_preds):
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
        
        return cd_matrix
    
    def precompute_shapely_geometries(self,
                                     vectors: np.ndarray,
                                     linewidth: float = 2.0) -> List:
        """
        Pre-compute buffered Shapely geometries for a set of vectors.
        This is a performance optimization to avoid recomputing geometries
        for each threshold evaluation.
        
        Args:
            vectors: Array of shape (N, num_points, 2)
            linewidth: Buffer width for LineString
        
        Returns:
            List of buffered Shapely polygons
        """
        if len(vectors) == 0:
            return []
        
        geometries = [
            LineString(vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(len(vectors))
        ]
        return geometries
    
    def compute_chamfer_distance_torch(self,
                                       pred_vectors: np.ndarray,
                                       gt_vectors: np.ndarray) -> float:
        """
        Compute Chamfer Distance for monitoring (returns POSITIVE distance).
        """
        if len(pred_vectors) == 0 or len(gt_vectors) == 0:
            return float('inf')
        
        pred_points = pred_vectors.reshape(-1, 2)
        gt_points = gt_vectors.reshape(-1, 2)
        
        dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')
        
        valid_ab = dist_matrix.min(axis=1).mean()
        valid_ba = dist_matrix.min(axis=0).mean()
        
        chamfer_dist = (valid_ab + valid_ba) / 2.0
        
        return chamfer_dist
    
    def accumulate_sample(
        self,
        sample_info: Dict,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        apply_clipping: bool = True
    ):
        """
        Process one sample and accumulate results for each camera.
        """
        for camera_name in self.camera_names:
            # Process GT with optional FOV clipping
            gt_vectors, gt_labels = self.process_gt_with_fov_clipping(
                sample_info, camera_name, apply_clipping=apply_clipping)
            
            # Process predictions with optional FOV clipping
            pred_vectors_clipped, pred_labels_clipped, pred_scores_clipped = \
                self.process_predictions_with_fov_clipping_and_rotation(
                    pred_vectors, pred_labels, pred_scores, sample_info, camera_name,
                    apply_clipping=apply_clipping)
            
            # Store for this camera
            self.predictions_per_camera[camera_name].append({
                'vectors': pred_vectors_clipped,
                'labels': pred_labels_clipped,
                'scores': pred_scores_clipped
            })
            
            self.ground_truths_per_camera[camera_name].append({
                'vectors': gt_vectors,
                'labels': gt_labels
            })
        
        self.num_samples_processed += 1
    
    def match_predictions_to_gt_maptr_official(self,
                                               pred_vectors: np.ndarray,
                                               pred_scores: np.ndarray,
                                               gt_vectors: np.ndarray,
                                               threshold: float,
                                               pred_geometries: List = None,
                                               gt_geometries: List = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predictions to GT using MapTR's EXACT OFFICIAL method.
        EXACT copy from MapTR's tpfp.py:custom_tpfp_gen()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        
        if num_gts == 0:
            fp[:] = 1
            return tp, fp
        
        if num_preds == 0:
            return tp, fp
        
        # Convert threshold to NEGATIVE
        if threshold > 0:
            threshold = -threshold
        
        # Compute CD matrix (with optional pre-computed geometries)
        cd_matrix = self.compute_chamfer_distance_matrix_maptr_official(
            pred_vectors, gt_vectors, linewidth=2.0,
            pred_geometries=pred_geometries, gt_geometries=gt_geometries)
        
        # Find best matching GT for each prediction
        matrix_max = cd_matrix.max(axis=1)
        matrix_argmax = cd_matrix.argmax(axis=1)
        
        # Sort by confidence (descending)
        sort_inds = np.argsort(-pred_scores)
        
        # Track matched GTs
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # Greedy matching
        for i in sort_inds:
            if matrix_max[i] >= threshold:
                matched_gt = matrix_argmax[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        return tp, fp
    
    def compute_ap_area_based(self,
                              recalls: np.ndarray,
                              precisions: np.ndarray) -> float:
        """
        Compute Average Precision using area under PR curve.
        """
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        
        return float(ap)
    
    def compute_ap_for_class(self,
                            pred_vectors_list: List[np.ndarray],
                            pred_scores_list: List[np.ndarray],
                            gt_vectors_list: List[np.ndarray],
                            threshold: float,
                            class_name: str = "",
                            pred_geoms_list: List = None,
                            gt_geoms_list: List = None) -> Tuple[float, float]:
        """
        Compute AP and average CD for a single class at given threshold.
        
        Args:
            pred_geoms_list: Optional pre-computed prediction geometries (avoids redundant buffering)
            gt_geoms_list: Optional pre-computed GT geometries (avoids redundant buffering)
        """
        num_gts = sum(len(gts) for gts in gt_vectors_list)
        
        if num_gts == 0:
            return 0.0, float('inf')
        
        all_tp = []
        all_fp = []
        all_scores = []
        chamfer_distances_per_sample = []
        
        # Use pre-computed geometries if provided, otherwise compute them
        if pred_geoms_list is None or gt_geoms_list is None:
            pred_geoms_list = []
            gt_geoms_list = []
            
            desc = f"Pre-computing geometries ({class_name})" if class_name else "Pre-computing geometries"
            for pred_vecs, gt_vecs in tqdm(zip(pred_vectors_list, gt_vectors_list), 
                                           total=len(pred_vectors_list),
                                           desc=desc,
                                           leave=False,
                                           disable=len(pred_vectors_list) < 100):
                if len(pred_vecs) > 0:
                    pred_geoms = self.precompute_shapely_geometries(pred_vecs, linewidth=2.0)
                else:
                    pred_geoms = []
                
                if len(gt_vecs) > 0:
                    gt_geoms = self.precompute_shapely_geometries(gt_vecs, linewidth=2.0)
                else:
                    gt_geoms = []
                
                pred_geoms_list.append(pred_geoms)
                gt_geoms_list.append(gt_geoms)
        
        # Match predictions using pre-computed geometries
        iterator = zip(pred_vectors_list, pred_scores_list, gt_vectors_list, 
                      pred_geoms_list, gt_geoms_list)
        
        for pred_vecs, pred_scores, gt_vecs, pred_geoms, gt_geoms in iterator:
            if len(pred_vecs) == 0:
                continue
            
            if len(gt_vecs) == 0:
                all_tp.append(np.zeros(len(pred_vecs), dtype=np.float32))
                all_fp.append(np.ones(len(pred_vecs), dtype=np.float32))
                all_scores.append(pred_scores)
                continue
            
            # Match predictions to GT (using pre-computed geometries)
            tp, fp = self.match_predictions_to_gt_maptr_official(
                pred_vecs, pred_scores, gt_vecs, threshold,
                pred_geometries=pred_geoms, gt_geometries=gt_geoms)
            
            all_tp.append(tp)
            all_fp.append(fp)
            all_scores.append(pred_scores)
            
            # Compute chamfer distance
            cd_sample = self.compute_chamfer_distance_torch(pred_vecs, gt_vecs)
            chamfer_distances_per_sample.append(cd_sample)
        
        if len(all_tp) == 0:
            return 0.0, float('inf')
        
        # Concatenate all predictions
        all_tp = np.concatenate(all_tp)
        all_fp = np.concatenate(all_fp)
        all_scores = np.concatenate(all_scores)
        
        # Sort by confidence
        sort_inds = np.argsort(-all_scores)
        tp = all_tp[sort_inds]
        fp = all_fp[sort_inds]
        
        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        eps = np.finfo(np.float32).eps
        recalls = tp_cumsum / np.maximum(num_gts, eps)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        
        # Compute AP
        ap = self.compute_ap_area_based(recalls, precisions)
        
        # Average CD
        avg_cd = np.mean(chamfer_distances_per_sample) if chamfer_distances_per_sample else float('inf')
        
        return ap, avg_cd
    
    def evaluate(self) -> Dict:
        """
        Compute final metrics across all cameras and classes.
        """
        results = {}
        class_names = ['divider', 'ped_crossing', 'boundary']
        
        for camera_name in self.camera_names:
            camera_results = {}
            all_aps = []
            
            camera_preds = self.predictions_per_camera[camera_name]
            camera_gts = self.ground_truths_per_camera[camera_name]
            
            # PRE-COMPUTE GEOMETRIES ONCE PER CAMERA (not per class!)
            # This avoids redundant buffering: 6 cameras × 3 classes × 3 thresholds = 54 runs
            # Optimized to: 6 cameras = 6 runs (9x speedup!)
            # NOW PARALLELIZED: Use multiprocessing for additional 8x speedup
            print(f"Pre-computing geometries for {camera_name}...")
            
            # Prepare data for parallel buffering
            buffer_tasks = [(pred_data['vectors'], gt_data['vectors']) 
                           for pred_data, gt_data in zip(camera_preds, camera_gts)]
            
            # Use multiprocessing to buffer geometries in parallel
            num_workers = self.num_workers  # Use configured num_workers
            if num_workers > 1:
                # Clamp to cpu_count() to avoid excessive overhead if user requests too many
                num_workers = min(num_workers, cpu_count())
            
            if num_workers > 1:
                print(f"  Using {num_workers} parallel workers for geometry buffering...")
                with Pool(processes=num_workers) as pool:
                    geom_results = list(tqdm(
                        pool.starmap(buffer_geometries_worker, buffer_tasks),
                        total=len(buffer_tasks),
                        desc=f"Buffering geometries ({camera_name})",
                        leave=False
                    ))
                all_pred_geoms = [r[0] for r in geom_results]
                all_gt_geoms = [r[1] for r in geom_results]
            else:
                # Serial fallback
                all_pred_geoms = []
                all_gt_geoms = []
                for pred_data, gt_data in tqdm(zip(camera_preds, camera_gts),
                                              total=len(camera_preds),
                                              desc=f"Buffering geometries ({camera_name})",
                                              leave=False):
                    if len(pred_data['vectors']) > 0:
                        pred_geoms = self.precompute_shapely_geometries(pred_data['vectors'], linewidth=2.0)
                    else:
                        pred_geoms = []
                    
                    if len(gt_data['vectors']) > 0:
                        gt_geoms = self.precompute_shapely_geometries(gt_data['vectors'], linewidth=2.0)
                    else:
                        gt_geoms = []
                    
                    all_pred_geoms.append(pred_geoms)
                    all_gt_geoms.append(gt_geoms)
            
            # Evaluate each class
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Extract predictions and GT for this class
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                # Also extract the corresponding pre-computed geometries for this class
                pred_geoms_for_class = []
                gt_geoms_for_class = []
                
                for pred_data, gt_data, pred_geoms, gt_geoms in zip(camera_preds, camera_gts, 
                                                                     all_pred_geoms, all_gt_geoms):
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                    
                    # Extract geometries for this class using the same mask
                    if len(pred_geoms) > 0:
                        pred_geoms_for_class.append([pred_geoms[i] for i in range(len(pred_geoms)) if pred_mask[i]])
                    else:
                        pred_geoms_for_class.append([])
                    
                    if len(gt_geoms) > 0:
                        gt_geoms_for_class.append([gt_geoms[i] for i in range(len(gt_geoms)) if gt_mask[i]])
                    else:
                        gt_geoms_for_class.append([])
                
                # Compute AP at each threshold
                avg_cd = None
                for threshold in self.thresholds_chamfer:
                    ap, cd = self.compute_ap_for_class(
                        pred_vectors_list, pred_scores_list, gt_vectors_list, 
                        threshold, class_name=class_name,
                        pred_geoms_list=pred_geoms_for_class,
                        gt_geoms_list=gt_geoms_for_class)
                    
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                    
                    if avg_cd is None:
                        avg_cd = cd
                
                class_results['avg_chamfer_distance'] = avg_cd if avg_cd is not None else float('inf')
                
                camera_results[class_name] = class_results
            
            # Compute mAP
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        # Compute average across cameras if multiple
        if len(self.camera_names) > 1:
            avg_results = {}
            
            for class_name in class_names:
                class_avg = {}
                for threshold in self.thresholds_chamfer:
                    threshold_key = f'AP@{threshold}m'
                    aps = [results[cam][class_name][threshold_key] 
                           for cam in self.camera_names 
                           if class_name in results[cam]]
                    class_avg[threshold_key] = np.mean(aps) if aps else 0.0
                
                cds = [results[cam][class_name]['avg_chamfer_distance'] 
                      for cam in self.camera_names 
                      if class_name in results[cam] and results[cam][class_name]['avg_chamfer_distance'] != float('inf')]
                class_avg['avg_chamfer_distance'] = np.mean(cds) if cds else float('inf')
                
                avg_results[class_name] = class_avg
            
            all_camera_maps = [results[cam]['mAP'] for cam in self.camera_names]
            avg_results['mAP'] = np.mean(all_camera_maps) if all_camera_maps else 0.0
            
            results['AVERAGE'] = avg_results
        
        return results


# ==================== PARALLEL GEOMETRY BUFFERING WORKER ====================
def buffer_geometries_worker(pred_vectors: np.ndarray, gt_vectors: np.ndarray, linewidth: float = 2.0):
    """
    Worker function to buffer geometries for one sample in parallel.
    Returns buffered shapely geometries for predictions and GT.
    """
    from shapely.geometry import LineString
    from shapely.geometry import CAP_STYLE, JOIN_STYLE
    
    def buffer_vectors(vectors):
        if len(vectors) == 0:
            return []
        geometries = []
        for vec in vectors:
            if len(vec) >= 2:
                line = LineString(vec)
                buffered = line.buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                geometries.append(buffered)
            else:
                geometries.append(None)
        return geometries
    
    pred_geoms = buffer_vectors(pred_vectors)
    gt_geoms = buffer_vectors(gt_vectors)
    
    return pred_geoms, gt_geoms


# ==================== PARALLEL PROCESSING WORKER ====================
def process_sample_worker(
    sample_info: Dict,
    pred_data: Dict,
    nuscenes_data_path: str,
    pc_range: List[float],
    num_sample_pts: int,
    camera_names: List[str],
    apply_clipping: bool
) -> Dict:
    """
    Worker function to process a single sample in parallel.
    Returns processed GT and predictions for all cameras.
    
    This function is picklable and can be used with multiprocessing.Pool
    """
    from camera_fov_utils import extract_gt_with_fov_clipping, process_predictions_with_fov_clipping
    from shapely.geometry import LineString
    import numpy as np
    
    def resample_vector(vector: np.ndarray, num_sample: int) -> np.ndarray:
        """Resample vector to fixed number of points"""
        if len(vector) < 2:
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        return sampled_points
    
    results_per_camera = {}
    
    for camera_name in camera_names:
        # Process GT
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=nuscenes_data_path,
            pc_range=pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        gt_vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        # Resample GT to num_sample_pts
        if len(gt_vectors) > 0:
            final_gt_vectors = []
            for vector in gt_vectors:
                if len(vector) >= 2:
                    resampled_vec = resample_vector(vector, num_sample_pts)
                    final_gt_vectors.append(resampled_vec)
            gt_vectors_array = np.array(final_gt_vectors) if final_gt_vectors else np.array([])
            gt_labels_array = np.array(gt_labels) if final_gt_vectors else np.array([])
        else:
            gt_vectors_array = np.array([])
            gt_labels_array = np.array([])
        
        # Process predictions
        pred_vectors_processed, pred_labels_processed, pred_scores_processed = \
            process_predictions_with_fov_clipping(
                pred_vectors=pred_data['vectors'],
                pred_labels=pred_data['labels'],
                pred_scores=pred_data['scores'],
                sample_info=sample_info,
                nuscenes_path=nuscenes_data_path,
                pc_range=pc_range,
                camera_name=camera_name,
                apply_clipping=apply_clipping
            )
        
        # Resample predictions to num_sample_pts
        if len(pred_vectors_processed) > 0:
            final_pred_vectors = []
            final_pred_labels = []
            final_pred_scores = []
            for vec, label, score in zip(pred_vectors_processed, pred_labels_processed, pred_scores_processed):
                if len(vec) >= 2:
                    resampled_vec = resample_vector(vec, num_sample_pts)
                    final_pred_vectors.append(resampled_vec)
                    final_pred_labels.append(label)
                    final_pred_scores.append(score)
            
            pred_vectors_array = np.array(final_pred_vectors) if final_pred_vectors else np.array([])
            pred_labels_array = np.array(final_pred_labels) if final_pred_labels else np.array([])
            pred_scores_array = np.array(final_pred_scores) if final_pred_labels else np.array([])
        else:
            pred_vectors_array = np.array([])
            pred_labels_array = np.array([])
            pred_scores_array = np.array([])
        
        results_per_camera[camera_name] = {
            'gt': {
                'vectors': gt_vectors_array,
                'labels': gt_labels_array
            },
            'pred': {
                'vectors': pred_vectors_array,
                'labels': pred_labels_array,
                'scores': pred_scores_array
            }
        }
    
    return results_per_camera


def main():
    parser = argparse.ArgumentParser(description='Unified StreamMapNet Eval with Noise')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cameras', type=str, nargs='+', default=['CAM_FRONT'])
    parser.add_argument('--noise-type', type=str, default='rotation', choices=['rotation', 'translation'])
    parser.add_argument('--noise-std', type=float, default=0.0)
    parser.add_argument('--noise-seed', type=int, default=42)
    parser.add_argument('--noise-trans-std', type=float, default=None, help='Translation noise std (meters)')
    parser.add_argument('--noise-rot-std', type=float, default=None, help='Rotation noise std (radians)')
    
    parser.add_argument('--pc-range', type=float, nargs=6, 
                       default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                       help='Point cloud range')
    parser.add_argument('--num-sample-pts', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='eval_results_noise_stream')
    parser.add_argument('--nuscenes-path', type=str, default=None)
    parser.add_argument('--predictions-pkl', type=str, default=None, help='Override path to predictions pickle file')
    parser.add_argument('--samples-pkl', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    parser.add_argument('--skip-inference', action='store_true', help='Skip inference and use existing predictions')
    args = parser.parse_args()
    
    # Handle noise arguments
    noise_trans_std = 0.0
    noise_rot_std = 0.0
    
    if args.noise_trans_std is not None:
        noise_trans_std = args.noise_trans_std
    if args.noise_rot_std is not None:
        noise_rot_std = args.noise_rot_std
        
    if args.noise_trans_std is None and args.noise_rot_std is None:
        if args.noise_type == 'translation':
            noise_trans_std = args.noise_std
        elif args.noise_type == 'rotation':
            noise_rot_std = args.noise_std
    
    camera_indices = parse_camera_config(args.cameras)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames based on config
    camera_suffix = "_".join(args.cameras).lower().replace("cam_", "")
    if "all" in args.cameras or len(camera_indices) == 6:
        camera_suffix = "all"
    
    noise_suffix = ""
    if noise_trans_std > 0:
        noise_suffix += f"_trans{noise_trans_std:.3f}"
    if noise_rot_std > 0:
        noise_suffix += f"_rot{noise_rot_std:.3f}"
    if noise_trans_std == 0 and noise_rot_std == 0:
        noise_suffix = "_baseline"
        
    if args.predictions_pkl:
        predictions_pkl = args.predictions_pkl
    else:
        predictions_pkl = os.path.join(args.output_dir, f"streammapnet_preds_{camera_suffix}{noise_suffix}.pkl")
    
    if not args.skip_inference:
        # If user provides a specific path but doesn't skip inference, use that path for output
        if args.predictions_pkl:
             print(f"Using provided path for output: {predictions_pkl}")
        
        run_streammapnet_inference(args.config, args.checkpoint, predictions_pkl, camera_indices, 
                                   samples_pkl=args.samples_pkl, 
                                   noise_trans_std=noise_trans_std, 
                                   noise_rot_std=noise_rot_std, 
                                   noise_seed=args.noise_seed)
    elif not os.path.exists(predictions_pkl):
        raise FileNotFoundError(f"Predictions file not found: {predictions_pkl}. Cannot skip inference.")
    
    # Evaluation
    print("\nSTEP 2: Eval")
    with open(predictions_pkl, 'rb') as f:
        predictions = pickle.load(f)

    # Remap StreamMapNet class IDs to MapTR class IDs
    # StreamMapNet: 0=ped_crossing, 1=divider, 2=boundary
    # MapTR GT: 0=divider, 1=ped_crossing, 2=boundary
    streammapnet_to_maptr = {0: 1, 1: 0, 2: 2}
    
    print(f"\nRemapping prediction class IDs from StreamMapNet to MapTR format...")
    for token, pred_data in predictions.items():
        pred_labels = pred_data['labels']
        remapped_labels = np.array([streammapnet_to_maptr[int(label)] for label in pred_labels])
        predictions[token]['labels'] = remapped_labels
    print(f"✓ Remapped class IDs for {len(predictions)} samples")
    
    cfg = Config.fromfile(args.config)
    if args.samples_pkl: cfg.data.test.ann_file = args.samples_pkl
    if args.nuscenes_path: cfg.data.test.data_root = args.nuscenes_path
    
    patch_nusc_dataset(cfg, get_root_logger())
    dataset = build_dataset(cfg.data.test)
    
    evaluator = CameraSpecificEvaluator(args.nuscenes_path or cfg.data.test.data_root, args.pc_range, 
                                        camera_names=[list(CAMERA_MAP.keys())[i] for i in camera_indices],
                                        num_sample_pts=args.num_sample_pts,
                                        num_workers=max(1, args.num_workers))
    
    # Parallelize
    process_args = []
    
    # helper to get info
    data_infos = getattr(dataset, 'data_infos', getattr(dataset, 'samples', None))
    if data_infos is None:
         # Fallback to iterating dataset if no list exposed (rare)
         print("Warning: Could not access data_infos or samples from dataset.")
         data_infos = []

    for i in range(len(dataset)):
        sample_info = data_infos[i]
        token = sample_info.get('token')
        if token in predictions:
            pred_data = predictions[token]
            process_args.append((sample_info, pred_data, evaluator.nuscenes_data_path, evaluator.pc_range, 
                                 evaluator.num_sample_pts, evaluator.camera_names, True)) # FOV clipping ON by default
    
    print(f"Using {args.num_workers} parallel workers...")
    with Pool(args.num_workers) as pool:
        results = list(tqdm(pool.starmap(process_sample_worker, process_args), total=len(process_args)))
        
    for res in results:
        for cam in evaluator.camera_names:
            evaluator.predictions_per_camera[cam].append(res[cam]['pred'])
            evaluator.ground_truths_per_camera[cam].append(res[cam]['gt'])
            
    final_results = evaluator.evaluate()
    print(json.dumps(final_results, indent=2))
    results_json = os.path.join(args.output_dir, f"streammapnet_results_{camera_suffix}{noise_suffix}.json")
    
    final_results = evaluator.evaluate()
    print(json.dumps(final_results, indent=2))
    print(f"See {results_json}")
    with open(results_json, 'w') as f:
        json.dump(final_results, f, indent=2)

if __name__ == '__main__':
    main()
