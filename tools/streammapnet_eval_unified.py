#!/usr/bin/env python3
"""
Unified StreamMapNet Evaluation Script
======================================
Runs StreamMapNet inference with configurable camera inputs AND evaluates with camera-specific FOV clipping.

This script combines:
1. save_streammapnet_predictions.py - Inference with camera masking
2. evaluate_with_fov_clipping_standalone.py - MapTR-style evaluation

Features:
- Run inference with selectable cameras (front, front+back, all 6)
- Evaluate predictions with camera-specific FOV clipping
- Compare different camera configurations
- Use EXACT MapTR official evaluation code

Usage Examples:
---------------
# Front camera only
python tools/streammapnet_eval_unified.py --cameras CAM_FRONT

# Front + Back cameras
python tools/streammapnet_eval_unified.py --cameras CAM_FRONT CAM_BACK

# All 6 cameras (baseline)
python tools/streammapnet_eval_unified.py --cameras all

# Skip inference if predictions already exist
python tools/streammapnet_eval_unified.py --skip-inference --predictions-pkl existing.pkl

# Full BEV evaluation without FOV clipping
python tools/streammapnet_eval_unified.py --no-clipping --cameras CAM_FRONT
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

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
                'scene_name': scene_name
            }
            
            return input_dict
        
        NuscDataset.get_sample = patched_get_sample
        logger.info('Applied comprehensive dataset patching for StreamMapNet')
        
        return True
    except Exception as e:
        logger.warning(f'Could not patch NuscDataset: {e}')
        return False


def patch_image_paths(cfg, logger):
    """Patch image paths after dataset creation."""
    try:
        from plugin.datasets.nusc_dataset import NuscDataset
        
        data_root = cfg.data.test.get('data_root', '')
        if data_root:
            original_get_sample = NuscDataset.get_sample
            
            def patched_get_sample_with_paths(self, idx):
                """Fix image paths to use correct data_root."""
                input_dict = original_get_sample(self, idx)
                
                # Fix image paths
                fixed_img_filenames = []
                for img_path in input_dict['img_filenames']:
                    if img_path:
                        # Remove leading ./ and 'data/nuscenes/'
                        img_path_clean = img_path.lstrip('./')
                        if 'data/nuscenes/' in img_path_clean:
                            img_path_clean = img_path_clean.split('data/nuscenes/', 1)[1]
                        
                        # Join with data_root
                        fixed_path = os.path.join(data_root, img_path_clean)
                        fixed_img_filenames.append(fixed_path)
                    else:
                        fixed_img_filenames.append(img_path)
                
                input_dict['img_filenames'] = fixed_img_filenames
                return input_dict
            
            NuscDataset.get_sample = patched_get_sample_with_paths
            logger.info(f'Applied image path fixing with data_root: {data_root}')
    except Exception as e:
        logger.warning(f'Could not patch image paths: {e}')


# ==================== INFERENCE CODE ====================
def run_streammapnet_inference(
    config_path: str,
    checkpoint_path: str,
    output_pkl: str,
    camera_indices: List[int],
    score_thresh: float = 0.0,
    samples_pkl: str = None
) -> str:
    """
    Run StreamMapNet inference with specified camera configuration.
    Includes StreamMapNet-specific denormalization and coordinate rotation.
    
    Note: Does NOT override cfg.data.test.data_root to avoid path doubling issues.
    The annotation file paths may already be absolute, or they will be relative
    to the config's data_root setting.
    """
    print("\n" + "="*80)
    print("STEP 1: Running StreamMapNet Inference")
    print("="*80)
    
    cfg = Config.fromfile(config_path)
    
    # Note: We do NOT override data_root here because the annotation file paths may already be absolute
    # If they are relative, they will be relative to the config's data_root setting
    # Overriding samples_pkl is still needed for --samples-pkl argument
    if samples_pkl is not None:
        cfg.data.test.ann_file = samples_pkl
        print(f"Overriding dataset annotation file to: {samples_pkl}")
    
    # Import plugin modules
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
    
    # Setup CUDA
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    
    # Setup logger
    logger = get_root_logger()
    logger.info('Building dataset...')
    
    # Apply dataset patches
    patch_nusc_dataset(cfg, logger)
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # Patch image paths to use correct data_root if needed (CRITICAL - from original script)
    # Get data_root from dataset config
    data_root = cfg.data.test.get('data_root', '')
    if not data_root:
        # Try to get from dataset object
        if hasattr(dataset, 'data_root'):
            data_root = dataset.data_root
        elif hasattr(dataset, 'nusc') and hasattr(dataset.nusc, 'dataroot'):
            data_root = dataset.nusc.dataroot
    
    if data_root:
        from plugin.datasets.nusc_dataset import NuscDataset
        original_get_sample = NuscDataset.get_sample
        
        def patched_get_sample_with_data_root(self, idx):
            """Patched version that fixes image paths with correct data_root."""
            input_dict = original_get_sample(self, idx)
            
            # Fix image paths to use correct data_root
            fixed_img_filenames = []
            for img_path in input_dict['img_filenames']:
                if img_path:
                    # Normalize the path
                    img_path_abs = os.path.abspath(img_path) if not os.path.isabs(img_path) else img_path
                    
                    # If path doesn't exist, try to fix it using data_root
                    if not os.path.exists(img_path_abs):
                        # Remove leading ./ or relative path components
                        img_path_clean = img_path.lstrip('./')
                        
                        # Check if it contains 'nuscenes/' to extract the relative part
                        if 'nuscenes/' in img_path_clean:
                            # Extract the part after 'nuscenes/'
                            parts = img_path_clean.split('nuscenes/', 1)
                            if len(parts) > 1:
                                # Reconstruct with correct data_root
                                fixed_path = os.path.join(data_root, parts[1])
                                if os.path.exists(fixed_path):
                                    fixed_img_filenames.append(fixed_path)
                                    continue
                        
                        # Try joining with data_root/samples/ directly
                        # Path format: ./data/nuscenes/samples/CAM_FRONT/...
                        # We want: data_root/samples/CAM_FRONT/...
                        if 'samples/' in img_path_clean:
                            parts = img_path_clean.split('samples/', 1)
                            if len(parts) > 1:
                                fixed_path = os.path.join(data_root, 'samples', parts[1])
                                if os.path.exists(fixed_path):
                                    fixed_img_filenames.append(fixed_path)
                                    continue
                        
                        # Last resort: try joining with data_root directly
                        fixed_path = os.path.join(data_root, img_path_clean)
                        if os.path.exists(fixed_path):
                            fixed_img_filenames.append(fixed_path)
                        else:
                            # Keep original path if fixed path doesn't exist
                            logger.warning(f"Could not find image at {img_path} or {fixed_path}")
                            fixed_img_filenames.append(img_path)
                    else:
                        # Path exists, use it as-is
                        fixed_img_filenames.append(img_path_abs)
                else:
                    fixed_img_filenames.append(img_path)
            
            input_dict['img_filenames'] = fixed_img_filenames
            return input_dict
        
        NuscDataset.get_sample = patched_get_sample_with_data_root
        logger.info(f'Patched NuscDataset.get_sample to fix image paths with data_root: {data_root}')
    
    # Note: We do NOT call patch_image_paths() here because it causes path doubling
    # The comprehensive path-fixing logic above handles all path scenarios correctly
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info(f'Built dataset with {len(dataset)} samples')
    
    # Build model
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
    
    # Print camera configuration
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    logger.info(f'\nCamera configuration:')
    logger.info(f'  Active cameras ({len(camera_indices)}/6): {", ".join(camera_names)}')
    if len(camera_indices) < 6:
        inactive_names = [name for name, idx in CAMERA_MAP.items() if idx not in camera_indices]
        logger.info(f'  Inactive cameras (zeroed out): {", ".join(inactive_names)}')
    
    # Note: pc_range will be determined inside the loop for flexibility
    # This allows fallback to roi_size if pc_range is not directly available
    
    # Storage for predictions
    predictions = {}
    
    # Run inference
    logger.info('\nRunning inference...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        try:
            # Handle DataContainer: data['img_metas'] is a DataContainer
            # data['img_metas'].data[0] is the batch, [0] is the first item in batch
            img_metas = data['img_metas'].data[0]
            
            # Get sample token - use the actual NuScenes sample_token if available
            # Debug: print available keys on first sample
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {list(img_metas[0].keys())}")
            
            # Try to get token from img_metas (StreamMapNet format)
            if 'token' in img_metas[0]:
                sample_token = img_metas[0]['token']
            elif 'sample_idx' in img_metas[0]:
                # sample_idx might be the token in some formats
                sample_token = str(img_metas[0]['sample_idx'])
            else:
                # Fallback: try to get from dataset directly
                if hasattr(dataset, 'samples') and i < len(dataset.samples):
                    sample_token = dataset.samples[i].get('token', f'sample_{i}')
                else:
                    # Last resort: use index
                    sample_token = f'sample_{i}'
                    logger.warning(f"Could not find token for sample {i}, using {sample_token}")
            
            if i == 0:
                logger.info(f"DEBUG: Using sample_token: {sample_token}")
                logger.info(f"DEBUG: img_metas[0] has token: {'token' in img_metas[0]}")
                if 'token' in img_metas[0]:
                    logger.info(f"DEBUG: token value: {img_metas[0]['token']}")
            
            # Zero out inactive cameras using in-place modification (matches original approach)
            # NuScenes camera order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
            #                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
            if len(camera_indices) < 6 and 'img' in data:
                # Handle DataContainer: data['img'] is a DataContainer
                if hasattr(data['img'], 'data'):
                    imgs = data['img'].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                else:
                    imgs = data['img'][0].data[0]
                
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                    logger.info(f"DEBUG: Keeping camera indices {camera_indices}, zeroing out others")
                
                # Zero out inactive cameras by setting them to zero in-place
                # This matches the original approach but generalized for any camera subset
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    for view_idx in range(imgs.shape[1]):
                        if view_idx not in camera_indices:
                            imgs[:, view_idx, :, :, :] = 0
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    for view_idx in range(imgs.shape[0]):
                        if view_idx not in camera_indices:
                            imgs[view_idx, :, :, :] = 0
                else:
                    logger.warning(f"Unexpected image tensor shape: {imgs.shape}")
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Debug: print result structure on first sample
            if i == 0:
                logger.info(f"DEBUG: result type: {type(result)}")
                logger.info(f"DEBUG: result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    logger.info(f"DEBUG: result[0] type: {type(result[0])}")
                    logger.info(f"DEBUG: result[0] keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict'}")
            
            # Extract predictions from StreamMapNet result format
            # StreamMapNet's post_process returns: [{'vectors': np.array, 'scores': np.array, 'labels': np.array, 'token': str}]
            # Since batch_size=1, result[0] is the dict for the first (and only) sample
            if isinstance(result, (list, tuple)) and len(result) > 0:
                result_item = result[0]
                if isinstance(result_item, dict):
                    # StreamMapNet format: {'vectors': np.array(shape=[num_preds, num_points, 2]),
                    #                      'scores': np.array(shape=[num_preds]),
                    #                      'labels': np.array(shape=[num_preds])}
                    if 'vectors' in result_item:
                        pred_vectors = result_item['vectors']  # Already numpy array
                        pred_scores = result_item['scores']    # Already numpy array
                        pred_labels = result_item['labels']    # Already numpy array
                    # MapTR format: {'pts_bbox': {'pts_3d': ..., 'scores_3d': ..., 'labels_3d': ...}}
                    elif 'pts_bbox' in result_item:
                        result_dic = result_item['pts_bbox']
                        pred_vectors = result_dic.get('pts_3d')
                        pred_scores = result_dic.get('scores_3d')
                        pred_labels = result_dic.get('labels_3d')
                        # Convert to numpy if tensors
                        if torch.is_tensor(pred_vectors):
                            pred_vectors = pred_vectors.cpu().numpy()
                        if torch.is_tensor(pred_scores):
                            pred_scores = pred_scores.cpu().numpy()
                        if torch.is_tensor(pred_labels):
                            pred_labels = pred_labels.cpu().numpy()
                    else:
                        # Debug: show what we have
                        if i == 0:
                            logger.error(f"DEBUG: Unknown result structure. Available keys: {list(result_item.keys())}")
                        raise KeyError(f"Could not find prediction data in result. Available keys: {list(result_item.keys())}")
                else:
                    raise TypeError(f"result[0] is not a dict, got {type(result_item)}")
            else:
                raise TypeError(f"result is not a list/tuple or is empty, got {type(result)}")
            
            # Ensure numpy arrays
            if not isinstance(pred_vectors, np.ndarray):
                pred_vectors = np.array(pred_vectors)
            if not isinstance(pred_scores, np.ndarray):
                pred_scores = np.array(pred_scores)
            if not isinstance(pred_labels, np.ndarray):
                pred_labels = np.array(pred_labels)
            
            # Filter by score threshold
            if len(pred_scores) > 0:
                keep = pred_scores > score_thresh
                pred_vectors = pred_vectors[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
            else:
                # Empty predictions
                pred_vectors = np.array([]).reshape(0, 20, 2) if len(pred_vectors.shape) == 3 else np.array([])
                pred_labels = np.array([])
                pred_scores = np.array([])
            
            # Denormalize coordinates from [0, 1] to world coordinates (meters)
            # StreamMapNet outputs normalized coordinates, need to convert back using pc_range
            # Formula: world_coord = normalized_coord * (max - min) + min
            # NOTE: MapTR denormalizes in bbox_coder.decode(), so MapTR saves denormalized results
            # StreamMapNet does NOT denormalize in post_process(), so we need to do it here
            if len(pred_vectors) > 0 and pred_vectors.shape[-1] == 2:
                # Get pc_range from config (it's defined as a variable, not a dict key)
                # Try multiple ways to get it
                if hasattr(cfg, 'pc_range'):
                    pc_range = cfg.pc_range
                elif 'pc_range' in cfg:
                    pc_range = cfg['pc_range']
                else:
                    # Default: calculate from roi_size if available
                    if hasattr(cfg, 'roi_size'):
                        roi_size = cfg.roi_size
                        pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]
                    else:
                        # Fallback to default
                        pc_range = [-30.0, -15.0, -3.0, 30.0, 15.0, 5.0]
                
                if len(pc_range) >= 6:
                    x_min, y_min = pc_range[0], pc_range[1]
                    x_max, y_max = pc_range[3], pc_range[4]
                    
                    if i == 0:
                        logger.info(f"DEBUG: Before denormalization - vectors shape: {pred_vectors.shape}")
                        logger.info(f"DEBUG: Before denormalization - X range: [{pred_vectors[..., 0].min():.4f}, {pred_vectors[..., 0].max():.4f}], "
                                   f"Y range: [{pred_vectors[..., 1].min():.4f}, {pred_vectors[..., 1].max():.4f}]")
                        logger.info(f"DEBUG: First normalized vector: {pred_vectors[0] if len(pred_vectors) > 0 else 'Empty'}")
                        logger.info(f"DEBUG: pc_range: {pc_range}, using x_range=[{x_min}, {x_max}], y_range=[{y_min}, {y_max}]")
                    
                    # Denormalize: world = normalized * (max - min) + min
                    pred_vectors = pred_vectors.copy()  # Avoid modifying original
                    pred_vectors[..., 0] = pred_vectors[..., 0] * (x_max - x_min) + x_min
                    pred_vectors[..., 1] = pred_vectors[..., 1] * (y_max - y_min) + y_min
                    
                    if i == 0:
                        logger.info(f"DEBUG: After denormalization - X range: [{pred_vectors[..., 0].min():.2f}, {pred_vectors[..., 0].max():.2f}], "
                                   f"Y range: [{pred_vectors[..., 1].min():.2f}, {pred_vectors[..., 1].max():.2f}]")
                        logger.info(f"DEBUG: First denormalized vector: {pred_vectors[0] if len(pred_vectors) > 0 else 'Empty'}")
                    
                    # Apply 90-degree counterclockwise rotation to align with MapTR coordinate system
                    # StreamMapNet uses: X ∈ [-30, 30], Y ∈ [-15, 15]
                    # MapTR uses: X ∈ [-15, 15], Y ∈ [-30, 30]
                    # Rotation: (x, y) -> (-y, x)
                    pred_vectors_rotated = pred_vectors.copy()
                    pred_vectors_rotated[..., 0] = -pred_vectors[..., 1]  # new_x = -old_y
                    pred_vectors_rotated[..., 1] = pred_vectors[..., 0]   # new_y = old_x
                    pred_vectors = pred_vectors_rotated
                    
                    if i == 0:
                        logger.info(f"DEBUG: After 90° rotation to MapTR coords - X range: [{pred_vectors[..., 0].min():.2f}, {pred_vectors[..., 0].max():.2f}], "
                                   f"Y range: [{pred_vectors[..., 1].min():.2f}, {pred_vectors[..., 1].max():.2f}]")
                        logger.info(f"DEBUG: First rotated vector: {pred_vectors[0] if len(pred_vectors) > 0 else 'Empty'}")
                        if len(pred_scores) > 0:
                            logger.info(f"DEBUG: Number of vectors after filtering: {len(pred_vectors)}, scores range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
            
            # Store predictions
            predictions[sample_token] = {
                'vectors': pred_vectors,
                'labels': pred_labels,
                'scores': pred_scores
            }
            
        except Exception as e:
            logger.warning(f'Error processing sample {i}: {str(e)}')
        
        prog_bar.update()
    
    # Save predictions
    logger.info(f'\nSaving {len(predictions)} predictions to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    
    logger.info('✓ Inference complete!')
    return output_pkl


# ==================== EVALUATION CODE ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 100,
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None
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
                                                        linewidth: float = 2.0) -> np.ndarray:
        """
        Compute Chamfer Distance matrix using EXACT MapTR official method.
        EXACT copy from MapTR's tpfp_chamfer.py:custom_polyline_score()
        
        Returns NEGATIVE CD values (higher = better match).
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        if num_preds == 0 or num_gts == 0:
            return np.full((num_preds, num_gts), -100.0)
        
        # Create buffered shapely geometries
        pred_lines_shapely = [
            LineString(pred_vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(num_preds)
        ]
        gt_lines_shapely = [
            LineString(gt_vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(num_gts)
        ]
        
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
                                               threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predictions to GT using MapTR's EXACT OFFICIAL method.
        EXACT copy from MapTR's tpfp.py:custom_tpfp_gen()
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
        
        # Compute CD matrix
        cd_matrix = self.compute_chamfer_distance_matrix_maptr_official(
            pred_vectors, gt_vectors, linewidth=2.0)
        
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
                            threshold: float) -> Tuple[float, float]:
        """
        Compute AP and average CD for a single class at given threshold.
        """
        num_gts = sum(len(gts) for gts in gt_vectors_list)
        
        if num_gts == 0:
            return 0.0, float('inf')
        
        all_tp = []
        all_fp = []
        all_scores = []
        chamfer_distances_per_sample = []
        
        for pred_vecs, pred_scores, gt_vecs in zip(pred_vectors_list, pred_scores_list, gt_vectors_list):
            if len(pred_vecs) == 0:
                continue
            
            if len(gt_vecs) == 0:
                all_tp.append(np.zeros(len(pred_vecs), dtype=np.float32))
                all_fp.append(np.ones(len(pred_vecs), dtype=np.float32))
                all_scores.append(pred_scores)
                continue
            
            # Match predictions to GT
            tp, fp = self.match_predictions_to_gt_maptr_official(
                pred_vecs, pred_scores, gt_vecs, threshold)
            
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
            
            # Evaluate each class
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Extract predictions and GT for this class
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                for pred_data, gt_data in zip(camera_preds, camera_gts):
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                
                # Compute AP at each threshold
                avg_cd = None
                for threshold in self.thresholds_chamfer:
                    ap, cd = self.compute_ap_for_class(
                        pred_vectors_list, pred_scores_list, gt_vectors_list, threshold)
                    
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                    
                    if avg_cd is None:
                        avg_cd = cd
                
                class_results['avg_chamfer_distance'] = avg_cd if avg_cd is not None else float('inf')
                
                camera_results[class_name] = class_results
            
            # Compute mAP
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        return results


# ==================== MAIN FUNCTION ====================
def main():
    parser = argparse.ArgumentParser(
        description='Unified StreamMapNet Inference + Evaluation with Multi-Camera Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Front camera only
  python tools/streammapnet_eval_unified.py --cameras CAM_FRONT
  
  # Front + Back cameras
  python tools/streammapnet_eval_unified.py --cameras CAM_FRONT CAM_BACK
  
  # All 6 cameras (baseline)
  python tools/streammapnet_eval_unified.py --cameras all
  
  # Skip inference if predictions already exist
  python tools/streammapnet_eval_unified.py --skip-inference --predictions-pkl existing.pkl
        """
    )
    
    # Inference arguments
    parser.add_argument('--config', type=str,
                       default='/home/runw/Project/StreamMapNet/config/nusc_baseline_480_60x30_30e.py',
                       help='StreamMapNet config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/runw/Project/StreamMapNet/ckpts/nusc_baseline_480_60x30_30e.pth',
                       help='StreamMapNet checkpoint file')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference step (use existing predictions)')
    parser.add_argument('--predictions-pkl', type=str, default='streammapnet_predictions.pkl',
                       help='Output/input pickle file for predictions')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                       help='Score threshold for predictions (default: 0.0)')
    
    # Evaluation arguments
    parser.add_argument('--nuscenes-path', type=str,
                       default='/home/runw/Project/data/mini/nuscenes',
                       help='Path to NuScenes dataset')
    parser.add_argument('--samples-pkl', type=str,
                       default='/home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl',
                       help='Path to samples pickle file')
    parser.add_argument('--output-json', type=str, default='evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--pc-range', type=float, nargs=6,
                       default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                       help='Point cloud range for evaluation')
    
    # Camera configuration
    parser.add_argument('--cameras', type=str, nargs='+', default=['all'],
                       help='Camera names to use (CAM_FRONT, CAM_BACK, etc.) or "all"')
    parser.add_argument('--num-sample-pts', type=int, default=100,
                       help='Number of points to resample vectors to (default: 100)')
    
    # FOV clipping control
    parser.add_argument('--apply-clipping', action='store_true',
                       help='Apply camera FOV clipping to GT and predictions')
    parser.add_argument('--no-clipping', dest='apply_clipping', action='store_false',
                       help='Disable FOV clipping (full BEV evaluation)')
    parser.set_defaults(apply_clipping=True)
    
    args = parser.parse_args()
    
    # Parse camera configuration
    camera_indices = parse_camera_config(args.cameras)
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    
    print("="*80)
    print("Unified StreamMapNet Evaluation Script")
    print("="*80)
    print(f"\nCamera configuration: {', '.join(camera_names)} ({len(camera_indices)}/6)")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"NuScenes path: {args.nuscenes_path}")
    print(f"Samples pickle: {args.samples_pkl}")
    print(f"FOV clipping: {'ENABLED' if args.apply_clipping else 'DISABLED (full BEV)'}")
    
    # STEP 1: Run inference (or skip if requested)
    if not args.skip_inference:
        predictions_pkl = run_streammapnet_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_pkl=args.predictions_pkl,
            camera_indices=camera_indices,
            score_thresh=args.score_thresh,
            samples_pkl=args.samples_pkl
            # Note: nuscenes_path removed - we don't override data_root to avoid path doubling
        )
    else:
        print("\n" + "="*80)
        print("STEP 1: Skipping Inference (using existing predictions)")
        print("="*80)
        predictions_pkl = args.predictions_pkl
        if not os.path.exists(predictions_pkl):
            raise FileNotFoundError(f"Predictions file not found: {predictions_pkl}")
        print(f"Using existing predictions: {predictions_pkl}")
    
    # STEP 2: Run evaluation
    print("\n" + "="*80)
    print("STEP 2: Running Evaluation")
    print("="*80)
    
    # Load samples
    print(f"\nLoading samples from {args.samples_pkl}...")
    with open(args.samples_pkl, 'rb') as f:
        samples_data = pickle.load(f)
    samples = samples_data['infos']
    print(f"Loaded {len(samples)} samples")
    
    # Load predictions
    print(f"\nLoading predictions from {predictions_pkl}...")
    with open(predictions_pkl, 'rb') as f:
        predictions_by_token = pickle.load(f)
    print(f"Loaded predictions for {len(predictions_by_token)} samples")
    
    # Remap StreamMapNet class IDs to MapTR class IDs
    # StreamMapNet: 0=ped_crossing, 1=divider, 2=boundary
    # MapTR GT: 0=divider, 1=ped_crossing, 2=boundary
    streammapnet_to_maptr = {0: 1, 1: 0, 2: 2}
    
    print(f"\nRemapping prediction class IDs from StreamMapNet to MapTR format...")
    for token, pred_data in predictions_by_token.items():
        pred_labels = pred_data['labels']
        remapped_labels = np.array([streammapnet_to_maptr[int(label)] for label in pred_labels])
        predictions_by_token[token]['labels'] = remapped_labels
    print(f"✓ Remapped class IDs for {len(predictions_by_token)} samples")
    
    # Create evaluator
    evaluator = CameraSpecificEvaluator(
        nuscenes_data_path=args.nuscenes_path,
        pc_range=args.pc_range,
        num_sample_pts=args.num_sample_pts,
        thresholds_chamfer=[0.5, 1.0, 1.5],
        camera_names=camera_names
    )
    
    print(f"\nInitialized evaluator:")
    print(f"  - PC range: {args.pc_range}")
    print(f"  - Cameras: {camera_names}")
    print(f"  - Sample points: {args.num_sample_pts}")
    print(f"  - Chamfer thresholds: {evaluator.thresholds_chamfer} meters")
    
    # Evaluate all samples
    print(f"\nEvaluating {len(samples)} samples...")
    mode_str = "camera-specific FOV clipping" if args.apply_clipping else "full BEV (no clipping)"
    print(f"Mode: {mode_str}")
    
    for sample_info in tqdm(samples, desc="Processing samples"):
        sample_token = sample_info['token']
        
        if sample_token not in predictions_by_token:
            continue
        
        pred_data = predictions_by_token[sample_token]
        evaluator.accumulate_sample(
            sample_info=sample_info,
            pred_vectors=pred_data['vectors'],
            pred_labels=pred_data['labels'],
            pred_scores=pred_data['scores'],
            apply_clipping=args.apply_clipping
        )
    
    # Compute metrics
    print("\nComputing metrics...")
    results = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    for camera_name, camera_results in results.items():
        print(f"\n{camera_name}:")
        
        if 'mAP' in camera_results:
            print(f"  mAP (all classes & thresholds): {camera_results['mAP']:.4f}")
        
        print()
        for class_name in class_names:
            if class_name not in camera_results:
                continue
            class_results = camera_results[class_name]
            print(f"  {class_name}:")
            for threshold in evaluator.thresholds_chamfer:
                ap = class_results[f'AP@{threshold}m']
                print(f"    AP@{threshold}m: {ap:.4f}")
            cd = class_results['avg_chamfer_distance']
            cd_str = f"{cd:.4f}m" if cd != float('inf') else "N/A"
            print(f"    Avg CD: {cd_str}")
    
    # Save results
    print(f"\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
