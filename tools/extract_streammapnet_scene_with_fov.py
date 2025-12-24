"""
Extract StreamMapNet Scene Predictions with Camera FOV Clipping

This script runs StreamMapNet inference with camera-specific FOV clipping to match
the VGGT evaluation protocol:
1. Run inference on specific camera view(s) only (e.g., CAM_FRONT)
2. Apply FOV clipping to filter predictions visible in each camera
3. Transform to global coordinates
4. Merge predictions from all timestamps to construct scene map

This enables fair comparison with VGGT which uses camera-specific inputs and FOV clipping.

Usage:
    # Single camera (CAM_FRONT only)
    python extract_streammapnet_scene_with_fov.py \\
        --config /home/runw/Project/StreamMapNet/plugin/configs/nusc_baseline_480_60x30_30e.py \\
        --checkpoint /home/runw/Project/StreamMapNet/ckpts/nusc_baseline_480_60x30_30e.pth \\
        --nuscenes_path /home/runw/Project/data/mini/nuscenes \\
        --output_dir output/streammapnet_scene_fov \\
        --version v1.0-mini \\
        --scene_idx 0 \\
        --camera CAM_FRONT
    
    # Multiple cameras
    python extract_streammapnet_scene_with_fov.py \\
        --config ... \\
        --camera CAM_FRONT,CAM_BACK
"""

import argparse
import mmcv
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Ensure StreamMapNet repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings('ignore')

# StreamMapNet imports
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from plugin.datasets.builder import build_dataloader

# NuScenes imports
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
except ImportError:
    print("Error: nuscenes-devkit not available. Please install it.")
    sys.exit(1)

# Import FOV clipping utilities (from MapTR tools)
maptr_tools_path = Path(__file__).resolve().parents[2] / 'MapTR' / 'tools'
if str(maptr_tools_path) not in sys.path:
    sys.path.insert(0, str(maptr_tools_path))

from camera_fov_utils import CameraFOVClipper


def load_streammapnet_model(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load StreamMapNet model"""
    print(f"Loading StreamMapNet config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
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
            importlib.import_module(_module_path)
    
    # Build model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Handle FP16
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Set model classes
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    
    # Move to device and set eval mode
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model, cfg


def run_inference_with_fov_clipping(
    model,
    cfg,
    nusc: NuScenes,
    scene_idx: int,
    camera_names: List[str],
    confidence_threshold: float = 0.3,
    device: str = 'cuda',
) -> Dict:
    """
    Run StreamMapNet inference on scene with FOV clipping per camera.
    
    Returns predictions in GLOBAL coordinates after FOV filtering.
    """
    scene = nusc.scene[scene_idx]
    scene_token = scene['token']
    scene_name = scene['name']
    
    print(f"\nProcessing scene {scene_idx}: {scene_name}")
    print(f"  Cameras: {', '.join(camera_names)}")
    
    # Get all samples in scene
    sample_token = scene['first_sample_token']
    samples = []
    while sample_token:
        sample = nusc.get('sample', sample_token)
        samples.append(sample)
        sample_token = sample['next']
    
    print(f"  Timestamps: {len(samples)}")
    
    # Get PC range from config
    pc_range = getattr(cfg, 'pc_range', [-30.0, -15.0, -3.0, 30.0, 15.0, 5.0])
    if len(pc_range) < 6:
        # Try roi_size if pc_range not available
        roi_size = getattr(cfg, 'roi_size', [60, 30])
        pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3.0, roi_size[0]/2, roi_size[1]/2, 5.0]
    
    print(f"  PC range: {pc_range}")
    
    # Initialize FOV clipper
    fov_clipper = CameraFOVClipper()
    
    # Storage for FOV-clipped predictions
    all_predictions = []
    all_labels = []
    all_scores = []
    all_timestamps = []
    all_cameras = []
    
    # Patch NuscDataset.load_annotations to handle dict format pickle files
    # Original code expects list format, but some pickle files have dict format
    try:
        from plugin.datasets.nusc_dataset import NuscDataset
        original_load_annotations = NuscDataset.load_annotations
        
        def patched_load_annotations(self, ann_file):
            """Patched version that handles both list and dict formats."""
            import time
            start_time = time.time()
            ann = mmcv.load(ann_file)
            
            # Handle dict format - check for common keys or single list value
            if isinstance(ann, dict):
                # Try common keys first
                if 'infos' in ann:
                    ann = ann['infos']
                elif 'samples' in ann:
                    ann = ann['samples']
                else:
                    # If dict has only one key and it's a list, use it
                    keys = list(ann.keys())
                    if len(keys) == 1 and isinstance(ann[keys[0]], list):
                        ann = ann[keys[0]]
                    else:
                        raise ValueError(
                            f"Unexpected dict format in annotation file. "
                            f"Expected list or dict with 'infos'/'samples' key. "
                            f"Got dict with keys: {keys}"
                        )
            
            # Original code expects ann to be a list at this point
            if not isinstance(ann, list):
                raise TypeError(
                    f"After processing, expected list but got {type(ann)}. "
                    f"Annotation file format may be incorrect."
                )
            
            # Apply interval slicing (original behavior)
            samples = ann[::self.interval]
            
            print(f'collected {len(samples)} samples in {(time.time() - start_time):.2f}s')
            self.samples = samples
        
        NuscDataset.load_annotations = patched_load_annotations
        print("  Patched NuscDataset.load_annotations to handle dict formats")
        
        # Also patch get_sample to handle different key name formats
        original_get_sample = NuscDataset.get_sample
        
        def patched_get_sample(self, idx):
            """Patched version that handles different key name formats."""
            sample = self.samples[idx]
            
            # Handle location: try 'location' first, fallback to 'map_location'
            location = sample.get('location') or sample.get('map_location')
            if location is None:
                raise KeyError(f"Sample at index {idx} missing both 'location' and 'map_location' keys")
            
            # Handle e2g_translation/rotation
            e2g_translation = sample.get('e2g_translation') or sample.get('ego2global_translation')
            e2g_rotation = sample.get('e2g_rotation') or sample.get('ego2global_rotation')
            
            if e2g_translation is None or e2g_rotation is None:
                raise KeyError(f"Sample at index {idx} missing required translation/rotation keys")
            
            # Get map geometries
            map_geoms = self.map_extractor.get_map_geom(location, e2g_translation, e2g_rotation)
            map_label2geom = {}
            for k, v in map_geoms.items():
                if k in self.cat2id.keys():
                    map_label2geom[self.cat2id[k]] = v
            
            from pyquaternion import Quaternion
            import numpy as np
            
            ego2img_rts = []
            img_filenames = []
            cam_intrinsics = []
            cam_extrinsics = []
            
            for cam_name, c in sample['cams'].items():
                # Get extrinsics
                if 'extrinsics' in c:
                    extrinsic = np.array(c['extrinsics'])
                elif 'ego2cam' in c:
                    extrinsic = np.array(c['ego2cam'])
                elif 'cam2ego_translation' in c and 'cam2ego_rotation' in c:
                    cam2ego_r = Quaternion(c['cam2ego_rotation']).rotation_matrix
                    cam2ego_t = np.array(c['cam2ego_translation'])
                    ego2cam_r = cam2ego_r.T
                    ego2cam_t = -ego2cam_r @ cam2ego_t
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = ego2cam_r
                    extrinsic[:3, 3] = ego2cam_t
                elif 'sensor2ego_translation' in c and 'sensor2ego_rotation' in c:
                    sensor2ego_r = Quaternion(c['sensor2ego_rotation']).rotation_matrix
                    sensor2ego_t = np.array(c['sensor2ego_translation'])
                    ego2cam_r = sensor2ego_r.T
                    ego2cam_t = -ego2cam_r @ sensor2ego_t
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = ego2cam_r
                    extrinsic[:3, 3] = ego2cam_t
                else:
                    raise KeyError(f"Camera '{cam_name}' missing extrinsics")
                
                # Get intrinsics
                if 'intrinsics' in c:
                    intrinsic = np.array(c['intrinsics'])
                elif 'camera_intrinsic' in c:
                    intrinsic = np.array(c['camera_intrinsic'])
                elif 'cam_intrinsic' in c:
                    intrinsic = np.array(c['cam_intrinsic'])
                else:
                    raise KeyError(f"Camera '{cam_name}' missing intrinsics")
                
                # Ensure intrinsics is 3x3
                if intrinsic.shape == (3, 4):
                    intrinsic = intrinsic[:, :3]
                elif intrinsic.shape == (4, 4):
                    intrinsic = intrinsic[:3, :3]
                elif intrinsic.shape != (3, 3):
                    if intrinsic.size == 9:
                        intrinsic = intrinsic.reshape(3, 3)
                
                # Get image path
                img_path = c.get('img_fpath') or c.get('data_path') or c.get('img_path') or ''
                
                # Build ego2img transform
                ego2cam_rt = extrinsic.copy()
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                ego2cam_rt = (viewpad @ ego2cam_rt)
                ego2img_rts.append(ego2cam_rt)
                
                img_filenames.append(img_path)
                cam_intrinsics.append(intrinsic.tolist())
                cam_extrinsics.append(extrinsic.tolist())
            
            # Handle sample_idx
            sample_idx = sample.get('sample_idx') or sample.get('frame_idx', idx)
            scene_name = sample.get('scene_name') or sample.get('scene_token', '')
            
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
        print("  Patched NuscDataset.get_sample to handle different key formats")
    except Exception as e:
        print(f"  Warning: Could not patch NuscDataset: {e}")
    
    # Build dataset (prefer val, fallback to train)
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = nusc.dataroot
    
    # Fix annotation file path - check if file exists, otherwise try alternative locations
    ann_file = cfg.data.test.ann_file
    if not os.path.isabs(ann_file) or not os.path.exists(ann_file):
        # Try common locations for mini dataset
        possible_paths = [
            os.path.join(nusc.dataroot, 'gemap', 'nuscenes_map_infos_temporal_val.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_map_infos_temporal_val.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_map_infos_val.pkl'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  Using annotation file: {path}")
                cfg.data.test.ann_file = path
                break
        else:
            print(f"  Warning: Could not find annotation file. Tried:")
            for path in possible_paths:
                print(f"    - {path}")
            print(f"  Will attempt to use original path: {ann_file}")
    
    # Build validation dataset
    dataset_val = build_dataset(cfg.data.test)
    
    # Try to build train-like test dataset as fallback
    import copy as _copy
    try:
        train_like_test_cfg = _copy.deepcopy(cfg.data.test)
        train_like_test_cfg['ann_file'] = cfg.data.train['ann_file']
        
        # Fix train annotation file path too
        ann_file_train = train_like_test_cfg['ann_file']
        if not os.path.isabs(ann_file_train) or not os.path.exists(ann_file_train):
            possible_paths_train = [
                os.path.join(nusc.dataroot, 'gemap', 'nuscenes_map_infos_temporal_train.pkl'),
                os.path.join(nusc.dataroot, 'nuscenes_map_infos_temporal_train.pkl'),
                os.path.join(nusc.dataroot, 'nuscenes_map_infos_train.pkl'),
            ]
            for path in possible_paths_train:
                if os.path.exists(path):
                    train_like_test_cfg['ann_file'] = path
                    break
        
        dataset_train = build_dataset(train_like_test_cfg)
    except Exception as e:
        print(f"  Warning: Could not build train dataset: {e}")
        dataset_train = None
    
    # Collect sample indices for this scene
    def _collect_indices(ds):
        indices = []
        if hasattr(ds, 'data_infos'):
            # MapTR-style dataset
            for idx, data_info in enumerate(ds.data_infos):
                if data_info.get('scene_token', None) == scene_token:
                    indices.append(idx)
        elif hasattr(ds, 'samples'):
            # StreamMapNet-style dataset
            for idx, sample_info in enumerate(ds.samples):
                # Try different key names for scene identification
                sample_scene_token = sample_info.get('scene_token', sample_info.get('scene_name', None))
                if sample_scene_token == scene_token or sample_scene_token == scene_name:
                    indices.append(idx)
        return indices
    
    scene_sample_indices = _collect_indices(dataset_val)
    selected_split = 'val'
    selected_dataset = dataset_val
    
    if len(scene_sample_indices) == 0 and dataset_train is not None:
        scene_sample_indices = _collect_indices(dataset_train)
        selected_split = 'train'
        selected_dataset = dataset_train
    
    print(f"  Found {len(scene_sample_indices)} samples in {selected_split} dataset")
    
    if len(scene_sample_indices) == 0:
        print("  Warning: No samples found for this scene")
        return {
            'predictions': [],
            'labels': [],
            'scores': [],
            'timestamps': [],
            'cameras': [],
            'scene_token': scene_token,
            'scene_name': scene_name,
        }
    
    # Build dataloader
    scene_dataset = torch.utils.data.Subset(selected_dataset, scene_sample_indices)
    data_loader = build_dataloader(
        scene_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')),
    )
    
    # Camera mapping
    camera_map = {
        'CAM_FRONT': 0,
        'CAM_FRONT_RIGHT': 1,
        'CAM_FRONT_LEFT': 2,
        'CAM_BACK': 3,
        'CAM_BACK_LEFT': 4,
        'CAM_BACK_RIGHT': 5
    }
    
    # Get target camera indices
    target_camera_indices = [camera_map[cam] for cam in camera_names if cam in camera_map]
    
    # Run inference on each timestamp
    with torch.no_grad():
        for ts_idx, data in enumerate(tqdm(data_loader, desc="Inference + FOV clipping")):
            if ts_idx >= len(samples):
                print(f"  Warning: timestamp index {ts_idx} exceeds samples length {len(samples)}")
                break
            
            sample = samples[ts_idx]
            
            # Zero out all camera views except the specified ones
            if 'img' in data:
                # Handle DataContainer
                if hasattr(data['img'], 'data'):
                    imgs = data['img'].data[0]
                else:
                    imgs = data['img'][0].data[0] if hasattr(data['img'][0], 'data') else data['img'][0]
                
                if ts_idx == 0:
                    print(f"  Image tensor shape: {imgs.shape}")
                    print(f"  Using camera indices {target_camera_indices} ({', '.join(camera_names)})")
                
                # Zero out all views except target cameras
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in target_camera_indices:
                        mask[:, cam_idx, :, :, :] = 1
                    if hasattr(data['img'], 'data'):
                        data['img'].data[0] = imgs * mask
                    else:
                        data['img'][0].data[0] = imgs * mask
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in target_camera_indices:
                        mask[cam_idx, :, :, :] = 1
                    if hasattr(data['img'], 'data'):
                        data['img'].data[0] = imgs * mask
                    else:
                        data['img'][0].data[0] = imgs * mask
            
            # Forward pass (with zeroed camera inputs)
            result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions
            # StreamMapNet returns: [{'vectors': np.array, 'scores': np.array, 'labels': np.array}]
            if not isinstance(result, (list, tuple)) or len(result) == 0:
                continue
            
            result_dic = result[0]
            
            # Handle both StreamMapNet and MapTR formats
            if 'vectors' in result_dic:
                # StreamMapNet format
                pred_vectors = result_dic['vectors']
                pred_scores = result_dic['scores']
                pred_labels = result_dic['labels']
            elif 'pts_bbox' in result_dic:
                # MapTR format
                bbox_dic = result_dic['pts_bbox']
                pred_vectors = bbox_dic['pts_3d']
                pred_scores = bbox_dic['scores_3d']
                pred_labels = bbox_dic['labels_3d']
                # Convert to numpy if tensors
                if torch.is_tensor(pred_vectors):
                    pred_vectors = pred_vectors.cpu().numpy()
                if torch.is_tensor(pred_scores):
                    pred_scores = pred_scores.cpu().numpy()
                if torch.is_tensor(pred_labels):
                    pred_labels = pred_labels.cpu().numpy()
            else:
                print(f"  Warning: Unknown result format at timestamp {ts_idx}")
                continue
            
            # Ensure numpy arrays
            if not isinstance(pred_vectors, np.ndarray):
                pred_vectors = np.array(pred_vectors)
            if not isinstance(pred_scores, np.ndarray):
                pred_scores = np.array(pred_scores)
            if not isinstance(pred_labels, np.ndarray):
                pred_labels = np.array(pred_labels)
            
            # Filter by confidence
            if len(pred_scores) > 0:
                keep = pred_scores > confidence_threshold
                if keep.sum() == 0:
                    continue
                
                pred_vectors = pred_vectors[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
            else:
                continue
            
            # Denormalize if needed (StreamMapNet typically outputs normalized coords)
            # Check if coordinates are in [0, 1] range (normalized)
            if len(pred_vectors) > 0 and pred_vectors.max() <= 1.0 and pred_vectors.min() >= 0.0:
                x_min, y_min = pc_range[0], pc_range[1]
                x_max, y_max = pc_range[3], pc_range[4]
                
                pred_vectors = pred_vectors.copy()
                pred_vectors[..., 0] = pred_vectors[..., 0] * (x_max - x_min) + x_min
                pred_vectors[..., 1] = pred_vectors[..., 1] * (y_max - y_min) + y_min
                
                # Apply 90-degree counterclockwise rotation to align with MapTR coordinate system
                # StreamMapNet: X ∈ [-30, 30], Y ∈ [-15, 15]
                # MapTR: X ∈ [-15, 15], Y ∈ [-30, 30]
                # Rotation: (x, y) -> (-y, x)
                pred_vectors_rotated = pred_vectors.copy()
                pred_vectors_rotated[..., 0] = -pred_vectors[..., 1]
                pred_vectors_rotated[..., 1] = pred_vectors[..., 0]
                pred_vectors = pred_vectors_rotated
            
            # Now pred_vectors is in lidar-centric coordinates (meters)
            pred_pts_lidar = pred_vectors
            
            # Process each camera
            for camera_name in camera_names:
                # Get camera calibration for FOV clipping
                sd_token = sample['data'][camera_name]
                sd = nusc.get('sample_data', sd_token)
                ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                
                # Build BEV-aligned camera extrinsics (EXACT logic from camera_fov_utils.py)
                # This is CRITICAL for correct FOV clipping!
                lidar_sd_token = sample['data']['LIDAR_TOP']
                lidar_sd = nusc.get('sample_data', lidar_sd_token)
                lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
                lidar_ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
                
                # Camera2ego transform
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
                cam2ego[:3, 3] = cs_record['translation']
                
                # Ego2global transform
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(lidar_ego_pose['rotation']).rotation_matrix
                ego2global[:3, 3] = lidar_ego_pose['translation']
                
                # Lidar2ego transform
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(lidar_cs['rotation']).rotation_matrix
                lidar2ego[:3, 3] = lidar_cs['translation']
                
                # Compute composed transforms
                lidar2global = ego2global @ lidar2ego
                cam2global = ego2global @ cam2ego
                
                # Get lidar2global rotation angle for BEV alignment
                lidar2global_rotation = Quaternion(matrix=lidar2global)
                patch_angle_deg = quaternion_yaw(lidar2global_rotation) / np.pi * 180
                patch_angle_rad = np.radians(patch_angle_deg)
                
                # Create BEV alignment rotation matrix
                cos_a = np.cos(-patch_angle_rad)
                sin_a = np.sin(-patch_angle_rad)
                rotation_matrix_bev = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                # Apply BEV alignment to camera extrinsics
                cam_extrinsics_bev = np.eye(4)
                cam_extrinsics_bev[:3, :3] = rotation_matrix_bev @ cam2global[:3, :3]
                cam_extrinsics_bev[:3, 3] = rotation_matrix_bev @ (cam2global[:3, 3] - lidar2global[:3, 3])
                
                # Camera intrinsics
                intrinsics = np.array(cs_record['camera_intrinsic'])
                
                # Prepare predictions in FOV clipper format
                # pred_pts_lidar shape: [N, num_points, 2]
                vectors_for_fov = [[pred] for pred in pred_pts_lidar]
                labels_for_fov = pred_labels.tolist()
                
                # Apply FOV clipping using BEV-aligned extrinsics (CRITICAL!)
                fov_clipped_vecs, fov_clipped_labels, _ = fov_clipper.crop_vectors_to_fov(
                    vectors=vectors_for_fov,
                    labels=labels_for_fov,
                    extrinsics=cam_extrinsics_bev,
                    intrinsics=intrinsics
                )
                
                if len(fov_clipped_vecs) == 0:
                    continue
                
                # Flatten back to single vectors
                fov_clipped_preds = [vecs[0] for vecs in fov_clipped_vecs if len(vecs) > 0 and len(vecs[0]) >= 2]
                fov_clipped_labels_flat = [label for vecs, label in zip(fov_clipped_vecs, fov_clipped_labels) if len(vecs) > 0 and len(vecs[0]) >= 2]
                
                # Get corresponding scores
                fov_clipped_scores = []
                for i, (vecs, label) in enumerate(zip(fov_clipped_vecs, fov_clipped_labels)):
                    if len(vecs) > 0 and len(vecs[0]) >= 2:
                        fov_clipped_scores.append(pred_scores[i])
                
                if len(fov_clipped_preds) == 0:
                    continue
                
                # Transform FOV-clipped predictions from lidar to global coordinates
                # Transform each prediction to global
                for pred, label, score in zip(fov_clipped_preds, fov_clipped_labels_flat, fov_clipped_scores):
                    pred_global = np.zeros_like(pred)
                    for i, pt in enumerate(pred):
                        pt_3d = np.array([pt[0], pt[1], 0.0, 1.0])
                        pt_global = lidar2global @ pt_3d
                        pred_global[i] = pt_global[:2]
                    
                    all_predictions.append(pred_global)
                    all_labels.append(int(label))
                    all_scores.append(float(score))
                    all_timestamps.append(ts_idx)
                    all_cameras.append(camera_name)
    
    print(f"  Collected {len(all_predictions)} FOV-clipped predictions")
    
    # Count per class
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Class distribution:")
    print(f"    Divider: {label_counts.get(0, 0)}")
    print(f"    Ped Crossing: {label_counts.get(1, 0)}")
    print(f"    Boundary: {label_counts.get(2, 0)}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'scores': all_scores,
        'timestamps': all_timestamps,
        'cameras': all_cameras,
        'scene_token': scene_token,
        'scene_name': scene_name,
        'camera_names': camera_names,
    }


def visualize_scene_predictions(
    pred_data: Dict,
    output_path: Path,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (20, 20),
):
    """Visualize FOV-clipped scene predictions"""
    predictions = pred_data['predictions']
    labels = pred_data['labels']
    
    if len(predictions) == 0:
        print(f"  Warning: No predictions to visualize")
        return
    
    class_colors = {0: 'orange', 1: 'blue', 2: 'green'}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute bounds
    if xlim is None or ylim is None:
        all_coords = np.concatenate([p.reshape(-1, 2) for p in predictions])
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        if xlim is None:
            xlim = (x_min - x_margin, x_max + x_margin)
        if ylim is None:
            ylim = (y_min - y_margin, y_max + y_margin)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot predictions
    for pred, label in zip(predictions, labels):
        color = class_colors.get(label, 'gray')
        ax.plot(pred[:, 0], pred[:, 1], color=color, linewidth=1.5, alpha=0.7)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract StreamMapNet scene predictions with FOV clipping')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--nuscenes_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'])
    parser.add_argument('--scene_idx', type=int, help='Specific scene index')
    parser.add_argument('--num_scenes', type=int, help='Number of scenes to process')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                       help='Camera(s) to use, comma-separated (e.g., CAM_FRONT,CAM_BACK)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Parse camera names
    camera_names = [cam.strip() for cam in args.camera.split(',')]
    print(f"Using cameras: {', '.join(camera_names)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, cfg = load_streammapnet_model(args.config, args.checkpoint, args.device)
    
    # Load NuScenes
    print(f"\nLoading nuScenes {args.version} from {args.nuscenes_path}...")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_path, verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes")
    
    # Determine scenes to process
    if args.scene_idx is not None:
        scene_indices = [args.scene_idx]
    elif args.num_scenes is not None:
        scene_indices = list(range(min(args.num_scenes, len(nusc.scene))))
    else:
        scene_indices = list(range(len(nusc.scene)))
        print(f"Processing all {len(scene_indices)} scenes")
    
    # Process each scene
    for scene_idx in scene_indices:
        print(f"\n{'='*80}")
        print(f"Processing scene {scene_idx}/{len(nusc.scene)-1}")
        print(f"{'='*80}")
        
        # Run inference with FOV clipping
        pred_data = run_inference_with_fov_clipping(
            model=model,
            cfg=cfg,
            nusc=nusc,
            scene_idx=scene_idx,
            camera_names=camera_names,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
        
        if len(pred_data['predictions']) == 0:
            print(f"  Skipping scene {scene_idx} - no predictions after FOV clipping")
            continue
        
        # Create scene output directory
        scene_name = pred_data['scene_name']
        camera_suffix = '_'.join(camera_names)
        scene_output_dir = output_dir / f"scene_{scene_idx:04d}_{scene_name}_{camera_suffix}"
        scene_output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        save_data = {
            'predictions': np.array([p for p in pred_data['predictions']], dtype=object),
            'labels': np.array(pred_data['labels']),
            'scores': np.array(pred_data['scores']),
            'timestamps': np.array(pred_data['timestamps']),
            'cameras': pred_data['cameras'],
            'scene_token': pred_data['scene_token'],
            'scene_name': scene_name,
            'camera_names': camera_names,
        }
        np.save(scene_output_dir / 'streammapnet_fov_predictions.npy', save_data, allow_pickle=True)
        print(f"  Saved to: {scene_output_dir / 'streammapnet_fov_predictions.npy'}")
        
        # Visualize
        visualize_scene_predictions(
            pred_data=pred_data,
            output_path=scene_output_dir / 'streammapnet_fov_pred_map.png',
        )
    
    print(f"\n{'='*80}")
    print(f"Done! Processed {len(scene_indices)} scenes")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
