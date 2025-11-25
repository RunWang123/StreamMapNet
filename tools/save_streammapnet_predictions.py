#!/usr/bin/env python3
"""
Run StreamMapNet inference and save predictions for evaluation.
Based on MapTR's save_maptr_predictions.py but adapted for StreamMapNet.
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import get_root_logger
import os.path as osp
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run StreamMapNet inference and save predictions')
    parser.add_argument('--config', type=str, 
                       default='/home/runw/Project/StreamMapNet/plugin/configs/nusc_baseline_480_60x30_30e.py',
                       help='test config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/runw/Project/StreamMapNet/ckpts/nusc_baseline_480_60x30_30e.pth',
                       help='checkpoint file')
    parser.add_argument('--output-pkl', type=str, default='streammapnet_predictions.pkl',
                       help='Output pickle file for predictions')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                       help='Score threshold for predictions (default: 0.0 = keep ALL for proper evaluation, range: 0.0-1.0)')
    parser.add_argument('--front-camera-only', action='store_true',
                       help='Zero out all camera views except CAM_FRONT to simulate corrupted cameras')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

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
    
    # Patch NuscDataset.load_annotations to handle dict format pickle files
    # Original code expects list format, but some pickle files may have dict format
    # (e.g., {'infos': [...]} or {'samples': [...]})
    try:
        from plugin.datasets.nusc_dataset import NuscDataset
        original_load_annotations = NuscDataset.load_annotations
        
        def patched_load_annotations(self, ann_file):
            """Patched version that handles both list and dict formats.
            
            Original implementation expects a list, but handles dict formats:
            - {'infos': [...]} 
            - {'samples': [...]}
            - Or any dict with a single list value
            """
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
            
            # Validate sample structure - check first sample has required keys
            # Note: We handle both key name formats, so we check for either variant
            if len(samples) > 0:
                first_sample = samples[0]
                # Check for required keys (either format is acceptable)
                has_location = 'location' in first_sample or 'map_location' in first_sample
                has_translation = 'e2g_translation' in first_sample or 'ego2global_translation' in first_sample
                has_rotation = 'e2g_rotation' in first_sample or 'ego2global_rotation' in first_sample
                has_cams = 'cams' in first_sample
                has_token = 'token' in first_sample
                
                missing_required = []
                if not has_location:
                    missing_required.append('location/map_location')
                if not has_translation:
                    missing_required.append('e2g_translation/ego2global_translation')
                if not has_rotation:
                    missing_required.append('e2g_rotation/ego2global_rotation')
                if not has_cams:
                    missing_required.append('cams')
                if not has_token:
                    missing_required.append('token')
                
                if missing_required:
                    available_keys = list(first_sample.keys())
                    logger.warning(
                        f"First sample missing required keys: {missing_required}. "
                        f"Available keys: {available_keys}. "
                        f"This may cause errors during data loading."
                    )
                else:
                    logger.info(
                        f"Sample structure validated. Found keys compatible with StreamMapNet format."
                    )
            
            print(f'collected {len(samples)} samples in {(time.time() - start_time):.2f}s')
            self.samples = samples
        
        NuscDataset.load_annotations = patched_load_annotations
        logger.info('Patched NuscDataset.load_annotations to handle both list and dict formats')
        
        # Also patch get_sample to handle missing keys gracefully
        original_get_sample = NuscDataset.get_sample
        
        def patched_get_sample(self, idx):
            """Patched version that handles different key name formats.
            
            Handles both formats:
            - Original StreamMapNet format: 'location', 'e2g_translation', 'e2g_rotation', 'sample_idx', 'scene_name'
            - Alternative format: 'map_location', 'ego2global_translation', 'ego2global_rotation', 'frame_idx', 'scene_token'
            """
            sample = self.samples[idx]
            
            # Handle location: try 'location' first, fallback to 'map_location'
            location = sample.get('location') or sample.get('map_location')
            if location is None:
                available_keys = list(sample.keys())
                raise KeyError(
                    f"Sample at index {idx} missing both 'location' and 'map_location' keys. "
                    f"Available keys: {available_keys}"
                )
            
            # Handle e2g_translation: try 'e2g_translation' first, fallback to 'ego2global_translation'
            e2g_translation = sample.get('e2g_translation') or sample.get('ego2global_translation')
            
            # Handle e2g_rotation: try 'e2g_rotation' first, fallback to 'ego2global_rotation'
            e2g_rotation = sample.get('e2g_rotation') or sample.get('ego2global_rotation')
            
            if e2g_translation is None or e2g_rotation is None:
                available_keys = list(sample.keys())
                raise KeyError(
                    f"Sample at index {idx} missing required translation/rotation keys. "
                    f"Missing: e2g_translation={e2g_translation is None}, e2g_rotation={e2g_rotation is None}. "
                    f"Available keys: {available_keys}"
                )
            
            # Handle sample_idx: try 'sample_idx' first, fallback to 'frame_idx'
            sample_idx = sample.get('sample_idx') or sample.get('frame_idx')
            if sample_idx is None:
                # Use idx as fallback if neither key exists
                sample_idx = idx
            
            # Handle scene_name: try 'scene_name' first, fallback to 'scene_token' or use empty string
            scene_name = sample.get('scene_name') or sample.get('scene_token', '')
            
            # Continue with original logic but use the extracted values
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
            
            # Handle camera data - support multiple formats
            for cam_name, c in sample['cams'].items():
                # Try to get extrinsics (ego2cam transform matrix)
                extrinsic = None
                if 'extrinsics' in c:
                    extrinsic = np.array(c['extrinsics'])
                elif 'ego2cam' in c:
                    extrinsic = np.array(c['ego2cam'])
                elif 'cam2ego_translation' in c and 'cam2ego_rotation' in c:
                    # Build from cam2ego
                    cam2ego_r = Quaternion(c['cam2ego_rotation']).rotation_matrix
                    cam2ego_t = np.array(c['cam2ego_translation'])
                    # ego2cam = inverse of cam2ego
                    ego2cam_r = cam2ego_r.T
                    ego2cam_t = -ego2cam_r @ cam2ego_t
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = ego2cam_r
                    extrinsic[:3, 3] = ego2cam_t
                elif 'sensor2ego_translation' in c and 'sensor2ego_rotation' in c:
                    # Build from sensor2ego (for cameras, sensor2ego = cam2ego)
                    sensor2ego_r = Quaternion(c['sensor2ego_rotation']).rotation_matrix
                    sensor2ego_t = np.array(c['sensor2ego_translation'])
                    # ego2cam = inverse of sensor2ego
                    ego2cam_r = sensor2ego_r.T
                    ego2cam_t = -ego2cam_r @ sensor2ego_t
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = ego2cam_r
                    extrinsic[:3, 3] = ego2cam_t
                else:
                    available_keys = list(c.keys())
                    raise KeyError(
                        f"Camera '{cam_name}' missing extrinsics. "
                        f"Expected 'extrinsics', 'ego2cam', 'cam2ego_translation'/'cam2ego_rotation', "
                        f"or 'sensor2ego_translation'/'sensor2ego_rotation'. "
                        f"Available keys: {available_keys}"
                    )
                
                # Try to get intrinsics
                intrinsic = None
                if 'intrinsics' in c:
                    intrinsic = np.array(c['intrinsics'])
                elif 'camera_intrinsic' in c:
                    intrinsic = np.array(c['camera_intrinsic'])
                elif 'cam_intrinsic' in c:
                    intrinsic = np.array(c['cam_intrinsic'])
                else:
                    available_keys = list(c.keys())
                    raise KeyError(
                        f"Camera '{cam_name}' missing intrinsics. "
                        f"Expected 'intrinsics', 'camera_intrinsic', or 'cam_intrinsic'. "
                        f"Available keys: {available_keys}"
                    )
                
                # Ensure intrinsics is 3x3
                if intrinsic.shape == (3, 4):
                    intrinsic = intrinsic[:, :3]
                elif intrinsic.shape == (4, 4):
                    intrinsic = intrinsic[:3, :3]
                elif intrinsic.shape != (3, 3):
                    # Try to reshape if it's a flat array
                    if intrinsic.size == 9:
                        intrinsic = intrinsic.reshape(3, 3)
                    else:
                        raise ValueError(
                            f"Camera '{cam_name}' intrinsics has unexpected shape: {intrinsic.shape}. "
                            f"Expected (3, 3), (3, 4), or (4, 4)."
                        )
                
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
        logger.info('Patched NuscDataset.get_sample to handle missing keys with better error messages')
    except Exception as e:
        logger.warning(f'Could not patch NuscDataset: {e}')
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # Patch image paths to use correct data_root if needed
    # Get data_root from dataset config
    data_root = cfg.data.test.get('data_root', '')
    if not data_root:
        # Try to get from dataset object
        if hasattr(dataset, 'data_root'):
            data_root = dataset.data_root
        elif hasattr(dataset, 'nusc') and hasattr(dataset.nusc, 'dataroot'):
            data_root = dataset.nusc.dataroot
    
    if data_root:
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
    
    logger.info(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    logger.info('Model loaded and ready')

    # Storage for predictions
    predictions = {}
    
    # Run inference
    logger.info('Running inference...')
    if args.front_camera_only:
        logger.info('NOTE: Zeroing out all camera views EXCEPT CAM_FRONT to simulate corrupted cameras')
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        try:
            # Handle DataContainer: data['img_metas'] is a DataContainer
            # Based on evaluation code: data['img_metas'].data[0][0]['token']
            # data['img_metas'].data[0] is the batch, [0] is the first item in batch
            img_metas = data['img_metas'].data[0]
            
            # Get sample token - use the actual NuScenes sample_token if available
            # Debug: print available keys on first sample
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {list(img_metas[0].keys())}")
            
            # Try to get token from img_metas (StreamMapNet format)
            # Based on evaluation code: data['img_metas'].data[0][0]['token']
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
            
            # Zero out all camera views except CAM_FRONT (index 0) if requested
            # NuScenes camera order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, 
            #                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
            if args.front_camera_only and 'img' in data:
                # Handle DataContainer: data['img'] is a DataContainer
                if hasattr(data['img'], 'data'):
                    imgs = data['img'].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                else:
                    imgs = data['img'][0].data[0]
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                
                # Zero out views 1-5 (keep only view 0 = CAM_FRONT)
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    imgs[:, 1:, :, :, :] = 0
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    imgs[1:, :, :, :] = 0
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
                keep = pred_scores > args.score_thresh
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
                        logger.info(f"DEBUG: Number of vectors after filtering: {len(pred_vectors)}, scores range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
            
            # Store predictions
            predictions[sample_token] = {
                'vectors': pred_vectors,
                'labels': pred_labels,
                'scores': pred_scores
            }
            
        except Exception as e:
            import traceback
            logger.error(f'Error processing sample {i}: {str(e)}')
            logger.error(f'Traceback: {traceback.format_exc()}')
        
        prog_bar.update()
    
    # Save predictions
    logger.info(f'Saving {len(predictions)} predictions to {args.output_pkl}...')
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    
    logger.info('Done! Predictions saved.')
    logger.info(f'\nTo evaluate, run:')
    logger.info(f'python tools/evaluate_with_fov_clipping_standalone.py \\')
    logger.info(f'  --nuscenes-path /path/to/nuscenes \\')
    logger.info(f'  --samples-pkl /path/to/nuscenes_infos_temporal_val.pkl \\')
    logger.info(f'  --predictions-pkl {args.output_pkl} \\')
    logger.info(f'  --output-json evaluation_results.json')


if __name__ == '__main__':
    main()

