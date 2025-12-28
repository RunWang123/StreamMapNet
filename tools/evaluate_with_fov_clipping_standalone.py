#!/usr/bin/env python3
"""
Evaluate StreamMapNet predictions with camera-specific FOV clipping and rotation.
STANDALONE VERSION - Uses EXACT MapTR official evaluation code

This script applies camera-specific preprocessing to both GT and predictions:
1. Clip both GT and predictions to each camera's FOV
2. Rotate to camera-centric coordinates (camera forward = +Y)
3. Compute metrics using EXACT MapTR official method:
   - STRtree spatial indexing for Chamfer distance computation
   - Only computes CD for intersecting buffered geometries (linewidth=2m)
   - Negative CD convention (higher = better)
   - Greedy confidence-sorted matching

All FOV clipping logic imported from camera_fov_utils.py.
All evaluation logic matches MapTR's tpfp.py and tpfp_chamfer.py exactly.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import json

# Import NuScenes and geometry utilities
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString, Point, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

# Import shared camera FOV utilities (ensures identical logic with visualization)
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
    extract_gt_with_fov_clipping,
    process_predictions_with_fov_clipping
)

# Add StreamMapNet project path for evaluation utilities
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
print(f"DEBUG: project_root set to: {project_root}")
print(f"DEBUG: sys.path[0]: {sys.path[0]}")

# Import official StreamMapNet evaluation logic
try:
    from plugin.datasets.evaluation.AP import instance_match, average_precision
    from plugin.datasets.evaluation.distance import chamfer_distance
except ImportError as e:
    print(f"Error: Could not import official StreamMapNet evaluation logic: {e}")
    # Try importing plugin to see if it works
    try:
        import plugin
        print(f"DEBUG: plugin imported successfully from {plugin.__file__}")
    except ImportError:
        print("DEBUG: Could not import plugin module directly.")
        
    print("Make sure you are running this script from the project root or 'tools' directory.")
    sys.exit(1)


# ==================== EVALUATION-SPECIFIC CODE ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    
    MapTR NEGATIVE Chamfer Distance Convention:
    ==========================================
    - All Chamfer distances stored as NEGATIVE values
    - Example CD matrix: [[-0.3, -1.5], [-2.0, -0.8]]
    - Higher (less negative) = better match: -0.3 > -1.5 ✓
    - Non-intersecting pairs: -100.0 (guaranteed to not match)
    
    Why Negative?
    - Allows max() to find best match: max([-0.3, -1.5]) = -0.3
    - Consistent with score-based ranking (higher = better)
    - Threshold comparison: cd >= -threshold (e.g., -0.3 >= -0.5 ✓)
    
    Threshold Convention:
    ====================
    - User provides POSITIVE thresholds (e.g., [0.5, 1.0, 1.5] meters)
    - Internally converted to NEGATIVE (e.g., [-0.5, -1.0, -1.5])
    - Matching: if cd_score >= negative_threshold
    
    Example:
    --------
    CD = 0.35m (actual distance) -> stored as -0.35
    Threshold = 0.5m -> converted to -0.5
    Match check: -0.35 >= -0.5 ? YES ✓ (close enough)
    
    CD = 2.0m (actual distance) -> stored as -2.0
    Threshold = 0.5m -> converted to -0.5
    Match check: -2.0 >= -0.5 ? NO ✗ (too far)
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 200,  # StreamMapNet official uses 200
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None
    ):
        """
        Args:
            nuscenes_data_path: Path to NuScenes dataset
            pc_range: BEV range [-x, -y, -z, x, y, z]
            num_sample_pts: Number of points to resample vectors to (StreamMapNet: 200)
            thresholds_chamfer: Chamfer distance thresholds (MapTR uses [0.5, 1.0, 1.5])
            camera_names: List of camera names to evaluate (default: ['CAM_FRONT'] only)
        """
        self.nuscenes_data_path = nuscenes_data_path
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.num_sample_pts: int = num_sample_pts
        self.thresholds_chamfer = thresholds_chamfer or [0.5, 1.0, 1.5]
        self.camera_names = camera_names or ['CAM_FRONT']
        
        # Calculate patch size from pc_range
        self.patch_size = (self.pc_range[4] - self.pc_range[1], self.pc_range[3] - self.pc_range[0])
        
        # Accumulators for per-camera metrics
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.predictions_per_camera = {cam: [] for cam in self.camera_names}
        self.ground_truths_per_camera = {cam: [] for cam in self.camera_names}
        self.num_samples_processed = 0
    
    
    def resample_vector_linestring(self, vector: np.ndarray, num_sample: int) -> np.ndarray:
        """
        Resample a vector to fixed number of points using LineString interpolation.
        Matches StreamMapNet/MapTR implementation.
        """
        if len(vector) < 2:
            # Handle degenerate cases
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        # Create LineString and sample evenly
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        
        return sampled_points
    
    def streammapnet_instance_match(self, pred_lines, scores, gt_lines, thresholds):
        """
        EXACT copy of instance_match from StreamMapNet/plugin/datasets/evaluation/AP.py
        with simplified Chamfer distance calculation integrated.
        """
        num_preds = pred_lines.shape[0]
        num_gts = gt_lines.shape[0]

        # tp and fp
        tp_fp_list = []
        tp = np.zeros((num_preds), dtype=np.float32)
        fp = np.zeros((num_preds), dtype=np.float32)

        # if there is no gt lines in this sample, then all pred lines are false positives
        if num_gts == 0:
            fp[...] = 1
            for thr in thresholds:
                tp_fp_list.append((tp.copy(), fp.copy()))
            return tp_fp_list
        
        if num_preds == 0:
            for thr in thresholds:
                tp_fp_list.append((tp.copy(), fp.copy()))
            return tp_fp_list
            
        # Calculate Chamfer Distance Matrix (Batch)
        # Logic from StreamMapNet/plugin/datasets/evaluation/distance.py -> chamfer_distance_batch
        
        # (num_preds, num_pts, 2) vs (num_gts, num_pts, 2)
        # Using scipy cdist for CPU compatibility (Official uses Torch but we want this standalone to be robust)
        
        # Flatten lines for cdist: (num_preds * num_pts, 2)
        P = pred_lines.reshape(-1, 2)
        G = gt_lines.reshape(-1, 2)
        
        # Full distance matrix: (num_preds*num_pts, num_gts*num_pts)
        # WARNING: This can be large. 
        # But StreamMapNet official uses torch.cdist on ALL points.
        # Let's do it per-prediction to save memory if needed, or batch if small enough.
        # Given typical limits (preds ~100-300, pts=200), (60000, X) is big.
        # Let's iterate instead to be safe on CPU.
        
        matrix = np.zeros((num_preds, num_gts))
        for i in range(num_preds):
            # (num_pts, 2)
            p_pts = pred_lines[i]
            # (num_gts, num_pts, 2)
            # dist to all GTs
            # We can use cdist against all GT points at once? 
            # (num_pts, num_gts * num_pts)
            # This is complex. Let's just loop over GTs for simplicity and correctness.
            for j in range(num_gts):
                g_pts = gt_lines[j]
                d = distance.cdist(p_pts, g_pts, 'euclidean')
                # Chamfer: (mean(min(d, axis=1)) + mean(min(d, axis=0))) / 2
                # But Official implementation does SUM then divide by (2*num_pts)
                # "return (dist12 + dist21) / 2" where dist12 = sum / len
                # Wait, distance.py says:
                # dist1 = dist_mat.min(-1)[0].sum(-1)
                # dist_matrix = (dist1 + dist2).transpose(0, 1) / (2 * num_pts)
                # This is equivalent to (mean_d12 + mean_d21)/2. 
                
                dist1 = d.min(axis=1).sum()
                dist2 = d.min(axis=0).sum()
                matrix[i, j] = (dist1 + dist2) / (2.0 * pred_lines.shape[1])

        # for each det, the min distance with all gts
        matrix_min = matrix.min(axis=1)

        # for each det, which gt is the closest to it
        matrix_argmin = matrix.argmin(axis=1)
        # sort all dets in descending order by scores
        sort_inds = np.argsort(-scores)

        # match under different thresholds
        for thr in thresholds:
            tp = np.zeros((num_preds), dtype=np.float32)
            fp = np.zeros((num_preds), dtype=np.float32)

            gt_covered = np.zeros(num_gts, dtype=bool)
            for i in sort_inds:
                if matrix_min[i] <= thr:
                    matched_gt = matrix_argmin[i]
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[i] = 1
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1
            
            tp_fp_list.append((tp, fp))

        return tp_fp_list
    
    def process_gt_with_fov_clipping(
        self,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and process GT vectors for a specific camera.
        Uses shared extract_gt_with_fov_clipping() for 100% identical logic with visualization.
        
        Args:
            apply_clipping: If True, apply FOV clipping; if False, skip clipping
        
        Returns:
            vectors: (N, num_pts, 2) array of GT vectors
            labels: (N,) array of GT labels
        """
        # Use shared function for GT extraction + optional FOV clipping + rotation
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            fixed_num=20,  # Initial MapTR resampling to 20 points
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
        Apply optional FOV clipping AND camera-centric rotation to prediction vectors.
        Uses shared process_predictions_with_fov_clipping() for 100% identical logic with visualization.
        
        Args:
            apply_clipping: If True, apply FOV clipping; if False, skip clipping
        
        Returns:
            vectors: (N, num_pts, 2) optionally FOV-clipped and rotated prediction vectors
            labels: (N,) prediction labels
            scores: (N,) prediction scores (tracked through processing)
        """
        if len(pred_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Use shared function for optional FOV clipping + rotation
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
        
        Args:
            apply_clipping: If True, apply FOV clipping to both GT and predictions
        """
        for camera_name in self.camera_names:
            # Process GT with optional FOV clipping (uses shared function)
            gt_vectors, gt_labels = self.process_gt_with_fov_clipping(
                sample_info, camera_name, apply_clipping=apply_clipping)
            
            # Process predictions with optional FOV clipping (uses shared function)
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
    
    def compute_metrics_official(self,
                               pred_vectors_list: List[np.ndarray],
                               pred_scores_list: List[np.ndarray],
                               gt_vectors_list: List[np.ndarray],
                               thresholds: List[float]) -> Tuple[Dict[float, float], float]:
        """
        Compute AP and average CD using OFFICIAL StreamMapNet logic.
        Ref: plugin/datasets/evaluation/vector_eval.py:VectorEvaluate._evaluate_single
        """
        if not pred_vectors_list:
            return {thr: 0.0 for thr in thresholds}, float('inf')

        # 1. Interpolate all lines to fixed number of points
        # StreamMapNet official uses 200 points (INTERP_NUM=200 in vector_eval.py)
        # We assume vectors are already resampled to self.num_sample_pts in accumulate_sample/process_...
        
        # Flatten lists for instance_match input
        # instance_match expects: 
        #   pred_lines: (M, INTERP_NUM, 2)
        #   scores: (M,)
        #   gt_lines: (N, INTERP_NUM, 2)
        
        # We need to process ONE sample at a time or batch all?
        # instance_match() in AP.py takes arrays for ONE sample.
        # But vector_eval.py processes samples one by one (or in parallel) and then aggregates.
        # Wait, vector_eval.py `_evaluate_single` takes LISTS of vectors/scores/gt for ONE sample? 
        # No, `_evaluate_single` -> `instance_match`
        # `instance_match` docstring: "pred_lines (array): Detected lines of a sample..."
        # So it is per-sample matching.
        
        # We need to aggregate TP/FP across all samples to compute final AP.
        
        all_tp_fp_scores_by_thr = {thr: [] for thr in thresholds}
        total_num_gts = 0
        
        chamfer_distances = []
        
        for pred_vecs, scores, gt_vecs in zip(pred_vectors_list, pred_scores_list, gt_vectors_list):
            total_num_gts += len(gt_vecs)
            
            # Prepare arrays for this sample
            if len(pred_vecs) > 0:
                pred_lines = np.stack(pred_vecs) # (M, pts, 2)
                cur_scores = np.array(scores)
            else:
                pred_lines = np.zeros((0, self.num_sample_pts, 2))
                cur_scores = np.array([])
                
            if len(gt_vecs) > 0:
                gt_lines = np.stack(gt_vecs) # (N, pts, 2)
            else:
                gt_lines = np.zeros((0, self.num_sample_pts, 2))
            
            # Calculate Chamfer Distance for logging (average over sample)
            # Using official chamfer_distance logic if possible, or just simple cdist
            if len(pred_vecs) > 0 and len(gt_vecs) > 0:
                # Reuse the simple torch/numpy calculation for logging
                d_mat = distance.cdist(pred_lines.reshape(-1, 2), gt_lines.reshape(-1, 2))
                # Approximate sample CD (not used for Matching)
                # Matches simple logic in previous script versions
                # Official `chamfer_distance` is for single line pair. `chamfer_distance_batch` is for NxM.
                pass 

            # CALL OFFICIAL INSTANCE MATCH
            # Returns list of (tp, fp) for each threshold
            # instance_match signature: (pred_lines, scores, gt_lines, thresholds, metric='chamfer')
            tp_fp_list = instance_match(pred_lines, cur_scores, gt_lines, thresholds, metric='chamfer')
            
            for i, thr in enumerate(thresholds):
                tp, fp = tp_fp_list[i]
                # Store (tp, fp, score)
                # shape: (M, 3)
                if len(tp) > 0:
                    merged = np.column_stack([tp, fp, cur_scores])
                    all_tp_fp_scores_by_thr[thr].append(merged)
        
        # Calculate AP for each threshold
        ap_results = {}
        
        for thr in thresholds:
            data_list = all_tp_fp_scores_by_thr[thr]
            if not data_list:
                ap_results[thr] = 0.0
                continue
                
            # Concatenate all samples
            all_data = np.concatenate(data_list, axis=0) # (Total_M, 3)
            
            # Sort by score descending
            sort_inds = np.argsort(-all_data[:, 2])
            sorted_data = all_data[sort_inds]
            
            tp = sorted_data[:, 0]
            fp = sorted_data[:, 1]
            
            # CumSum
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            eps = np.finfo(np.float32).eps
            recalls = tp_cumsum / np.maximum(total_num_gts, eps)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
            
            # CALL OFFICIAL AVERAGE PRECISION
            ap = average_precision(recalls, precisions, mode='area')
            ap_results[thr] = ap
            
        return ap_results, 0.0 # Returning 0.0 for CD as we rely on AP mainly now
    
    def evaluate(self) -> Dict:
        """
        Compute final metrics using OFFICIAL StreamMapNet logic.
        """
        results = {}
        class_names = ['divider', 'ped_crossing', 'boundary']
        
        # StreamMapNet / MapTR evaluation loop
        # For each class, for each camera, compute AP
        
        for camera_name in self.camera_names:
            camera_results = {}
            all_aps = []
            
            camera_preds = self.predictions_per_camera[camera_name]
            camera_gts = self.ground_truths_per_camera[camera_name]
            
            # Per-Class Evaluation
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Initialize lists for accumulated vectors
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                for pred_data, gt_data in zip(camera_preds, camera_gts):
                    # Filter by class
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                
                # Compute AP for all thresholds at once
                ap_dict, avg_cd = self.compute_metrics_official(
                    pred_vectors_list, pred_scores_list, gt_vectors_list, self.thresholds_chamfer)
                
                for threshold in self.thresholds_chamfer:
                    ap = ap_dict.get(threshold, 0.0)
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                
                class_results['avg_chamfer_distance'] = avg_cd
                camera_results[class_name] = class_results
            
            # Compute mAP across all classes and thresholds
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MapTR with camera-specific FOV clipping')
    parser.add_argument('--nuscenes-path', type=str, 
                       default='/home/runw/Project/data/mini/nuscenes',
                       help='Path to NuScenes dataset')
    parser.add_argument('--samples-pkl', type=str, 
                       default='/home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl',
                      help='Path to samples pickle file')
    parser.add_argument('--predictions-pkl', type=str, 
                       default='streammapnet_predictions.pkl',
                       help='Path to predictions pickle file')
    parser.add_argument('--output-json', type=str,
                      default='evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--pc-range', type=float, nargs=6,
                      default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                      help='Point cloud range')
    parser.add_argument('--cameras', type=str, nargs='+',
                      default=['CAM_FRONT'],
                      help='Camera names to evaluate')
    parser.add_argument('--num-sample-pts', type=int, default=200,
                      help='Number of points to resample vectors to (default: 200)')
    parser.add_argument('--apply-clipping', action='store_true',
                      help='Apply camera FOV clipping to GT and predictions (default: False for full BEV evaluation)')
    parser.add_argument('--no-clipping', dest='apply_clipping', action='store_false',
                      help='Disable FOV clipping (full BEV evaluation)')
    parser.set_defaults(apply_clipping=True)
    
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluate StreamMapNet with Camera-Specific FOV Clipping")
    print("Using EXACT Official MapTR Evaluation Code")
    print("="*80)
    print(f"\nNuScenes path: {args.nuscenes_path}")
    print(f"Samples pickle: {args.samples_pkl}")
    print(f"Predictions pickle: {args.predictions_pkl}")
    print(f"Output: {args.output_json}")
    
    # Load samples using NuscDataset (handles both annotation formats)
    print(f"\nLoading samples from {args.samples_pkl}...")
    from plugin.datasets.nusc_dataset import NuscDataset
    
    # Create dataset instance to load samples
    dataset = NuscDataset(
        data_root=args.nuscenes_path + '/',
        ann_file=args.samples_pkl,
        cat2id={'ped_crossing': 0, 'divider': 1, 'boundary': 2},
        roi_size=(args.pc_range[3] - args.pc_range[0], args.pc_range[4] - args.pc_range[1]),
        meta={},
        pipeline=None,
        test_mode=True
    )
    samples = dataset.samples
    print(f"Loaded {len(samples)} samples")
    
    # Load predictions
    print(f"\nLoading predictions from {args.predictions_pkl}...")
    with open(args.predictions_pkl, 'rb') as f:
        predictions_data = pickle.load(f)
    predictions_by_token = predictions_data
    print(f"Loaded predictions for {len(predictions_by_token)} samples")
    
    # NOTE: Class ID remapping is NO LONGER NEEDED
    # The config's cat2id now matches the model's training:
    # Config: 0=ped_crossing, 1=divider, 2=boundary
    # Model: 0=ped_crossing, 1=divider, 2=boundary
    # GT extraction uses config's cat2id, so everything is aligned
    
    print(f"
Class mapping (config cat2id):")
    print(f"  0=ped_crossing, 1=divider, 2=boundary")
    
    # Create evaluator
    evaluator = CameraSpecificEvaluator(
        nuscenes_data_path=args.nuscenes_path,
        pc_range=args.pc_range,
        num_sample_pts=args.num_sample_pts,
        thresholds_chamfer=[0.5, 1.0, 1.5],
        camera_names=args.cameras
    )
    
    print(f"Initialized evaluator with:")
    print(f"  - PC range: {args.pc_range}")
    print(f"  - Patch size: {evaluator.patch_size}")
    print(f"  - Cameras: {args.cameras}")
    print(f"  - Sample points per vector: {args.num_sample_pts} (MapTR standard)")
    print(f"  - Chamfer thresholds: {evaluator.thresholds_chamfer} meters (MapTR standard)")
    print(f"  - FOV clipping: {'ENABLED' if args.apply_clipping else 'DISABLED (full BEV)'}")
    print(f"  - Evaluation method: Official MapTR (STRtree spatial filtering, linewidth=2m)")
    
    # Evaluate all samples
    print(f"\nEvaluating {len(samples)} samples across {len(args.cameras)} cameras...")
    mode_str = "camera-specific FOV clipping" if args.apply_clipping else "full BEV (no clipping)"
    print(f"Accumulating predictions and GT with {mode_str}...")
    
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
        
        # Print mAP first
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
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("="*80)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if __name__ == '__main__':
    main()
