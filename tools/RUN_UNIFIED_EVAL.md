# How to Run StreamMapNet Unified Evaluation Script

## Quick Start

### 1. Basic Usage (All 6 Cameras)

```bash
cd /home/runw/Project/StreamMapNet

python tools/streammapnet_eval_unified.py \
  --config plugin/configs/nusc_baseline_480_60x30_30e.py \
  --checkpoint ckpts/nusc_baseline_480_60x30_30e.pth \
  --nuscenes-path /home/runw/Project/data/mini/nuscenes \
  --samples-pkl /home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl \
  --cameras all \
  --predictions-pkl predictions_all_cameras.pkl \
  --output-json results_all_cameras.json
```

### 2. Front Camera Only

```bash
python tools/streammapnet_eval_unified.py \
  --config plugin/configs/nusc_baseline_480_60x30_30e.py \
  --checkpoint ckpts/nusc_baseline_480_60x30_30e.pth \
  --nuscenes-path /home/runw/Project/data/mini/nuscenes \
  --samples-pkl /home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl \
  --cameras CAM_FRONT \
  --predictions-pkl predictions_front_only.pkl \
  --output-json results_front_only.json
```

### 3. Front + Back Cameras

```bash
python tools/streammapnet_eval_unified.py \
  --config plugin/configs/nusc_baseline_480_60x30_30e.py \
  --checkpoint ckpts/nusc_baseline_480_60x30_30e.pth \
  --nuscenes-path /home/runw/Project/data/mini/nuscenes \
  --samples-pkl /home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl \
  --cameras CAM_FRONT CAM_BACK \
  --predictions-pkl predictions_front_back.pkl \
  --output-json results_front_back.json
```

### 4. Front Hemisphere (3 Cameras)

```bash
python tools/streammapnet_eval_unified.py \
  --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
  --predictions-pkl predictions_front_hemisphere.pkl \
  --output-json results_front_hemisphere.json
```

## Required Arguments

### Mandatory (must provide these)

1. **--config**: Path to StreamMapNet config file
   - Default: `plugin/configs/nusc_baseline_480_60x30_30e.py`
   - Your path: `/home/runw/Project/StreamMapNet/plugin/configs/nusc_baseline_480_60x30_30e.py`

2. **--checkpoint**: Path to StreamMapNet checkpoint
   - Default: `ckpts/nusc_baseline_480_60x30_30e.pth`
   - Your path: `/home/runw/Project/StreamMapNet/ckpts/nusc_baseline_480_60x30_30e.pth`

3. **--nuscenes-path**: Path to NuScenes dataset root
   - Default: `/home/runw/Project/data/mini/nuscenes`
   - Full dataset: `/home/runw/Project/data/nuscenes` (if you have it)

4. **--samples-pkl**: Path to annotation pickle file
   - Default: `/home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl`
   - Make sure this file exists!

### Optional Arguments

5. **--cameras**: Which cameras to use (space-separated)
   - Options: `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `all`
   - Default: `all`
   - Examples:
     ```bash
     --cameras CAM_FRONT
     --cameras CAM_FRONT CAM_BACK
     --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT
     --cameras all
     ```

6. **--predictions-pkl**: Where to save/load predictions
   - Default: `streammapnet_predictions.pkl`
   - Example: `predictions_front_only.pkl`

7. **--output-json**: Where to save evaluation results
   - Default: `evaluation_results.json`
   - Example: `results_front_only.json`

8. **--score-thresh**: Score threshold for filtering predictions
   - Default: `0.0` (keep all predictions)
   - Range: `0.0` to `1.0`
   - Example: `--score-thresh 0.3`

9. **--skip-inference**: Skip inference step (use existing predictions)
   - Use this if you already have predictions from a previous run
   - Example:
     ```bash
     python tools/streammapnet_eval_unified.py \
       --skip-inference \
       --predictions-pkl existing_predictions.pkl \
       --cameras CAM_FRONT \
       --output-json results.json
     ```

10. **--apply-clipping / --no-clipping**: Enable/disable camera FOV clipping
    - Default: `--apply-clipping` (enabled)
    - Use `--no-clipping` for full BEV evaluation without FOV restrictions
    - Example:
      ```bash
      python tools/streammapnet_eval_unified.py \
        --no-clipping \
        --cameras CAM_FRONT
      ```

## Advanced Usage

### Compare Different Camera Configurations

Run multiple evaluations to compare performance:

```bash
# 1. Front camera only
python tools/streammapnet_eval_unified.py \
  --cameras CAM_FRONT \
  --predictions-pkl preds_1cam.pkl \
  --output-json results_1cam.json

# 2. Front + Back
python tools/streammapnet_eval_unified.py \
  --cameras CAM_FRONT CAM_BACK \
  --predictions-pkl preds_2cam.pkl \
  --output-json results_2cam.json

# 3. All 6 cameras (baseline)
python tools/streammapnet_eval_unified.py \
  --cameras all \
  --predictions-pkl preds_6cam.pkl \
  --output-json results_6cam.json

# Compare results
python -c "
import json
for name, file in [('1 cam', 'results_1cam.json'), 
                   ('2 cam', 'results_2cam.json'), 
                   ('6 cam', 'results_6cam.json')]:
    with open(file) as f:
        data = json.load(f)
    cam = list(data.keys())[0]
    print(f'{name}: mAP = {data[cam][\"mAP\"]:.4f}')
"
```

### Reuse Predictions for Different Evaluations

```bash
# Step 1: Run inference once (all cameras)
python tools/streammapnet_eval_unified.py \
  --cameras all \
  --predictions-pkl predictions.pkl \
  --output-json results_all_with_clipping.json

# Step 2: Re-evaluate same predictions without FOV clipping
python tools/streammapnet_eval_unified.py \
  --skip-inference \
  --predictions-pkl predictions.pkl \
  --cameras all \
  --no-clipping \
  --output-json results_all_no_clipping.json
```

### Evaluate on Different Camera Subsets (from same predictions)

```bash
# Run inference once with all cameras
python tools/streammapnet_eval_unified.py \
  --cameras all \
  --predictions-pkl predictions_all.pkl \
  --output-json results_all.json

# Then evaluate different camera subsets using same predictions
# Note: The predictions contain data from all cameras, but evaluation 
# will apply per-camera FOV clipping based on --cameras argument
python tools/streammapnet_eval_unified.py \
  --skip-inference \
  --predictions-pkl predictions_all.pkl \
  --cameras CAM_FRONT \
  --output-json results_front_from_all.json
```

## Troubleshooting

### Common Issues

1. **File not found errors**
   ```bash
   # Check if files exist
   ls -lh plugin/configs/nusc_baseline_480_60x30_30e.py
   ls -lh ckpts/nusc_baseline_480_60x30_30e.pth
   ls -lh /home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl
   ```

2. **CUDA out of memory**
   - The script uses `samples_per_gpu=1` by default
   - Make sure only 1 GPU process is running
   - Check: `nvidia-smi`

3. **Wrong camera names**
   - Valid camera names (case-sensitive):
     - `CAM_FRONT`
     - `CAM_FRONT_RIGHT`
     - `CAM_FRONT_LEFT`
     - `CAM_BACK`
     - `CAM_BACK_LEFT`
     - `CAM_BACK_RIGHT`
     - `all`

4. **Debug output**
   - The script has extensive debug logging
   - On first sample, it will print:
     - Token extraction info
     - Image tensor shape
     - Result structure
     - Denormalization steps (before/after values)
     - Rotation steps (before/after ranges)

## Output Files

### 1. Predictions File (`*.pkl`)

Binary pickle file containing predictions for all samples:
```python
{
  'sample_token_1': {
    'vectors': np.array([num_preds, 20, 2]),  # Coordinates in meters
    'labels': np.array([num_preds]),          # Class IDs (0=divider, 1=ped_crossing, 2=boundary)
    'scores': np.array([num_preds])           # Confidence scores
  },
  'sample_token_2': { ... },
  ...
}
```

### 2. Results File (`*.json`)

JSON file with per-camera evaluation metrics:
```json
{
  "CAM_FRONT": {
    "mAP": 0.4523,
    "divider": {
      "AP@0.5m": 0.5234,
      "AP@1.0m": 0.6123,
      "AP@1.5m": 0.6821,
      "avg_chamfer_distance": 0.3421
    },
    "ped_crossing": { ... },
    "boundary": { ... }
  }
}
```

## Example Workflow

Complete workflow for evaluation on mini dataset:

```bash
cd /home/runw/Project/StreamMapNet

# Test 1: Front camera only
python tools/streammapnet_eval_unified.py \
  --cameras CAM_FRONT \
  --predictions-pkl predictions_front.pkl \
  --output-json results_front.json

# Test 2: All cameras (baseline)
python tools/streammapnet_eval_unified.py \
  --cameras all \
  --predictions-pkl predictions_all.pkl \
  --output-json results_all.json

# Compare results
echo "Front camera only:"
python -c "import json; print('mAP:', json.load(open('results_front.json'))['CAM_FRONT']['mAP'])"

echo "All cameras:"
python -c "import json; print('mAP:', json.load(open('results_all.json'))['CAM_FRONT']['mAP'])"
```

## Next Steps

1. **Start with a quick test** (front camera only, small dataset)
2. **Check the debug output** to verify everything is working
3. **Run full evaluation** with all cameras
4. **Compare results** across different camera configurations
