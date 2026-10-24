# Probability Calibration Guide

## Overview

Probability calibration has been added to improve the confidence-based manual-review feature. The system now:
- Calibrates model probabilities using temperature scaling
- Selects confidence thresholds based on precision-coverage analysis
- Shows top-k predictions with confidence scores
- Persists calibration parameters consistently
- Keeps test set out of threshold selection

## Features

### 1. Temperature Scaling
- Learns a single temperature parameter to calibrate model outputs
- Improves probability calibration
- Fitted on validation set only
- Saves temperature to checkpoint

### 2. Precision-Coverage Analysis
- Analyzes precision vs coverage at various confidence thresholds
- Selects threshold that meets target precision (default: 0.95)
- Plots precision-coverage curve for visualization
- Uses validation set only (test set isolation)

### 3. Top-K Predictions
- Shows top-k predictions with confidence scores (default: k=3)
- Helps understand model uncertainty
- Useful for ambiguous cases
- Configurable k value

### 4. Consistent Persistence
- Calibration parameters saved to checkpoint
- Threshold and temperature persisted
- Automatically loaded in demo
- Consistent between training and inference

## Usage

### During Training

Run calibration after training:

```bash
python -m src.train.trainer --calibrate --target_precision 0.95 --k 3
```

This will:
1. Train the model as usual
2. Calibrate probabilities on validation set
3. Select threshold based on target precision
4. Save calibration to checkpoint

### In Python Code

```python
from src.train.trainer import Trainer

trainer = Trainer(config_path="configs/config.yaml")
best_acc = trainer.train()

# Calibrate after training
calibration_results = trainer.calibrate_model(
    target_precision=0.95,
    k=3
)
```

### In Evaluation

The evaluator automatically loads calibration from checkpoint:

```python
from src.evaluate.evaluator import Evaluator

evaluator = Evaluator(
    checkpoint_path="outputs/checkpoints/best_checkpoint.pth",
    config_path="configs/config.yaml"
)

# Calibration automatically loaded if available
results = evaluator.evaluate(test_loader, val_loader=val_loader)
```

### In Demo App

The demo automatically loads calibration and uses calibrated probabilities:

```bash
python app/demo.py
```

The demo shows:
- Predicted class
- Calibrated confidence
- Manual review flag (based on calibrated threshold)
- Top-k predictions with confidence scores
- Grad-CAM visualizations (if enabled)

## Configuration

Edit `configs/calibration_config.yaml`:

```yaml
temperature_scaling:
  enabled: true
  max_epochs: 10
  learning_rate: 0.01

threshold_selection:
  target_precision: 0.95
  method: precision_coverage

top_k:
  k: 3
  show_all_probabilities: true

persistence:
  save_dir: "outputs/calibration"
  save_to_checkpoint: true
```

## Calibration Process

### Step 1: Temperature Scaling
1. Load validation set
2. Initialize temperature scaling module
3. Optimize temperature using LBFGS
4. Minimize cross-entropy loss
5. Converges when temperature change < 1e-4

### Step 2: Precision-Coverage Analysis
1. Get calibrated probabilities on validation set
2. Calculate precision and coverage at 100 thresholds (0.0 to 1.0)
3. Select highest threshold that meets target precision
4. Plot precision-coverage curve

### Step 3: Persistence
1. Save temperature to checkpoint
2. Save threshold to checkpoint
3. Save calibration config to JSON
4. Save precision-coverage curve plot

## Output Files

Created in `outputs/calibration/`:
- `calibration_config.json` - Calibration parameters
- `temperature_scaling.pth` - Temperature scaling state
- `precision_coverage_curve.png` - Precision-coverage plot

## Test Set Isolation

**Critical**: Calibration uses only the validation set:
- Temperature scaling fitted on validation set
- Threshold selection based on validation precision
- Test set never used in calibration
- Test set evaluated only once after model selection

This prevents data leakage and ensures honest performance estimation.

## Limitations

### 1. Confidence Not a Guarantee
- High confidence does NOT guarantee correctness
- Calibration improves probability estimates but doesn't eliminate errors
- Manual review should still be performed for critical applications
- Calibration assumes validation set is representative

### 2. Temperature Scaling Limitations
- Single temperature may not calibrate all classes equally
- May not work well for highly imbalanced data
- Assumes model is reasonably trained
- Fitting quality depends on validation set size

### 3. Threshold Selection
- Target precision is a heuristic
- Higher precision = lower coverage (more manual review)
- Lower precision = higher coverage (more auto-accept)
- Trade-off depends on business requirements

### 4. Calibration Drift
- Calibration tied to specific model checkpoint
- If model is retrained, must recalibrate
- Distribution shift may require recalibration
- Periodic recalibration recommended for production

### 5. Top-K Limitations
- Top-k may include very low-confidence predictions
- Not useful if model is very uncertain
- High k may confuse users
- Default k=3 is reasonable for 3-class problem

## Integration with Existing Pipeline

The calibration module:
- Reuses existing model architectures
- Uses same data splits (validation only)
- Does not require retraining
- Integrates with trainer, evaluator, and demo
- Preserves existing functionality

## Testing

Run calibration tests:

```bash
python tests/test_calibration.py
```

Tests verify:
- Temperature scaling initialization and forward pass
- Calibrator initialization
- Calibrated probability computation
- Prediction with confidence scores
- Calibration persistence (save/load)
- Precision-coverage analysis structure
- Threshold selection logic
- Behavior without temperature scaling

## Important Notes

### Calibration is Optional
- System works without calibration (uses raw softmax)
- Can be disabled by not running calibration
- Demo falls back to manual threshold if calibration not found
- Backward compatible with existing workflow

### Confidence Scores
- Calibrated probabilities are better estimates of true correctness
- Still not guarantees of correctness
- Should be used as heuristics, not absolute measures
- Always verify uncertain predictions manually

### Manual Review Flagging
- Predictions below threshold flagged for manual review
- Threshold selected to meet target precision on validation set
- Adjust target_precision based on business requirements
- Higher precision = more conservative (more manual review)

## Troubleshooting

### Issue: Calibration fails to converge
**Solution**: Increase max_epochs, check validation set size, ensure model is trained

### Issue: Threshold too low/high
**Solution**: Adjust target_precision, check precision-coverage curve

### Issue: Calibration not loaded in demo
**Solution**: Ensure calibration was run after training, check checkpoint contains calibration data

### Issue: Top-k shows low confidences
**Solution**: Model may be underconfident, check calibration quality, reduce k

## Examples

### Calibration Command

```bash
# Calibrate with target precision 0.95
python -m src.train.trainer --calibrate --target_precision 0.95

# Calibrate with higher precision (more conservative)
python -m src.train.trainer --calibrate --target_precision 0.98

# Calibrate with more top predictions
python -m src.train.trainer --calibrate --k 5
```

### Programmatic Usage

```python
from src.calibration.calibration import Calibrator

calibrator = Calibrator(model, num_classes=3, device=device)
results = calibrator.calibrate(val_loader, target_precision=0.95, k=3)

print(f"Temperature: {results['temperature']}")
print(f"Threshold: {results['threshold']}")
print(f"Precision at threshold: {results['precision_at_threshold']}")
print(f"Coverage at threshold: {results['coverage_at_threshold']}")
```

## References

- Temperature Scaling: https://arxiv.org/abs/1706.04599
- Confidence Calibration: https://arxiv.org/abs/1706.04599
- Precision-Recall Trade-off: Standard ML practice
