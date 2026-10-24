# Calibration Extension Summary

## Overview

Extended the confidence-based manual-review feature with probability calibration, precision-coverage analysis, and top-k predictions. The system now calibrates model probabilities, selects confidence thresholds based on validation precision, and shows top-k predictions with confidence scores.

## Implementation

### New Module: `src/calibration/`
- **File**: `src/calibration/calibration.py` (425 lines)
- **Classes**: `TemperatureScaling`, `Calibrator`
- **Features**:
  - Temperature scaling for probability calibration
  - Precision-coverage analysis for threshold selection
  - Configurable target precision (default: 0.95)
  - Top-k predictions with confidence scores (default: k=3)
  - Calibration persistence to checkpoint
  - Consistent loading in trainer, evaluator, and demo

### Temperature Scaling

**Class**: `TemperatureScaling`
- Learns single temperature parameter
- Optimized using LBFGS
- Minimizes cross-entropy on validation set
- Converges when temperature change < 1e-4
- Improves probability calibration

### Precision-Coverage Analysis

**Method**: Analyze precision vs coverage at 100 thresholds
1. Calculate precision and coverage at each threshold
2. Select highest threshold meeting target precision
3. Plots precision-coverage curve
4. Uses validation set only (test set isolation)

### Threshold Selection

**Algorithm**:
1. Generate calibrated probabilities on validation set
2. Calculate precision at each threshold (0.0 to 1.0)
3. Select highest threshold where precision >= target_precision
4. Default target_precision: 0.95
5. Ensures auto-accept decisions meet precision goal

### Top-K Predictions

**Feature**: Show top-k predictions with confidence scores
- Default k=3 (appropriate for 3-class problem)
- Shows uncertainty in model predictions
- Helps identify ambiguous cases
- Configurable via parameter

## Integration

### Modified Files

1. **src/train/trainer.py**
   - Added `calibrate_model()` method
   - Added `--calibrate` CLI flag
   - Saves calibration to checkpoint
   - Integrated temperature scaling with validation set

2. **src/evaluate/evaluator.py**
   - Loads calibration from checkpoint
   - Uses calibrated probabilities if available
   - Falls back to raw softmax if not calibrated

3. **app/demo.py**
   - Loads calibration from checkpoint
   - Uses calibrated threshold
   - Shows top-k predictions
   - Displays confidence for all classes

### Configuration

**File**: `configs/calibration_config.yaml`
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

## Test Set Isolation

**Critical**: Calibration uses only validation set:
- Temperature scaling fitted on validation set
- Threshold selection based on validation precision
- Test set never used in calibration
- Test set evaluated only once after selection

This prevents data leakage and ensures unbiased performance estimation.

## Output Files

Created in `outputs/calibration/`:
- `calibration_config.json` - Calibration parameters (temperature, threshold, k)
- `temperature_scaling.pth` - Temperature scaling state
- `precision_coverage_curve.png` - Precision-coverage plot

## Usage

### After Training
```bash
python -m src.train.trainer --calibrate --target_precision 0.95 --k 3
```

### In Demo
```bash
python app/demo.py
```
Automatically loads calibration if available in checkpoint.

### In Code
```python
from src.calibration.calibration import Calibrator

calibrator = Calibrator(model, num_classes=3, device=device)
results = calibrator.calibrate(val_loader, target_precision=0.95, k=3)
```

## Test Coverage

### New Test File: `tests/test_calibration.py` (275 lines)

8 focused tests:
1. Temperature scaling initialization and forward pass
2. Calibrator initialization
3. Calibrated probability computation
4. Prediction with confidence scores
5. Calibration persistence (save/load)
6. Precision-coverage analysis structure
7. Threshold selection logic
8. Behavior without temperature scaling

## Key Features

### 1. Improved Confidence Scores
- Temperature scaling improves probability calibration
- Better estimates of true correctness
- More reliable confidence thresholds

### 2. Data-Driven Threshold Selection
- Threshold selected based on validation precision
- Configurable target precision
- Precision-coverage curve for analysis

### 3. Top-K Predictions
- Shows multiple predictions with confidence
- Helps understand model uncertainty
- Useful for ambiguous cases

### 4. Consistent Persistence
- Calibration saved to checkpoint
- Automatically loaded in demo
- Consistent between training and inference

### 5. Test Set Isolation
- Calibration uses validation set only
- Test set never used in threshold selection
- Honest performance estimation

## Limitations

### 1. Confidence Not a Guarantee
- Calibrated confidence scores are better estimates
- Still do not guarantee correctness
- Manual review still required for uncertain predictions
- Documented in README and CALIBRATION_README

### 2. Temperature Scaling Limitations
- Single temperature may not calibrate all classes equally
- May not work well for highly imbalanced data
- Assumes model is reasonably trained
- Fitting quality depends on validation set size

### 3. Threshold Selection Trade-off
- Higher precision = lower coverage (more manual review)
- Lower precision = higher coverage (more auto-accept)
- Target precision is a heuristic
- Business requirements may vary

### 4. Calibration Drift
- Calibration tied to specific model checkpoint
- If model retrained, must recalibrate
- Distribution shift may require recalibration
- Periodic recalibration recommended

### 5. Top-K Limitations
- Top-k may include very low-confidence predictions
- Not useful if model is very uncertain
- High k may confuse users
- Default k=3 is reasonable for 3-class problem

## Verification

### ✅ Syntax Validation
All new files compile successfully:
- `src/calibration/calibration.py` ✓
- `src/train/trainer.py` ✓
- `src/evaluate/evaluator.py` ✓
- `app/demo.py` ✓
- `tests/test_calibration.py` ✓

### ✅ YAML Validation
- `configs/calibration_config.yaml` ✓ (valid YAML)

### ⚠️ Runtime Testing
Tests could NOT be executed due to missing dependencies (torch, numpy, etc.) in current environment.

### ❌ Training Execution
No training was executed. Calibration runs after training but doesn't require retraining.

## Backward Compatibility

All changes are backward compatible:
- Calibration is optional (can be disabled)
- System works without calibration (uses raw softmax)
- Demo falls back to manual threshold if calibration not found
- Existing training/evaluation workflow unchanged
- Checkpoints remain compatible

## Documentation

- **CALIBRATION_README.md** - User guide with usage examples and limitations
- **CALIBRATION_EXTENSION_SUMMARY.md** - This file
- Updated **README.md** with calibration usage and limitations

## Files Added

1. `src/calibration/__init__.py` - Module init
2. `src/calibration/__main__.py` - Main entry point
3. `src/calibration/calibration.py` - Calibration implementation (425 lines)
4. `configs/calibration_config.yaml` - Calibration configuration
5. `tests/test_calibration.py` - Calibration tests (275 lines)
6. `CALIBRATION_README.md` - User guide (287 lines)
7. `CALIBRATION_EXTENSION_SUMMARY.md` - This file

## Files Modified

1. `src/train/trainer.py` - Added calibration method and CLI flag
2. `src/evaluate/evaluator.py` - Added calibration loading support
3. `app/demo.py` - Integrated calibration and top-k predictions
4. `README.md` - Added calibration usage and documentation references

## Next Steps

To use calibration:
```bash
# Install dependencies (already in requirements.txt)
pip install -r requirements.txt

# Train model
python -m src.train.trainer

# Calibrate probabilities
python -m src.train.trainer --calibrate --target_precision 0.95

# Run tests
python tests/test_calibration.py

# Launch demo (auto-loads calibration)
python app/demo.py
```

## Integration with Existing Pipeline

The calibration module:
- Reuses existing model architectures
- Uses same data splits (validation only for calibration)
- Does not require retraining
- Integrates with trainer, evaluator, and demo
- Preserves Grad-CAM and benchmark functionality
- Maintains test set isolation

## Summary

The calibration extension adds:
- ✅ Temperature scaling for probability calibration
- ✅ Precision-coverage analysis for threshold selection
- ✅ Top-k predictions with confidence scores
- ✅ Consistent calibration persistence
- ✅ Test set isolation in threshold selection
- ✅ Comprehensive test coverage
- ✅ Detailed documentation with limitations
- ✅ Backward compatible with existing workflow

The system now provides better confidence estimates, data-driven threshold selection, and top-k predictions while maintaining test set isolation and documenting that confidence is not a guarantee of correctness.
