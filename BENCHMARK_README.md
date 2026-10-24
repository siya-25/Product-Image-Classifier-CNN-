# Model Benchmarking Guide

## Overview

The benchmark module compares multiple CNN architectures (Custom CNN, MobileNetV2, ResNet18, EfficientNet-B0) using identical dataset splits and evaluation settings.

## Key Features

1. **Identical Data Splits**: All models use the same train/val/test split via `use_saved_split=True`
2. **Fair Comparison**: Same batch size, augmentation, and evaluation metrics for all models
3. **Comprehensive Metrics**: Records accuracy, precision, recall, F1, parameter count, model size, and inference latency
4. **Model Selection**: Selects best model based on validation F1 macro (not test set)
5. **Test Set Isolation**: Evaluates selected model on test set only once after selection
6. **Machine-Readable Results**: Saves JSON results and comparison chart

## Usage

### Run Full Benchmark

```bash
python -m src.benchmark.benchmark
```

This will:
1. Load the base config from `configs/config.yaml`
2. Load benchmark config from `configs/benchmark_config.yaml`
3. Train each model (skipping if checkpoint exists and `skip_existing=True`)
4. Evaluate each model on validation and test sets
5. Save individual model results
6. Select best model based on validation F1 macro
7. Generate comparison chart

### Configuration

Edit `configs/benchmark_config.yaml`:

```yaml
# Models to benchmark
models:
  - custom
  - mobilenet_v2
  - resnet18
  - efficientnet_b0

# Training settings (reduced for benchmarking)
max_epochs: 20
early_stopping_patience: 5
batch_size: 32

# Data settings
use_saved_split: true  # Critical: use identical splits

# Execution settings
skip_existing: true  # Skip if checkpoint exists
device: auto  # auto, cuda, cpu
```

### Output Files

After running, the following files are created in `outputs/benchmark/`:

```
outputs/benchmark/
├── checkpoints/
│   ├── custom_best.pth
│   ├── mobilenet_v2_best.pth
│   ├── resnet18_best.pth
│   ├── efficientnet_b0_best.pth
│   ├── custom_config.yaml
│   ├── mobilenet_v2_config.yaml
│   ├── resnet18_config.yaml
│   └── efficientnet_b0_config.yaml
├── custom_results.json
├── mobilenet_v2_results.json
├── resnet18_results.json
├── efficientnet_b0_results.json
├── benchmark_results.json  # Combined results
├── model_selection.json    # Best model selection
└── model_comparison.png    # Comparison chart
```

### Results Format

Each model's results JSON contains:

```json
{
  "model_type": "custom",
  "best_val_acc": 0.8567,
  "epochs_trained": 15,
  "final_train_loss": 0.345,
  "final_val_loss": 0.412,
  "num_parameters": 2500000,
  "model_size_mb": 9.8,
  "inference_latency_ms": 12.5,
  "val_accuracy": 0.8567,
  "val_precision_macro": 0.8512,
  "val_recall_macro": 0.8489,
  "val_f1_macro": 0.8500,
  "test_accuracy": 0.8434,
  "test_precision_macro": 0.8389,
  "test_recall_macro": 0.8367,
  "test_f1_macro": 0.8378,
  "val_per_class": {...},
  "test_per_class": {...}
}
```

### Model Selection

The best model is selected based on **validation F1 macro** (not test accuracy) to prevent test set leakage. The selection is saved in `model_selection.json`:

```json
{
  "best_model": "mobilenet_v2",
  "selection_metric": "val_f1_macro",
  "selection_value": 0.8723,
  "test_accuracy": 0.8656,
  "test_f1_macro": 0.8689,
  "all_models_ranked": [
    {"model": "mobilenet_v2", "val_f1_macro": 0.8723, "test_f1_macro": 0.8689},
    {"model": "resnet18", "val_f1_macro": 0.8656, "test_f1_macro": 0.8612},
    ...
  ]
}
```

## Important Notes

### 1. Test Set Isolation
- **Critical**: The benchmark uses validation metrics for model selection
- Test set is evaluated **after** selection for final reporting only
- This prevents test set leakage and overfitting

### 2. Identical Splits
- All models use the same data split via `use_saved_split=True`
- Split indices are saved in `data/processed/split_indices.json`
- Ensures fair comparison between models

### 3. Training Time
- Benchmark uses reduced epochs (default: 20) for faster comparison
- Early stopping (patience: 5) prevents overfitting
- Adjust `max_epochs` in benchmark config if needed

### 4. Checkpoint Management
- Each model's checkpoint is saved with model-specific name
- Set `skip_existing: false` to retrain all models
- Set `skip_existing: true` to skip trained models

### 5. Resource Requirements
- Training 4 models requires significant GPU time
- Consider training on subset or using cloud GPU
- Monitor disk space for checkpoints

## Testing

Run benchmark tests:

```bash
python tests/test_benchmark.py
```

Tests verify:
- Benchmark initialization
- Config loading
- Model size calculation
- Parameter count ranges
- Results structure
- Device selection
- Identical splits enforcement

## Integration with Training Pipeline

The benchmark module reuses existing modules:
- `src.model.cnn` - Model architectures
- `src.data.dataset` - Data loading with saved splits
- `src.train.trainer` - Training loop
- `src.evaluate.evaluator` - Evaluation metrics

No changes to existing modules were made.

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `batch_size` in benchmark config or use smaller models

### Issue: Training Too Slow
**Solution**: Reduce `max_epochs` or train fewer models

### Issue: Different Splits
**Solution**: Ensure `use_saved_split: true` and split indices exist

### Issue: Checkpoint Not Found
**Solution**: Check `outputs/checkpoints/` has `best_checkpoint.pth`

## Example Workflow

```bash
# 1. Download data (if not already done)
python -m src.data.download

# 2. Run benchmark
python -m src.benchmark.benchmark

# 3. View results
cat outputs/benchmark/benchmark_results.json
cat outputs/benchmark/model_selection.json

# 4. View comparison chart
# Open outputs/benchmark/model_comparison.png

# 5. Use best model for final evaluation
# The best checkpoint is in outputs/benchmark/checkpoints/{best_model}_best.pth
```

## Next Steps

After benchmarking:
1. Review `model_selection.json` for best model
2. Use best model's checkpoint for production
3. Fine-tune best model with full training if needed
4. Deploy best model via demo app
