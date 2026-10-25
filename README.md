# Product Image Classifier (CNN)

A Flipkart-style e-commerce product image classifier that categorizes products into three classes: **Apparel**, **Electronics**, and **Home**. Built with PyTorch and trained on Indian e-commerce data.

## Problem Statement

E-commerce catalogues need automated product image categorization to scale operations. This project builds a CNN classifier that:
- Categorizes product images into Apparel, Electronics, or Home
- Achieves >85% validation accuracy
- Identifies misclassified items for manual review
- Provides confidence thresholds for auto-accept vs manual routing

## Dataset

### Source
- **Primary**: Hugging Face dataset `jason1966/PromptCloudHQ_flipkart-products` (20k Flipkart products)
- **License**: Unknown (public dataset)
- **Download Date**: 2026-10-10

### Label Mapping
The dataset's category tree is mapped to our 3 classes using keyword matching:

| Our Class | Mapped From Categories |
|-----------|----------------------|
| Apparel | Clothing, Footwear, Bags, Accessories, Jewellery, Fashion, Beauty |
| Electronics | Mobiles, Laptops, Cameras, Audio, Computers, Gaming, Smart Wearables |
| Home | Furniture, Home Decor, Kitchen, Dining, Appliances, Lighting, Garden |

See `configs/label_map.yaml` for the complete mapping.

### Dataset Statistics
- **Target**: 1,500 images per class (minimum)
- **Balance**: No class exceeds 2x another
- **Preprocessing**: 
  - Perceptual hash deduplication
  - Stratified train/val/test split (70/15/15)
  - Image resizing to 224x224
  - Data augmentation for training

## Model

### Architecture
We use a small custom CNN baseline:

```
Input (3x224x224)
├─ Conv2d(3, 64, 3x3) + BatchNorm + ReLU + MaxPool
├─ Conv2d(64, 128, 3x3) + BatchNorm + ReLU + MaxPool
├─ Conv2d(128, 256, 3x3) + BatchNorm + ReLU + MaxPool
├─ Conv2d(256, 512, 3x3) + BatchNorm + ReLU + MaxPool
├─ AdaptiveAvgPool + Flatten
├─ Dropout + Linear(512, 256) + ReLU
└─ Dropout + Linear(256, 3)
```

**Parameters**: ~2.5M trainable parameters

### Transfer Learning Options
If baseline doesn't reach 85% accuracy, the code supports:
- MobileNetV2 (lightweight, good for mobile)
- ResNet18 (balanced performance)
- EfficientNet-B0 (state-of-the-art efficiency)

These can be enabled via `configs/config.yaml`.

## Training Setup

### Configuration
All settings are in `configs/config.yaml`:
- **Batch size**: 32
- **Learning rate**: 0.001 (Adam optimizer)
- **Weight decay**: 0.0001
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early stopping**: 10 epochs patience
- **Mixed precision**: Enabled (when GPU available)

### Augmentation
Training augmentation includes:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)

### Reproducibility
- Fixed random seed (42)
- Deterministic CuDNN (may impact performance)
- Checkpoint saving with full state

## Results

### Performance Metrics
See `REPORT.md` for detailed results.

### Training Curves
- Loss and accuracy curves saved to `outputs/figures/training_curves.png`
- TensorBoard logs in `outputs/log_dir/`

### Confusion Matrix
- Confusion matrix heatmap in `outputs/figures/confusion_matrix.png`
- Top confused pairs identified with visual examples

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

Tested on:
- Python 3.8+
- PyTorch 2.3.0
- CUDA 11.8+ (for GPU)

## Usage

### 1. Download Data
```bash
python -m src.data.download
```
This downloads from Hugging Face, parses categories, downloads images, deduplicates, and balances classes.

### 2. Train Model
```bash
python -m src.train.trainer
```
Training progress is logged to console and TensorBoard. Best checkpoint saved to `outputs/checkpoints/best_checkpoint.pth`.

### 3. Calibrate Probabilities (Optional)
```bash
python -m src.train.trainer --calibrate --target_precision 0.95
```
Calibrates model probabilities and selects confidence threshold based on validation set. See `CALIBRATION_README.md` for details.

### 4. Evaluate
```bash
python -m src.evaluate.evaluator
```
Generates metrics, confusion matrix, and identifies confused pairs.

### 5. Benchmark Models (Optional)
```bash
python -m src.benchmark.benchmark
```
Compares Custom CNN, MobileNetV2, ResNet18, and EfficientNet-B0 with identical data splits. See `BENCHMARK_README.md` for details.

### 6. Launch Demo (with Grad-CAM and Calibration)
```bash
python app/demo.py
```
Opens a Gradio interface for interactive classification with Grad-CAM explainability and calibrated confidence scores. See `GRADCAM_README.md` and `CALIBRATION_README.md` for details.

## Colab Reproduction

A complete end-to-end notebook is provided in `notebooks/colab_run.ipynb`:

1. Clone the repository
2. Run all cells sequentially
3. The notebook will:
   - Install dependencies
   - Download data
   - Train the model
   - Evaluate and display results
   - Launch the demo

**Expected runtime on free T4**: ~30-45 minutes (depends on download speed)

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── download.py      # Dataset download and preprocessing
│   │   └── dataset.py       # PyTorch Dataset class
│   ├── model/
│   │   └── cnn.py           # Model architectures
│   ├── train/
│   │   └── trainer.py       # Training loop
│   ├── evaluate/
│   │   └── evaluator.py     # Evaluation and error analysis
│   ├── benchmark/
│   │   └── benchmark.py     # Model comparison benchmark
│   ├── explainability/
│   │   └── gradcam.py       # Grad-CAM explainability
│   ├── calibration/
│   │   └── calibration.py   # Probability calibration
│   └── utils/
│       └── helpers.py       # Utility functions
├── configs/
│   ├── config.yaml          # Training configuration
│   ├── label_map.yaml       # Category to class mapping
│   ├── benchmark_config.yaml # Benchmark configuration
│   └── calibration_config.yaml # Calibration configuration
├── tests/
│   ├── test_dataset.py      # Dataset tests
│   ├── test_model.py        # Model tests
│   ├── test_smoke_train.py  # Smoke test
│   ├── test_critical_fixes.py # Data pipeline tests
│   ├── test_benchmark.py    # Benchmark tests
│   ├── test_gradcam.py      # Grad-CAM tests
│   └── test_calibration.py   # Calibration tests
├── notebooks/
│   └── colab_run.ipynb      # Colab notebook
├── app/
│   └── demo.py              # Gradio demo with Grad-CAM and calibration
├── outputs/
│   ├── checkpoints/         # Model checkpoints
│   ├── figures/             # Plots and visualizations
│   ├── metrics/             # Evaluation metrics
│   ├── benchmark/           # Benchmark results
│   └── calibration/        # Calibration results
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── processed/           # Processed images
│   └── cache/               # Download cache
├── requirements.txt
├── README.md
├── DATA_CARD.md
├── REPORT.md
├── FIXES_SUMMARY.md
├── BENCHMARK_README.md
└── GRADCAM_README.md
```

## Testing

Run the test suite:
```bash
python tests/test_dataset.py
python tests/test_model.py
python tests/test_smoke_train.py
python tests/test_critical_fixes.py
python tests/test_benchmark.py
python tests/test_gradcam.py
python tests/test_calibration.py
```

## Why PyTorch?

PyTorch was chosen for this project because:
1. **Research-friendly**: First-class support for rapid experimentation
2. **Flexibility**: Easy to implement custom architectures and training loops
3. **Community**: Large ecosystem of pretrained models and tools
4. **Colab support**: Excellent integration with Google Colab's free GPUs
5. **Debugging**: Pythonic debugging compared to framework graph modes

## Limitations

1. **Dataset License**: The Flipkart dataset has an unknown license; for production use, verify licensing or use a properly licensed dataset
2. **Class Imbalance**: Despite balancing, some categories may have inherent visual overlap
3. **Image Quality**: Downloaded images vary in quality and resolution
4. **Generalization**: Model trained on Indian e-commerce data may not generalize to other markets
5. **Background Context**: Model may rely on background context rather than product features
6. **Confidence Not a Guarantee**: Calibrated confidence scores are better estimates but do not guarantee correctness; manual review still required for uncertain predictions

## Future Work

1. **Active Learning**: Continuously improve model with human-in-the-loop feedback
2. **Multi-label Classification**: Products may belong to multiple categories
3. **Fine-grained Classification**: Sub-categories within each main class
4. **Explainability Enhancement**: Visual attention maps to understand model decisions (Grad-CAM implemented)
5. **Deployment**: Optimize for edge deployment (ONNX, TensorRT)

## Citation

If you use this code, please cite:
```
@software{product_image_classifier,
  title={Product Image Classifier (CNN)},
  author={Your Name},
  year={2026},
  url={https://github.com/your-username/product-image-classifier}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue.

## Additional Documentation

- **FIXES_SUMMARY.md**: Critical data pipeline fixes for reproducibility
- **BENCHMARK_README.md**: Model benchmarking guide
- **BENCHMARK_EXTENSION_SUMMARY.md**: Benchmark implementation details
- **GRADCAM_README.md**: Grad-CAM explainability guide
- **GRADCAM_EXTENSION_SUMMARY.md**: Grad-CAM implementation details
- **CALIBRATION_README.md**: Probability calibration guide
- **CALIBRATION_EXTENSION_SUMMARY.md**: Calibration implementation details
