# Grad-CAM Explainability Guide

## Overview

Grad-CAM (Gradient-weighted Class Activation Mapping) has been added to the Product Image Classifier to provide visual explanations of model predictions. This helps understand which image regions influence the predicted class.

## Features

1. **Grad-CAM and Grad-CAM++**: Two explainability methods
2. **Multi-Architecture Support**: Works with Custom CNN, MobileNetV2, ResNet18, EfficientNet-B0
3. **Automatic Layer Detection**: Auto-selects appropriate target layer for each model
4. **Gradio Integration**: Visual explanations in the demo app
5. **Graceful Error Handling**: Fallbacks for unsupported layers or invalid inputs
6. **Three Visualizations**: Original image, heatmap, and overlay

## Usage

### In Python Code

```python
from src.model.cnn import get_model
from src.explainability.gradcam import get_gradcam, visualize_explanation
from PIL import Image
import torch

# Load model
model = get_model("custom", num_classes=3, pretrained=False)
model.eval()

# Initialize Grad-CAM
gradcam = get_gradcam(model, model_type="custom", method="gradcam")

# Load and preprocess image
image = Image.open("path/to/image.jpg")
# ... apply transforms ...

# Generate CAM
input_tensor = transforms(image).unsqueeze(0)
cam = gradcam.generate_cam(input_tensor, target_class=0)

# Visualize
original_np, heatmap_np, overlay_np = visualize_explanation(image, cam, alpha=0.5)
```

### In Gradio Demo

The demo app now includes Grad-CAM visualizations:

```bash
python app/demo.py
```

The interface shows:
- **Original Image**: The uploaded product image
- **Heatmap**: Grad-CAM heatmap highlighting influential regions
- **Overlay**: Heatmap overlaid on original image

### Configuration

Enable/disable Grad-CAM in the demo:

```python
demo = ProductClassifierDemo(
    checkpoint_path="outputs/checkpoints/best_checkpoint.pth",
    config_path="configs/config.yaml",
    enable_gradcam=True,  # Set to False to disable
    gradcam_method="gradcam"  # or "gradcam++"
)
```

## Supported Architectures

| Architecture | Target Layer | Status |
|--------------|--------------|--------|
| Custom CNN | features.6 | ✅ Supported |
| MobileNetV2 | features.18 | ✅ Supported |
| ResNet18 | layer4 | ✅ Supported |
| EfficientNet-B0 | features.8 | ✅ Supported |

## Implementation Details

### Grad-CAM vs Grad-CAM++

**Grad-CAM**:
- Uses gradient-weighted class activation mapping
- Faster computation
- Good localization

**Grad-CAM++**:
- Improved gradient weighting
- Better localization for fine-grained features
- Slightly more computation

### Automatic Layer Detection

If no target layer is specified, the system:
1. Tries model-specific default layers
2. Falls back to finding the last convolutional layer
3. Raises error if no conv layer found

### Preprocessing Consistency

Grad-CAM uses the same preprocessing as training/evaluation:
- Resize to model input size (224x224)
- Normalization (ImageNet stats)
- Same transforms pipeline

## Limitations

### 1. Resolution
- Heatmaps are generated at the spatial resolution of the target layer
- Typically 7x7 or 14x14 for most models
- Upsampled to image size for visualization
- May lose fine-grained details

### 2. Global Explanations
- Grad-CAM provides global explanations (what's important overall)
- Does not provide per-pixel importance
- May miss small but important features

### 3. Class-Specific
- Explanations are for a specific predicted class
- Different classes may highlight different regions
- Cannot explain why one class was NOT chosen

### 4. Architecture Dependence
- Quality of explanations depends on model architecture
- Some architectures (e.g., very deep networks) may produce less interpretable heatmaps
- Transfer learning models may focus on generic features

### 5. OpenCV Dependency
- Full functionality requires OpenCV (opencv-python)
- Fallback to simple overlay if OpenCV unavailable
- Heatmap colormaps may not be available without OpenCV

### 6. Gradio Performance
- Generating heatmaps adds ~50-100ms per prediction
- May slow down demo slightly
- Can be disabled if performance is critical

## Error Handling

### Unsupported Layers
If a model has an unsupported architecture:
- System attempts to find last conv layer automatically
- Falls back to simple visualization if Grad-CAM fails
- Logs warning and continues with basic prediction

### Invalid Inputs
- Wrong image sizes are handled gracefully
- Non-RGB images are converted to RGB
- Corrupted images raise clear error messages

### Missing Dependencies
- If OpenCV is missing, falls back to simple overlay
- Logs warning but doesn't crash
- Grad-CAM still generates heatmaps (just without colormaps)

## Testing

Run Grad-CAM tests:

```bash
python tests/test_gradcam.py
```

Tests verify:
- Grad-CAM initialization for all architectures
- CAM generation
- Grad-CAM++ generation
- Visualization
- Factory function
- Overlay generation
- Auto layer detection
- Graceful error handling

## Integration with Existing Pipeline

The Grad-CAM module:
- Reuses existing model architectures
- Uses same preprocessing pipeline
- Does not require retraining
- Preserves existing functionality
- Can be disabled without affecting predictions

## Future Improvements

1. **Layer Selection**: Allow manual layer specification
2. **Multiple Classes**: Show explanations for all classes
3. **Integrated Gradients**: Add complementary explainability method
4. **Interactive Heatmaps**: Click-to-explore regions
5. **Batch Processing**: Generate explanations for multiple images
6. **Quantitative Metrics**: Add explainability quality metrics

## Troubleshooting

### Issue: Grad-CAM fails to initialize
**Solution**: Check model architecture is supported, or manually specify target layer

### Issue: Heatmaps are all uniform
**Solution**: Model may not have learned discriminative features, check training quality

### Issue: OpenCV errors
**Solution**: Install opencv-python, or system will use fallback mode

### Issue: Heatmaps don't match image
**Solution**: Ensure preprocessing matches training pipeline exactly

## References

- Grad-CAM Paper: https://arxiv.org/abs/1610.02391
- Grad-CAM++ Paper: https://arxiv.org/abs/1710.11063
- PyTorch Implementation: Based on standard Grad-CAM formulation
