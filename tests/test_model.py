"""
Tests for model architecture and forward pass.
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.model.cnn import get_model, count_parameters, CustomCNN, TransferLearningModel


def test_custom_cnn_forward():
    """Test custom CNN forward pass."""
    model = CustomCNN(num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 3)
    assert not torch.isnan(output).any()
    print("✓ Custom CNN forward pass test passed")


def test_custom_cnn_parameter_count():
    """Test custom CNN parameter count."""
    model = CustomCNN(num_classes=3)
    params = count_parameters(model)
    
    assert params > 0
    assert params < 10_000_000  # Should be relatively small
    print(f"✓ Custom CNN parameter count test passed: {params:,} parameters")


def test_mobilenet_forward():
    """Test MobileNetV2 forward pass."""
    model = get_model("mobilenet_v2", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 3)
    assert not torch.isnan(output).any()
    print("✓ MobileNetV2 forward pass test passed")


def test_resnet_forward():
    """Test ResNet18 forward pass."""
    model = get_model("resnet18", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 3)
    assert not torch.isnan(output).any()
    print("✓ ResNet18 forward pass test passed")


def test_model_factory():
    """Test model factory function."""
    # Test custom
    custom = get_model("custom", num_classes=3)
    assert isinstance(custom, CustomCNN)
    
    # Test transfer learning
    mobilenet = get_model("mobilenet_v2", num_classes=3, pretrained=False)
    assert isinstance(mobilenet, TransferLearningModel)
    
    print("✓ Model factory test passed")


def test_invalid_model_type():
    """Test that invalid model type raises error."""
    try:
        model = get_model("invalid_model", num_classes=3)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Invalid model type test passed")


if __name__ == "__main__":
    test_custom_cnn_forward()
    test_custom_cnn_parameter_count()
    test_mobilenet_forward()
    test_resnet_forward()
    test_model_factory()
    test_invalid_model_type()
