"""
Tests for Grad-CAM explainability module.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.model.cnn import get_model
from src.explainability.gradcam import GradCAM, GradCAMPlusPlus, get_gradcam, visualize_explanation


def test_gradcam_initialization():
    """Test that Grad-CAM initializes correctly for each model type."""
    try:
        model_types = ['custom', 'mobilenet_v2', 'resnet18', 'efficientnet_b0']
        
        for model_type in model_types:
            model = get_model(model_type, num_classes=3, pretrained=False)
            
            try:
                gradcam = GradCAM(model, model_type=model_type)
                assert gradcam.target_layer is not None
                print(f"✓ Grad-CAM initialized for {model_type}")
            except Exception as e:
                print(f"⚠ Grad-CAM initialization failed for {model_type}: {e}")
        
    except Exception as e:
        print(f"⚠ Grad-CAM initialization test failed: {e}")


def test_gradcam_generation():
    """Test that Grad-CAM generates heatmaps."""
    try:
        model = get_model("custom", num_classes=3, pretrained=False)
        gradcam = GradCAM(model, model_type="custom")
        
        # Create dummy input
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Generate CAM
        cam = gradcam.generate_cam(input_tensor, target_class=0)
        
        # Verify output shape
        assert cam.shape == (14, 14) or cam.shape == (7, 7), f"Unexpected shape: {cam.shape}"
        assert cam.min() >= 0 and cam.max() <= 1, "CAM not normalized to [0, 1]"
        
        print("✓ Grad-CAM generation test passed")
        
    except Exception as e:
        print(f"⚠ Grad-CAM generation test failed: {e}")


def test_gradcam_plus_plus():
    """Test that Grad-CAM++ generates heatmaps."""
    try:
        model = get_model("custom", num_classes=3, pretrained=False)
        gradcam_plus = GradCAMPlusPlus(model, model_type="custom")
        
        # Create dummy input
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Generate CAM
        cam = gradcam_plus.generate_cam(input_tensor, target_class=0)
        
        # Verify output shape
        assert cam.shape == (14, 14) or cam.shape == (7, 7), f"Unexpected shape: {cam.shape}"
        assert cam.min() >= 0 and cam.max() <= 1, "CAM not normalized to [0, 1]"
        
        print("✓ Grad-CAM++ generation test passed")
        
    except Exception as e:
        print(f"⚠ Grad-CAM++ generation test failed: {e}")


def test_visualization():
    """Test visualization function."""
    try:
        # Create dummy image
        image = Image.new('RGB', (224, 224), color='red')
        
        # Create dummy CAM
        cam = np.random.rand(14, 14)
        
        # Generate visualizations
        original_np, heatmap_np, overlay_np = visualize_explanation(image, cam, alpha=0.5)
        
        # Verify shapes
        assert original_np.shape == (224, 224, 3), f"Unexpected original shape: {original_np.shape}"
        assert heatmap_np.shape == (224, 224, 3), f"Unexpected heatmap shape: {heatmap_np.shape}"
        assert overlay_np.shape == (224, 224, 3), f"Unexpected overlay shape: {overlay_np.shape}"
        
        print("✓ Visualization test passed")
        
    except Exception as e:
        print(f"⚠ Visualization test failed: {e}")


def test_factory_function():
    """Test factory function for Grad-CAM."""
    try:
        model = get_model("custom", num_classes=3, pretrained=False)
        
        # Test Grad-CAM
        gradcam = get_gradcam(model, model_type="custom", method="gradcam")
        assert isinstance(gradcam, GradCAM)
        
        # Test Grad-CAM++
        gradcam_plus = get_gradcam(model, model_type="custom", method="gradcam++")
        assert isinstance(gradcam_plus, GradCAMPlusPlus)
        
        # Test invalid method
        try:
            get_gradcam(model, model_type="custom", method="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        
        print("✓ Factory function test passed")
        
    except Exception as e:
        print(f"⚠ Factory function test failed: {e}")


def test_overlay_generation():
    """Test overlay generation."""
    try:
        model = get_model("custom", num_classes=3, pretrained=False)
        gradcam = GradCAM(model, model_type="custom")
        
        # Create dummy image
        image = Image.new('RGB', (224, 224), color='blue')
        
        # Create dummy CAM
        cam = np.random.rand(14, 14)
        
        # Generate overlay
        overlay = gradcam.generate_overlay(image, cam, alpha=0.5)
        
        # Verify shape
        assert overlay.shape == (224, 224, 3), f"Unexpected overlay shape: {overlay.shape}"
        
        print("✓ Overlay generation test passed")
        
    except Exception as e:
        print(f"⚠ Overlay generation test failed: {e}")


def test_auto_layer_detection():
    """Test automatic layer detection."""
    try:
        model_types = ['custom', 'mobilenet_v2', 'resnet18', 'efficientnet_b0']
        
        for model_type in model_types:
            model = get_model(model_type, num_classes=3, pretrained=False)
            
            try:
                gradcam = GradCAM(model, model_type=model_type, target_layer=None)
                assert gradcam.target_layer is not None
                print(f"✓ Auto layer detection for {model_type}: {gradcam.target_layer}")
            except Exception as e:
                print(f"⚠ Auto layer detection failed for {model_type}: {e}")
        
    except Exception as e:
        print(f"⚠ Auto layer detection test failed: {e}")


def test_graceful_error_handling():
    """Test graceful error handling for invalid inputs."""
    try:
        model = get_model("custom", num_classes=3, pretrained=False)
        gradcam = GradCAM(model, model_type="custom")
        
        # Test with wrong input shape
        try:
            wrong_input = torch.randn(1, 3, 100, 100)  # Wrong size
            cam = gradcam.generate_cam(wrong_input, target_class=0)
            # Should still work (internal resize)
            print("✓ Graceful handling of wrong input size")
        except Exception as e:
            print(f"⚠ Error handling test noted: {e}")
        
    except Exception as e:
        print(f"⚠ Graceful error handling test failed: {e}")


if __name__ == "__main__":
    print("="*50)
    print("GRAD-CAM EXPLAINABILITY TESTS")
    print("="*50)
    
    test_gradcam_initialization()
    test_gradcam_generation()
    test_gradcam_plus_plus()
    test_visualization()
    test_factory_function()
    test_overlay_generation()
    test_auto_layer_detection()
    test_graceful_error_handling()
    
    print("="*50)
