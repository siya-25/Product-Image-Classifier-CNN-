"""
Grad-CAM and Grad-CAM++ implementation for model explainability.
Supports Custom CNN, MobileNetV2, ResNet18, and EfficientNet-B0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from PIL import Image
import logging

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available, Grad-CAM visualization will be limited")

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps highlighting image regions that influence the predicted class.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        model_type: str = "custom"
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of target layer (auto-detected if None)
            model_type: Model architecture type
        """
        self.model = model
        self.model_type = model_type
        self.model.eval()
        
        # Auto-detect target layer if not specified
        self.target_layer = self._find_target_layer(target_layer)
        
        # Store gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"Grad-CAM initialized for {model_type} with target layer: {self.target_layer}")
    
    def _find_target_layer(self, target_layer: Optional[str]) -> str:
        """
        Auto-detect the target layer for Grad-CAM.
        
        Args:
            target_layer: Manually specified layer name
            
        Returns:
            Layer name to use for Grad-CAM
        """
        if target_layer:
            return target_layer
        
        # Default layers for each model type
        default_layers = {
            'custom': 'features.6',  # Last conv block in custom CNN
            'mobilenet_v2': 'features.18',  # Near end of MobileNetV2
            'resnet18': 'layer4',  # Last residual block
            'efficientnet_b0': 'features.8'  # Near end of EfficientNet
        }
        
        if self.model_type in default_layers:
            layer_name = default_layers[self.model_type]
            # Verify layer exists
            try:
                _ = self._get_layer(layer_name)
                return layer_name
            except (AttributeError, KeyError):
                logger.warning(f"Default layer {layer_name} not found, searching...")
        
        # Fallback: find last conv layer
        return self._find_last_conv_layer()
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                return name
        
        raise ValueError("No convolutional layer found in model")
    
    def _get_layer(self, layer_name: str) -> nn.Module:
        """Get a layer by name."""
        module = self.model
        parts = layer_name.split('.')
        
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        return module
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        target_layer = self._get_layer(self.target_layer)
        
        # Forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (uses prediction if None)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # (1, 1, H, W)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def generate_overlay(
        self,
        original_image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate overlay of heatmap on original image.
        
        Args:
            original_image: PIL Image
            cam: Grad-CAM heatmap (H, W)
            alpha: Transparency of heatmap (0-1)
            colormap: OpenCV colormap (ignored if cv2 unavailable)
            
        Returns:
            Overlay image as numpy array (H, W, 3)
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using simple overlay")
            return self._simple_overlay(original_image, cam, alpha)
        
        # Resize heatmap to match image size
        cam_resized = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
        
        # Apply colormap
        cam_colored = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        original_np = np.array(original_image)
        
        # Blend
        overlay = cv2.addWeighted(original_np, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay
    
    def _simple_overlay(
        self,
        original_image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Simple overlay without OpenCV (fallback).
        
        Args:
            original_image: PIL Image
            cam: Grad-CAM heatmap (H, W)
            alpha: Transparency of heatmap (0-1)
            
        Returns:
            Overlay image as numpy array (H, W, 3)
        """
        # Resize heatmap
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray(np.uint8(cam * 255))
        cam_resized = cam_pil.resize(original_image.size)
        cam_np = np.array(cam_resized)
        
        # Convert to RGB
        if cam_np.ndim == 2:
            cam_np = np.stack([cam_np] * 3, axis=-1)
        
        # Convert original to numpy
        original_np = np.array(original_image)
        
        # Simple blend
        overlay = (original_np * (1 - alpha) + cam_np * alpha).astype(np.uint8)
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Grad-CAM with better localization.
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (uses prediction if None)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create one-hot encoding
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Grad-CAM++ calculation
        # See: https://arxiv.org/abs/1710.11063
        
        # Second derivative
        grads_square = gradients.pow(2)
        grads_third = grads_square * gradients
        
        # Sum over spatial dimensions
        sum_grads = gradients.sum(dim=(2, 3), keepdim=True) + 1e-7
        sum_grads_square = grads_square.sum(dim=(2, 3), keepdim=True) + 1e-7
        sum_grads_third = grads_third.sum(dim=(2, 3), keepdim=True) + 1e-7
        
        # Alpha weights
        alpha = sum_grads_square / (2 * sum_grads_square + sum_grads * (sum_grads_square).sqrt())
        alpha = alpha * (gradients >= 0).float()
        
        # Weighted activations
        weights = (alpha * gradients) / sum_grads
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def get_gradcam(
    model: nn.Module,
    model_type: str = "custom",
    method: str = "gradcam",
    target_layer: Optional[str] = None
) -> GradCAM:
    """
    Factory function to get Grad-CAM instance.
    
    Args:
        model: PyTorch model
        model_type: Model architecture type
        method: 'gradcam' or 'gradcam++'
        target_layer: Target layer name (auto-detected if None)
        
    Returns:
        Grad-CAM instance
    """
    if method == "gradcam":
        return GradCAM(model, target_layer, model_type)
    elif method == "gradcam++":
        return GradCAMPlusPlus(model, target_layer, model_type)
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_explanation(
    original_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate visualization of explanation.
    
    Args:
        original_image: PIL Image
        cam: Grad-CAM heatmap (H, W)
        alpha: Transparency for overlay
        
    Returns:
        Tuple of (original_np, heatmap_np, overlay_np)
    """
    # Original image as numpy
    original_np = np.array(original_image)
    
    if not CV2_AVAILABLE:
        # Simple fallback without colormap
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray(np.uint8(cam * 255))
        cam_resized = cam_pil.resize(original_image.size)
        heatmap_np = np.array(cam_resized)
        
        # Convert to RGB
        if heatmap_np.ndim == 2:
            heatmap_np = np.stack([heatmap_np] * 3, axis=-1)
        
        # Simple overlay
        overlay = (original_np * (1 - alpha) + heatmap_np * alpha).astype(np.uint8)
        
        return original_np, heatmap_np, overlay
    
    # Heatmap as colored image
    cam_resized = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    heatmap_colored = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    
    return original_np, heatmap_colored, overlay
