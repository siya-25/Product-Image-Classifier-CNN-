"""
CNN model architectures for product image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """
    Small custom CNN baseline for product image classification.
    Architecture: 4 convolutional blocks + 2 FC layers.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        """
        Initialize custom CNN.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout rate for regularization
        """
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 64 filters, 3x3
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 128 filters, 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 256 filters, 3x3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 512 filters, 3x3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


class TransferLearningModel(nn.Module):
    """
    Transfer learning model using pretrained backbones.
    Supports: MobileNetV2, ResNet18, EfficientNet-B0.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "mobilenet_v2",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize transfer learning model.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            freeze_backbone: Whether to freeze backbone weights
        """
        super(TransferLearningModel, self).__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Frozen {backbone} backbone")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"Initialized {backbone} with pretrained={pretrained}, freeze={freeze_backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def get_model(
    model_type: str = "custom",
    num_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('custom', 'mobilenet_v2', 'resnet18', 'efficientnet_b0')
        num_classes: Number of output classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_type == "custom":
        return CustomCNN(num_classes=num_classes, **kwargs)
    elif model_type in ["mobilenet_v2", "resnet18", "efficientnet_b0"]:
        return TransferLearningModel(
            num_classes=num_classes,
            backbone=model_type,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Test model creation."""
    # Test custom CNN
    custom_model = get_model("custom", num_classes=3)
    print(f"Custom CNN parameters: {count_parameters(custom_model):,}")
    
    # Test transfer learning
    mobilenet = get_model("mobilenet_v2", num_classes=3, pretrained=False)
    print(f"MobileNetV2 parameters: {count_parameters(mobilenet):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = custom_model(x)
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    main()
