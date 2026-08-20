"""
FakeInversion - ResNet Classifier Model
==========================================
Modified ResNet50 for 9-channel input (original + noise map + reconstruction).
Binary classification: real (0) vs fake (1).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision.models as models

import config


class FakeInversionClassifier(nn.Module):
    """ResNet-based classifier for 9-channel FakeInversion features.

    Architecture:
        - Modified ResNet50 with 9-channel input conv layer
        - First 3 channels initialized from ImageNet-pretrained weights
        - Remaining 6 channels initialized with Kaiming He initialization
        - Final FC layer: 2048 → 2 (binary classification)
    """

    def __init__(
        self,
        backbone: str = None,
        input_channels: int = None,
        num_classes: int = None,
        pretrained: bool = True,
    ):
        """Initialize the classifier.

        Args:
            backbone: ResNet variant ('resnet50', 'resnet18', etc.).
            input_channels: Number of input channels (9 for FakeInversion).
            num_classes: Number of output classes (2 for binary).
            pretrained: Whether to use ImageNet-pretrained backbone weights.
        """
        super().__init__()

        self.backbone_name = backbone or config.CLASSIFIER_BACKBONE
        self.input_channels = input_channels or config.INPUT_CHANNELS
        self.num_classes = num_classes or config.NUM_CLASSES

        # Load backbone
        self.backbone = self._create_backbone(pretrained)

        # Modify first conv layer for 9-channel input
        self._modify_first_conv(pretrained)

        # Modify final classification layer
        self._modify_classifier()

    def _create_backbone(self, pretrained: bool) -> nn.Module:
        """Create the ResNet backbone."""
        weights = "IMAGENET1K_V2" if pretrained else None

        backbone_fn = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }.get(self.backbone_name)

        if backbone_fn is None:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        return backbone_fn(weights=weights)

    def _modify_first_conv(self, pretrained: bool):
        """Modify the first convolutional layer for 9-channel input.

        Strategy: Copy pretrained weights for the first 3 channels,
        initialize the remaining 6 channels with Kaiming He initialization.
        """
        old_conv = self.backbone.conv1

        # Create new conv layer with 9 input channels
        new_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                # Copy pretrained weights for first 3 channels (original image)
                new_conv.weight[:, :3, :, :] = old_conv.weight.clone()

                # Initialize remaining channels (noise map + reconstruction)
                # Each set of 3 channels gets a copy of pretrained weights
                # scaled down to avoid dominating the pretrained channels
                for i in range(3, self.input_channels, 3):
                    end = min(i + 3, self.input_channels)
                    num_ch = end - i
                    new_conv.weight[:, i:end, :, :] = (
                        old_conv.weight[:, :num_ch, :, :].clone() * 0.1
                    )

                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        self.backbone.conv1 = new_conv

    def _modify_classifier(self):
        """Replace the final FC layer for binary classification."""
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            Probabilities of shape (B, num_classes).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.

        Args:
            x: Input tensor of shape (B, 9, H, W).

        Returns:
            Predicted labels of shape (B,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def build_model(pretrained: bool = True) -> FakeInversionClassifier:
    """Factory function to build the FakeInversion classifier."""
    model = FakeInversionClassifier(pretrained=pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.CLASSIFIER_BACKBONE} ({config.INPUT_CHANNELS}ch → {config.NUM_CLASSES} classes)")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable_params:,}")
    return model
