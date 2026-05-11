"""ResNet backbone for RT-DETR.

Extracts multi-scale features from C3, C4, C5 stages of ResNet.
Supports ResNet-18 (student) and ResNet-50 (teacher).

Output channels per backbone:
  ResNet-18: C3=128, C4=256, C5=512
  ResNet-50: C3=512, C4=1024, C5=2048
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics (used when freeze_bn=True)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


def _replace_bn_with_frozen(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm2d with FrozenBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features)
            frozen.weight.copy_(child.weight.data)
            frozen.bias.copy_(child.bias.data)
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            setattr(module, name, frozen)
        else:
            _replace_bn_with_frozen(child)
    return module


# Output channels for each backbone variant at C3/C4/C5
BACKBONE_OUT_CHANNELS = {
    "resnet18": [128, 256, 512],
    "resnet50": [512, 1024, 2048],
}


class ResNetBackbone(nn.Module):
    """ResNet backbone returning multi-scale feature maps.

    Returns features from three stages:
      '0' -> C3 (stride 8 relative to input)
      '1' -> C4 (stride 16 relative to input)
      '2' -> C5 (stride 32 relative to input)

    Args:
        name: One of 'resnet18' or 'resnet50'.
        pretrained: Load ImageNet pretrained weights.
        freeze_bn: Replace BatchNorm with FrozenBatchNorm.
        freeze_stages: Number of early stages to freeze (0=none, 1=stem+layer1, ...).
    """

    def __init__(
        self,
        name: str = "resnet18",
        pretrained: bool = True,
        freeze_bn: bool = False,
        freeze_stages: int = 1,
    ):
        super().__init__()
        assert name in ("resnet18", "resnet50"), f"Unsupported backbone: {name}"

        self.name = name
        self.out_channels = BACKBONE_OUT_CHANNELS[name]

        if name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
        else:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = resnet50(weights=weights)

        if freeze_bn:
            base = _replace_bn_with_frozen(base)

        # Decompose ResNet into named stages
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # C2 — not returned but needed for C3
        self.layer2 = base.layer2  # C3: stride-8 features
        self.layer3 = base.layer3  # C4: stride-16 features
        self.layer4 = base.layer4  # C5: stride-32 features

        # Freeze early stages if requested
        self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze parameters of the first `num_stages` stages."""
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for i, stage in enumerate(stages):
            if i < num_stages:
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through backbone.

        Args:
            x: Input images of shape [B, 3, H, W].

        Returns:
            Dictionary mapping scale index (str) to feature tensor:
              '0': C3 features [B, C3, H/8, W/8]
              '1': C4 features [B, C4, H/16, W/16]
              '2': C5 features [B, C5, H/32, W/32]
        """
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)   # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32

        return {"0": c3, "1": c4, "2": c5}
