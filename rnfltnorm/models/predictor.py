"""Residual ResNet34-FPN with a polar TSNIT head and optional vessel input."""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from rnfltnorm.data.dataset import make_polar_bin_masks, polar_profile


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class FpnStage(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels),
        )

    def forward(self, lateral: torch.Tensor, top: torch.Tensor | None) -> torch.Tensor:
        if top is not None:
            top = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            lateral = lateral + top
        return self.refine(lateral)


class ResidualPolarRnfltPredictor(nn.Module):
    """Predict RNFLT residual from the training-set mean, plus a TSNIT profile."""

    def __init__(
        self,
        output_size: int = 200,
        decoder_channels: int = 96,
        pretrained: bool = True,
        in_channels: int = 4,
        residual_scale: float = 0.35,
        polar_bins: int = 72,
        polar_inner_radius: float = 0.20,
        polar_outer_radius: float = 0.46,
    ):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.output_size = int(output_size)
        self.residual_scale = float(residual_scale)
        self.in_channels = int(in_channels)
        self.polar_bins = int(polar_bins)

        rgb_mean = (0.485, 0.456, 0.406)
        rgb_std = (0.229, 0.224, 0.225)
        if self.in_channels == 4:
            mean = rgb_mean + (0.12,)
            std = rgb_std + (0.18,)
        else:
            mean = rgb_mean[: self.in_channels]
            std = rgb_std[: self.in_channels]
        self.register_buffer(
            "input_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, self.in_channels, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(std, dtype=torch.float32).view(1, self.in_channels, 1, 1),
        )
        self.register_buffer(
            "polar_masks",
            make_polar_bin_masks(
                self.output_size,
                self.polar_bins,
                polar_inner_radius,
                polar_outer_radius,
            ),
        )

        old = backbone.conv1
        if self.in_channels == 3:
            self.conv1 = old
        else:
            conv1 = nn.Conv2d(
                self.in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            with torch.no_grad():
                conv1.weight[:, :3] = old.weight
                extra = self.in_channels - 3
                if extra > 0:
                    conv1.weight[:, 3:] = old.weight[:, 1:2].repeat(1, extra, 1, 1)
            self.conv1 = conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        encoder_channels = (64, 64, 128, 256, 512)
        self.laterals = nn.ModuleList(
            [nn.Conv2d(ch, decoder_channels, kernel_size=1) for ch in encoder_channels]
        )
        self.fpn_stages = nn.ModuleList([FpnStage(decoder_channels) for _ in encoder_channels])
        self.fusion = nn.Sequential(
            ConvNormAct(decoder_channels * 2, decoder_channels, kernel_size=1),
            nn.Dropout2d(0.1),
            ConvNormAct(decoder_channels, decoder_channels),
        )
        self.residual_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.mask_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.polar_head = nn.Sequential(
            nn.Conv1d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(decoder_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        fundus: torch.Tensor,
        mean_thickness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fundus = (fundus - self.input_mean) / self.input_std
        c0 = self.relu(self.bn1(self.conv1(fundus)))
        c1 = self.layer1(self.maxpool(c0))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        features = (c0, c1, c2, c3, c4)

        pyramid = [None] * len(features)
        top = None
        for idx in reversed(range(len(features))):
            lateral = self.laterals[idx](features[idx])
            top = self.fpn_stages[idx](lateral, top)
            pyramid[idx] = top

        target_size = (self.output_size, self.output_size)
        fused = torch.cat(
            [
                F.interpolate(pyramid[0], size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(pyramid[1], size=target_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        decoded = self.fusion(fused)
        residual = torch.tanh(self.residual_head(decoded))
        if mean_thickness.ndim == 3:
            mean_thickness = mean_thickness.unsqueeze(0)
        if mean_thickness.shape[0] == 1 and fundus.shape[0] > 1:
            mean_thickness = mean_thickness.expand(fundus.shape[0], -1, -1, -1)
        thickness = (mean_thickness + self.residual_scale * residual).clamp(0.0, 1.0)

        polar_masks = self.polar_masks.to(device=decoded.device, dtype=decoded.dtype)
        ones = torch.ones(
            decoded.shape[0],
            1,
            decoded.shape[-2],
            decoded.shape[-1],
            device=decoded.device,
            dtype=decoded.dtype,
        )
        feat_profile = self._angular_pool(decoded, polar_masks)
        polar_residual = torch.tanh(self.polar_head(feat_profile)).squeeze(1)
        mean_polar = polar_profile(mean_thickness, ones, polar_masks)
        polar = (mean_polar + self.residual_scale * polar_residual).clamp(0.0, 1.0)
        return {
            "thickness": thickness,
            "residual": residual,
            "mask_logits": self.mask_head(decoded),
            "polar": polar,
            "polar_residual": polar_residual,
        }

    def _angular_pool(self, features: torch.Tensor, bin_masks: torch.Tensor) -> torch.Tensor:
        """Pool decoder features into ``[B, C, bins]`` angular bins."""
        weights = bin_masks[None, None]
        denom = weights.sum(dim=(-2, -1)).clamp_min(1.0)
        pooled = (features.unsqueeze(2) * weights).sum(dim=(-2, -1)) / denom
        return pooled

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        for module in modules:
            yield from module.parameters()

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        modules = (
            self.laterals,
            self.fpn_stages,
            self.fusion,
            self.residual_head,
            self.mask_head,
            self.polar_head,
        )
        for module in modules:
            yield from module.parameters()
