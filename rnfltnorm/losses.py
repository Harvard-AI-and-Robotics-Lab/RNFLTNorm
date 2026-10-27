"""Residual and polar-profile losses for v5."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnfltnorm.data.dataset import make_polar_bin_masks, polar_profile


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def ssim_map(pred: torch.Tensor, target: torch.Tensor, window: int = 7) -> torch.Tensor:
    padding = window // 2
    mu_x = F.avg_pool2d(pred, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(1e-8)


def multiscale_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    scales: int = 3,
) -> torch.Tensor:
    losses = []
    x, y, mask = pred, target, valid_mask
    for _ in range(scales):
        losses.append(1.0 - masked_mean(ssim_map(x, y), mask))
        if min(x.shape[-2:]) < 16:
            break
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)
        mask = F.avg_pool2d(mask, 2, stride=2)
    return torch.stack(losses).mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    sobel_x = pred.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    pred_dx = F.conv2d(pred, sobel_x, padding=1)
    pred_dy = F.conv2d(pred, sobel_y, padding=1)
    target_dx = F.conv2d(target, sobel_x, padding=1)
    target_dy = F.conv2d(target, sobel_y, padding=1)
    edge_mask = -F.max_pool2d(-valid_mask, 3, stride=1, padding=1)
    error = (pred_dx - target_dx).abs() + (pred_dy - target_dy).abs()
    return masked_mean(error, edge_mask)


def mask_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denominator = prob.sum(dim=dims) + target.sum(dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def peak_weights(
    target: torch.Tensor,
    mean_thickness: torch.Tensor,
    valid_mask: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Upweight thick ISNT peaks on the mean map and on this eye."""
    if mean_thickness.shape[0] == 1 and target.shape[0] > 1:
        mean_thickness = mean_thickness.expand(target.shape[0], -1, -1, -1)
    baseline = masked_mean(mean_thickness, valid_mask).detach().clamp_min(1e-3)
    mean_peak = (mean_thickness - baseline).clamp(min=0.0) / baseline
    target_peak = (target - baseline).clamp(min=0.0) / baseline
    return valid_mask * (1.0 + float(strength) * (mean_peak + target_peak))


class ResidualLoss(nn.Module):
    def __init__(
        self,
        image_size: int,
        *,
        l1_weight: float = 1.0,
        peak_weight: float = 2.0,
        residual_l1_weight: float = 1.0,
        ms_ssim_weight: float = 0.4,
        gradient_weight: float = 0.35,
        polar_weight: float = 0.6,
        polar_consistency_weight: float = 0.2,
        mask_weight: float = 0.1,
        polar_bins: int = 72,
        polar_inner_radius: float = 0.20,
        polar_outer_radius: float = 0.46,
        residual_scale: float = 0.35,
    ):
        super().__init__()
        self.weights = {
            "l1": float(l1_weight),
            "residual": float(residual_l1_weight),
            "ms_ssim": float(ms_ssim_weight),
            "gradient": float(gradient_weight),
            "polar": float(polar_weight),
            "polar_consistency": float(polar_consistency_weight),
            "mask": float(mask_weight),
        }
        self.peak_weight = float(peak_weight)
        self.residual_scale = float(residual_scale)
        self.register_buffer(
            "polar_masks",
            make_polar_bin_masks(
                image_size,
                polar_bins,
                polar_inner_radius,
                polar_outer_radius,
            ),
        )

    def forward(
        self,
        thickness: torch.Tensor,
        residual: torch.Tensor,
        polar: torch.Tensor,
        mask_logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        mean_thickness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if mean_thickness.shape[0] == 1 and target.shape[0] > 1:
            mean_thickness = mean_thickness.expand(target.shape[0], -1, -1, -1)
        weights = peak_weights(target, mean_thickness, valid_mask, self.peak_weight)
        residual_target = ((target - mean_thickness) / self.residual_scale).clamp(-1.0, 1.0)
        polar_target = polar_profile(target, valid_mask, self.polar_masks)
        polar_from_map = polar_profile(thickness, valid_mask, self.polar_masks)
        usable = (self.polar_masks[None] * valid_mask).sum(dim=(-2, -1)) > 0
        polar_error = (polar - polar_target).abs() * usable.to(polar.dtype)
        consistency = (polar_from_map - polar.detach()).abs() * usable.to(polar.dtype)
        parts = {
            "l1": (thickness - target).abs().mul(weights).sum() / weights.sum().clamp_min(1.0),
            "residual": masked_mean((residual - residual_target).abs(), valid_mask),
            "ms_ssim": multiscale_ssim_loss(thickness, target, valid_mask),
            "gradient": gradient_loss(thickness, target, valid_mask),
            "polar": polar_error.sum() / usable.to(polar.dtype).sum().clamp_min(1.0),
            "polar_consistency": consistency.sum() / usable.to(polar.dtype).sum().clamp_min(1.0),
            "mask": mask_bce_dice_loss(mask_logits, valid_mask),
        }
        parts["total"] = sum(self.weights[name] * value for name, value in parts.items())
        return parts


@torch.no_grad()
def batch_metrics(
    thickness: torch.Tensor,
    mask_logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    rnflt_scale: float,
) -> Dict[str, torch.Tensor]:
    batch_size = thickness.shape[0]
    maes = []
    correlations = []
    ssims = []
    for idx in range(batch_size):
        mask = valid_mask[idx].bool()
        pred_values = thickness[idx][mask]
        target_values = target[idx][mask]
        maes.append((pred_values - target_values).abs().mean() * rnflt_scale)
        pred_centered = pred_values - pred_values.mean()
        target_centered = target_values - target_values.mean()
        denom = torch.sqrt(pred_centered.square().sum() * target_centered.square().sum())
        correlations.append(
            (pred_centered * target_centered).sum() / denom.clamp_min(1e-8)
        )
        ssims.append(masked_mean(ssim_map(thickness[idx : idx + 1], target[idx : idx + 1]), valid_mask[idx : idx + 1]))
    predicted_mask = torch.sigmoid(mask_logits) >= 0.5
    truth = valid_mask.bool()
    dims = tuple(range(1, truth.ndim))
    intersection = (predicted_mask & truth).sum(dim=dims).float()
    dice = (2.0 * intersection + 1.0) / (
        predicted_mask.sum(dim=dims).float() + truth.sum(dim=dims).float() + 1.0
    )
    return {
        "mae_um": torch.stack(maes).mean(),
        "correlation": torch.stack(correlations).mean(),
        "ssim": torch.stack(ssims).mean(),
        "mask_dice": dice.mean(),
    }
