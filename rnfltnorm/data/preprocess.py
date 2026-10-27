"""Fundus preprocessing, laterality, and vessel high-pass."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


def eye_from_pid(pid: str) -> Optional[int]:
    """Extract the 0/1 eye field from ``patient_eye_timestamp`` identifiers."""
    parts = str(pid).split("_")
    if len(parts) >= 2 and parts[1] in ("0", "1"):
        return int(parts[1])
    return None


def _crop_retinal_field(rgb: np.ndarray) -> np.ndarray:
    """Remove black camera borders while preserving the complete retinal field."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return rgb
    intensity = rgb[..., :3].astype(np.float32).mean(axis=2)
    foreground = intensity > 8.0
    if foreground.sum() < 0.1 * foreground.size:
        return rgb
    ys, xs = np.where(foreground)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = int(round(0.03 * max(x1 - x0, y1 - y0)))
    x0, x1 = max(0, x0 - margin), min(rgb.shape[1], x1 + margin)
    y0, y1 = max(0, y0 - margin), min(rgb.shape[0], y1 + margin)
    return rgb[y0:y1, x0:x1]


def _letterbox_square(rgb: np.ndarray) -> Image.Image:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), color=(0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def preprocess_fundus(
    rgb: np.ndarray,
    size: int,
    *,
    crop_retina: bool = True,
    flip: bool = False,
    augment: bool = False,
) -> torch.Tensor:
    """Crop borders, canonicalize laterality, resize, and photometrically augment."""
    arr = np.asarray(rgb, dtype=np.uint8)
    if crop_retina:
        arr = _crop_retinal_field(arr)
    if flip:
        arr = np.fliplr(arr).copy()
    image = _letterbox_square(arr)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    if augment:
        image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(0.90, 1.10)))
        image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(0.90, 1.10)))
        image = ImageEnhance.Color(image).enhance(float(np.random.uniform(0.90, 1.10)))
        gamma = float(np.random.uniform(0.90, 1.10))
        aug = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.clip(aug**gamma * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
    arr_float = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr_float).permute(2, 0, 1)


def vessel_highpass_channel(fundus: torch.Tensor, kernel: int = 15) -> torch.Tensor:
    """Green-channel high-pass; vessels are brighter after inversion."""
    green = fundus[1:2].unsqueeze(0)
    pad = kernel // 2
    blur = F.avg_pool2d(green, kernel, stride=1, padding=pad)
    vessels = (blur - green).clamp(min=0.0)[0]
    peak = vessels.max().clamp_min(1e-6)
    return vessels / peak


def attach_vessel_channel(fundus: torch.Tensor) -> torch.Tensor:
    return torch.cat([fundus, vessel_highpass_channel(fundus)], dim=0)
