"""RNFLT thickness normalization (valid tissue pixels only)."""
from __future__ import annotations

import numpy as np
import torch


def valid_rnflt_mask(rnflt: np.ndarray) -> np.ndarray:
    """True where pixel is measured RNFLT thickness (excludes cup/rim sentinels)."""
    return np.asarray(rnflt) > 0


def normalize_rnflt_map(
    rnflt: np.ndarray,
    rnflt_min: float,
    rnflt_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    rnflt = np.asarray(rnflt, dtype=np.float32)
    mask = valid_rnflt_mask(rnflt)
    out = np.zeros_like(rnflt, dtype=np.float32)
    denom = max(float(rnflt_max) - float(rnflt_min), 1e-6)
    out[mask] = np.clip((rnflt[mask] - rnflt_min) / denom, 0.0, 1.0)
    return out, mask


def denormalize_rnflt_clamped(
    x: torch.Tensor | np.ndarray,
    rnflt_min: float,
    rnflt_max: float,
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().float().cpu().numpy()
    else:
        arr = np.asarray(x, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    arr = np.clip(arr, 0.0, 1.0)
    return arr * (float(rnflt_max) - float(rnflt_min)) + float(rnflt_min)
