"""Healthy-baseline fundus/RNFLT pairs with vessel high-pass and polar helpers."""
from __future__ import annotations

import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rnfltnorm.data.preprocess import (
    attach_vessel_channel,
    eye_from_pid,
    preprocess_fundus,
)
from rnfltnorm.rnflt_norm import normalize_rnflt_map


def make_polar_bin_masks(
    image_size: int,
    bins: int,
    inner_radius: float,
    outer_radius: float,
) -> torch.Tensor:
    coords = torch.arange(image_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (image_size - 1) / 2.0
    dx = xx - center
    dy = center - yy
    radius = torch.sqrt(dx.square() + dy.square()) / float(image_size)
    angle = torch.remainder(torch.atan2(dy, dx), 2.0 * torch.pi)
    ring = (radius >= inner_radius) & (radius <= outer_radius)
    masks = []
    for idx in range(bins):
        lo = 2.0 * torch.pi * idx / bins
        hi = 2.0 * torch.pi * (idx + 1) / bins
        masks.append((ring & (angle >= lo) & (angle < hi)).float())
    return torch.stack(masks, dim=0)


def polar_profile(
    thickness: torch.Tensor,
    valid_mask: torch.Tensor,
    bin_masks: torch.Tensor,
) -> torch.Tensor:
    """Mean thickness in each peripapillary angular bin. ``[B, bins]``."""
    weights = valid_mask * bin_masks[None]
    denom = weights.sum(dim=(-2, -1)).clamp_min(1.0)
    return (thickness * weights).sum(dim=(-2, -1)) / denom


def _read_rnflt_pid(payload: tuple[str, str]) -> tuple[str, Optional[np.ndarray], Optional[str]]:
    path, key = payload
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"{key}.npy") as handle:
                rnflt = np.load(handle).astype(np.float32)
            with archive.open("pid.npy") as handle:
                pid = str(np.load(handle, allow_pickle=True))
        return path, rnflt, pid
    except Exception:
        return path, None, None


def compute_mean_rnflt_map(
    rows: pd.DataFrame,
    data_dir: Path,
    *,
    rnflt_key: str,
    rnflt_min: float,
    rnflt_max: float,
    canonicalize_eye: bool,
    workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Masked mean / count of canonical normalized RNFLT maps."""
    data_dir = Path(data_dir)
    payloads = [
        (str(data_dir / str(name)), rnflt_key)
        for name in rows["combined_npz"].astype(str)
    ]
    acc = None
    count = None
    if payloads:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for _path, rnflt, pid in pool.map(_read_rnflt_pid, payloads, chunksize=16):
                if rnflt is None or pid is None:
                    continue
                eye = eye_from_pid(pid)
                if canonicalize_eye and eye == 1:
                    rnflt = np.fliplr(rnflt).copy()
                norm, valid = normalize_rnflt_map(rnflt, rnflt_min, rnflt_max)
                if acc is None:
                    acc = np.zeros_like(norm, dtype=np.float64)
                    count = np.zeros_like(norm, dtype=np.float64)
                acc += norm * valid
                count += valid
    if acc is None or count is None:
        raise RuntimeError("failed to accumulate a training-set mean RNFLT map")
    mean = (acc / np.clip(count, 1.0, None)).astype(np.float32)
    support = (count > 0).astype(np.float32)
    return mean, support


class FundusRnfltDataset(Dataset):
    """Healthy-baseline pairs with an optional vessel high-pass channel."""

    def __init__(
        self,
        rows: pd.DataFrame,
        data_dir: Path,
        fundus_size: int,
        rnflt_min: float,
        rnflt_max: float,
        *,
        rnflt_key: str = "rnflt_artifact_correct",
        crop_retina: bool = True,
        canonicalize_eye: bool = True,
        use_vessel_channel: bool = True,
        augment: bool = False,
    ):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.fundus_size = int(fundus_size)
        self.rnflt_min = float(rnflt_min)
        self.rnflt_max = float(rnflt_max)
        self.rnflt_key = rnflt_key
        self.crop_retina = bool(crop_retina)
        self.canonicalize_eye = bool(canonicalize_eye)
        self.use_vessel_channel = bool(use_vessel_channel)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        path = self.data_dir / str(row["combined_npz"])
        with np.load(path, allow_pickle=True) as data:
            rgb = data["colorfundus"].astype(np.uint8)
            rnflt = data[self.rnflt_key].astype(np.float32)
            pid = str(data["pid"])
            glaucoma = int(data["glaucoma"])

        eye = eye_from_pid(pid)
        flip = self.canonicalize_eye and eye == 1
        if flip:
            rnflt = np.fliplr(rnflt).copy()

        fundus = preprocess_fundus(
            rgb,
            self.fundus_size,
            crop_retina=self.crop_retina,
            flip=flip,
            augment=self.augment,
        )
        if self.use_vessel_channel:
            fundus = attach_vessel_channel(fundus)
        rnflt_norm, valid = normalize_rnflt_map(rnflt, self.rnflt_min, self.rnflt_max)
        target = torch.from_numpy(rnflt_norm)[None]
        valid_mask = torch.from_numpy(valid.astype(np.float32))[None]
        meta = {
            "pid": pid,
            "eye": -1 if eye is None else eye,
            "canonical_flipped": bool(flip),
            "glaucoma": glaucoma,
            "combined_npz": str(row["combined_npz"]),
        }
        return fundus, target, valid_mask, meta
