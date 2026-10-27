"""Load the v7 checkpoint and predict a healthy RNFLT-norm map from a color fundus."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rnfltnorm.config import REPO_ROOT
from rnfltnorm.data.preprocess import attach_vessel_channel, preprocess_fundus
from rnfltnorm.models.predictor import ResidualPolarRnfltPredictor

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint/checkpoint_best.pt"
DEFAULT_MEAN_MAP = REPO_ROOT / "checkpoint/mean_rnflt.npz"


def load_predictor(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    mean_map: str | Path = DEFAULT_MEAN_MAP,
    device: str | torch.device | None = None,
    raw_weights: bool = False,
) -> tuple[ResidualPolarRnfltPredictor, dict[str, Any], np.ndarray, torch.device]:
    """Load EMA (or raw) weights, training config, and the mean RNFLT map."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    payload = torch.load(Path(checkpoint), map_location="cpu")
    cfg = payload["config"]
    mean_path = Path(mean_map)
    if not mean_path.is_file():
        mean_path = Path(cfg["output_dir"]) / cfg.get("mean_map_name", "mean_rnflt.npz")
    mean_np = np.load(mean_path)["mean"].astype(np.float32)
    in_channels = 4 if bool(cfg.get("use_vessel_channel", True)) else 3
    model = ResidualPolarRnfltPredictor(
        output_size=int(cfg.get("image_size", 200)),
        decoder_channels=int(cfg.get("decoder_channels", 96)),
        pretrained=False,
        in_channels=in_channels,
        residual_scale=float(cfg.get("residual_scale", 1.0)),
        polar_bins=int(cfg.get("polar_bins", 72)),
        polar_inner_radius=float(cfg.get("polar_inner_radius", 0.20)),
        polar_outer_radius=float(cfg.get("polar_outer_radius", 0.46)),
    ).to(device)
    state_key = "model" if raw_weights else "ema_model"
    model.load_state_dict(payload.get(state_key, payload["model"]))
    model.eval()
    return model, cfg, mean_np, device


def load_fundus_rgb(path: str | Path) -> np.ndarray:
    """Read a color fundus PNG/JPEG as ``H×W×3`` uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def prepare_fundus(
    rgb: np.ndarray,
    cfg: dict[str, Any],
    *,
    flip: bool = False,
    crop_retina: bool | None = None,
) -> torch.Tensor:
    """Crop, optionally mirror OS→OD, resize, and attach the vessel channel.

    Packaged sample PNGs under ``image/`` are already cropped and OD-canonical;
    pass ``crop_retina=False`` and ``flip=False`` for those files.
    """
    if crop_retina is None:
        crop_retina = bool(cfg.get("crop_retina", True))
    fundus = preprocess_fundus(
        rgb,
        int(cfg.get("fundus_size", 384)),
        crop_retina=crop_retina,
        flip=flip,
        augment=False,
    )
    if bool(cfg.get("use_vessel_channel", True)):
        fundus = attach_vessel_channel(fundus)
    return fundus


@torch.no_grad()
def predict_rnflt(
    model: ResidualPolarRnfltPredictor,
    fundus: torch.Tensor,
    mean_np: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    mask_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted thickness in µm and the predicted valid-tissue mask."""
    mean_thickness = torch.from_numpy(mean_np)[None, None].to(device)
    output = model(fundus[None].to(device), mean_thickness)
    pred_norm = output["thickness"][0, 0].cpu().numpy()
    pred_mask = torch.sigmoid(output["mask_logits"])[0, 0].cpu().numpy() >= mask_threshold
    pred_um = pred_norm * (float(cfg["rnflt_max"]) - float(cfg["rnflt_min"])) + float(cfg["rnflt_min"])
    return pred_um, pred_mask


def plot_2dmap(
    thickness: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    vmin: float = 0.0,
    vmax: float = 350.0,
    title: str = "Predicted RNFLT Norm",
    cmap: str = "turbo",
):
    """Plot a 200×200 RNFLT map on the 0–350 µm turbo scale."""
    plot = np.asarray(thickness, dtype=np.float32)
    if valid is not None:
        plot = np.where(valid, plot, np.nan)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    image = ax.imshow(plot, cmap=palette, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("µm")
    fig.tight_layout()
    plt.show()
    return fig
