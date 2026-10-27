"""Predict healthy RNFLT from a glaucoma fundus and write the GitHub demo panel."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from rnfltnorm.config import REPO_ROOT, V5Config
from rnfltnorm.data.preprocess import attach_vessel_channel, eye_from_pid, preprocess_fundus
from rnfltnorm.infer import mean_from_checkpoint
from rnfltnorm.models.predictor import ResidualPolarRnfltPredictor
from rnfltnorm.rnflt_norm import normalize_rnflt_map


DEMO_CASES = [
    {
        "eye_id": "1496028_1",
        "glaucoma_npz": "fundusrnflt_051576.npz",
        "healthy_npz": "fundusrnflt_051582.npz",
    },
    {
        "eye_id": "9386127_0",
        "glaucoma_npz": "fundusrnflt_098647.npz",
        "healthy_npz": "fundusrnflt_098652.npz",
    },
]

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoint/checkpoint_best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--mean-map", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=V5Config.data_dir)
    p.add_argument("--image-dir", type=Path, default=REPO_ROOT / "image")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--raw-weights", action="store_true")
    return p.parse_args()


def load_model(
    checkpoint_path: Path,
    mean_map_path: Path | None,
    device: torch.device,
    raw_weights: bool,
) -> tuple[ResidualPolarRnfltPredictor, dict[str, Any], np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    mean_np = mean_from_checkpoint(checkpoint, mean_map_path)
    in_channels = 4 if bool(cfg.get("use_vessel_channel", True)) else 3
    model = ResidualPolarRnfltPredictor(
        output_size=int(cfg.get("image_size", 200)),
        decoder_channels=int(cfg.get("decoder_channels", 96)),
        pretrained=False,
        in_channels=in_channels,
        residual_scale=float(cfg.get("residual_scale", 0.35)),
        polar_bins=int(cfg.get("polar_bins", 72)),
        polar_inner_radius=float(cfg.get("polar_inner_radius", 0.20)),
        polar_outer_radius=float(cfg.get("polar_outer_radius", 0.46)),
    ).to(device)
    state_key = "model" if raw_weights else "ema_model"
    model.load_state_dict(checkpoint.get(state_key, checkpoint["model"]))
    model.eval()
    return model, cfg, mean_np


def load_exam(path: Path, rnflt_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        rgb = data["colorfundus"].astype(np.uint8)
        rnflt = data[rnflt_key].astype(np.float32)
        pid = str(data["pid"])
    return rgb, rnflt, pid


def canonical_rnflt(rnflt: np.ndarray, flip: bool) -> np.ndarray:
    return np.fliplr(rnflt).copy() if flip else rnflt


def plot_map(thickness: np.ndarray, valid: np.ndarray, rnflt_min: float, rnflt_max: float) -> np.ndarray:
    um = thickness * (rnflt_max - rnflt_min) + rnflt_min
    return np.where(valid, um, np.nan)


def masked_mean_um(thickness: np.ndarray, valid: np.ndarray, rnflt_min: float, rnflt_max: float) -> float:
    um = thickness[valid] * (rnflt_max - rnflt_min) + rnflt_min
    return float(np.mean(um)) if um.size else float("nan")


@torch.no_grad()
def predict_case(
    model: ResidualPolarRnfltPredictor,
    cfg: dict[str, Any],
    mean_np: np.ndarray,
    glaucoma_path: Path,
    healthy_path: Path,
    device: torch.device,
    mask_threshold: float,
) -> dict[str, Any]:
    rnflt_key = str(cfg.get("rnflt_key", "rnflt_artifact_correct"))
    rgb, glc_rnflt, pid = load_exam(glaucoma_path, rnflt_key)
    _, hlt_rnflt, _ = load_exam(healthy_path, rnflt_key)
    eye = eye_from_pid(pid)
    flip = bool(cfg.get("canonicalize_eye", True)) and eye == 1
    fundus = preprocess_fundus(
        rgb,
        int(cfg.get("fundus_size", 384)),
        crop_retina=bool(cfg.get("crop_retina", True)),
        flip=flip,
        augment=False,
    )
    in_channels = 4 if bool(cfg.get("use_vessel_channel", True)) else 3
    model_input = attach_vessel_channel(fundus) if in_channels == 4 else fundus
    mean_thickness = torch.from_numpy(mean_np)[None, None].to(device)
    output = model(model_input[None].to(device), mean_thickness)
    pred_norm = output["thickness"][0, 0].cpu().numpy()
    pred_mask = torch.sigmoid(output["mask_logits"])[0, 0].cpu().numpy() >= mask_threshold

    rnflt_min = float(cfg["rnflt_min"])
    rnflt_max = float(cfg["rnflt_max"])
    glc_norm, glc_valid = normalize_rnflt_map(canonical_rnflt(glc_rnflt, flip), rnflt_min, rnflt_max)
    hlt_norm, hlt_valid = normalize_rnflt_map(canonical_rnflt(hlt_rnflt, flip), rnflt_min, rnflt_max)
    pred_um = pred_norm * (rnflt_max - rnflt_min) + rnflt_min
    hlt_um = hlt_norm * (rnflt_max - rnflt_min) + rnflt_min
    mae = float(np.mean(np.abs(pred_um[hlt_valid] - hlt_um[hlt_valid]))) if hlt_valid.any() else float("nan")
    laterality = "OS mirrored" if flip else "OD"
    return {
        "pid": pid,
        "eye_id": "_".join(str(pid).split("_")[:2]),
        "laterality": laterality,
        "glaucoma_name": glaucoma_path.name,
        "healthy_name": healthy_path.name,
        "fundus_rgb": np.clip(fundus[:3].permute(1, 2, 0).numpy(), 0.0, 1.0),
        "glaucoma_plot": plot_map(glc_norm, glc_valid, rnflt_min, rnflt_max),
        "healthy_plot": plot_map(hlt_norm, hlt_valid, rnflt_min, rnflt_max),
        "pred_plot": np.where(pred_mask, pred_um, np.nan),
        "glaucoma_mean": masked_mean_um(glc_norm, glc_valid, rnflt_min, rnflt_max),
        "healthy_mean": masked_mean_um(hlt_norm, hlt_valid, rnflt_min, rnflt_max),
        "pred_mean": float(np.nanmean(np.where(pred_mask, pred_um, np.nan))),
        "mae": mae,
    }


def save_fundus_png(rgb: np.ndarray, path: Path) -> None:
    image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_panel(cases: list[dict[str, Any]], path: Path) -> None:
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(16.4, 4.15 * n), squeeze=False)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgray")
    last_im = None
    for row, case in enumerate(cases):
        axes[row, 0].imshow(case["fundus_rgb"])
        axes[row, 0].set_title("Input fundus (Glaucoma)", fontsize=12, pad=8)
        last_im = axes[row, 1].imshow(case["glaucoma_plot"], cmap=cmap, vmin=0, vmax=350)
        axes[row, 1].set_title("Paired RNFLT (Glaucoma)", fontsize=12, pad=8)
        axes[row, 2].imshow(case["healthy_plot"], cmap=cmap, vmin=0, vmax=350)
        axes[row, 2].set_title("True RNFLT Norm (Non-Glaucoma)", fontsize=12, pad=8)
        axes[row, 3].imshow(case["pred_plot"], cmap=cmap, vmin=0, vmax=350)
        axes[row, 3].set_title("Predicted RNFLT Norm", fontsize=12, pad=8)
        for axis in axes[row]:
            axis.axis("off")
    fig.subplots_adjust(left=0.02, right=0.90, top=0.92, bottom=0.04, wspace=0.08, hspace=0.28)
    cax = fig.add_axes([0.915, 0.12, 0.015, 0.70])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("µm")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, mean_np = load_model(args.checkpoint, args.mean_map, device, args.raw_weights)
    cases = []
    for spec in DEMO_CASES:
        glaucoma_path = args.data_dir / spec["glaucoma_npz"]
        healthy_path = args.data_dir / spec["healthy_npz"]
        case = predict_case(
            model,
            cfg,
            mean_np,
            glaucoma_path,
            healthy_path,
            device,
            args.mask_threshold,
        )
        fundus_path = args.image_dir / f"fundus_{spec['eye_id']}.png"
        save_fundus_png(case["fundus_rgb"], fundus_path)
        print(
            f"{spec['eye_id']} {case['laterality']}: "
            f"glaucoma mean={case['glaucoma_mean']:.1f} µm, "
            f"healthy mean={case['healthy_mean']:.1f} µm, "
            f"pred mean={case['pred_mean']:.1f} µm, MAE={case['mae']:.1f} µm"
        )
        print(f"wrote {fundus_path}")
        cases.append(case)
    panel_path = args.image_dir / "sample_predictions.png"
    save_panel(cases, panel_path)
    print(f"wrote {panel_path}")


if __name__ == "__main__":
    main()
