"""Configuration for residual healthy-baseline v5."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("RNFLTNORM_DATA_ROOT", REPO_ROOT.parent))


@dataclass
class V5Config:
    data_dir: Path = DATA_ROOT / "colorfundus_rnflt_match_dataset"
    index_csv: Path = DATA_ROOT / "metatables/color_fundus_rnflt_combined_npz_index.csv"
    stats_json: Path = DATA_ROOT / "metatables/rnflt_norm_stats.json"
    patient_split_csv: Path = DATA_ROOT / "metatables/diffusion_patient_split_holdout20.csv"
    rnflt_key: str = "rnflt_artifact_correct"
    imagetype_filter: tuple = ("ONH", "macula")
    viewingangle_max: float = 50.0
    delta_days_max: int = 30
    holdout_ratio: float = 0.2
    val_ratio: float = 0.1
    seed: int = 42
    dedupe_rnflt: bool = True
    healthy_baseline: bool = True
    healthy_label: int = 0
    glaucoma_cache_csv: Path = DATA_ROOT / "metatables/combined_npz_glaucoma_cache.csv"
    reuse_patient_split: bool = True

    fundus_size: int = 384
    image_size: int = 200
    crop_retina: bool = True
    canonicalize_eye: bool = True
    use_vessel_channel: bool = True

    rnflt_min: float = 0.0
    rnflt_max: float = 220.0

    backbone: str = "resnet34"
    decoder_channels: int = 96
    pretrained: bool = True
    residual_scale: float = 0.35
    polar_bins: int = 72
    polar_inner_radius: float = 0.20
    polar_outer_radius: float = 0.46

    l1_weight: float = 1.0
    peak_weight: float = 2.0
    residual_l1_weight: float = 1.0
    ms_ssim_weight: float = 0.4
    gradient_weight: float = 0.35
    polar_weight: float = 0.6
    polar_consistency_weight: float = 0.2
    mask_weight: float = 0.1

    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 120
    decoder_lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999

    output_dir: Path = REPO_ROOT / "runs/fundus2rnflt_v5_healthy_baseline"
    mean_map_name: str = "mean_rnflt.npz"
    log_every_steps: int = 50
    preview_every_epochs: int = 5
    preview_cases: int = 4
    device: str = "cuda"

    def to_dict(self):
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result


def load_stats_into_v5_config(cfg: V5Config) -> V5Config:
    if not cfg.stats_json.is_file():
        return cfg
    import json

    stats = json.loads(cfg.stats_json.read_text())
    cfg.rnflt_min = float(stats["p01"])
    cfg.rnflt_max = float(stats["p99"])
    return cfg
