"""Fundus-to-RNFLT v7: residual polar predictor with high-frequency detail loss."""

from rnfltnorm.infer import (
    load_fundus_rgb,
    load_predictor,
    plot_2dmap,
    prepare_fundus,
    predict_rnflt,
)

__version__ = "0.7.0"
__all__ = [
    "load_fundus_rgb",
    "load_predictor",
    "plot_2dmap",
    "prepare_fundus",
    "predict_rnflt",
]
