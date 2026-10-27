"""Fundus-to-RNFLT v5: residual polar predictor."""

from rnfltnorm.infer import (
    load_fundus_rgb,
    load_predictor,
    plot_2dmap,
    prepare_fundus,
    predict_rnflt,
)

__version__ = "0.5.0"
__all__ = [
    "load_fundus_rgb",
    "load_predictor",
    "plot_2dmap",
    "prepare_fundus",
    "predict_rnflt",
]
