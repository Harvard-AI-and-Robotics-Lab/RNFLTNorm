#!/usr/bin/env python3
"""Infer a healthy RNFLT-norm map from a color fundus image or npz."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rnfltnorm.infer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MEAN_MAP,
    load_fundus_rgb,
    load_predictor,
    plot_2dmap,
    prepare_fundus,
    predict_rnflt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="color fundus PNG/JPEG or a fundusrnflt .npz")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--mean-map", type=Path, default=DEFAULT_MEAN_MAP)
    p.add_argument("--output", type=Path, default=None, help="optional .npy path for the prediction")
    p.add_argument("--flip", action="store_true", help="mirror OS to OD-like orientation")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--show", action="store_true", help="plot the predicted RNFLT map")
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return data["colorfundus"].astype(np.uint8)
    return load_fundus_rgb(path)


def main() -> None:
    args = parse_args()
    rgb = load_rgb(Path(args.input))
    model, cfg, mean_np, device = load_predictor(
        args.checkpoint, args.mean_map, device=args.device
    )
    fundus = prepare_fundus(rgb, cfg, flip=args.flip)
    pred, mask = predict_rnflt(model, fundus, mean_np, cfg, device)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, pred)
        print("Saved predicted RNFLT map to", args.output)
    if args.show:
        plot_2dmap(pred, mask, title="Predicted RNFLT Norm")
    print("Predicted RNFLT shape:", pred.shape)
    print("Mean RNFLT: {:.2f} um".format(float(np.nanmean(np.where(mask, pred, np.nan)))))


if __name__ == "__main__":
    main()
