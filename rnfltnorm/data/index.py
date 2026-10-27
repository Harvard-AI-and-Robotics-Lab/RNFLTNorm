"""Index tables and patient-level train/val/holdout splits."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "holdout"]


def patient_id(full_pid_rnflt: str) -> str:
    return str(full_pid_rnflt).split("_")[0]


def apply_index_filters(
    df: pd.DataFrame,
    imagetypes: Optional[Tuple[str, ...]] = ("ONH", "macula"),
    viewingangle_max: float = 50.0,
    delta_days_max: int = 30,
) -> pd.DataFrame:
    out = df
    if imagetypes is not None and "imagetype.min" in out.columns:
        out = out[out["imagetype.min"].isin(imagetypes)]
    if viewingangle_max is not None and "viewingangle.min" in out.columns:
        out = out[out["viewingangle.min"] <= viewingangle_max]
    if delta_days_max is not None and "delta_days" in out.columns:
        out = out[out["delta_days"] <= delta_days_max]
    return out.reset_index(drop=True)


def make_patient_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by patient: holdout, val, train."""
    if len(df) == 0:
        empty = df.iloc[0:0].copy()
        return empty, empty, empty

    patients = sorted({patient_id(x) for x in df["full_pid_rnflt"]})
    rng = np.random.default_rng(seed)
    patients = list(patients)
    rng.shuffle(patients)

    n_holdout = max(1, int(round(len(patients) * holdout_ratio))) if holdout_ratio > 0 else 0
    if n_holdout >= len(patients):
        n_holdout = max(0, len(patients) - 1)

    holdout_set = set(patients[:n_holdout])
    dev_patients = [p for p in patients if p not in holdout_set]

    n_val = 0
    val_set: set = set()
    if dev_patients and val_ratio > 0:
        n_val = max(1, int(round(len(dev_patients) * val_ratio)))
        if n_val >= len(dev_patients):
            n_val = max(1, len(dev_patients) - 1) if len(dev_patients) > 1 else 1
        val_set = set(dev_patients[:n_val])
    train_set = set(dev_patients) - val_set

    pid = df["full_pid_rnflt"].map(patient_id)
    holdout_df = df.loc[pid.isin(holdout_set)].reset_index(drop=True)
    val_df = df.loc[pid.isin(val_set)].reset_index(drop=True)
    train_df = df.loc[pid.isin(train_set)].reset_index(drop=True)
    return train_df, val_df, holdout_df


def patient_split_table(
    df: pd.DataFrame,
    holdout_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per patient with split assignment (reproducible)."""
    patients = sorted({patient_id(x) for x in df["full_pid_rnflt"]})
    rng = np.random.default_rng(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)

    n_holdout = max(1, int(round(len(shuffled) * holdout_ratio))) if holdout_ratio > 0 else 0
    if n_holdout >= len(shuffled):
        n_holdout = max(0, len(shuffled) - 1)
    holdout_set = set(shuffled[:n_holdout])
    dev = [p for p in shuffled if p not in holdout_set]

    n_val = 0
    val_set: set = set()
    if dev and val_ratio > 0:
        n_val = max(1, int(round(len(dev) * val_ratio)))
        if n_val >= len(dev):
            n_val = max(1, len(dev) - 1) if len(dev) > 1 else 1
        val_set = set(dev[:n_val])

    rows = []
    for p in shuffled:
        if p in holdout_set:
            split: SplitName = "holdout"
        elif p in val_set:
            split = "val"
        else:
            split = "train"
        rows.append({"patient_id": p, "split": split})
    return pd.DataFrame(rows)


def dedupe_rnflt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one fundus row per RNFLT exam (smallest delta_days, then jpgfile)."""
    out = df.sort_values(["rnflt_npz", "delta_days", "jpgfile"]).reset_index(drop=True)
    return out.drop_duplicates(subset=["rnflt_npz"], keep="first").reset_index(drop=True)


def load_index_table(
    index_csv: Path,
    dedupe_rnflt: bool = True,
    export_status: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    if export_status is not None and "export_status" in df.columns:
        df = df[df["export_status"] == export_status].copy()
    if dedupe_rnflt:
        df = dedupe_rnflt_rows(df)
    return df
