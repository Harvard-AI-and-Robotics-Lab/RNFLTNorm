"""Filter paired exams to the first healthy visit per eye."""
from __future__ import annotations

import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from rnfltnorm.data.index import patient_id


def eye_id_from_full_pid(full_pid: str) -> str:
    """Return ``patient_eye`` from ``patient_eye_timestamp`` identifiers."""
    parts = str(full_pid).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _read_glaucoma_from_npz(path: str) -> tuple[str, Optional[int]]:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("glaucoma.npy") as handle:
                value = np.load(handle, allow_pickle=True)
        return path, int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return path, None


def build_glaucoma_cache(
    rows: pd.DataFrame,
    data_dir: Path,
    cache_csv: Path,
    *,
    workers: int = 16,
) -> pd.DataFrame:
    """Read ``glaucoma`` from combined npz zip members and cache one row per file."""
    data_dir = Path(data_dir)
    names = sorted({str(name) for name in rows["combined_npz"].dropna().astype(str)})
    paths = [str(data_dir / name) for name in names]
    records = []
    if paths:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for path, label in pool.map(_read_glaucoma_from_npz, paths, chunksize=32):
                records.append(
                    {
                        "combined_npz": Path(path).name,
                        "glaucoma": label,
                    }
                )
    cache = pd.DataFrame.from_records(records)
    cache_csv = Path(cache_csv)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(cache_csv, index=False)
    return cache


def load_or_build_glaucoma_cache(
    rows: pd.DataFrame,
    data_dir: Path,
    cache_csv: Path,
    *,
    workers: int = 16,
) -> pd.DataFrame:
    cache_csv = Path(cache_csv)
    needed = set(rows["combined_npz"].dropna().astype(str))
    if cache_csv.is_file():
        cache = pd.read_csv(cache_csv)
        have = set(cache["combined_npz"].astype(str))
        if needed <= have:
            return cache
    return build_glaucoma_cache(rows, data_dir, cache_csv, workers=workers)


def apply_healthy_baseline_filter(
    df: pd.DataFrame,
    glaucoma_cache: pd.DataFrame,
    *,
    healthy_label: int = 0,
) -> pd.DataFrame:
    """Keep the earliest healthy RNFLT exam per patient-eye, with its paired fundus.

    ``glaucoma`` in the Harvard RNFLT export is 0=healthy, 1=glaucoma, 2=suspect.
    Baseline is the first remaining healthy exam for that eye.
    """
    out = df.merge(glaucoma_cache, on="combined_npz", how="left")
    healthy = out[out["glaucoma"] == int(healthy_label)].copy()
    if healthy.empty:
        return healthy.reset_index(drop=True)
    healthy["patient_id"] = healthy["full_pid_rnflt"].map(patient_id)
    healthy["eye_id"] = healthy["full_pid_rnflt"].map(eye_id_from_full_pid)
    healthy["rnflt_time"] = pd.to_datetime(healthy["rnflt_time"], errors="coerce")
    healthy = healthy.sort_values(
        ["eye_id", "rnflt_time", "delta_days", "jpgfile"]
    ).reset_index(drop=True)
    baseline = healthy.drop_duplicates(subset=["eye_id"], keep="first").reset_index(drop=True)
    return baseline


def apply_patient_split_from_table(
    df: pd.DataFrame,
    split_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse an existing patient split so healthy-baseline and all-pairs share patients."""
    split = pd.read_csv(split_csv)
    mapping = {
        str(pid): str(name)
        for pid, name in zip(split["patient_id"], split["split"])
    }
    labels = df["full_pid_rnflt"].map(patient_id).map(mapping)
    train_df = df.loc[labels == "train"].reset_index(drop=True)
    val_df = df.loc[labels == "val"].reset_index(drop=True)
    holdout_df = df.loc[labels == "holdout"].reset_index(drop=True)
    return train_df, val_df, holdout_df
