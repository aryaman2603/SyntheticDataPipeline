"""
covertype_loader.py — Forest Covertype dataset.

Source: UCI (archive.ics.uci.edu/dataset/31/covertype), 581,012 rows,
54 features (10 quantitative + 4 binary wilderness-area one-hot + 40
binary soil-type one-hot), target Cover_Type ∈ {1..7} — a 7-CLASS
problem, not binary. Your existing tstr.py already branches on n_classes
(multi_class="ovr", average="weighted" for roc_auc_score), so no changes
needed there — just be aware TSTR AUC here is a multiclass OVR score,
not directly the same quantity as your binary-target datasets.

Paper's "Dataset:CT" (Yan et al., VAE-GAN, Knowledge-Based Systems 2025):
"randomly subsampled to 30,000 instances" — done here via
config['datasets']['covertype']['subsample_n'], plain random sample
(not stratified), matching the paper's stated wording.

Two ways to get the raw data, both supported:
  1. (Recommended, no manual download) sklearn.datasets.fetch_covtype() —
     pulls from sklearn's own mirror on first call and caches locally.
     Set config['datasets']['covertype']['use_sklearn_fetch'] = true.
  2. Manual UCI download: covtype.data.gz, 55 comma-separated columns
     (54 features + target), no header. Point 'raw_file' at the
     decompressed CSV and leave use_sklearn_fetch unset/false.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion.replication_common import save_processed

logger = logging.getLogger(__name__)

QUANT_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type_{i}" for i in range(1, 41)]
ALL_FEATURE_COLS = QUANT_COLS + WILDERNESS_COLS + SOIL_COLS
TARGET_COL = "Cover_Type"


def _load_via_sklearn() -> pd.DataFrame:
    from sklearn.datasets import fetch_covtype
    bunch = fetch_covtype(as_frame=False, download_if_missing=True)
    df = pd.DataFrame(bunch.data, columns=ALL_FEATURE_COLS)
    df[TARGET_COL] = bunch.target
    return df


def _load_via_raw_file(raw_file: str) -> pd.DataFrame:
    df = pd.read_csv(raw_file, header=None, names=ALL_FEATURE_COLS + [TARGET_COL])
    return df


def load_and_process(config: dict, dataset_name: str = "covertype") -> None:
    ds_cfg = config["datasets"][dataset_name]

    if ds_cfg.get("use_sklearn_fetch", True):
        logger.info("[covertype] Fetching via sklearn.datasets.fetch_covtype()")
        df = _load_via_sklearn()
    else:
        df = _load_via_raw_file(ds_cfg["raw_file"])

    subsample_n = ds_cfg.get("subsample_n")
    if subsample_n and len(df) > subsample_n:
        rng = np.random.RandomState(config["global"]["seed"])
        df = df.sample(n=subsample_n, random_state=rng).reset_index(drop=True)
        logger.info(f"[covertype] Subsampled to {subsample_n} rows (paper: 'randomly subsampled to 30,000')")

    # Collapse the one-hot wilderness/soil blocks back to single categorical
    # columns before handing off — CTGAN/TVAE's conditional generator
    # handles a small number of true categorical columns far better than
    # 44 near-constant binary columns, and this is a standard, reversible
    # transform (not a paper-specified choice, but a defensible one for
    # generator input quality).
    df["Wilderness_Area"] = df[WILDERNESS_COLS].idxmax(axis=1)
    df["Soil_Type"] = df[SOIL_COLS].idxmax(axis=1)
    df = df.drop(columns=WILDERNESS_COLS + SOIL_COLS)

    categorical_cols = ["Wilderness_Area", "Soil_Type", TARGET_COL]

    logger.info(f"[covertype] Loaded {len(df)} rows, target has {df[TARGET_COL].nunique()} classes")

    save_processed(
        df=df,
        processed_dir=ds_cfg["processed_dir"],
        target_col=TARGET_COL,
        categorical_cols=categorical_cols,
        test_size=config["global"]["test_size"],
        seed=config["global"]["seed"],
        source_note="UCI Covertype (archive.ics.uci.edu/dataset/31/covertype)",
    )