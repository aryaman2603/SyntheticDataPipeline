"""
cardio_loader.py — Cardiovascular Disease dataset.

Source: Kaggle, sulianova/cardiovascular-disease-dataset
(https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
Paper's "Dataset:CD" (Yan et al., VAE-GAN, Knowledge-Based Systems 2025).
70,000 rows, 11 features + target `cardio` (binary). Raw file is
SEMICOLON-delimited (a known gotcha with this specific Kaggle CSV) —
sep=";" below, don't drop this if you re-source the file.

Columns: id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;
alco;active;cardio
  - age is in DAYS, not years (e.g. 18393 ≈ 50 years). Left as-is here
    since the paper doesn't state a conversion; note this explicitly if
    you want to add an /365.25 transform for interpretability — it
    won't change what the generator learns, just how humans read it.
  - gender, cholesterol, gluc are integer-coded categoricals (not
    continuous, despite being numeric dtype) — same pattern as your
    existing acs_income config's categorical_cols for integer-coded
    census columns.

Paper: "randomly subsampled to 30,000 instances" — done here via
config['datasets']['cardio']['subsample_n'].
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion.replication_common import save_processed

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]


def load_and_process(config: dict, dataset_name: str = "cardio") -> None:
    ds_cfg = config["datasets"][dataset_name]

    df = pd.read_csv(ds_cfg["raw_file"], sep=";")

    missing_expected = {"id", "age", "cardio"} - set(df.columns)
    if missing_expected:
        raise ValueError(
            f"[cardio] Expected columns not found: {missing_expected}. "
            f"Got columns: {list(df.columns)}. This dataset's raw CSV is "
            f"semicolon-delimited — if you re-downloaded it and this fires, "
            f"check the file wasn't re-saved with comma delimiters."
        )

    df = df.drop(columns=["id"])

    subsample_n = ds_cfg.get("subsample_n")
    if subsample_n and len(df) > subsample_n:
        rng = np.random.RandomState(config["global"]["seed"])
        df = df.sample(n=subsample_n, random_state=rng).reset_index(drop=True)
        logger.info(f"[cardio] Subsampled to {subsample_n} rows (paper: 'randomly subsampled to 30,000')")

    logger.info(f"[cardio] Loaded {len(df)} rows")

    save_processed(
        df=df,
        processed_dir=ds_cfg["processed_dir"],
        target_col="cardio",
        categorical_cols=CATEGORICAL_COLS,
        test_size=config["global"]["test_size"],
        seed=config["global"]["seed"],
        source_note="Kaggle sulianova/cardiovascular-disease-dataset",
    )