"""
adult_loader.py — UCI Adult / Census Income dataset.

Source: https://archive.ics.uci.edu/dataset/2/adult
Paper's "Dataset:AD" (Yan et al., VAE-GAN, Knowledge-Based Systems 2025).

Standard UCI Adult format: 14 features + income target, comma-separated,
"?" as missing-value marker, target values have a trailing period in the
official test split (adult.test) but not in adult.data — this loader
strips that so both files can be concatenated into one raw CSV before
running (recommended: cat adult.data adult.test > adult_combined.csv,
skip the ">50K.\n" header line adult.test starts with, point
config['datasets']['adult']['raw_file'] at the combined file).

Known, standard preprocessing choice (not paper-specified, but universal
in Adult-dataset literature): fnlwgt is a census sampling weight, not a
real feature — dropped. Rows with "?" in any column are dropped rather
than imputed, matching the simplest common treatment (~7% of rows).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ingestion.replication_common import save_processed

logger = logging.getLogger(__name__)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country", "income",
]


def load_and_process(config: dict, dataset_name: str = "adult") -> None:
    ds_cfg = config["datasets"][dataset_name]

    df = pd.read_csv(
        ds_cfg["raw_file"], header=None, names=COLUMNS,
        skipinitialspace=True, na_values="?",
    )

    # Normalise target: strip trailing "." present in the official test
    # split (">50K." / "<=50K.") so train/test values match.
    df["income"] = df["income"].str.rstrip(".")

    df = df.drop(columns=["fnlwgt"])
    df = df.dropna()  # drop rows with any "?" (~7% of rows, standard treatment)

    logger.info(f"[adult] Loaded {len(df)} rows after dropping missing values")

    save_processed(
        df=df,
        processed_dir=ds_cfg["processed_dir"],
        target_col="income",
        categorical_cols=[c for c in CATEGORICAL_COLS if c != "fnlwgt"],
        test_size=config["global"]["test_size"],
        seed=config["global"]["seed"],
        source_note="UCI Adult (archive.ics.uci.edu/dataset/2/adult)",
    )