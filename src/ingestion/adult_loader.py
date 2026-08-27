"""
adult_loader.py — UCI Adult / Census Income dataset.

Source: https://archive.ics.uci.edu/dataset/2/adult
Paper's "Dataset:AD" (Yan et al., VAE-GAN, Knowledge-Based Systems 2025).

Standard UCI Adult format: 14 features + income target, comma-separated,
"?" as missing-value marker. UCI ships this as TWO separate files
(adult.data + adult.test) with two quirks this loader handles
automatically so no manual file-surgery is needed:
  1. adult.test's first line is a stray comment
     ("|1x3 Cross validator") — not data, skipped automatically.
  2. adult.test's target values have a trailing "." (">50K." vs ">50K")
     that adult.data's don't — stripped so train/test values match
     before the two files are concatenated.

Point config['datasets']['adult']['raw_file'] at adult.data and
config['datasets']['adult']['raw_file_test'] at adult.test — just the
two files as downloaded from UCI, unmodified. If raw_file_test isn't
set, falls back to treating raw_file as a single already-combined file
(old behaviour, still supported).

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


def _load_single(path: str, skip_first_line: bool = False) -> pd.DataFrame:
    return pd.read_csv(
        path, header=None, names=COLUMNS,
        skipinitialspace=True, na_values="?",
        skiprows=1 if skip_first_line else 0,
        comment=None,
    )


def load_and_process(config: dict, dataset_name: str = "adult") -> None:
    ds_cfg = config["datasets"][dataset_name]

    if ds_cfg.get("raw_file_test"):
        # Two separate UCI files, unmodified — handle both quirks here.
        train_part = _load_single(ds_cfg["raw_file"], skip_first_line=False)
        # adult.test's first line is a stray comment, not data — detect
        # and skip it rather than assuming it's always present, in case
        # someone already stripped it.
        with open(ds_cfg["raw_file_test"]) as f:
            first_line = f.readline()
        test_needs_skip = first_line.startswith("|")
        test_part = _load_single(ds_cfg["raw_file_test"], skip_first_line=test_needs_skip)
        df = pd.concat([train_part, test_part], ignore_index=True)
        logger.info(
            f"[adult] Combined {len(train_part)} (adult.data) + {len(test_part)} "
            f"(adult.test, header_skipped={test_needs_skip}) = {len(df)} rows"
        )
    else:
        # Fallback: a single already-combined raw file (old behaviour).
        df = _load_single(ds_cfg["raw_file"])

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