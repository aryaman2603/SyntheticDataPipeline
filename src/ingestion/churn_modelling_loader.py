"""
churn_modelling_loader.py — Bank customer churn dataset.

Source: Kaggle, shrutimechlearn/churn-modelling
(https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
Paper's "Dataset:CM" (Yan et al., VAE-GAN, Knowledge-Based Systems 2025).
10,000 rows, no missing values, standard raw column names:
  RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age,
  Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
  EstimatedSalary, Exited

Identifier columns (RowNumber, CustomerId, Surname) are dropped — they
carry no generalisable signal and Surname in particular is a near-unique
free-text field that would badly distort a tabular generator.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ingestion.replication_common import save_processed

logger = logging.getLogger(__name__)

DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_COLS = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "Exited"]


def load_and_process(config: dict, dataset_name: str = "churn_modelling") -> None:
    ds_cfg = config["datasets"][dataset_name]

    df = pd.read_csv(ds_cfg["raw_file"])
    missing_expected = {"RowNumber", "CustomerId", "Surname", "Exited"} - set(df.columns)
    if missing_expected:
        raise ValueError(
            f"[churn_modelling] Expected columns not found: {missing_expected}. "
            f"Got columns: {list(df.columns)}. Confirm you downloaded the "
            f"shrutimechlearn/churn-modelling version of this dataset."
        )

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    n_missing = df.isna().sum().sum()
    if n_missing:
        logger.warning(f"[churn_modelling] {n_missing} missing values found (unexpected for this dataset) — dropping rows")
        df = df.dropna()

    logger.info(f"[churn_modelling] Loaded {len(df)} rows")

    save_processed(
        df=df,
        processed_dir=ds_cfg["processed_dir"],
        target_col="Exited",
        categorical_cols=CATEGORICAL_COLS,
        test_size=config["global"]["test_size"],
        seed=config["global"]["seed"],
        source_note="Kaggle shrutimechlearn/churn-modelling",
    )