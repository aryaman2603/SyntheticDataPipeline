"""
home_credit_loader.py — Loader for the Home Credit Default Risk dataset.

Preprocessing decisions (from EDA):
  - Drop identifier: SK_ID_CURR
  - Drop entire AVG/MODE/MEDI property block (~45 cols, structurally missing
    for ~48% of rows — same rows null together)
  - Drop OWN_CAR_AGE (66% null)
  - Handle DAYS_EMPLOYED sentinel: 365243 means "unemployed" — replace with
    NaN and add binary flag IS_UNEMPLOYED
  - Impute EXT_SOURCE_1/2/3 and credit bureau request cols with median
  - Impute OCCUPATION_TYPE and NAME_TYPE_SUITE with 'Unknown'
  - Log1p transform heavily right-skewed amount columns
  - Cast object columns as categorical
  - Target: TARGET (binary 0/1, 91.9% / 8.1% imbalance — stratified split)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class HomeCreditLoader(BaseLoader):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ds_config = config["datasets"]["home_credit"]
        self.global_config = config["global"]

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        path = self.ds_config["raw_file"]
        logger.info(f"Loading Home Credit from {path}")

        df = pd.read_csv(path)
        logger.info(f"Raw shape: {df.shape}")

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Step 2 — Preprocess
    # ------------------------------------------------------------------

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load() before preprocess()")

        df = self.df.copy()

        # 1. Drop configured columns (identifier + property block + OWN_CAR_AGE)
        drop_cols = [c for c in self.ds_config["drop_cols"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        logger.info(f"Dropped {len(drop_cols)} columns")

        # 2. Handle DAYS_EMPLOYED sentinel
        sentinel = self.ds_config.get("days_employed_sentinel", 365243)
        if "DAYS_EMPLOYED" in df.columns:
            df["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] == sentinel).astype(int)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(sentinel, np.nan)
            # Impute DAYS_EMPLOYED NaNs with median of the non-sentinel values
            median_employed = df["DAYS_EMPLOYED"].median()
            df["DAYS_EMPLOYED"].fillna(median_employed, inplace=True)
            logger.info(
                f"DAYS_EMPLOYED: replaced sentinel {sentinel} with NaN, "
                f"added IS_UNEMPLOYED flag ({df['IS_UNEMPLOYED'].sum():,} unemployed), "
                f"imputed with median={median_employed:.0f}"
            )

        # 3. Median impute numeric columns with moderate nulls
        median_cols = [c for c in self.ds_config.get("median_impute_cols", []) if c in df.columns]
        for col in median_cols:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        logger.info(f"Median-imputed {len(median_cols)} columns")

        # 4. Categorical impute — fill with 'Unknown' rather than mode
        #    to preserve the fact that missingness is informative
        mode_cols = [c for c in self.ds_config.get("mode_impute_cols", []) if c in df.columns]
        for col in mode_cols:
            df[col].fillna("Unknown", inplace=True)
        logger.info(f"Filled {len(mode_cols)} categorical columns with 'Unknown'")

        # 5. Log1p transform right-skewed amount columns
        log_cols = [c for c in self.ds_config.get("log_transform_cols", []) if c in df.columns]
        for col in log_cols:
            df[col] = np.log1p(df[col])
        logger.info(f"Log1p transformed: {log_cols}")

        # 6. Cast object columns as categorical
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("category")

        logger.info(
            f"Preprocessed shape: {df.shape} | "
            f"dtypes — numeric: {len(df.select_dtypes(include='number').columns)}, "
            f"categorical: {len(df.select_dtypes(include='category').columns)}"
        )

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Step 3 — Split (delegates to base class)
    # ------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience method: load → preprocess → split."""
        self.load()
        df = self.preprocess()
        return self.split(
            df=df,
            target_col=self.ds_config["target_col"],
            processed_dir=self.ds_config["processed_dir"],
            test_size=self.global_config["test_size"],
            seed=self.global_config["seed"],
        )