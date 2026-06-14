"""
diabetes_loader.py — Loader for the Diabetes 130-US Hospitals dataset.

Preprocessing decisions (from EDA):
  - Replace '?' with NaN (dataset-specific missing marker)
  - Drop identifier columns: encounter_id, patient_nbr
  - Drop high-null columns (>40%): weight, max_glu_serum, A1Cresult,
    payer_code, medical_specialty
  - Drop constant columns (zero variance): examide, citoglipton
  - Impute low-null categoricals with mode: race, diag_1, diag_2, diag_3
  - Group ICD-9 codes to 3-character prefixes: diag_1, diag_2, diag_3
  - Cast drug dosage columns as explicit categorical dtype
  - Cast remaining object columns as categorical
  - Target: readmitted — keep as-is (3-class: NO, >30, <30)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.ingestion.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class DiabetesLoader(BaseLoader):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ds_config = config["datasets"]["diabetes_130us"]
        self.global_config = config["global"]

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        path = self.ds_config["raw_file"]
        logger.info(f"Loading Diabetes 130-US from {path}")

        df = pd.read_csv(path)
        logger.info(f"Raw shape: {df.shape}")

        # Replace dataset-specific missing value markers with NaN
        for marker in self.ds_config.get("missing_values", []):
            df.replace(marker, np.nan, inplace=True)
        logger.info("Replaced missing value markers with NaN")

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Step 2 — Preprocess
    # ------------------------------------------------------------------

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Call load() before preprocess()")

        df = self.df.copy()

        # 1. Drop configured columns (identifiers, high-null, constants)
        drop_cols = [c for c in self.ds_config["drop_cols"] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)
        logger.info(f"Dropped {len(drop_cols)} columns: {drop_cols}")

        # 2. Group ICD-9 codes to 3-character prefix
        #    e.g. '250.83' → '250', 'V27' → 'V27', 'E11' → 'E11'
        icd_cols = [c for c in self.ds_config.get("icd_cols", []) if c in df.columns]
        for col in icd_cols:
            df[col] = df[col].apply(self._group_icd_code)
        logger.info(f"Grouped ICD-9 codes in: {icd_cols}")

        # 3. Impute low-null columns
        impute_map = self.ds_config.get("high_null_impute", {})
        for col, strategy in impute_map.items():
            if col not in df.columns:
                continue
            if strategy == "mode":
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                logger.info(f"Imputed '{col}' with mode='{fill_val}'")

        # 4. Cast drug/ordinal columns as categorical dtype
        ordinal_cols = [c for c in self.ds_config.get("ordinal_cols", []) if c in df.columns]
        for col in ordinal_cols:
            df[col] = df[col].astype("category")

        # 5. Cast remaining object columns as categorical
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_icd_code(code: str | float) -> str | float:
        """
        Truncate ICD-9 code to 3-character disease category prefix.
        Handles numeric codes (250.83 → '250'), V-codes (V27 → 'V27'),
        E-codes (E11 → 'E11'), and NaN passthrough.
        """
        if pd.isna(code):
            return code
        code = str(code).strip()
        # V-codes and E-codes: keep first 3 chars
        if code.startswith(("V", "E")):
            return code[:3]
        # Numeric codes: take digits before the decimal point, max 3 chars
        return code.split(".")[0][:3]