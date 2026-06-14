"""
acs_income_loader.py — Loader for the ACS Income dataset (via folktables).

Preprocessing decisions (from EDA):
  - No nulls, no duplicates — cleanest of the three datasets
  - Cast integer-coded census columns as categorical dtype:
    COW, SCHL, MAR, RELP, SEX, RAC1P, POBP
  - Group OCCP (529 unique occupation codes) to 23 SOC major groups
    using first-2-digit mapping — reduces cardinality to manageable level
  - AGEP and WKHP stay as continuous numeric
  - Target: PINCP (bool) — cast to int (1/0)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ingestion.base_loader import BaseLoader

logger = logging.getLogger(__name__)

# SOC major group mapping — first 2 digits of OCCP code → group label
# Based on ACS PUMS occupation code documentation
SOC_MAJOR_GROUPS = {
    (0,    99):  "MGR",   # Management
    (100,  199): "BUS",   # Business & Financial Operations
    (200,  299): "CMM",   # Computer & Mathematical
    (300,  399): "ENG",   # Architecture & Engineering
    (400,  499): "SCI",   # Life, Physical & Social Science
    (500,  599): "CMS",   # Community & Social Services
    (600,  699): "LGL",   # Legal
    (700,  749): "EDU",   # Educational Instruction
    (750,  799): "ENT",   # Arts, Design, Entertainment
    (800,  899): "MED",   # Healthcare Practitioners
    (900,  999): "HLS",   # Healthcare Support
    (1000, 1099): "PRT",  # Protective Service
    (1100, 1199): "EAT",  # Food Preparation
    (1200, 1299): "CLN",  # Building & Grounds Cleaning
    (1300, 1499): "PRS",  # Personal Care & Service
    (1500, 1599): "SAL",  # Sales
    (1600, 1799): "OFF",  # Office & Administrative Support
    (1800, 1999): "FFF",  # Farming, Fishing & Forestry
    (2000, 2099): "CON",  # Construction & Extraction
    (2100, 2199): "EXT",  # Extraction Workers
    (2200, 2599): "RPR",  # Installation, Maintenance & Repair
    (2600, 2999): "PRD",  # Production
    (3000, 3999): "TRN",  # Transportation & Material Moving
    (4000, 9999): "MIL",  # Military & Other
}


def _map_occp_to_soc(code: float) -> str:
    """Map a single OCCP code to its SOC major group label."""
    if pd.isna(code):
        return "UNK"
    code_int = int(code)
    for (low, high), label in SOC_MAJOR_GROUPS.items():
        if low <= code_int <= high:
            return label
    return "OTH"


class ACSIncomeLoader(BaseLoader):

    def __init__(self, config: dict):
        super().__init__(config)
        self.ds_config = config["datasets"]["acs_income"]
        self.global_config = config["global"]

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        path = self.ds_config["raw_file"]
        logger.info(f"Loading ACS Income from {path}")

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

        # 1. Cast target from bool to int
        target_col = self.ds_config["target_col"]
        df[target_col] = df[target_col].astype(int)
        logger.info(f"Cast target '{target_col}' from bool to int")

        # 2. Group OCCP (529 codes) → 23 SOC major groups
        occp_col = self.ds_config.get("occp_col", "OCCP")
        if occp_col in df.columns:
            df[occp_col] = df[occp_col].apply(_map_occp_to_soc)
            logger.info(
                f"Grouped '{occp_col}' to SOC major groups: "
                f"{df[occp_col].nunique()} unique groups"
            )

        # 3. Cast integer-coded census columns as categorical
        cat_cols = [c for c in self.ds_config.get("categorical_cols", []) if c in df.columns]
        for col in cat_cols:
            df[col] = df[col].astype("category")
        logger.info(f"Cast {len(cat_cols)} census code columns to categorical: {cat_cols}")

        # 4. Cast OCCP to categorical now that it's been grouped
        if occp_col in df.columns:
            df[occp_col] = df[occp_col].astype("category")

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