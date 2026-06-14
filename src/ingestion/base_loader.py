"""
base_loader.py — Abstract base class for all dataset loaders.

Every loader must implement:
  - load()   → reads raw CSV, applies missing value markers, returns raw DataFrame
  - preprocess() → applies all dataset-specific cleaning decisions
  - split()  → stratified train/test split, writes train.csv, test.csv, meta.json
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BaseLoader(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Abstract interface — every loader must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Read raw CSV and return a DataFrame with missing markers replaced."""

    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Apply all dataset-specific cleaning, dropping, and transformation."""

    # ------------------------------------------------------------------
    # Shared split logic — same for all datasets
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        target_col: str,
        processed_dir: str,
        test_size: float,
        seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified train/test split.
        Writes train.csv, test.csv, and meta.json to processed_dir.
        Returns (train_df, test_df).
        """
        processed_path = Path(processed_dir)
        processed_path.mkdir(parents=True, exist_ok=True)

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[target_col],
        )

        train_df.to_csv(processed_path / "train.csv", index=False)
        test_df.to_csv(processed_path / "test.csv", index=False)
        logger.info(
            f"Split complete — train: {len(train_df):,} rows, "
            f"test: {len(test_df):,} rows → {processed_dir}"
        )

        meta = self._build_meta(df, target_col)
        with open(processed_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"meta.json written → {processed_dir}")

        return train_df, test_df

    # ------------------------------------------------------------------
    # Meta builder — produces the column schema consumed by generators
    # ------------------------------------------------------------------

    def _build_meta(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Build meta.json schema.

        meta.json structure:
        {
          "target_col": "readmitted",
          "columns": {
            "col_name": {
              "type": "categorical" | "continuous",
              "cardinality": 4,          # categorical only
              "values": ["No","Steady"],  # categorical only
              "min": 0.0,                # continuous only
              "max": 99.0                # continuous only
            }
          }
        }
        """
        columns = {}
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                unique_vals = df[col].dropna().unique().tolist()
                columns[col] = {
                    "type": "categorical",
                    "cardinality": df[col].nunique(),
                    "values": sorted([str(v) for v in unique_vals]),
                }
            else:
                columns[col] = {
                    "type": "continuous",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }

        # Target column metadata
        target_meta = {
            "type": "categorical" if df[target_col].dtype == "object" else "continuous",
            "cardinality": df[target_col].nunique(),
            "values": sorted([str(v) for v in df[target_col].dropna().unique().tolist()]),
        }

        return {
            "target_col": target_col,
            "target": target_meta,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": columns,
        }