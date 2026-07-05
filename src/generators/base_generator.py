"""
base_generator.py — Abstract base class for all synthetic data generators.

Every generator must implement:
  - fit(train_df, metadata)  → trains the model on real data
  - sample(n)                → generates n synthetic rows
  - save(path)               → persists the trained model
  - load(path)               → restores a trained model from disk

The base class provides:
  - prepare_dataframe()      → handles category dtype for SDV compatibility
  - build_metadata()         → builds SDV Metadata from DataFrame dtypes
  - save_synthetic()         → writes synthetic CSV with standard naming
  - load_train_data()        → reads train.csv and restores categorical dtypes
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):

    def __init__(self, config: dict, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.ds_config = config["datasets"][dataset_name]
        self.model = None
        self._int_coded_cat_cols: list[str] = []

    # ------------------------------------------------------------------
    # Abstract interface — every generator must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        """Fit the generator on the real training data."""

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        """Sample n synthetic rows. Must be called after fit()."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a trained model from disk."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        SDV's synthesizers do not accept pandas 'category' dtype.

        Two cases for category columns:
          - String-valued (e.g. 'No/Steady/Up/Down') → cast to object.
          - Integer-coded (e.g. ACS census codes 1.0, 2.0) → cast to float.
            We then mark them as sdtype='categorical' in build_metadata()
            via self._int_coded_cat_cols. Casting to object would cause
            SDV's type detection to fail on numeric strings.
        """
        df = df.copy()
        self._int_coded_cat_cols = []

        for col in df.select_dtypes(include=["category"]).columns:
            sample_vals = df[col].dropna().unique()
            try:
                pd.to_numeric(sample_vals)
                # Numeric-coded categorical — cast to float, track for metadata override
                df[col] = pd.to_numeric(df[col].astype(str), errors="coerce")
                self._int_coded_cat_cols.append(col)
            except (ValueError, TypeError):
                # String-valued categorical — cast to object
                df[col] = df[col].astype(object)

        return df

    def build_metadata(self, train_df: pd.DataFrame) -> Metadata:
        """
        Build an SDV Metadata object from the DataFrame dtypes.
        Expects train_df to have already been passed through prepare_dataframe().

        object columns            → sdtype='categorical'
        numeric columns           → sdtype='numerical'
        integer-coded cat columns → sdtype='categorical' (override)
        """
        metadata = Metadata()
        metadata.detect_table_from_dataframe(
            table_name=self.dataset_name,
            data=train_df,
        )

        # Override: object columns → categorical
        for col in train_df.select_dtypes(include=["object"]).columns:
            metadata.update_column(
                table_name=self.dataset_name,
                column_name=col,
                sdtype="categorical",
            )

        # Override: numeric columns → numerical
        for col in train_df.select_dtypes(include=["number"]).columns:
            metadata.update_column(
                table_name=self.dataset_name,
                column_name=col,
                sdtype="numerical",
            )

        # Override: integer-coded categoricals → categorical despite numeric dtype
        for col in self._int_coded_cat_cols:
            if col in train_df.columns:
                metadata.update_column(
                    table_name=self.dataset_name,
                    column_name=col,
                    sdtype="categorical",
                )

        logger.info(
            f"[{self.dataset_name}] Metadata built — "
            f"{len(train_df.columns)} columns "
            f"({len(self._int_coded_cat_cols)} int-coded categoricals)"
        )
        return metadata

    def save_synthetic(self, synthetic_df: pd.DataFrame, label: str) -> Path:
        """
        Write synthetic CSV to data/synthetic/<dataset>/<label>.csv
        label examples: 'ctgan_nodp', 'tvae_eps1', 'copulagan_eps10'
        """
        out_dir = Path(self.ds_config["synthetic_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}.csv"
        synthetic_df.to_csv(out_path, index=False)
        logger.info(
            f"[{self.dataset_name}] Synthetic data saved → {out_path} "
            f"({len(synthetic_df):,} rows)"
        )
        return out_path

    def load_train_data(self) -> pd.DataFrame:
        """
        Load the preprocessed train.csv for this dataset.
        Restores categorical dtypes from meta.json since CSV
        round-trips lose dtype information.
        """
        train_path = Path(self.ds_config["processed_dir"]) / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(
                f"train.csv not found at {train_path}. "
                f"Run the ingest stage first."
            )
        df = pd.read_csv(train_path)

        # Restore categorical dtypes from meta.json
        meta_path = Path(self.ds_config["processed_dir"]) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for col, col_meta in meta.get("columns", {}).items():
                if col in df.columns and col_meta["type"] == "categorical":
                    df[col] = df[col].astype("category")

        logger.info(
            f"[{self.dataset_name}] Loaded train data: "
            f"{df.shape[0]:,} rows x {df.shape[1]} cols"
        )
        return df