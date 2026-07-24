"""
ctgan_generator.py — CTGAN generator wrapping SDV's CTGANSynthesizer.

Hyperparameters are pulled from config.yaml → generators.ctgan
"""

from __future__ import annotations

import logging

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from src.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class CTGANGenerator(BaseGenerator):

    def __init__(self, config: dict, dataset_name: str):
        super().__init__(config, dataset_name)
        self.gen_config = config["generators"]["ctgan"]
        self.seed = config["global"]["seed"]

    def fit(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        # SDV requires object dtype, not category
        train_df = self.prepare_dataframe(train_df)

        logger.info(
            f"[{self.dataset_name}] Fitting CTGAN — "
            f"epochs={self.gen_config['epochs']}, "
            f"batch_size={self.gen_config['batch_size']}"
        )
        self.model = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.gen_config["epochs"],
            batch_size=self.gen_config["batch_size"],
            generator_dim=tuple(self.gen_config["generator_dim"]),
            discriminator_dim=tuple(self.gen_config["discriminator_dim"]),
            enable_gpu=True,  # set to True on cloud GPU
            verbose=True,
        )
        self.model.fit(train_df)
        logger.info(f"[{self.dataset_name}] CTGAN fitting complete")

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before sample()")
        logger.info(f"[{self.dataset_name}] CTGAN sampling {n:,} rows")
        return self.model.sample(num_rows=n)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first")
        self.model.save(path)
        logger.info(f"[{self.dataset_name}] CTGAN model saved → {path}")

    def load(self, path: str) -> None:
        self.model = CTGANSynthesizer.load(path)
        logger.info(f"[{self.dataset_name}] CTGAN model loaded from {path}")