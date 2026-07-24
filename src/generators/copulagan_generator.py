"""
copulagan_generator.py — CopulaGAN generator wrapping SDV's CopulaGANSynthesizer.

Hyperparameters are pulled from config.yaml → generators.copulagan
"""

from __future__ import annotations

import logging

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer

from src.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class CopulaGANGenerator(BaseGenerator):

    def __init__(self, config: dict, dataset_name: str):
        super().__init__(config, dataset_name)
        self.gen_config = config["generators"]["copulagan"]
        self.seed = config["global"]["seed"]

    def fit(self, train_df: pd.DataFrame, metadata: Metadata) -> None:
        # SDV requires object dtype, not category
        train_df = self.prepare_dataframe(train_df)

        logger.info(
            f"[{self.dataset_name}] Fitting CopulaGAN — "
            f"epochs={self.gen_config['epochs']}, "
            f"batch_size={self.gen_config['batch_size']}"
        )
        self.model = CopulaGANSynthesizer(
            metadata=metadata,
            epochs=self.gen_config["epochs"],
            batch_size=self.gen_config["batch_size"],
            enable_gpu=True,  # set to True on cloud GPU
            verbose=True,
        )
        self.model.fit(train_df)
        logger.info(f"[{self.dataset_name}] CopulaGAN fitting complete")

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before sample()")
        logger.info(f"[{self.dataset_name}] CopulaGAN sampling {n:,} rows")
        return self.model.sample(num_rows=n)

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first")
        self.model.save(path)
        logger.info(f"[{self.dataset_name}] CopulaGAN model saved → {path}")

    def load(self, path: str) -> None:
        self.model = CopulaGANSynthesizer.load(path)
        logger.info(f"[{self.dataset_name}] CopulaGAN model loaded from {path}")