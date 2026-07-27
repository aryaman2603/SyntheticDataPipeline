"""
distance_metrics.py — DCR (Distance to Closest Record) and
NNDR (Nearest Neighbour Distance Ratio) evaluators.

DCR:
  For each synthetic record, find its nearest real training record.
  Report the mean and median distance. Higher DCR = synthetic records
  are further from real records = less memorisation = better privacy.

NNDR:
  For each synthetic record, compute:
    NNDR = dist_to_nearest_train / dist_to_nearest_test

  NNDR close to 1.0 → synthetic record is equally close to train and test
                       → no evidence of memorisation → private
  NNDR << 1.0        → synthetic record is much closer to train than test
                       → evidence of memorisation → privacy leak

Both metrics sample a subset of rows for efficiency on large datasets.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DistanceMetricsEvaluator:

    def __init__(self, config: dict):
        self.config = config
        self.dcr_config  = config["evaluation"]["privacy"]["dcr"]
        self.nndr_config = config["evaluation"]["privacy"]["nndr"]
        self.seed = config["global"]["seed"]

    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        dataset_name: str,
        label: str,
    ) -> dict:
        """
        Compute DCR and NNDR for a synthetic dataset.

        Returns a dict with:
          - dcr_mean, dcr_median
          - nndr_mean, nndr_median
          - privacy_score: combined scalar (higher = more private)
        """
        logger.info(f"[{dataset_name}] Distance metrics evaluation — {label}")

        sample_size = self.dcr_config["sample_size"]

        # Sample for efficiency
        synth_sample = synthetic_df.sample(
            n=min(sample_size, len(synthetic_df)),
            random_state=self.seed,
        )
        train_sample = real_train_df.sample(
            n=min(sample_size, len(real_train_df)),
            random_state=self.seed,
        )
        test_sample = real_test_df.sample(
            n=min(sample_size, len(real_test_df)),
            random_state=self.seed,
        )

        # Encode to numeric
        synth_enc = self._encode_numeric(synth_sample, target_col)
        train_enc = self._encode_numeric(train_sample, target_col)
        test_enc  = self._encode_numeric(test_sample,  target_col)

        # ------------------------------------------------------------------
        # DCR — distance from each synthetic record to nearest train record
        # ------------------------------------------------------------------
        nn_train = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        nn_train.fit(train_enc)
        dcr_dists, _ = nn_train.kneighbors(synth_enc)
        dcr_dists = dcr_dists.flatten()

        dcr_mean   = float(np.mean(dcr_dists))
        dcr_median = float(np.median(dcr_dists))

        # ------------------------------------------------------------------
        # NNDR — ratio of dist-to-train vs dist-to-test
        # ------------------------------------------------------------------
        nn_test = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        nn_test.fit(test_enc)
        test_dists, _ = nn_test.kneighbors(synth_enc)
        test_dists = test_dists.flatten()

        # Avoid division by zero
        epsilon = 1e-10
        nndr = dcr_dists / (test_dists + epsilon)

        nndr_mean   = float(np.mean(nndr))
        nndr_median = float(np.median(nndr))

        # ------------------------------------------------------------------
        # Privacy score — combines DCR and NNDR
        # DCR: normalise to (0,1] via sigmoid-like transform
        # NNDR: closer to 1.0 is better, so score = 1 - |nndr_median - 1| clamped
        # ------------------------------------------------------------------
        dcr_score  = float(1 - np.exp(-dcr_mean))        # 0 if dcr→0, →1 as dcr grows
        nndr_score = float(max(0, 1 - abs(nndr_median - 1.0)))  # 1 if nndr=1, 0 if very different
        privacy_score = float(np.mean([dcr_score, nndr_score]))

        results = {
            "dataset":       dataset_name,
            "label":         label,
            "dcr_mean":      dcr_mean,
            "dcr_median":    dcr_median,
            "nndr_mean":     nndr_mean,
            "nndr_median":   nndr_median,
            "privacy_score": privacy_score,
        }

        logger.info(
            f"[{dataset_name}] {label} — "
            f"DCR mean: {dcr_mean:.4f}, "
            f"NNDR median: {nndr_median:.4f}, "
            f"privacy_score: {privacy_score:.4f}"
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_numeric(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        """Encode all columns to numeric for distance computation."""
        df = df.copy()
        feature_cols = [c for c in df.columns if c != target_col]

        for col in feature_cols:
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                df[col] = df[col].astype(str)
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col])

        return df[feature_cols].fillna(0).values.astype(float)