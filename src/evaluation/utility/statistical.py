"""
statistical.py — Column-wise statistics and correlation comparison
between real and synthetic data.

Metrics computed:
  - Per-column mean, std, min, max for numeric columns
  - Per-column value distribution (normalized counts) for categorical columns
  - Pairwise Pearson correlation matrix for numeric columns
  - Column-wise statistical similarity score (1 - normalised mean absolute error)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


class StatisticalEvaluator:

    def __init__(self, config: dict):
        self.config = config

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        dataset_name: str,
        label: str,
    ) -> dict:
        """
        Compute statistical similarity between real and synthetic data.

        Returns a dict with:
          - numeric_stats: per-column mean/std/min/max comparison
          - wasserstein: per-column Wasserstein distance (numeric cols)
          - correlation_diff: mean absolute difference in correlation matrices
          - overall_score: single scalar summary (higher = more similar)
        """
        logger.info(f"[{dataset_name}] Statistical evaluation — {label}")

        results = {
            "dataset": dataset_name,
            "label": label,
        }

        num_cols = real_df.select_dtypes(include="number").columns.tolist()
        cat_cols = real_df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ------------------------------------------------------------------
        # 1. Numeric column statistics
        # ------------------------------------------------------------------
        numeric_stats = {}
        wasserstein_scores = {}

        for col in num_cols:
            r = real_df[col].dropna()
            s = synthetic_df[col].dropna()

            numeric_stats[col] = {
                "real_mean":  float(r.mean()),
                "synth_mean": float(s.mean()),
                "real_std":   float(r.std()),
                "synth_std":  float(s.std()),
                "real_min":   float(r.min()),
                "synth_min":  float(s.min()),
                "real_max":   float(r.max()),
                "synth_max":  float(s.max()),
            }

            # Wasserstein distance — measures distributional difference
            # Normalise by real std to make it scale-independent
            std = r.std()
            if std > 0:
                wasserstein_scores[col] = float(
                    wasserstein_distance(r, s) / std
                )
            else:
                wasserstein_scores[col] = 0.0

        results["numeric_stats"] = numeric_stats
        results["wasserstein_per_col"] = wasserstein_scores
        results["wasserstein_mean"] = float(np.mean(list(wasserstein_scores.values()))) if wasserstein_scores else 0.0

        # ------------------------------------------------------------------
        # 2. Categorical column distributions
        # ------------------------------------------------------------------
        cat_stats = {}
        cat_similarity_scores = []

        for col in cat_cols:
            real_dist  = real_df[col].value_counts(normalize=True)
            synth_dist = synthetic_df[col].value_counts(normalize=True)

            # Align on same categories
            all_cats = real_dist.index.union(synth_dist.index)
            real_aligned  = real_dist.reindex(all_cats, fill_value=0)
            synth_aligned = synth_dist.reindex(all_cats, fill_value=0)

            # Total variation distance — 0 means identical, 1 means completely different
            tvd = float(0.5 * np.abs(real_aligned - synth_aligned).sum())
            cat_similarity_scores.append(1 - tvd)

            cat_stats[col] = {
                "tvd": tvd,
                "real_top3":  real_dist.head(3).to_dict(),
                "synth_top3": synth_dist.head(3).to_dict(),
            }

        results["categorical_stats"] = cat_stats
        results["categorical_similarity_mean"] = float(np.mean(cat_similarity_scores)) if cat_similarity_scores else 1.0

        # ------------------------------------------------------------------
        # 3. Pairwise correlation matrix difference (numeric cols only)
        # ------------------------------------------------------------------
        if len(num_cols) >= 2:
            real_corr  = real_df[num_cols].corr()
            synth_corr = synthetic_df[num_cols].corr()
            corr_diff  = (real_corr - synth_corr).abs()
            # Mask diagonal (always 0)
            np.fill_diagonal(corr_diff.values, 0)
            mean_corr_diff = float(corr_diff.values.mean())
        else:
            mean_corr_diff = 0.0

        results["correlation_matrix_diff"] = mean_corr_diff

        # ------------------------------------------------------------------
        # 4. Overall statistical similarity score
        # ------------------------------------------------------------------
        # Combine: categorical similarity, inverse wasserstein, inverse corr diff
        # All components in [0, 1] range where higher = better
        wasserstein_score = float(1 / (1 + results["wasserstein_mean"]))  # maps to (0, 1]
        corr_score        = float(1 / (1 + mean_corr_diff))               # maps to (0, 1]
        cat_score         = results["categorical_similarity_mean"]

        results["overall_score"] = float(
            np.mean([wasserstein_score, corr_score, cat_score])
        )

        logger.info(
            f"[{dataset_name}] {label} — "
            f"wasserstein: {results['wasserstein_mean']:.4f}, "
            f"corr_diff: {mean_corr_diff:.4f}, "
            f"cat_sim: {cat_score:.4f}, "
            f"overall: {results['overall_score']:.4f}"
        )

        return results