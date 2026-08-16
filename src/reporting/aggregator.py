"""
aggregator.py — Layer 4 (Reporting), consolidation step.

Reads outputs/results/full_benchmark.csv (produced by the evaluation
notebooks) and builds the master comparison matrix described in
project_context.md Section 4, Layer 4: rows are generator-epsilon
combinations, columns are utility + privacy metrics, one matrix per
dataset plus a combined cross-dataset view.

SCOPE (as of the two-generator, two-dataset study decision):
  - Generators: ctgan, tvae only. copulagan is excluded — it showed an
    unresolved anomaly (DCR blowing up ~6x under DP and staying flat
    across epsilon, unlike ctgan/tvae) that wasn't worth chasing further
    for this study. Its rows are not deleted from full_benchmark.csv,
    just filtered out here, so they remain available for an appendix note.
  - Datasets: diabetes_130us, acs_income only. home_credit is excluded —
    DP generation timed out against Kaggle's 12-hour session limit before
    a full combo set could be produced. Its nodp-only rows are filtered
    out here for the same reason (an incomplete row set would make
    cross-dataset comparisons misleading).

This scoping lives in SCOPED_GENERATORS / SCOPED_DATASETS below — change
those two lists to widen the study again later without touching any
other logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCOPED_GENERATORS = ["ctgan", "tvae"]
SCOPED_DATASETS = ["diabetes_130us", "acs_income"]

# Column groupings used throughout — kept in one place so visualiser.py
# and any future reporting code stay in sync with what's actually in
# full_benchmark.csv.
UTILITY_METRICS = ["utility_ratio", "tstr_auc", "trtr_auc", "tstr_f1",
                    "tstr_accuracy", "stat_overall", "wasserstein_mean",
                    "corr_diff", "cat_similarity"]
PRIVACY_METRICS = ["mia_auc", "mia_privacy_score", "dcr_mean", "dcr_median",
                    "nndr_mean", "nndr_median", "dist_privacy_score"]

# Primary metrics for the Pareto frontier / headline comparisons —
# matches config['reporting']['pareto'] (utility: roc_auc-family, privacy:
# mia_auc, lower/closer-to-0.5 is better).
PRIMARY_UTILITY_METRIC = "utility_ratio"
PRIMARY_PRIVACY_METRIC = "mia_auc"


class BenchmarkAggregator:
    """Consolidates full_benchmark.csv into report-ready comparison tables."""

    def __init__(self, config: dict):
        self.config = config
        self.reporting_cfg = config["reporting"]

    # ------------------------------------------------------------------
    # Loading + scoping
    # ------------------------------------------------------------------

    def load(self, path: str | Path | None = None) -> pd.DataFrame:
        """
        Load full_benchmark.csv and apply the study's generator/dataset
        scope. Does NOT mutate the underlying CSV — filtering happens
        in-memory so the excluded rows (copulagan, home_credit) stay on
        disk for the appendix.
        """
        path = Path(path) if path else Path(self.reporting_cfg["output_file"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the evaluation notebook(s) first."
            )

        df = pd.read_csv(path)
        n_total = len(df)

        df = df[
            df["generator"].isin(SCOPED_GENERATORS)
            & df["dataset"].isin(SCOPED_DATASETS)
        ].copy()

        logger.info(
            f"Loaded {n_total} rows from {path}, "
            f"{len(df)} in scope (generators={SCOPED_GENERATORS}, "
            f"datasets={SCOPED_DATASETS})."
        )

        missing = self._check_completeness(df)
        if missing:
            logger.warning(f"Missing expected combos: {missing}")

        # Consistent epsilon ordering for every downstream table/plot:
        # nodp first (baseline), then decreasing epsilon (increasing
        # privacy strength) — 10, 5, 1.
        df["epsilon"] = pd.Categorical(
            df["epsilon"].astype(str),
            categories=["nodp", "10", "5", "1"],
            ordered=True,
        )
        return df.sort_values(["dataset", "generator", "epsilon"]).reset_index(drop=True)

    def _check_completeness(self, df: pd.DataFrame) -> list[tuple[str, str, str]]:
        """Returns any (dataset, generator, epsilon) combos missing from df."""
        expected_eps = ["nodp"] + [str(e) for e in self.config["differential_privacy"]["epsilons"]]
        expected = {
            (d, g, e)
            for d in SCOPED_DATASETS
            for g in SCOPED_GENERATORS
            for e in expected_eps
        }
        present = set(zip(df["dataset"], df["generator"], df["epsilon"].astype(str)))
        return sorted(expected - present)

    # ------------------------------------------------------------------
    # Master comparison matrix
    # ------------------------------------------------------------------

    def master_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The Layer 4 master comparison matrix: one row per
        (dataset, generator, epsilon), all utility + privacy columns.
        This is just `df` with columns ordered/selected consistently —
        kept as an explicit method since it's the artifact
        project_context.md Section 4 describes as the Layer 4 output.
        """
        cols = ["dataset", "generator", "epsilon", "label"] + UTILITY_METRICS + PRIVACY_METRICS
        cols = [c for c in cols if c in df.columns]
        return df[cols].reset_index(drop=True)

    def per_dataset_matrix(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        return self.master_matrix(df[df["dataset"] == dataset_name])

    # ------------------------------------------------------------------
    # Pareto frontier
    # ------------------------------------------------------------------

    def pareto_frontier(self, df: pd.DataFrame, dataset_name: str | None = None) -> pd.DataFrame:
        """
        Flags Pareto-optimal (generator, epsilon) combos: a combo is on
        the frontier if no other combo has BOTH higher utility_ratio AND
        privacy at least as good (mia_auc at least as close to 0.5).

        Returns the input rows with an added `is_pareto_optimal` bool column.
        Computed per-dataset by default (privacy/utility scales aren't
        necessarily comparable across datasets); pass dataset_name=None
        to compute over all scoped datasets combined instead.
        """
        subset = df if dataset_name is None else df[df["dataset"] == dataset_name]
        subset = subset.copy()

        # "Privacy goodness" — distance of MIA AUC from 0.5, inverted so
        # higher = more private, on the same "higher is better" convention
        # as utility_ratio.
        subset["_privacy_goodness"] = 1 - 2 * (subset[PRIMARY_PRIVACY_METRIC] - 0.5).abs()

        is_optimal = []
        records = subset[[PRIMARY_UTILITY_METRIC, "_privacy_goodness"]].to_numpy()
        for i, (u_i, p_i) in enumerate(records):
            dominated = False
            for j, (u_j, p_j) in enumerate(records):
                if j == i:
                    continue
                if u_j >= u_i and p_j >= p_i and (u_j > u_i or p_j > p_i):
                    dominated = True
                    break
            is_optimal.append(not dominated)

        subset["is_pareto_optimal"] = is_optimal
        return subset.drop(columns=["_privacy_goodness"])

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def epsilon_trend_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per (dataset, generator): does utility_ratio decrease and does
        mia_auc move toward 0.5 as epsilon decreases from 10 to 1?
        Reports the nodp->eps10->eps5->eps1 deltas and a simple
        monotonicity flag for each metric, so this can be scanned quickly
        rather than re-deriving it from the raw pivot each time.
        """
        rows = []
        for (dataset, generator), group in df.groupby(["dataset", "generator"], observed=True):
            g = group.set_index("epsilon").reindex(["nodp", "10", "5", "1"])
            utility_seq = g[PRIMARY_UTILITY_METRIC].tolist()
            mia_seq = g[PRIMARY_PRIVACY_METRIC].tolist()

            utility_monotonic_down = all(
                a >= b for a, b in zip(utility_seq, utility_seq[1:])
                if pd.notna(a) and pd.notna(b)
            )
            # "improves" for MIA AUC means moving toward 0.5 as epsilon shrinks
            mia_dist = [abs(x - 0.5) if pd.notna(x) else np.nan for x in mia_seq]
            mia_monotonic_improving = all(
                a >= b for a, b in zip(mia_dist, mia_dist[1:])
                if pd.notna(a) and pd.notna(b)
            )

            rows.append({
                "dataset": dataset,
                "generator": generator,
                "utility_nodp": utility_seq[0],
                "utility_eps1": utility_seq[3],
                "utility_delta": (utility_seq[3] - utility_seq[0]
                                   if pd.notna(utility_seq[0]) and pd.notna(utility_seq[3]) else np.nan),
                "utility_monotonic_decreasing": utility_monotonic_down,
                "mia_nodp": mia_seq[0],
                "mia_eps1": mia_seq[3],
                "mia_delta_toward_0.5": (mia_dist[0] - mia_dist[3]
                                          if pd.notna(mia_dist[0]) and pd.notna(mia_dist[3]) else np.nan),
                "mia_monotonic_improving": mia_monotonic_improving,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        out_dir = Path(self.reporting_cfg["output_file"]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {out_path} ({len(df)} rows)")
        return out_path