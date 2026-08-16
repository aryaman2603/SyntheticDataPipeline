"""
visualiser.py — Layer 4 (Reporting), plotting step.

Builds the plots described in project_context.md Section 5:
  outputs/plots/tradeoff_curves/       — utility & privacy vs epsilon
  outputs/plots/correlation_heatmaps/  — metric heatmap (see note below)
  outputs/plots/tstr_comparisons/      — TSTR utility bar comparisons

NOTE on correlation_heatmaps: full_benchmark.csv stores per-combo SCALAR
summaries (corr_diff, wasserstein_mean, etc.), not the full per-column
correlation matrices computed inside StatisticalEvaluator.evaluate() —
those live only in that method's return dict and aren't persisted. So
this module's "heatmap" output is a (generator x epsilon) x dataset
metric heatmap, not a feature-correlation heatmap. If you want true
per-column correlation heatmaps, that requires re-running
StatisticalEvaluator and saving its `numeric_stats`/correlation output
per combo — a separate, larger change to the eval notebooks, not done here.

All functions take a matplotlib Axes or Figure-producing pattern and
save to disk. Designed to be called from a notebook for inline display
AND to leave files behind under outputs/plots/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS_ORDER = ["nodp", "10", "5", "1"]
GENERATOR_COLORS = {"ctgan": "#4C72B0", "tvae": "#DD8452", "copulagan": "#55A868"}


class BenchmarkVisualiser:

    def __init__(self, config: dict):
        self.config = config
        self.plots_cfg = config["reporting"]["plots"]

    def _ensure_dirs(self):
        for key in ("tradeoff_curves", "correlation_heatmaps", "tstr_comparisons"):
            Path(self.plots_cfg[key]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Trade-off curves: metric vs epsilon, one line per generator
    # ------------------------------------------------------------------

    def plot_tradeoff_curves(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """
        Two-panel figure for one dataset: utility_ratio vs epsilon (left)
        and mia_auc vs epsilon (right), one line per generator. A
        horizontal reference line at mia_auc=0.5 marks "perfectly private".
        """
        self._ensure_dirs()
        subset = df[df["dataset"] == dataset_name]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

        for generator, group in subset.groupby("generator"):
            g = group.set_index("epsilon").reindex(EPS_ORDER)
            color = GENERATOR_COLORS.get(generator, None)
            axes[0].plot(EPS_ORDER, g["utility_ratio"], marker="o", label=generator, color=color)
            axes[1].plot(EPS_ORDER, g["mia_auc"], marker="o", label=generator, color=color)

        axes[0].set_title(f"{dataset_name} — Utility ratio vs ε")
        axes[0].set_xlabel("ε (nodp = no privacy)")
        axes[0].set_ylabel("utility_ratio (TSTR AUC / TRTR AUC)")
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="perfectly private")
        axes[1].set_title(f"{dataset_name} — MIA AUC vs ε")
        axes[1].set_xlabel("ε (nodp = no privacy)")
        axes[1].set_ylabel("MIA AUC (0.5 = private, 1.0 = compromised)")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.suptitle(f"Privacy–utility trade-off — {dataset_name}", fontsize=12)
        fig.tight_layout()

        out_path = Path(self.plots_cfg["tradeoff_curves"]) / f"{dataset_name}_tradeoff.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        return out_path

    def plot_tradeoff_curves_combined(self, df: pd.DataFrame) -> Path:
        """
        Same as plot_tradeoff_curves but both scoped datasets side by
        side in one figure (4 panels: utility x 2 datasets, MIA x 2
        datasets) — useful for a single "headline" figure in the report.
        """
        self._ensure_dirs()
        datasets = sorted(df["dataset"].unique())

        fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8), squeeze=False)

        for col, dataset_name in enumerate(datasets):
            subset = df[df["dataset"] == dataset_name]
            for generator, group in subset.groupby("generator"):
                g = group.set_index("epsilon").reindex(EPS_ORDER)
                color = GENERATOR_COLORS.get(generator, None)
                axes[0][col].plot(EPS_ORDER, g["utility_ratio"], marker="o", label=generator, color=color)
                axes[1][col].plot(EPS_ORDER, g["mia_auc"], marker="o", label=generator, color=color)

            axes[0][col].set_title(f"{dataset_name}")
            axes[0][col].set_ylabel("utility_ratio")
            axes[0][col].set_ylim(0, 1.05)
            axes[0][col].grid(alpha=0.3)
            axes[0][col].legend()

            axes[1][col].axhline(0.5, color="gray", linestyle="--", linewidth=1)
            axes[1][col].set_xlabel("ε (nodp = no privacy)")
            axes[1][col].set_ylabel("mia_auc")
            axes[1][col].set_ylim(0.4, 1.0)
            axes[1][col].grid(alpha=0.3)

        fig.suptitle("Privacy–utility trade-off — CTGAN vs TVAE", fontsize=13)
        fig.tight_layout()

        out_path = Path(self.plots_cfg["tradeoff_curves"]) / "combined_tradeoff.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Pareto scatter: utility vs privacy, frontier highlighted
    # ------------------------------------------------------------------

    def plot_pareto_scatter(self, df_with_pareto_flag: pd.DataFrame, dataset_name: str) -> Path:
        """
        Scatter of utility_ratio (x) vs MIA-based privacy goodness (y),
        one point per (generator, epsilon), Pareto-optimal points
        highlighted with a ring. Expects the `is_pareto_optimal` column
        from BenchmarkAggregator.pareto_frontier().
        """
        self._ensure_dirs()
        subset = df_with_pareto_flag[df_with_pareto_flag["dataset"] == dataset_name].copy()
        subset["privacy_goodness"] = 1 - 2 * (subset["mia_auc"] - 0.5).abs()

        fig, ax = plt.subplots(figsize=(6, 5.5))

        for generator, group in subset.groupby("generator"):
            color = GENERATOR_COLORS.get(generator, None)
            ax.scatter(group["utility_ratio"], group["privacy_goodness"],
                       label=generator, color=color, s=70, zorder=3)
            for _, row in group.iterrows():
                ax.annotate(str(row["epsilon"]), (row["utility_ratio"], row["privacy_goodness"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)

        optimal = subset[subset["is_pareto_optimal"]]
        ax.scatter(optimal["utility_ratio"], optimal["privacy_goodness"],
                   facecolors="none", edgecolors="black", s=160, linewidths=1.5,
                   label="Pareto-optimal", zorder=4)

        ax.set_xlabel("utility_ratio (higher = better)")
        ax.set_ylabel("privacy goodness = 1 - 2|MIA AUC - 0.5| (higher = more private)")
        ax.set_title(f"Privacy–utility Pareto frontier — {dataset_name}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_path = Path(self.plots_cfg["tstr_comparisons"]) / f"{dataset_name}_pareto.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Metric heatmap (see module docstring re: naming vs true corr heatmaps)
    # ------------------------------------------------------------------

    def plot_metric_heatmap(self, df: pd.DataFrame, metric: str = "utility_ratio") -> Path:
        """
        Heatmap of `metric` with rows = (generator, epsilon) combos,
        columns = dataset. Quick way to eyeball a metric across the
        whole scoped study in one image.
        """
        self._ensure_dirs()
        pivot = df.pivot_table(
            index=["generator", "epsilon"], columns="dataset", values=metric, observed=True
        )
        # Keep epsilon order sensible within each generator block
        pivot = pivot.reindex(
            pd.MultiIndex.from_product(
                [sorted(df["generator"].unique()), EPS_ORDER], names=["generator", "epsilon"]
            )
        ).dropna(how="all")

        fig, ax = plt.subplots(figsize=(1.8 * len(pivot.columns) + 2, 0.45 * len(pivot) + 2))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([f"{g} / ε={e}" for g, e in pivot.index])

        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color="white" if val < pivot.values[~np.isnan(pivot.values)].mean() else "black",
                            fontsize=8)

        ax.set_title(f"{metric} across scoped study")
        fig.colorbar(im, ax=ax, shrink=0.7, label=metric)
        fig.tight_layout()

        out_path = Path(self.plots_cfg["correlation_heatmaps"]) / f"{metric}_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # TSTR bar comparison
    # ------------------------------------------------------------------

    def plot_tstr_comparison(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """
        Grouped bar chart: TSTR AUC per generator per epsilon, with the
        TRTR (real-data ceiling) AUC drawn as a horizontal reference line
        per generator (they should be ~equal across epsilon since TRTR
        doesn't depend on the synthetic data).
        """
        self._ensure_dirs()
        subset = df[df["dataset"] == dataset_name]
        generators = sorted(subset["generator"].unique())

        x = np.arange(len(EPS_ORDER))
        width = 0.8 / len(generators)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, generator in enumerate(generators):
            g = subset[subset["generator"] == generator].set_index("epsilon").reindex(EPS_ORDER)
            color = GENERATOR_COLORS.get(generator, None)
            ax.bar(x + i * width, g["tstr_auc"], width, label=f"{generator} (TSTR)", color=color)
            trtr_mean = g["trtr_auc"].mean()
            ax.axhline(trtr_mean, color=color, linestyle=":", linewidth=1.2, alpha=0.7)

        ax.set_xticks(x + width * (len(generators) - 1) / 2)
        ax.set_xticklabels(EPS_ORDER)
        ax.set_xlabel("ε (nodp = no privacy)")
        ax.set_ylabel("AUC")
        ax.set_title(f"TSTR AUC vs ε — {dataset_name}\n(dotted lines = TRTR ceiling per generator)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()

        out_path = Path(self.plots_cfg["tstr_comparisons"]) / f"{dataset_name}_tstr_bars.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {out_path}")
        return out_path