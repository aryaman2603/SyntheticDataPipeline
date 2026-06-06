"""
run_pipeline.py — Single entry point for the full synthetic data pipeline.

Usage:
    python run_pipeline.py                        # run all stages
    python run_pipeline.py --stages ingest        # run a specific stage
    python run_pipeline.py --stages ingest gen    # run multiple stages
    python run_pipeline.py --datasets diabetes_130us acs_income  # subset of datasets

Stages:
    ingest   → Layer 1: data ingestion and preprocessing
    generate → Layer 2: synthetic data generation (non-DP + DP)
    evaluate → Layer 3: utility and privacy evaluation
    report   → Layer 4: benchmark aggregation and visualisation
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a"),
        ],
    )
    return logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def bootstrap_directories(config: dict, logger: logging.Logger) -> None:
    """Create all required output directories if they don't exist."""
    dirs = [
        config["paths"]["processed"],
        config["paths"]["synthetic"],
        config["paths"]["outputs"],
        config["paths"]["plots"],
        config["reporting"]["plots"]["tradeoff_curves"],
        config["reporting"]["plots"]["correlation_heatmaps"],
        config["reporting"]["plots"]["tstr_comparisons"],
    ]
    # Per-dataset processed and synthetic subdirs
    for dataset in config["datasets"].values():
        dirs.append(dataset["processed_dir"])
        dirs.append(dataset["synthetic_dir"])

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory structure verified ({len(dirs)} directories).")


# ---------------------------------------------------------------------------
# Stage 1 — Data ingestion and preprocessing
# ---------------------------------------------------------------------------

def run_ingest(config: dict, datasets: list[str], logger: logging.Logger) -> None:
    """
    Layer 1: Load raw CSVs, clean, encode, split into train/test, write meta.json.
    Placeholder — to be implemented in Stage 2.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — Data Ingestion & Preprocessing")
    logger.info("=" * 60)

    for name in datasets:
        logger.info(f"  [{name}] Loading and preprocessing ...")
        # TODO (Stage 2): instantiate loader, call load(), preprocess(), split()
        # from src.ingestion.diabetes_loader import DiabetesLoader
        # from src.ingestion.home_credit_loader import HomeCreditLoader
        # from src.ingestion.acs_income_loader import ACSIncomeLoader
        # loader = <loader_class>(config)
        # loader.load()
        # loader.split()
        logger.info(f"  [{name}] ✓ placeholder complete")

    logger.info("Stage 1 complete.\n")


# ---------------------------------------------------------------------------
# Stage 2 — Synthetic data generation
# ---------------------------------------------------------------------------

def run_generate(config: dict, datasets: list[str], logger: logging.Logger) -> None:
    """
    Layer 2: Fit CTGAN, TVAE, CopulaGAN (no-DP + DP at each epsilon),
    sample synthetic data, write labelled CSVs.
    Placeholder — to be implemented in Stages 3 and 5.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 — Synthetic Data Generation")
    logger.info("=" * 60)

    generators  = config["generators"]["list"]
    epsilons    = config["differential_privacy"]["epsilons"]

    for name in datasets:
        logger.info(f"  [{name}] Running generators ...")

        # Non-DP generators
        for gen in generators:
            logger.info(f"    [{name}] {gen} (no-DP) — placeholder")
            # TODO (Stage 3): instantiate generator, fit on train, sample, save CSV

        # DP generators
        for gen in generators:
            for eps in epsilons:
                logger.info(f"    [{name}] {gen} (ε={eps}) — placeholder")
                # TODO (Stage 5): wrap generator with dp_wrapper, fit, sample, save CSV

    logger.info("Stage 2 complete.\n")


# ---------------------------------------------------------------------------
# Stage 3 — Evaluation
# ---------------------------------------------------------------------------

def run_evaluate(config: dict, datasets: list[str], logger: logging.Logger) -> None:
    """
    Layer 3: Run utility evaluator (TSTR, stats, correlations) and
    privacy evaluator (MIA, DCR, NNDR) on every synthetic CSV.
    Placeholder — to be implemented in Stage 4.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3 — Evaluation (Utility + Privacy)")
    logger.info("=" * 60)

    for name in datasets:
        logger.info(f"  [{name}] Running utility evaluator ... placeholder")
        # TODO (Stage 4): from src.evaluation.utility.statistical import StatisticalEvaluator
        # TODO (Stage 4): from src.evaluation.utility.tstr import TSTREvaluator
        # TODO (Stage 4): from src.evaluation.privacy.mia import MIAEvaluator
        # TODO (Stage 4): from src.evaluation.privacy.distance_metrics import DistanceEvaluator

        logger.info(f"  [{name}] Running privacy evaluator ... placeholder")

    logger.info("Stage 3 complete.\n")


# ---------------------------------------------------------------------------
# Stage 4 — Reporting
# ---------------------------------------------------------------------------

def run_report(config: dict, logger: logging.Logger) -> None:
    """
    Layer 4: Aggregate all evaluation results into benchmark tables,
    compute Pareto frontier, generate plots.
    Placeholder — to be implemented in Stage 6.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4 — Benchmark Reporting")
    logger.info("=" * 60)

    logger.info("  Aggregating results ... placeholder")
    # TODO (Stage 6): from src.reporting.aggregator import Aggregator
    # TODO (Stage 6): from src.reporting.visualiser import Visualiser

    logger.info("  Generating plots ... placeholder")
    logger.info("Stage 4 complete.\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Synthetic Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["ingest", "generate", "evaluate", "report"],
        default=["ingest", "generate", "evaluate", "report"],
        help="Pipeline stages to run (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to process (default: all defined in config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config["global"]["log_level"])

    logger.info("Pipeline starting.")
    logger.info(f"  Config : {args.config}")
    logger.info(f"  Stages : {args.stages}")

    # Resolve dataset list
    all_datasets = list(config["datasets"].keys())
    datasets = args.datasets if args.datasets else all_datasets

    invalid = set(datasets) - set(all_datasets)
    if invalid:
        logger.error(f"Unknown dataset(s): {invalid}. Available: {all_datasets}")
        sys.exit(1)

    logger.info(f"  Datasets: {datasets}\n")

    # Bootstrap directory structure
    bootstrap_directories(config, logger)

    # Run requested stages
    start = time.time()

    stage_map = {
        "ingest":   lambda: run_ingest(config, datasets, logger),
        "generate": lambda: run_generate(config, datasets, logger),
        "evaluate": lambda: run_evaluate(config, datasets, logger),
        "report":   lambda: run_report(config, logger),
    }

    for stage in args.stages:
        stage_map[stage]()

    elapsed = time.time() - start
    logger.info(f"Pipeline finished in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()