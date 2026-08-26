"""
replication_common.py — shared save/meta helper for the four new
replication-study dataset loaders (adult, churn_modelling, covertype, cardio).

Writes train.csv / test.csv / meta.json in the exact shape
notebooks/03_evaluation_debug.ipynb's load_dataset() and
05/06_dp_evaluation.ipynb already expect:

    meta.json:
      {
        "columns": {
          "<col_name>": {"type": "categorical" | "numeric"},
          ...
        },
        "target_col": "...",
        "source": "..."   # provenance note, not read by existing notebooks
      }

NOTE ON INTERFACE: this project's BaseLoader (per project_context.md,
"Every loader implements load() and split()") wasn't available in this
session, so this helper is written as a standalone function rather than
a BaseLoader subclass. If your actual base_loader.py has a different
save/meta convention, adapt save_processed() below to match it — the
important part is that the OUTPUT SHAPE (train.csv/test.csv/meta.json
with this categorical/numeric column typing) is what the rest of your
pipeline already consumes unmodified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def save_processed(
    df: pd.DataFrame,
    processed_dir: str | Path,
    target_col: str,
    categorical_cols: list[str],
    test_size: float,
    seed: int,
    source_note: str = "",
) -> None:
    """
    Splits df into train/test, writes both plus meta.json to processed_dir.

    Args:
      categorical_cols: columns to mark type="categorical" in meta.json
                         (everything else is marked "numeric"). Does NOT
                         include target_col by default — pass it explicitly
                         in categorical_cols if your target is categorical
                         (all four new datasets here have categorical/binary
                         targets, so callers do include target_col).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df[target_col] if df[target_col].nunique() < 20 else None,
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    meta = {
        "columns": {
            col: {"type": "categorical" if col in categorical_cols else "numeric"}
            for col in df.columns
        },
        "target_col": target_col,
        "source": source_note,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Saved {processed_dir}: {len(train_df)} train, {len(test_df)} test rows, "
        f"target={target_col}, {len(categorical_cols)} categorical cols"
    )