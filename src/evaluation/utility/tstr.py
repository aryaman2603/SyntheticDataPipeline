"""
tstr.py — Train on Synthetic, Test on Real (TSTR) evaluator.

Trains an XGBoost classifier on synthetic data, evaluates on the
real held-out test set. Also computes TRTR (Train on Real, Test on Real)
as a utility ceiling baseline.

Metrics: ROC-AUC, F1 (weighted), Accuracy
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class TSTREvaluator:

    def __init__(self, config: dict):
        self.config = config
        self.tstr_config = config["evaluation"]["utility"]["tstr"]

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
        Compute TSTR and TRTR scores.

        Returns a dict with roc_auc, f1_weighted, accuracy for both
        TSTR and TRTR, plus a utility_ratio (TSTR/TRTR) as the
        primary summary metric.
        """
        logger.info(f"[{dataset_name}] TSTR evaluation — {label}")

        # Encode categorical columns for XGBoost
        X_synth, y_synth, encoders = self._encode(synthetic_df, target_col)
        X_real_train, y_real_train, _   = self._encode(real_train_df, target_col, encoders)
        X_real_test,  y_real_test,  _   = self._encode(real_test_df,  target_col, encoders)

        n_classes = len(np.unique(y_real_train))

        # TSTR — train on synthetic, test on real
        tstr_scores = self._train_and_evaluate(
            X_train=X_synth,
            y_train=y_synth,
            X_test=X_real_test,
            y_test=y_real_test,
            n_classes=n_classes,
        )

        # TRTR — train on real, test on real (utility ceiling)
        trtr_scores = self._train_and_evaluate(
            X_train=X_real_train,
            y_train=y_real_train,
            X_test=X_real_test,
            y_test=y_real_test,
            n_classes=n_classes,
        )

        # Utility ratio — how close TSTR gets to TRTR (1.0 = perfect)
        utility_ratio = float(
            tstr_scores["roc_auc"] / trtr_scores["roc_auc"]
            if trtr_scores["roc_auc"] > 0 else 0.0
        )

        results = {
            "dataset":       dataset_name,
            "label":         label,
            "tstr":          tstr_scores,
            "trtr":          trtr_scores,
            "utility_ratio": utility_ratio,
        }

        logger.info(
            f"[{dataset_name}] {label} — "
            f"TSTR AUC: {tstr_scores['roc_auc']:.4f}, "
            f"TRTR AUC: {trtr_scores['roc_auc']:.4f}, "
            f"utility_ratio: {utility_ratio:.4f}"
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int,
    ) -> dict:

        model = XGBClassifier(
            n_estimators=self.tstr_config["n_estimators"],
            max_depth=self.tstr_config["max_depth"],
            learning_rate=self.tstr_config["learning_rate"],
            random_state=self.config["global"]["seed"],
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # ROC-AUC — handle binary and multiclass
        if n_classes == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_test, y_proba,
                multi_class="ovr",
                average="weighted",
            ))

        return {
            "roc_auc":    auc,
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "accuracy":   float(accuracy_score(y_test, y_pred)),
        }

    def _encode(
        self,
        df: pd.DataFrame,
        target_col: str,
        encoders: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Label-encode categorical columns and separate features from target.
        If encoders is provided, use them (for consistent encoding across
        train/test/synthetic). Otherwise fit new encoders.
        """
        df = df.copy()
        fit_new = encoders is None
        encoders = encoders or {}

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in cat_cols:
            cat_cols.remove(target_col)

        for col in cat_cols:
            df[col] = df[col].astype(str)
            if fit_new:
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col])
                encoders[col] = enc
            else:
                enc = encoders.get(col)
                if enc:
                    # Handle unseen categories gracefully
                    known = set(enc.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known else enc.classes_[0]
                    )
                    df[col] = enc.transform(df[col])

        # Encode target
        target_vals = df[target_col].astype(str)
        if fit_new:
            target_enc = LabelEncoder()
            y = target_enc.fit_transform(target_vals)
            encoders["__target__"] = target_enc
        else:
            target_enc = encoders.get("__target__", LabelEncoder())
            known = set(target_enc.classes_)
            target_vals = target_vals.apply(
                lambda x: x if x in known else target_enc.classes_[0]
            )
            y = target_enc.transform(target_vals)

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values.astype(float)

        return X, y, encoders