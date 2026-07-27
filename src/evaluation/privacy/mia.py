"""
mia.py — Membership Inference Attack (MIA) evaluator.

Attack setup:
  - Real training records are labelled as members (1)
  - Real test records are labelled as non-members (0)
  - A shadow XGBoost model is trained to distinguish them
    using the synthetic data as its knowledge source

The attacker's feature vector for each real record is its
distance to the nearest synthetic record — the intuition being
that if a real record is very close to a synthetic record,
it was likely in the training set (memorisation).

MIA AUC close to 0.5 → random guessing → private
MIA AUC close to 1.0 → attacker succeeds → privacy leak
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MIAEvaluator:

    def __init__(self, config: dict):
        self.config = config
        self.mia_config = config["evaluation"]["privacy"]["mia"]
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
        Run MIA and return AUC score.

        MIA AUC interpretation:
          0.5 → perfectly private (attacker can't do better than random)
          1.0 → completely compromised (attacker perfectly identifies members)

        Returns a dict with mia_auc and privacy_score (1 - 2*|auc - 0.5|)
        where privacy_score=1.0 means perfect privacy, 0.0 means no privacy.
        """
        logger.info(f"[{dataset_name}] MIA evaluation — {label}")

        attack_samples = self.mia_config["attack_samples"]

        # Sample equal numbers of members and non-members
        n_each = min(attack_samples // 2, len(real_train_df), len(real_test_df))

        members     = real_train_df.sample(n=n_each, random_state=self.seed)
        non_members = real_test_df.sample(n=n_each, random_state=self.seed)

        # Labels: 1 = member (was in training), 0 = non-member
        member_labels     = np.ones(n_each)
        non_member_labels = np.zeros(n_each)

        # Attacker feature: distance to nearest synthetic record
        # The intuition: memorised records will be closer to their
        # synthetic counterparts than non-training records
        synth_encoded  = self._encode_numeric(synthetic_df, target_col)
        member_encoded = self._encode_numeric(members, target_col)
        non_member_enc = self._encode_numeric(non_members, target_col)

        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
        nn.fit(synth_encoded)

        member_dists,     _ = nn.kneighbors(member_encoded)
        non_member_dists, _ = nn.kneighbors(non_member_enc)

        # Build attacker's dataset
        X_attack = np.concatenate([member_dists, non_member_dists], axis=0)
        y_attack = np.concatenate([member_labels, non_member_labels])

        # Train shadow attack model
        attack_model = XGBClassifier(
            n_estimators=self.mia_config["n_estimators"],
            max_depth=4,
            random_state=self.seed,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
        attack_model.fit(X_attack, y_attack)
        y_proba = attack_model.predict_proba(X_attack)[:, 1]

        mia_auc = float(roc_auc_score(y_attack, y_proba))

        # Privacy score: 1.0 = perfect privacy, 0.0 = completely compromised
        privacy_score = float(1 - 2 * abs(mia_auc - 0.5))

        results = {
            "dataset":       dataset_name,
            "label":         label,
            "mia_auc":       mia_auc,
            "privacy_score": privacy_score,
            "n_members":     n_each,
            "n_non_members": n_each,
        }

        logger.info(
            f"[{dataset_name}] {label} — "
            f"MIA AUC: {mia_auc:.4f}, "
            f"privacy_score: {privacy_score:.4f}"
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_numeric(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        """
        Encode all columns to numeric for distance computation.
        Categoricals are label-encoded, numerics kept as-is.
        Target column is excluded.
        """
        df = df.copy()
        feature_cols = [c for c in df.columns if c != target_col]

        for col in feature_cols:
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                df[col] = df[col].astype(str)
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col])

        return df[feature_cols].fillna(0).values.astype(float)