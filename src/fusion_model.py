"""Interpretable learned fusion for ML, technical evidence, and Local LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .security_matrix import matrix_feature_names, technical_feature_vector


SEED = 20260715
MATRIX_COLUMNS = tuple(matrix_feature_names())
FALLBACK_COLUMNS = ("ml_probability", *MATRIX_COLUMNS, "matrix_coverage")
FULL_COLUMNS = (
    "ml_probability",
    *MATRIX_COLUMNS,
    "matrix_coverage",
    "llm_probability",
    "llm_confidence",
    "llm_available",
)


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return float(default)


def build_fusion_features(
    raw_texts: Iterable[object],
    ml_probabilities: Iterable[float],
    llm_probabilities: Iterable[float | None] | None = None,
    llm_confidences: Iterable[float | None] | None = None,
) -> pd.DataFrame:
    """Create one feature row per message without treating missing LLM as safe."""

    texts = [str(value or "") for value in raw_texts]
    ml_values = list(ml_probabilities)
    if len(texts) != len(ml_values):
        raise ValueError("raw_texts and ml_probabilities must have equal length")
    llm_values = list(llm_probabilities) if llm_probabilities is not None else [None] * len(texts)
    confidence_values = list(llm_confidences) if llm_confidences is not None else [None] * len(texts)
    if len(llm_values) != len(texts) or len(confidence_values) != len(texts):
        raise ValueError("LLM feature arrays must match raw_texts length")

    rows = []
    for raw_text, ml_value, llm_value, confidence_value in zip(
        texts, ml_values, llm_values, confidence_values
    ):
        technical = technical_feature_vector(raw_text)
        observed = [technical[name] for name in MATRIX_COLUMNS if name.endswith("_observed")]
        llm_available = llm_value is not None and not pd.isna(llm_value)
        row = {
            "ml_probability": _bounded(ml_value, 0.5),
            **technical,
            "matrix_coverage": float(np.mean(observed)) if observed else 0.0,
            # 0.5 is neutral and is always accompanied by llm_available=0.
            "llm_probability": _bounded(llm_value, 0.5) if llm_available else 0.5,
            "llm_confidence": _bounded(confidence_value, 0.0) if llm_available else 0.0,
            "llm_available": float(llm_available),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FULL_COLUMNS)


@dataclass
class HybridFusionModel:
    """Two learned fusion paths: full tri-modal and ML+matrix fallback."""

    full_model: Pipeline | None = None
    fallback_model: Pipeline | None = None
    full_training_rows: int = 0
    fallback_training_rows: int = 0

    @staticmethod
    def _new_model() -> Pipeline:
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=SEED,
                        solver="liblinear",
                    ),
                ),
            )
        )

    def fit(self, features: pd.DataFrame, labels) -> "HybridFusionModel":
        labels = np.asarray(labels, dtype=int)
        if len(features) != len(labels) or len(np.unique(labels)) < 2:
            raise ValueError("Fusion training requires aligned rows containing both labels")
        self.fallback_model = self._new_model().fit(features[list(FALLBACK_COLUMNS)], labels)
        self.fallback_training_rows = int(len(features))

        available = features["llm_available"].eq(1.0).to_numpy()
        if available.any() and len(np.unique(labels[available])) == 2:
            self.full_model = self._new_model().fit(features.loc[available, list(FULL_COLUMNS)], labels[available])
            self.full_training_rows = int(available.sum())
        else:
            self.full_model = None
            self.full_training_rows = 0
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.fallback_model is None:
            raise RuntimeError("Fusion model has not been fitted")
        fallback_scores = self.fallback_model.predict_proba(features[list(FALLBACK_COLUMNS)])[:, 1]
        scores = np.asarray(fallback_scores, dtype=float)
        if self.full_model is not None:
            available = features["llm_available"].eq(1.0).to_numpy()
            if available.any():
                scores[available] = self.full_model.predict_proba(
                    features.loc[available, list(FULL_COLUMNS)]
                )[:, 1]
        return np.column_stack((1.0 - scores, scores))

    def modes(self, features: pd.DataFrame) -> np.ndarray:
        full_available = self.full_model is not None
        return np.where(
            full_available & features["llm_available"].eq(1.0).to_numpy(),
            "full_ml_matrix_llm",
            "fallback_ml_matrix",
        )

    def coefficient_table(self, mode: str = "full") -> pd.DataFrame:
        model = self.full_model if mode == "full" else self.fallback_model
        columns = FULL_COLUMNS if mode == "full" else FALLBACK_COLUMNS
        if model is None:
            return pd.DataFrame(columns=("feature", "coefficient"))
        coefficients = model.named_steps["logistic"].coef_[0]
        return pd.DataFrame({"feature": columns, "coefficient": coefficients}).sort_values(
            "coefficient", ascending=False
        )
