"""Strict loader and inference facade for promoted hybrid-paper-v1 runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np

from .deployment_validator import DeploymentValidationError, resolve_run_pointer, validate_deployment_run
from .fusion_model import FALLBACK_COLUMNS, FULL_COLUMNS, build_fusion_features


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "research_results"


def hybrid_deployment_signature() -> tuple:
    """Return mtimes that invalidate the Streamlit hybrid bundle cache."""
    pointer = RESULTS_ROOT / "latest_hybrid.json"
    if not pointer.exists():
        return ()
    try:
        run_dir, _ = resolve_run_pointer(pointer, RESULTS_ROOT)
    except (OSError, ValueError, KeyError, DeploymentValidationError, json.JSONDecodeError):
        return (pointer.stat().st_mtime_ns,)
    paths = (
        pointer,
        run_dir / "hybrid_summary.json",
        run_dir / "artifact_manifest.json",
        run_dir / "tfidf_vectorizer.joblib",
        run_dir / "ml_model.joblib",
        run_dir / "fallback_fusion.joblib",
        run_dir / "full_fusion.joblib",
    )
    return tuple(path.stat().st_mtime_ns if path.exists() else None for path in paths)


@dataclass(frozen=True)
class HybridPrediction:
    ml_probability: float
    llm_probability: float | None
    final_probability: float
    mode: str
    threshold_review: float
    threshold_high: float
    state: str


@dataclass
class HybridDeploymentBundle:
    vectorizer: object
    ml_model: object
    fallback_fusion: object
    full_fusion: object
    summary: dict
    run_dir: Path

    def predict(
        self,
        *,
        raw_text: str,
        research_text: str,
        llm_probability: float | None,
        llm_confidence: float | None = None,
    ) -> HybridPrediction:
        ml_probability = float(
            self.ml_model.predict_proba(self.vectorizer.transform([research_text]))[0, 1]
        )
        features = build_fusion_features(
            [raw_text], [ml_probability], [llm_probability], [llm_confidence]
        )
        full_available = llm_probability is not None and self.full_fusion is not None
        if full_available:
            final_probability = float(
                self.full_fusion.predict_proba(features[list(FULL_COLUMNS)])[0, 1]
            )
            mode = "full_ml_matrix_llm"
            configuration = self.summary["configurations"]["Full_ML_Matrix_LLM"]
        else:
            final_probability = float(
                self.fallback_fusion.predict_proba(features[list(FALLBACK_COLUMNS)])[0, 1]
            )
            mode = "fallback_ml_matrix"
            configuration = self.summary["configurations"]["ML_plus_Matrix"]
        threshold_high = float(configuration["threshold"])
        # Review threshold is an explicit conservative operating band.  It is
        # never used for paper metrics, which continue to use threshold_high.
        threshold_review = float(np.clip(threshold_high * 0.6, 0.0, threshold_high))
        if final_probability >= threshold_high:
            state = "HIGH_RISK"
        elif final_probability >= threshold_review:
            state = "MANUAL_REVIEW"
        else:
            state = "LOW_RISK"
        return HybridPrediction(
            ml_probability,
            llm_probability,
            final_probability,
            mode,
            threshold_review,
            threshold_high,
            state,
        )


def load_hybrid_bundle(pointer_path: str | Path | None = None) -> HybridDeploymentBundle:
    pointer_path = Path(pointer_path) if pointer_path else RESULTS_ROOT / "latest_hybrid.json"
    run_dir, _pointer = resolve_run_pointer(pointer_path, RESULTS_ROOT)
    required = (
        "tfidf_vectorizer.joblib",
        "ml_model.joblib",
        "fallback_fusion.joblib",
        "full_fusion.joblib",
    )
    validated = validate_deployment_run(run_dir, required_artifacts=required, require_promoted=True)
    return HybridDeploymentBundle(
        vectorizer=joblib.load(run_dir / "tfidf_vectorizer.joblib"),
        ml_model=joblib.load(run_dir / "ml_model.joblib"),
        fallback_fusion=joblib.load(run_dir / "fallback_fusion.joblib"),
        full_fusion=joblib.load(run_dir / "full_fusion.joblib"),
        summary=validated["summary"],
        run_dir=run_dir,
    )
