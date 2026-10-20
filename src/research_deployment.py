"""Load the frozen research detector for the interactive application.

The benchmark artifacts used to be isolated under ``research_results`` while
the dashboard continued loading the legacy ``models/*.pkl`` pair. This module
is the explicit deployment bridge: artifact, preprocessing and threshold all
come from the same benchmark run.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from .deployment_validator import DeploymentValidationError, resolve_run_pointer, validate_deployment_run
except ImportError:  # pragma: no cover
    from deployment_validator import DeploymentValidationError, resolve_run_pointer, validate_deployment_run

try:
    from .research_data import build_research_document, normalize_research_text
except ImportError:  # pragma: no cover
    from research_data import build_research_document, normalize_research_text


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "research_results"
DEPLOYED_MODEL = "logistic_regression"


def latest_research_run() -> Path:
    run_dir, _pointer = resolve_run_pointer(RESULTS_ROOT / "latest_benchmark.json", RESULTS_ROOT)
    return run_dir


def deployment_signature() -> tuple:
    """Return mtimes that invalidate Streamlit's cached model resource."""
    try:
        run_dir = latest_research_run()
    except (OSError, ValueError, KeyError, json.JSONDecodeError, DeploymentValidationError):
        return ()
    paths = (
        RESULTS_ROOT / "latest_benchmark.json",
        run_dir / "benchmark_summary.json",
        run_dir / "benchmark_metrics.csv",
        run_dir / "tfidf_vectorizer.joblib",
        run_dir / f"{DEPLOYED_MODEL}.joblib",
    )
    return tuple(path.stat().st_mtime_ns if path.exists() else None for path in paths)


def _metric(metrics: pd.DataFrame, dataset: str, variant: str) -> dict:
    selected = metrics.loc[
        metrics["model"].eq(DEPLOYED_MODEL)
        & metrics["dataset"].eq(dataset)
        & metrics["variant"].eq(variant)
    ]
    if selected.empty:
        return {}
    row = selected.iloc[0]
    return {
        "samples": int(row["samples"]),
        "average_precision": float(row["average_precision"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
    }


def load_research_detector(require_promoted: bool = False) -> tuple[object, object, dict]:
    """Load one frozen detector after validating its bundle.

    ``require_promoted=False`` exists only for reproducible evaluation of
    historical runs.  The interactive application uses strict mode.
    """
    run_dir = latest_research_run()
    validated = validate_deployment_run(
        run_dir,
        required_artifacts=("tfidf_vectorizer.joblib", f"{DEPLOYED_MODEL}.joblib"),
        require_promoted=require_promoted,
    )
    summary = validated["summary"]
    model_spec = summary["models"][DEPLOYED_MODEL]
    vectorizer = joblib.load(run_dir / "tfidf_vectorizer.joblib")
    model = joblib.load(run_dir / f"{DEPLOYED_MODEL}.joblib")
    metrics = pd.read_csv(run_dir / "benchmark_metrics.csv")
    threshold = float(model_spec["threshold"])
    manifest = {
        "mode": "research",
        "run_id": summary["run_id"],
        "model": DEPLOYED_MODEL,
        "threshold": threshold,
        "warning_threshold": threshold * 0.6,
        "preprocessing": str(summary.get("preprocessing", "normalize_research_text")),
        "task": summary["task"],
        "selection_reason": (
            "The frozen research Logistic Regression is deployed because the current UI provides faithful "
            "word-level coefficients. RF/XGBoost/SVM remain benchmark comparators, not silently mixed into inference."
        ),
        "metrics": {
            # New promoted runs report the independent locked test.  The
            # fallback preserves read-only access to historical runs so they
            # can be audited without being promoted.
            "locked_internal_test": _metric(metrics, "locked_internal_test", "all")
            or _metric(metrics, "internal_calibration", "all"),
            "spaphish_v5": _metric(metrics, "spaphish_v5", "unique_template"),
            "uot_validation_2024": _metric(metrics, "uot_validation_2024", "unique_template"),
        },
        "run_dir": str(run_dir),
        "deployment_validation": {
            "strict": bool(require_promoted),
            "checked_hashes": validated["checked_hashes"],
        },
    }
    return vectorizer, model, manifest


def prepare_research_document(text: str, preprocessing: str | None = None) -> str:
    """Use the exact preprocessing contract of the active frozen bundle.

    Historical bundles deliberately keep their legacy normalizer.  New
    bilingual-v3 bundles receive their matching enriched representation, so a
    dashboard restart cannot accidentally feed a new model an old document.
    """
    contract = preprocessing
    if contract is None:
        try:
            summary = json.loads((latest_research_run() / "benchmark_summary.json").read_text(encoding="utf-8"))
            contract = str(summary.get("preprocessing", "normalize_research_text"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, DeploymentValidationError):
            contract = "normalize_research_text"
    if contract in {
        "build_research_document_bilingual_v3",
        "build_research_document_bilingual_v4",
        "build_research_document_bilingual_v5",
        "build_research_document_bilingual_v6",
        "build_research_document_bilingual_v7",
    }:
        return build_research_document(str(text or ""))
    return normalize_research_text(str(text or ""))
