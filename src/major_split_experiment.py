"""MeAJOR split experiment for domain-shift remediation.

This experiment keeps the normal research train sources, splits MeAJOR into
train/calibration/holdout partitions by template group, and evaluates whether
including MeAJOR training examples improves holdout robustness without testing
on examples used for fitting.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .research_benchmark import (
        ALL_MODEL_NAMES,
        CHARACTER_FEATURE_TEXT_MAX_CHARS,
        SEED,
        _evaluate_variant,
        _fit_models,
        _json_default,
        _make_vectorizer,
        _split_train_calibration,
    )
    from .research_data import DEFAULT_REGISTRY, MODEL_DOCUMENT_MAX_CHARS, split_primary_corpora
    from .research_metrics import tune_threshold
except ImportError:  # pragma: no cover
    from research_benchmark import (
        ALL_MODEL_NAMES,
        CHARACTER_FEATURE_TEXT_MAX_CHARS,
        SEED,
        _evaluate_variant,
        _fit_models,
        _json_default,
        _make_vectorizer,
        _split_train_calibration,
    )
    from research_data import DEFAULT_REGISTRY, MODEL_DOCUMENT_MAX_CHARS, split_primary_corpora
    from research_metrics import tune_threshold


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research_results"
MAJOR_ID = "major_corpus_2026"


def _first_group_split(
    frame: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    left_index, right_index = next(
        splitter.split(frame, frame["binary_label"], groups=frame["template_group"])
    )
    return frame.iloc[left_index].reset_index(drop=True), frame.iloc[right_index].reset_index(drop=True)


def split_major(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_calibration, holdout = _first_group_split(frame, n_splits=5, random_state=SEED + 2)
    train, calibration = _first_group_split(train_calibration, n_splits=8, random_state=SEED + 3)
    return train, calibration, holdout


def split_historical(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/calibration/holdout partitions without template leakage.

    The previous experiment tuned the threshold on the complete historical
    calibration partition and therefore had no untouched historical holdout.
    This 80/10/10 split keeps a source-disjoint retention check alongside the
    independent MeAJOR holdout.
    """
    train, calibration_holdout = _split_train_calibration(frame)
    calibration, holdout = _first_group_split(
        calibration_holdout,
        n_splits=2,
        random_state=SEED + 1,
    )
    return train, calibration, holdout


def _group_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    return len(set(left["template_group"]).intersection(right["template_group"]))


def _partition_integrity(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, int]:
    integrity = {
        "train_calibration_group_overlap": _group_overlap(train, calibration),
        "train_holdout_group_overlap": _group_overlap(train, holdout),
        "calibration_holdout_group_overlap": _group_overlap(calibration, holdout),
    }
    if any(integrity.values()):
        raise ValueError(f"Template-group leakage detected: {integrity}")
    return integrity


def run_major_split_experiment(
    model_names: list[str],
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_root: str | Path = DEFAULT_OUTPUT,
    save_models: bool = True,
) -> dict:
    started_at = datetime.now(timezone.utc)
    corpus, external_sets, data_manifest = split_primary_corpora(registry_path, include_raw=False)
    if MAJOR_ID not in external_sets:
        raise ValueError(f"{MAJOR_ID} must be registered as an external dataset.")

    historical_train, historical_calibration, historical_holdout = split_historical(corpus)
    major_train, major_calibration, major_holdout = split_major(external_sets[MAJOR_ID])
    historical_integrity = _partition_integrity(
        historical_train,
        historical_calibration,
        historical_holdout,
    )
    major_integrity = _partition_integrity(major_train, major_calibration, major_holdout)

    combined_train = pd.concat([historical_train, major_train], ignore_index=True)
    combined_calibration = pd.concat([historical_calibration, major_calibration], ignore_index=True)
    evaluation_sets = {
        "historical_holdout": historical_holdout,
        "major_corpus_2026_holdout": major_holdout,
        **{name: frame for name, frame in external_sets.items() if name != MAJOR_ID},
    }

    vectorizer = _make_vectorizer()
    fit_started = time.perf_counter()
    train_matrix = vectorizer.fit_transform(combined_train["research_text"])
    vectorizer_seconds = time.perf_counter() - fit_started
    models, svd = _fit_models(model_names, train_matrix, combined_train["binary_label"].to_numpy())

    run_id = started_at.strftime("major-split-%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    all_metrics = []
    all_predictions = []
    calibration_matrix = vectorizer.transform(combined_calibration["research_text"])
    thresholds = {}
    for model in models:
        calibration_scores = model.predict_scores(calibration_matrix)
        threshold, calibration_f2 = tune_threshold(combined_calibration["binary_label"], calibration_scores, beta=2.0)
        thresholds[model.name] = {"threshold": threshold, "calibration_f2": calibration_f2}
        metrics, predictions = _evaluate_variant(
            model, vectorizer, combined_calibration, threshold, 0, "all", "combined_calibration"
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        for dataset_name, frame in evaluation_sets.items():
            for variant, variant_frame in (
                ("all", frame),
                ("unique_template", frame.drop_duplicates("template_group", keep="first")),
            ):
                metrics, predictions = _evaluate_variant(
                    model, vectorizer, variant_frame.reset_index(drop=True), threshold, 0, variant, dataset_name
                )
                all_metrics.append(metrics)
                all_predictions.append(predictions)

    pd.json_normalize(all_metrics, sep=".").to_csv(run_dir / "major_split_metrics.csv", index=False, encoding="utf-8")
    pd.concat(all_predictions, ignore_index=True).to_csv(run_dir / "major_split_predictions.csv", index=False, encoding="utf-8")

    model_manifest = {
        model.name: {
            "feature_space": model.feature_space,
            "notes": model.notes,
            "fit_seconds": model.fit_seconds,
            **thresholds[model.name],
        }
        for model in models
    }
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "task": (
            "MeAJOR split remediation experiment with independent historical and MeAJOR holdouts; "
            "no holdout examples are used for vectorizer fitting, model fitting, or threshold tuning."
        ),
        "threshold_policy": "Maximise F2 on historical calibration plus MeAJOR calibration split.",
        "data_manifest": data_manifest,
        "split": {
            "historical_train_rows": int(len(historical_train)),
            "historical_calibration_rows": int(len(historical_calibration)),
            "historical_holdout_rows": int(len(historical_holdout)),
            "major_train_rows": int(len(major_train)),
            "major_calibration_rows": int(len(major_calibration)),
            "major_holdout_rows": int(len(major_holdout)),
            "group_key": "normalized template hash",
        },
        "partition_integrity": {
            "historical": historical_integrity,
            "major_corpus_2026": major_integrity,
        },
        "vectorizer": {
            "name": "FeatureUnion(TF-IDF word + char)",
            "word_ngram_range": [1, 2],
            "character_ngram_range": [3, 5],
            "max_features": {"word": 30000, "char": 10000},
            "model_document_max_chars": MODEL_DOCUMENT_MAX_CHARS,
            "character_input_max_chars": CHARACTER_FEATURE_TEXT_MAX_CHARS,
            "fit_seconds": vectorizer_seconds,
        },
        "preprocessing": "build_research_document_bilingual_v6",
        "models": model_manifest,
        "metrics": all_metrics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (run_dir / "major_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    if save_models:
        joblib.dump(vectorizer, run_dir / "tfidf_vectorizer.joblib")
        if svd is not None:
            joblib.dump(svd, run_dir / "truncated_svd.joblib")
        for model in models:
            joblib.dump(model.estimator, run_dir / f"{model.name}.joblib")
    try:
        portable_run_dir = str(run_dir.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        portable_run_dir = str(run_dir.resolve())
    (Path(output_root) / "latest_major_split.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "path": portable_run_dir,
                "deployment_status": "research_only_not_promoted",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MeAJOR train/calibration/holdout split experiment.")
    parser.add_argument("--models", default="logistic_regression")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    if names == ["all"]:
        names = ALL_MODEL_NAMES
    summary = run_major_split_experiment(
        names,
        registry_path=args.registry,
        output_root=args.output,
        save_models=not args.no_save_models,
    )
    print(json.dumps({"run_id": summary["run_id"], "models": list(summary["models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
