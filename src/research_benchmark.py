"""Reproducible multi-model and cross-dataset phishing benchmark.

Usage examples:
    python -m src.research_benchmark --models logistic_regression,linear_svm,multinomial_nb
    python -m src.research_benchmark --models all --bootstrap 200

External sets are evaluation-only.  Thresholds are selected on the internal
calibration split and then frozen before any external metric is computed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion

try:
    from .classifiers import HybridPhishingClassifier
    from .research_data import DEFAULT_REGISTRY, MODEL_DOCUMENT_MAX_CHARS, split_primary_corpora
    from .research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold
    from .deployment_validator import sha256_file
    from .artifact_signing import require_signing_key, sign_manifest
except ImportError:  # pragma: no cover - direct script compatibility
    from classifiers import HybridPhishingClassifier
    from research_data import DEFAULT_REGISTRY, MODEL_DOCUMENT_MAX_CHARS, split_primary_corpora
    from research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold
    from deployment_validator import sha256_file
    from artifact_signing import require_signing_key, sign_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research_results"
SEED = 20260715
ALL_MODEL_NAMES = [
    "logistic_regression",
    "hybrid_logistic_nb",
    "linear_svm",
    "multinomial_nb",
    "random_forest",
    "xgboost",
]
MIN_SELECTION_RECALL = 0.90
MIN_SELECTION_PRECISION = 0.80
MIN_LANGUAGE_EVALUATION_SAMPLES = 100
MIN_LANGUAGE_CLASS_SAMPLES = 30
CHARACTER_FEATURE_TEXT_MAX_CHARS = 1_000


@dataclass
class FittedModel:
    name: str
    estimator: object
    predict_scores: Callable[[object], np.ndarray]
    feature_space: str
    notes: str
    fit_seconds: float


def _positive_probability(matrix: object, *, model: object) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1])


def _svm_margin_probability(matrix: object, *, model: object) -> np.ndarray:
    return np.asarray(expit(model.decision_function(matrix)))


def _reduced_positive_probability(matrix: object, *, model: object, reducer: object) -> np.ndarray:
    return np.asarray(model.predict_proba(reducer.transform(matrix))[:, 1])


def _split_train_calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, calibration_index = next(
        splitter.split(frame, frame["binary_label"], groups=frame["template_group"])
    )
    return frame.iloc[train_index].reset_index(drop=True), frame.iloc[calibration_index].reset_index(drop=True)


def _split_train_calibration_holdout(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create disjoint grouped fit, threshold-calibration and locked-test sets."""

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    development_index, holdout_index = next(
        outer.split(frame, frame["binary_label"], groups=frame["template_group"])
    )
    development = frame.iloc[development_index].reset_index(drop=True)
    holdout = frame.iloc[holdout_index].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    fit_index, calibration_index = next(
        inner.split(development, development["binary_label"], groups=development["template_group"])
    )
    return (
        development.iloc[fit_index].reset_index(drop=True),
        development.iloc[calibration_index].reset_index(drop=True),
        holdout,
    )


def _limit_training(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame
    positives = frame.loc[frame["binary_label"].eq(1)]
    negatives = frame.loc[frame["binary_label"].eq(0)]
    positive_target = min(len(positives), max(1, max_rows // 4))
    negative_target = max_rows - positive_target
    selected = pd.concat(
        [
            positives.sample(positive_target, random_state=SEED),
            negatives.sample(min(len(negatives), negative_target), random_state=SEED),
        ],
        ignore_index=True,
    )
    return selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def _character_feature_view(document: object) -> str:
    """Bound character features while retaining start, end and security tokens.

    The word branch already observes the full capped research document.  A
    separate bounded view prevents one very long MIME/newsletter body from
    dominating character n-gram fitting time and memory; the end is retained
    because ``build_research_document`` appends its structural security tokens
    there.
    """
    value = str(document or "")
    if len(value) <= CHARACTER_FEATURE_TEXT_MAX_CHARS:
        return value
    tail = min(350, CHARACTER_FEATURE_TEXT_MAX_CHARS // 3)
    head = CHARACTER_FEATURE_TEXT_MAX_CHARS - tail
    return value[:head] + " [TRUNCATED_FOR_CHARACTER_FEATURES] " + value[-tail:]


def _make_vectorizer() -> FeatureUnion:
    """Return the shared multilingual sparse representation.

    Word n-grams retain explainable phrases, while character n-grams make the
    classifier resilient to Vietnamese diacritics, no-space obfuscation and
    previously unseen spelling variants.  Both branches consume the same
    leakage-resistant document created by ``build_research_document``.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=10_000,
                    sublinear_tf=True,
                    preprocessor=_character_feature_view,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def _fit_models(names, train_matrix, train_labels, dense_train=None) -> tuple[list[FittedModel], object | None]:
    fitted: list[FittedModel] = []
    svd = None
    if any(name in {"random_forest", "xgboost"} for name in names):
        svd = TruncatedSVD(n_components=min(128, train_matrix.shape[1] - 1), random_state=SEED)
        dense_train = svd.fit_transform(train_matrix)

    for name in names:
        started = time.perf_counter()
        if name == "logistic_regression":
            estimator = LogisticRegression(
                C=4.0, class_weight="balanced", max_iter=1200, solver="liblinear", random_state=SEED
            ).fit(train_matrix, train_labels)
            predictor = partial(_positive_probability, model=estimator)
            feature_space = "TF-IDF word (1,2)-grams"
            notes = "Interpretable linear baseline; spam is a hard negative."
        elif name == "hybrid_logistic_nb":
            estimator = HybridPhishingClassifier(logistic_weight=0.90, nb_alpha=0.35).fit(
                train_matrix, train_labels
            )
            predictor = partial(_positive_probability, model=estimator)
            feature_space = "TF-IDF word (1,2)-grams"
            notes = "Production-compatible 90% logistic and 10% multinomial-NB probability ensemble."
        elif name == "linear_svm":
            estimator = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(train_matrix, train_labels)
            predictor = partial(_svm_margin_probability, model=estimator)
            feature_space = "TF-IDF word (1,2)-grams"
            notes = "Sigmoid of the SVM margin is rank-preserving but not probability-calibrated."
        elif name == "multinomial_nb":
            estimator = MultinomialNB(alpha=0.35).fit(train_matrix, train_labels)
            predictor = partial(_positive_probability, model=estimator)
            feature_space = "TF-IDF word (1,2)-grams"
            notes = "Classic sparse-text baseline; useful for showing whether linear discriminative models add value."
        elif name == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=240,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=1,
            ).fit(dense_train, train_labels)
            predictor = partial(_reduced_positive_probability, model=estimator, reducer=svd)
            feature_space = "128-dimensional TruncatedSVD of the shared TF-IDF space"
            notes = "Tree baseline uses the same text representation after fixed dimensionality reduction."
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as error:
                raise RuntimeError("xgboost requested but not installed; install requirements-research.txt") from error
            estimator = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=1,
            ).fit(dense_train, train_labels)
            predictor = partial(_reduced_positive_probability, model=estimator, reducer=svd)
            feature_space = "128-dimensional TruncatedSVD of the shared TF-IDF space"
            notes = "XGBoost comparison on the same leakage-resistant text split."
        else:
            raise ValueError(f"Unknown classical model: {name}")
        fitted.append(
            FittedModel(name, estimator, predictor, feature_space, notes, time.perf_counter() - started)
        )
    return fitted, svd


def _selection_cross_validation(
    frame: pd.DataFrame,
    model_names: list[str],
    folds: int = 5,
) -> tuple[list[dict], dict]:
    """Select a model without consulting any external evaluation set.

    Each outer validation fold receives a threshold tuned on a disjoint inner
    calibration split.  This keeps F2 model selection separate from the locked
    external corpora and prevents repeated templates crossing fold boundaries.
    """

    if folds < 2:
        raise ValueError("selection folds must be at least 2")
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED + 100)
    rows: list[dict] = []
    for fold, (outer_train_index, validation_index) in enumerate(
        outer.split(frame, frame["binary_label"], groups=frame["template_group"]),
        start=1,
    ):
        outer_train = frame.iloc[outer_train_index].reset_index(drop=True)
        validation = frame.iloc[validation_index].reset_index(drop=True)
        inner_splits = min(4, max(2, outer_train["template_group"].nunique()))
        inner = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=SEED + 100 + fold,
        )
        fit_index, calibration_index = next(
            inner.split(
                outer_train,
                outer_train["binary_label"],
                groups=outer_train["template_group"],
            )
        )
        fit_frame = outer_train.iloc[fit_index].reset_index(drop=True)
        calibration = outer_train.iloc[calibration_index].reset_index(drop=True)
        vectorizer = _make_vectorizer()
        fit_matrix = vectorizer.fit_transform(fit_frame["research_text"])
        calibration_matrix = vectorizer.transform(calibration["research_text"])
        validation_matrix = vectorizer.transform(validation["research_text"])
        models, _ = _fit_models(
            model_names,
            fit_matrix,
            fit_frame["binary_label"].to_numpy(),
        )
        for model in models:
            calibration_scores = model.predict_scores(calibration_matrix)
            threshold, calibration_f2 = tune_threshold(
                calibration["binary_label"], calibration_scores, beta=2.0
            )
            validation_scores = model.predict_scores(validation_matrix)
            metrics = classification_metrics(
                validation["binary_label"], validation_scores, threshold
            )
            rows.append(
                {
                    "fold": fold,
                    "model": model.name,
                    "fit_rows": int(len(fit_frame)),
                    "calibration_rows": int(len(calibration)),
                    "validation_rows": int(len(validation)),
                    "threshold": float(threshold),
                    "calibration_f2": float(calibration_f2),
                    **metrics,
                }
            )

    metrics_frame = pd.DataFrame(rows)
    summaries: list[dict] = []
    metric_names = (
        "f2",
        "recall",
        "precision",
        "average_precision",
        "specificity",
        "mcc",
    )
    for name, group in metrics_frame.groupby("model", sort=False):
        record: dict = {"model": name, "folds": int(len(group))}
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            record[f"mean_{metric}"] = float(values.mean()) if len(values) else None
            record[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        record["eligible"] = bool(
            record["mean_recall"] is not None
            and record["mean_precision"] is not None
            and record["mean_recall"] >= MIN_SELECTION_RECALL
            and record["mean_precision"] >= MIN_SELECTION_PRECISION
        )
        summaries.append(record)

    eligible = [record for record in summaries if record["eligible"]]
    pool = eligible or summaries
    winner = max(
        pool,
        key=lambda record: (
            record["mean_f2"] if record["mean_f2"] is not None else -1.0,
            record["mean_average_precision"] if record["mean_average_precision"] is not None else -1.0,
            record["mean_recall"] if record["mean_recall"] is not None else -1.0,
            record["mean_precision"] if record["mean_precision"] is not None else -1.0,
        ),
    )
    selection = {
        "selected_model": winner["model"],
        "primary_metric": "mean outer-fold F2 with inner-fold threshold tuning",
        "minimum_recall": MIN_SELECTION_RECALL,
        "minimum_precision": MIN_SELECTION_PRECISION,
        "fallback_used": not bool(eligible),
        "external_data_consulted": False,
        "group_key": "normalized template hash",
        "summaries": summaries,
    }
    return rows, selection


def _evaluate_variant(model, vectorizer, frame, threshold, bootstrap, variant, dataset_name):
    matrix = vectorizer.transform(frame["research_text"])
    started = time.perf_counter()
    scores = np.asarray(model.predict_scores(matrix), dtype=float)
    predict_seconds = time.perf_counter() - started
    metrics = classification_metrics(frame["binary_label"], scores, threshold)
    metrics["bootstrap_95_ci"] = grouped_bootstrap_ci(
        frame["binary_label"], scores, frame["template_group"], threshold, iterations=bootstrap, seed=SEED
    )
    metrics.update(
        {
            "dataset": dataset_name,
            "variant": variant,
            "model": model.name,
            "predict_seconds": predict_seconds,
            "milliseconds_per_email": 1000.0 * predict_seconds / max(1, len(frame)),
        }
    )
    predictions = pd.DataFrame(
        {
            "dataset": dataset_name,
            "variant": variant,
            "model": model.name,
            "message_id": frame["message_id"].astype(str),
            "template_group": frame["template_group"].astype(str),
            "threat_type": frame["threat_type"].astype(str),
            "y_true": frame["binary_label"].astype(int),
            "score": scores,
            "threshold": threshold,
            "y_pred": (scores >= threshold).astype(int),
        }
    )
    return metrics, predictions


def _evaluate_by_language(
    model: FittedModel,
    vectorizer: object,
    frame: pd.DataFrame,
    threshold: float,
    bootstrap: int,
    variant: str,
    dataset_name: str,
) -> list[dict]:
    """Report language slices separately; never manufacture a missing slice."""
    if "language" not in frame.columns:
        return []
    rows: list[dict] = []
    for language in ("vi", "en", "mixed", "unknown"):
        subset = frame.loc[frame["language"].astype(str).eq(language)].reset_index(drop=True)
        if subset.empty or subset["binary_label"].nunique() != 2:
            continue
        metrics, _ = _evaluate_variant(
            model, vectorizer, subset, threshold, bootstrap, variant, f"{dataset_name}:language={language}"
        )
        metrics["language"] = language
        rows.append(metrics)
    return rows


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _deployment_gate(
    model_selection: dict,
    metrics: list[dict],
    data_manifest: dict | None = None,
) -> dict:
    """Apply conservative absolute gates before changing dashboard inference."""

    selected_model = model_selection["selected_model"]

    def metric(dataset: str, variant: str) -> dict:
        return next(
            (
                row
                for row in metrics
                if row.get("model") == selected_model
                and row.get("dataset") == dataset
                and row.get("variant") == variant
            ),
            {},
        )

    internal = metric("locked_internal_test", "all")
    spaphish = metric("spaphish_v5", "unique_template")
    trec = metric("trec_2007", "unique_template")
    major = metric("major_corpus_2026", "unique_template")
    language_rows = [
        row for row in metrics
        if row.get("dataset") == "locked_internal_test:language=vi" and row.get("variant") == "all"
    ]
    vietnamese = language_rows[0] if language_rows else {}
    training_governance = (
        (data_manifest or {}).get("train", {}).get("source_governance", [])
    )
    evaluation_governance = (
        (data_manifest or {}).get("external", {}).get("source_governance", [])
    )
    non_promotable_sources = [
        record
        for record in training_governance
        if not (
            record.get("license_approved")
            and record.get("provenance_url")
            and record.get("sha256_declared")
        )
    ]
    non_promotable_ids = [str(record.get("id", "unknown")) for record in non_promotable_sources]
    non_promotable_evaluation_sources = [
        record
        for record in evaluation_governance
        if not (
            record.get("license_approved")
            and record.get("provenance_url")
            and record.get("sha256_declared")
        )
    ]
    non_promotable_evaluation_ids = [
        str(record.get("id", "unknown"))
        for record in non_promotable_evaluation_sources
    ]
    checks = [
        {
            "name": "training_data_governance",
            "passed": bool(training_governance) and not non_promotable_sources,
            "detail": (
                "Every training source needs an approved reusable licence, provenance URL and SHA-256. "
                f"Non-promotable sources={non_promotable_ids or 'none'}."
            ),
        },
        {
            "name": "evaluation_data_governance",
            "passed": bool(evaluation_governance) and not non_promotable_evaluation_sources,
            "detail": (
                "Every external source used for a promotion claim needs an approved reusable licence, "
                "provenance URL and SHA-256. "
                f"Non-promotable sources={non_promotable_evaluation_ids or 'none'}."
            ),
        },
        {
            "name": "selection_constraints",
            "passed": not model_selection.get("fallback_used", True),
            "detail": "Winner must satisfy the predeclared CV recall/precision constraints.",
        },
        {
            "name": "deployment_compatibility",
            "passed": selected_model == "logistic_regression",
            "detail": f"dashboard-compatible winner={selected_model}",
        },
        {
            "name": "internal_recall",
            "passed": internal.get("recall", 0.0) >= 0.90,
            "detail": f"recall={internal.get('recall')}",
        },
        {
            "name": "internal_precision",
            "passed": internal.get("precision", 0.0) >= 0.80,
            "detail": f"precision={internal.get('precision')}",
        },
        {
            "name": "spaphish_recall",
            "passed": spaphish.get("recall", 0.0) >= 0.70,
            "detail": f"unique-template recall={spaphish.get('recall')}",
        },
        {
            "name": "spaphish_specificity",
            "passed": spaphish.get("specificity", 0.0) >= 0.80,
            "detail": f"unique-template specificity={spaphish.get('specificity')}",
        },
        {
            "name": "trec_hard_negative_specificity",
            "passed": trec.get("specificity", 0.0) >= 0.95,
            "detail": f"unique-template specificity={trec.get('specificity')}",
        },
        {
            "name": "major_recall",
            "passed": major.get("recall", 0.0) >= 0.70,
            "detail": f"unique-template recall={major.get('recall')}",
        },
        {
            "name": "vietnamese_coverage_and_quality",
            "passed": (
                vietnamese.get("samples", 0) >= MIN_LANGUAGE_EVALUATION_SAMPLES
                and vietnamese.get("positives", 0) >= MIN_LANGUAGE_CLASS_SAMPLES
                and vietnamese.get("negatives", 0) >= MIN_LANGUAGE_CLASS_SAMPLES
                and vietnamese.get("recall", 0.0) >= 0.85
                and vietnamese.get("precision", 0.0) >= 0.75
            ),
            "detail": (
                "Vietnamese locked test requires at least "
                f"{MIN_LANGUAGE_EVALUATION_SAMPLES} examples / {MIN_LANGUAGE_CLASS_SAMPLES} per class; "
                f"samples={vietnamese.get('samples', 0)}, positives={vietnamese.get('positives', 0)}, "
                f"negatives={vietnamese.get('negatives', 0)}, recall={vietnamese.get('recall')}, "
                f"precision={vietnamese.get('precision')}"
            ),
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "policy": "All predeclared internal and external safety checks must pass.",
        "checks": checks,
    }


def run_benchmark(
    model_names: list[str],
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_root: str | Path = DEFAULT_OUTPUT,
    bootstrap: int = 200,
    max_train_rows: int | None = None,
    save_models: bool = False,
    selection_folds: int = 5,
    selection_max_rows: int | None = None,
    final_selected_only: bool = False,
    promote_if_passed: bool = False,
    max_train_rows_per_source: int | None = None,
    max_external_rows_per_source: int | None = None,
) -> dict:
    if promote_if_passed and not save_models:
        raise ValueError("A deployment promotion requires --save-models so a complete immutable bundle exists.")
    if save_models or promote_if_passed:
        require_signing_key()
    started_at = datetime.now(timezone.utc)
    # Benchmarking needs the model document and provenance, not a second
    # full raw-email copy.  Avoid loading it for every source so the complete
    # multi-corpus run remains feasible on the supported workstation.
    corpus, external_sets, data_manifest = split_primary_corpora(
        registry_path,
        include_raw=False,
        max_train_rows_per_source=max_train_rows_per_source,
        max_external_rows_per_source=max_external_rows_per_source,
    )
    selection_corpus = _limit_training(corpus, selection_max_rows)
    selection_rows, model_selection = _selection_cross_validation(
        selection_corpus, model_names, folds=selection_folds
    )
    model_selection["selection_rows"] = int(len(selection_corpus))
    model_selection["full_corpus_rows"] = int(len(corpus))
    corpus = _limit_training(corpus, max_train_rows)
    final_model_names = (
        [model_selection["selected_model"]] if final_selected_only else model_names
    )
    base_train, calibration, locked_internal_test = _split_train_calibration_holdout(corpus)

    vectorizer = _make_vectorizer()
    fit_started = time.perf_counter()
    train_matrix = vectorizer.fit_transform(base_train["research_text"])
    vectorizer_seconds = time.perf_counter() - fit_started
    models, svd = _fit_models(final_model_names, train_matrix, base_train["binary_label"].to_numpy())

    run_id = started_at.strftime("benchmark-%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(selection_rows).to_csv(
        run_dir / "selection_cv_metrics.csv", index=False, encoding="utf-8"
    )
    (run_dir / "model_selection.json").write_text(
        json.dumps(model_selection, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    all_metrics = []
    all_predictions = []
    calibration_matrix = vectorizer.transform(calibration["research_text"])
    thresholds = {}

    for model in models:
        calibration_scores = model.predict_scores(calibration_matrix)
        threshold, calibration_f2 = tune_threshold(calibration["binary_label"], calibration_scores, beta=2.0)
        thresholds[model.name] = {"threshold": threshold, "calibration_f2": calibration_f2}
        metrics, predictions = _evaluate_variant(
            model, vectorizer, locked_internal_test, threshold, bootstrap, "all", "locked_internal_test"
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_metrics.extend(
            _evaluate_by_language(
                model, vectorizer, locked_internal_test, threshold, bootstrap, "all", "locked_internal_test"
            )
        )
        for dataset_name, external in external_sets.items():
            for variant, frame in (
                ("all", external),
                ("unique_template", external.drop_duplicates("template_group", keep="first")),
            ):
                metrics, predictions = _evaluate_variant(
                    model, vectorizer, frame.reset_index(drop=True), threshold, bootstrap, variant, dataset_name
                )
                all_metrics.append(metrics)
                all_predictions.append(predictions)
                all_metrics.extend(
                    _evaluate_by_language(
                        model, vectorizer, frame.reset_index(drop=True), threshold, bootstrap, variant, dataset_name
                    )
                )

    metrics_frame = pd.json_normalize(all_metrics, sep=".")
    metrics_frame.to_csv(run_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    predictions_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    model_manifest = {
        model.name: {
            "feature_space": model.feature_space,
            "notes": model.notes,
            "fit_seconds": model.fit_seconds,
            **thresholds[model.name],
        }
        for model in models
    }
    deployment_gate = _deployment_gate(model_selection, all_metrics, data_manifest)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "task": "phishing vs non-phishing; spam is a distinct hard-negative subtype",
        "threshold_policy": "Maximise F2 on the grouped calibration split, then freeze before locked internal and external tests.",
        "external_policy": "No external example is used for vectorizer fitting, model fitting, calibration, or threshold selection.",
        "model_selection": model_selection,
        "deployment_gate": deployment_gate,
        "deployment_status": (
            "promoted" if promote_if_passed and deployment_gate["passed"] else "research_only_not_promoted"
        ),
        "final_fit_policy": (
            "Refit only the cross-validation winner on the full train partition."
            if final_selected_only
            else "Refit every compared model on the full train partition."
        ),
        "data_manifest": data_manifest,
        "split": {
            "base_train_rows": int(len(base_train)),
            "calibration_rows": int(len(calibration)),
            "locked_internal_test_rows": int(len(locked_internal_test)),
            "base_train_label_counts": {str(k): int(v) for k, v in base_train["binary_label"].value_counts().items()},
            "calibration_label_counts": {str(k): int(v) for k, v in calibration["binary_label"].value_counts().items()},
            "locked_internal_test_label_counts": {str(k): int(v) for k, v in locked_internal_test["binary_label"].value_counts().items()},
            "group_key": "normalized template hash",
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
        "preprocessing": "build_research_document_bilingual_v7",
        "models": model_manifest,
        "metrics": all_metrics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Historical training positives are dominated by the Nazario corpus.",
            "UoT contains repeated synthetic templates, so unique-template results are the primary interpretation for that set.",
            "SpaPhish is Spanish whereas the historical phishing-positive train source is English; this intentionally stress-tests language transfer.",
        ],
    }
    (run_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    if save_models:
        joblib.dump(vectorizer, run_dir / "tfidf_vectorizer.joblib")
        if svd is not None:
            joblib.dump(svd, run_dir / "truncated_svd.joblib")
        for model in models:
            joblib.dump(model.estimator, run_dir / f"{model.name}.joblib")
        artifact_names = [
            "benchmark_summary.json",
            "tfidf_vectorizer.joblib",
            *(f"{model.name}.joblib" for model in models),
        ]
        (run_dir / "artifact_manifest.json").write_text(
            json.dumps(
                sign_manifest({
                    "schema_version": 1,
                    "run_id": run_id,
                    "artifacts": {name: sha256_file(run_dir / name) for name in artifact_names},
                }),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if promote_if_passed and deployment_gate["passed"]:
        try:
            portable_run_dir = str(run_dir.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            portable_run_dir = str(run_dir.resolve())
        pointer_path = Path(output_root) / "latest_benchmark.json"
        temporary_path = pointer_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                sign_manifest({"run_id": run_id, "path": portable_run_dir}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, pointer_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-grade multi-model phishing benchmark.")
    parser.add_argument(
        "--models",
        default="logistic_regression,hybrid_logistic_nb,linear_svm,multinomial_nb,random_forest,xgboost",
        help=(
            "Comma-separated: logistic_regression, hybrid_logistic_nb, linear_svm, "
            "multinomial_nb, random_forest, xgboost"
        ),
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument(
        "--selection-max-rows",
        type=int,
        default=None,
        help="Optional stratified cap for nested-CV selection only; final fitting still uses --max-train-rows.",
    )
    parser.add_argument(
        "--final-selected-only",
        action="store_true",
        help="After cross-validation, refit/evaluate only the locked winner on the full corpus.",
    )
    parser.add_argument(
        "--max-train-rows-per-source",
        type=int,
        default=None,
        help="Deterministic source-level cap before loading/concatenation; recorded in the manifest.",
    )
    parser.add_argument(
        "--max-external-rows-per-source",
        type=int,
        default=None,
        help="Deterministic external-source cap for resource-bounded evaluation; recorded in the manifest.",
    )
    parser.add_argument(
        "--promote-if-passed",
        action="store_true",
        help="Update latest_benchmark.json only when every deployment gate passes.",
    )
    args = parser.parse_args()
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    if names == ["all"]:
        names = ALL_MODEL_NAMES
    summary = run_benchmark(
        names,
        registry_path=args.registry,
        output_root=args.output,
        bootstrap=args.bootstrap,
        max_train_rows=args.max_train_rows,
        save_models=args.save_models,
        selection_folds=args.selection_folds,
        selection_max_rows=args.selection_max_rows,
        final_selected_only=args.final_selected_only,
        promote_if_passed=args.promote_if_passed,
        max_train_rows_per_source=args.max_train_rows_per_source,
        max_external_rows_per_source=args.max_external_rows_per_source,
    )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "models": list(summary["models"]),
                "selected_model": summary["model_selection"]["selected_model"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
