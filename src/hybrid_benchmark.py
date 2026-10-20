"""Paper-grade ML + technical matrix + Local-LLM hybrid benchmark.

This runner implements the four-part protocol used by ``hybrid-paper-v1``:
base ML fitting, fusion fitting, threshold selection, and a locked internal
test.  External data is transformed only after every trainable component and
operating threshold has been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .deployment_validator import sha256_file
from .artifact_signing import require_signing_key, sign_manifest
from .fusion_model import FALLBACK_COLUMNS, FULL_COLUMNS, MATRIX_COLUMNS, build_fusion_features
from .ml_detector import fit_ml_detector, predict_ml_probability
from .research_data import DEFAULT_REGISTRY, load_registry, split_primary_corpora
from .research_metrics import (
    classification_metrics,
    grouped_bootstrap_ci,
    grouped_paired_bootstrap_delta_ci,
    tune_threshold,
)
from .research_splits import partition_integrity, partition_manifest, split_hybrid_partitions
from .paper_protocol import DEFAULT_PAPER_PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research_results"
SEED = 20260715
ML_VARIANTS = ("word", "word_char")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _load_llm_cache(
    path: str | Path | None, expected_identity: dict[str, str] | None = None
) -> tuple[pd.DataFrame, dict]:
    columns = ("content_hash", "llm_probability", "llm_confidence", "llm_available")
    if not path:
        return pd.DataFrame(columns=columns), {"status": "NOT_RUN", "reason": "No LLM cache supplied"}
    cache_path = Path(path)
    if not cache_path.exists():
        return pd.DataFrame(columns=columns), {"status": "NOT_RUN", "reason": f"Missing cache: {cache_path}"}
    frame = pd.read_csv(cache_path)
    required = {"content_hash", "llm_probability"}
    missing = required - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=columns), {
            "status": "NOT_RUN",
            "reason": f"LLM cache missing columns: {sorted(missing)}",
        }
    if expected_identity:
        for field, expected in expected_identity.items():
            if field not in frame:
                raise ValueError(f"Paper LLM cache is missing identity field: {field}")
            observed = set(frame[field].dropna().astype(str))
            if observed != {str(expected)}:
                raise ValueError(
                    f"Paper LLM cache mixes or mismatches {field}: {sorted(observed)}"
                )
    clean = frame.drop_duplicates("content_hash", keep="last").copy()
    clean["llm_probability"] = pd.to_numeric(clean["llm_probability"], errors="coerce")
    valid = clean["llm_probability"].between(0.0, 1.0, inclusive="both")
    if "abstain" in clean:
        valid &= ~clean["abstain"].fillna(False).astype(bool)
    clean["llm_available"] = valid.astype(float)
    clean.loc[~valid, "llm_probability"] = np.nan
    if "llm_confidence" in clean:
        clean["llm_confidence"] = pd.to_numeric(clean["llm_confidence"], errors="coerce").clip(0, 1)
    elif "confidence" in clean:
        clean["llm_confidence"] = pd.to_numeric(clean["confidence"], errors="coerce").clip(0, 1)
    else:
        clean["llm_confidence"] = (clean["llm_probability"] - 0.5).abs() * 2.0
    manifest = {
        "status": "AVAILABLE",
        "path": str(cache_path),
        "sha256": sha256_file(cache_path),
        "rows": int(len(clean)),
        "valid_rows": int(valid.sum()),
        "invalid_or_abstained_rows": int((~valid).sum()),
        "models": sorted(clean["model"].dropna().astype(str).unique()) if "model" in clean else [],
        "prompt_versions": sorted(clean["prompt_version"].dropna().astype(str).unique()) if "prompt_version" in clean else [],
    }
    return clean[list(columns)], manifest


def _load_prepared_run(path: str | Path, registry_path: str | Path) -> tuple[dict, dict, pd.DataFrame]:
    prepared_path = Path(path)
    if prepared_path.is_dir():
        prepared_path = prepared_path / "prepared_run.json"
    run_dir = prepared_path.parent
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared.get("protocol") != "hybrid-paper-v2-frozen-matched-cohort":
        raise ValueError("Prepared run is not hybrid paper protocol v2.")
    if not prepared.get("paper_integrity_gate", {}).get("passed"):
        raise ValueError("Prepared paper inputs failed their integrity gate.")
    if sha256_file(registry_path) != prepared.get("registry_source_sha256"):
        raise ValueError("Current registry differs from the frozen prepared run.")
    artifact_manifest = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in artifact_manifest.get("artifacts", {}).items():
        candidate = run_dir / name
        if not candidate.exists() or sha256_file(candidate) != expected:
            raise ValueError(f"Prepared-run artifact failed checksum verification: {name}")
    protocol = json.loads((run_dir / "frozen_protocol.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(run_dir / "llm_cohort.csv", dtype={"content_hash": str})
    return prepared, protocol, cohort


def _load_llm_cache_amendment(
    prepared_run: str | Path | None,
    llm_cache_path: str | Path | None,
) -> dict | None:
    if prepared_run is None:
        return None
    prepared_location = Path(prepared_run)
    prepared_dir = prepared_location if prepared_location.is_dir() else prepared_location.parent
    amendment_path = prepared_dir / "llm_cache_amendment.json"
    if not amendment_path.exists():
        return None
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    failures = amendment.get("failures", [])
    cache_matches = bool(
        llm_cache_path
        and Path(llm_cache_path).is_file()
        and amendment.get("cache_after_sha256") == sha256_file(llm_cache_path)
    )
    amendment["integrity_check"] = {
        "passed": bool(
            amendment.get("complete_registered_cohort")
            and not failures
            and cache_matches
            and not amendment.get("raw_text_persisted")
        ),
        "cache_hash_matches": cache_matches,
        "failure_count": len(failures),
    }
    return amendment


def _cohort_frames(
    frames: dict[str, pd.DataFrame], cohort: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict]:
    selected: dict[str, pd.DataFrame] = {}
    violations: list[str] = []
    for name, cohort_part in cohort.groupby("evaluation_partition", sort=True):
        if name not in frames:
            violations.append(f"Cohort partition has no reconstructed frame: {name}")
            continue
        order = cohort_part[["content_hash", "cohort_order"]].copy()
        available = frames[name].sort_values(
            ["content_hash", "template_group", "message_id"]
        ).drop_duplicates("content_hash", keep="first")
        merged = order.merge(available, on="content_hash", how="left", validate="one_to_one", indicator=True)
        missing = int(merged["_merge"].ne("both").sum())
        if missing:
            violations.append(f"{name} has {missing} missing frozen cohort rows")
        selected[name] = merged.loc[merged["_merge"].eq("both")].drop(
            columns="_merge"
        ).sort_values("cohort_order").reset_index(drop=True)
    return selected, {
        "passed": not violations,
        "rows": {name: int(len(frame)) for name, frame in selected.items()},
        "violations": violations,
    }


def _attach_llm(frame: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if cache.empty:
        result["llm_probability"] = np.nan
        result["llm_confidence"] = np.nan
        result["llm_available"] = 0.0
        return result
    result = result.merge(cache, on="content_hash", how="left", validate="many_to_one")
    result["llm_available"] = result["llm_available"].fillna(0.0).astype(float)
    return result


def _select_ml_variant(frame: pd.DataFrame, variants: Iterable[str], folds: int = 5) -> tuple[str, list[dict]]:
    folds = max(2, min(int(folds), int(frame["template_group"].nunique())))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED + 20)
    rows: list[dict] = []
    for fold, (fit_index, validation_index) in enumerate(
        splitter.split(frame, frame["binary_label"], groups=frame["template_group"]), start=1
    ):
        fit_frame = frame.iloc[fit_index]
        validation = frame.iloc[validation_index]
        for variant in variants:
            started = time.perf_counter()
            vectorizer, model = fit_ml_detector(
                fit_frame["research_text"], fit_frame["binary_label"], kind=variant
            )
            scores = predict_ml_probability(vectorizer, model, validation["research_text"])
            rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "fit_rows": int(len(fit_frame)),
                    "validation_rows": int(len(validation)),
                    "average_precision": float(average_precision_score(validation["binary_label"], scores)),
                    "fit_predict_seconds": time.perf_counter() - started,
                }
            )
    metrics = pd.DataFrame(rows)
    winner = (
        metrics.groupby("variant", as_index=False)["average_precision"]
        .mean()
        .sort_values(["average_precision", "variant"], ascending=[False, True])
        .iloc[0]["variant"]
    )
    return str(winner), rows


def _module_frame(frame, vectorizer, ml_model, llm_cache) -> pd.DataFrame:
    result = _attach_llm(frame, llm_cache)
    result["ml_probability"] = predict_ml_probability(vectorizer, ml_model, result["research_text"])
    features = build_fusion_features(
        result["raw_text"],
        result["ml_probability"],
        result["llm_probability"],
        result["llm_confidence"],
    )
    for column in features:
        result[column] = features[column].to_numpy()
    return result


def _new_fusion_model() -> Pipeline:
    return Pipeline(
        (
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced", max_iter=1_000, solver="liblinear", random_state=SEED
                ),
            ),
        )
    )


def _fit_configuration(columns: tuple[str, ...], frame: pd.DataFrame):
    if len(columns) == 1:
        return None
    return _new_fusion_model().fit(frame[list(columns)], frame["binary_label"])


def _configuration_scores(frame: pd.DataFrame, columns: tuple[str, ...], model) -> np.ndarray:
    if model is None:
        return frame[columns[0]].astype(float).to_numpy()
    return model.predict_proba(frame[list(columns)])[:, 1]


def _requires_llm(columns: tuple[str, ...]) -> bool:
    return any(column.startswith("llm_") for column in columns)


def _evaluate_configuration(
    name: str,
    columns: tuple[str, ...],
    model,
    threshold: float,
    frames: dict[str, pd.DataFrame],
    external_names: set[str],
    bootstrap: int,
) -> tuple[list[dict], list[pd.DataFrame], dict]:
    metrics_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    for dataset, source_frame in frames.items():
        if dataset in {"ml_train", "fusion_train", "threshold_selection"}:
            continue
        variants = [("all", source_frame)]
        if dataset in external_names:
            variants.append(("unique_template", source_frame.drop_duplicates("template_group", keep="first")))
        for variant, frame in variants:
            key = f"{dataset}:{variant}"
            if _requires_llm(columns) and not frame["llm_available"].eq(1.0).all():
                skipped[key] = f"Incomplete LLM coverage: {int(frame['llm_available'].eq(0).sum())} missing/abstained"
                continue
            scores = _configuration_scores(frame, columns, model)
            metrics: dict[str, Any] = dict(
                classification_metrics(frame["binary_label"], scores, threshold)
            )
            metrics["bootstrap_95_ci"] = grouped_bootstrap_ci(
                frame["binary_label"], scores, frame["template_group"], threshold, bootstrap, SEED
            )
            metrics.update({"configuration": name, "dataset": dataset, "variant": variant})
            metrics_rows.append(metrics)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "configuration": name,
                        "dataset": dataset,
                        "variant": variant,
                        "message_id": frame["message_id"],
                        "content_hash": frame["content_hash"],
                        "template_group": frame["template_group"],
                        "y_true": frame["binary_label"].astype(int),
                        "score": scores,
                        "threshold": threshold,
                        "y_pred": (scores >= threshold).astype(int),
                    }
                )
            )
    return metrics_rows, prediction_rows, skipped


def _deployment_gate(metrics: list[dict], protocol_gate: dict, full_completed: bool) -> dict:
    def metric(dataset: str, variant: str = "unique_template") -> dict:
        return next(
            (
                row for row in metrics
                if row.get("configuration") == "Full_ML_Matrix_LLM"
                and row.get("dataset") == dataset
                and row.get("variant") == variant
            ),
            {},
        )

    internal = metric("locked_internal_test", "all")
    spaphish = metric("spaphish_v5")
    trec = metric("trec_2007")
    major = metric("major_corpus_2026")
    checks = [
        {"name": "protocol_integrity", "passed": bool(protocol_gate.get("passed")), "detail": protocol_gate.get("violations", [])},
        {"name": "full_fusion_completed", "passed": bool(full_completed), "detail": "Full tri-modal evaluation must have complete LLM coverage."},
        {"name": "internal_recall", "passed": internal.get("recall", 0.0) >= 0.90, "detail": internal.get("recall")},
        {"name": "internal_precision", "passed": internal.get("precision", 0.0) >= 0.80, "detail": internal.get("precision")},
        {"name": "spaphish_recall", "passed": spaphish.get("recall", 0.0) >= 0.70, "detail": spaphish.get("recall")},
        {"name": "spaphish_specificity", "passed": spaphish.get("specificity", 0.0) >= 0.80, "detail": spaphish.get("specificity")},
        {"name": "trec_specificity", "passed": trec.get("specificity", 0.0) >= 0.95, "detail": trec.get("specificity")},
        {"name": "major_recall", "passed": major.get("recall", 0.0) >= 0.70, "detail": major.get("recall")},
    ]
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _paired_ablation_deltas(predictions: pd.DataFrame, iterations: int) -> list[dict]:
    """Compare every matched-cohort configuration against ML-only."""

    if predictions.empty:
        return []
    rows: list[dict] = []
    matched = predictions.loc[
        ~predictions["configuration"].astype(str).str.endswith("_full_data")
    ]
    for (dataset, variant), baseline in matched.loc[
        matched["configuration"].eq("ML_only")
    ].groupby(["dataset", "variant"], sort=True):
        for configuration, candidate in matched.loc[
            matched["dataset"].eq(dataset)
            & matched["variant"].eq(variant)
            & ~matched["configuration"].eq("ML_only")
        ].groupby("configuration", sort=True):
            joined = baseline.merge(
                candidate,
                on=["content_hash", "template_group", "y_true"],
                suffixes=("_reference", "_candidate"),
                how="inner",
                validate="one_to_one",
            )
            if len(joined) != len(baseline) or len(joined) != len(candidate):
                raise ValueError(
                    f"Paired ablation mismatch for {configuration} on {dataset}:{variant}"
                )
            reference_metric = classification_metrics(
                joined["y_true"], joined["score_reference"], float(joined["threshold_reference"].iloc[0])
            )
            candidate_metric = classification_metrics(
                joined["y_true"], joined["score_candidate"], float(joined["threshold_candidate"].iloc[0])
            )
            names = ("average_precision", "roc_auc", "precision", "recall", "f1", "f2", "specificity")
            point_deltas: dict[str, float] = {}
            for name in names:
                candidate_value = candidate_metric.get(name)
                reference_value = reference_metric.get(name)
                if candidate_value is not None and reference_value is not None:
                    point_deltas[name] = float(candidate_value) - float(reference_value)
            rows.append(
                {
                    "reference": "ML_only",
                    "candidate": str(configuration),
                    "dataset": str(dataset),
                    "variant": str(variant),
                    "samples": int(len(joined)),
                    "point_delta_candidate_minus_reference": point_deltas,
                    "paired_grouped_bootstrap_95_ci": grouped_paired_bootstrap_delta_ci(
                        joined["y_true"],
                        joined["score_reference"],
                        joined["score_candidate"],
                        joined["template_group"],
                        float(joined["threshold_reference"].iloc[0]),
                        float(joined["threshold_candidate"].iloc[0]),
                        iterations=iterations,
                        seed=SEED,
                    ),
                }
            )
    return rows


def run_hybrid_benchmark(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    llm_cache_path: str | Path | None,
    output_root: str | Path = DEFAULT_OUTPUT,
    ml_variants: Iterable[str] = ML_VARIANTS,
    selection_folds: int = 5,
    bootstrap: int = 200,
    promote_if_passed: bool = False,
    prepared_run: str | Path | None = None,
    protocol_path: str | Path = DEFAULT_PAPER_PROTOCOL,
) -> dict:
    require_signing_key()
    started = datetime.now(timezone.utc)
    prepared: dict | None = None
    paper_protocol: dict | None = None
    cohort = pd.DataFrame()
    run_seed = SEED
    if prepared_run:
        prepared, paper_protocol, cohort = _load_prepared_run(prepared_run, registry_path)
        run_seed = int(prepared["seed"])
        minimum_bootstrap = int(paper_protocol["bootstrap"]["minimum_iterations"])
        if int(bootstrap) < minimum_bootstrap:
            raise ValueError(
                f"Paper protocol requires at least {minimum_bootstrap} grouped bootstrap iterations."
            )
    corpus, external_sets, data_manifest = split_primary_corpora(registry_path)
    partitions = split_hybrid_partitions(corpus, seed=run_seed)
    partition_frames = partitions.as_dict()
    protocol_gate = partition_integrity(partition_frames)
    if not protocol_gate["passed"]:
        raise ValueError(f"Protocol integrity failed: {protocol_gate['violations']}")

    selected_variant, selection_rows = _select_ml_variant(
        partitions.ml_train, tuple(ml_variants), folds=selection_folds
    )
    vectorizer, ml_model = fit_ml_detector(
        partitions.ml_train["research_text"], partitions.ml_train["binary_label"], selected_variant
    )
    expected_identity = None
    if prepared and paper_protocol:
        expected_identity = {
            "paper_run_id": prepared["paper_run_id"],
            "model_file_sha256": paper_protocol["llm"]["preferred_model"]["sha256"],
        }
    llm_cache, llm_manifest = _load_llm_cache(llm_cache_path, expected_identity)
    llm_cache_amendment = _load_llm_cache_amendment(prepared_run, llm_cache_path)
    frames = {
        name: _module_frame(frame, vectorizer, ml_model, llm_cache)
        for name, frame in partition_frames.items()
    }
    for name, frame in external_sets.items():
        frames[name] = _module_frame(frame, vectorizer, ml_model, llm_cache)
    llm_frames = frames
    cohort_gate = {"passed": True, "rows": {}, "violations": []}
    if prepared is not None:
        llm_frames, cohort_gate = _cohort_frames(frames, cohort)
        if not cohort_gate["passed"]:
            raise ValueError(f"Frozen LLM cohort reconstruction failed: {cohort_gate['violations']}")

    matrix_columns = (*MATRIX_COLUMNS, "matrix_coverage")
    configurations: dict[str, tuple[str, ...]] = {
        "ML_only": ("ml_probability",),
        "Technical_Matrix_only": matrix_columns,
        "LLM_only": ("llm_probability",),
        "ML_plus_Matrix": tuple(FALLBACK_COLUMNS),
        "ML_plus_LLM": ("ml_probability", "llm_probability", "llm_confidence", "llm_available"),
        "Matrix_plus_LLM": (*matrix_columns, "llm_probability", "llm_confidence", "llm_available"),
        "Full_ML_Matrix_LLM": tuple(FULL_COLUMNS),
    }
    full_data_configurations: set[str] = set()
    if prepared is not None:
        # Every named ablation above is fitted, thresholded, and evaluated on
        # the same frozen cohort.  Full-data ML baselines remain available as
        # secondary estimates but are never compared directly with the cohort
        # LLM configurations.
        configurations = {
            "ML_only_full_data": ("ml_probability",),
            "ML_plus_Matrix_full_data": tuple(FALLBACK_COLUMNS),
            **configurations,
        }
        full_data_configurations = {"ML_only_full_data", "ML_plus_Matrix_full_data"}
    metrics_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    statuses: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    for name, columns in configurations.items():
        needs_llm = _requires_llm(columns)
        configuration_frames = (
            frames
            if prepared is None or name in full_data_configurations
            else llm_frames
        )
        required = (configuration_frames["fusion_train"], configuration_frames["threshold_selection"])
        missing_required = sum(int(frame["llm_available"].eq(0).sum()) for frame in required) if needs_llm else 0
        if missing_required:
            statuses[name] = {
                "status": "NOT_RUN",
                "reason": "Fusion/threshold partitions do not have complete valid LLM coverage.",
                "missing_or_abstained": missing_required,
            }
            continue
        model = _fit_configuration(columns, configuration_frames["fusion_train"])
        threshold_scores = _configuration_scores(configuration_frames["threshold_selection"], columns, model)
        threshold, threshold_f2 = tune_threshold(
            configuration_frames["threshold_selection"]["binary_label"], threshold_scores, beta=2.0
        )
        rows, predictions, skipped = _evaluate_configuration(
            name, columns, model, threshold, configuration_frames, set(external_sets), bootstrap
        )
        metrics_rows.extend(rows)
        prediction_rows.extend(predictions)
        fitted_models[name] = model
        statuses[name] = {
            "status": "COMPLETED",
            "features": list(columns),
            "threshold": threshold,
            "threshold_f2": threshold_f2,
            "skipped_evaluations": skipped,
        }

    full_status = statuses.get("Full_ML_Matrix_LLM", {})
    full_completed = full_status.get("status") == "COMPLETED" and not full_status.get("skipped_evaluations")
    deployment_gate = _deployment_gate(metrics_rows, protocol_gate, full_completed)
    cache_complete = (
        llm_manifest.get("status") == "AVAILABLE"
        and llm_manifest.get("valid_rows") == len(cohort)
    ) if prepared is not None else full_completed
    paper_checks = []
    if prepared is not None:
        if paper_protocol is None or prepared_run is None:
            raise RuntimeError("Prepared paper-v2 inputs require a frozen protocol and run path.")
        paper_checks = [
            {"name": "prepared_input_integrity", "passed": bool(prepared["paper_integrity_gate"]["passed"])},
            {"name": "cohort_reconstruction", "passed": bool(cohort_gate["passed"]), "detail": cohort_gate},
            {"name": "complete_valid_llm_cache", "passed": bool(cache_complete), "detail": llm_manifest},
            {"name": "full_hybrid_completed", "passed": bool(full_completed)},
            {"name": "bootstrap_minimum", "passed": int(bootstrap) >= int(paper_protocol["bootstrap"]["minimum_iterations"]), "detail": bootstrap},
        ]
        if llm_cache_amendment is not None:
            paper_checks.append(
                {
                    "name": "llm_cache_amendment_integrity",
                    "passed": bool(llm_cache_amendment["integrity_check"]["passed"]),
                    "detail": llm_cache_amendment["integrity_check"],
                }
            )
    paper_gate = {
        "passed": bool(paper_checks) and all(check["passed"] for check in paper_checks),
        "checks": paper_checks,
        "independent_of_deployment_performance": True,
    }
    run_prefix = "hybrid-paper-v2" if prepared is not None else "hybrid-paper-v1"
    run_id = started.strftime(f"{run_prefix}-%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    partition_manifest(partition_frames).to_csv(run_dir / "partition_manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(selection_rows).to_csv(run_dir / "model_selection_cv.csv", index=False, encoding="utf-8")
    pd.json_normalize(metrics_rows, sep=".").to_csv(run_dir / "benchmark_metrics.csv", index=False, encoding="utf-8")
    all_predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )
    all_predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    paired_deltas = _paired_ablation_deltas(all_predictions, bootstrap) if prepared is not None else []
    (run_dir / "paired_deltas.json").write_text(
        json.dumps(paired_deltas, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    registry = load_registry(registry_path)
    (run_dir / "frozen_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "partition_integrity.json").write_text(
        json.dumps(protocol_gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if prepared is not None and paper_protocol is not None:
        if prepared_run is None:
            raise RuntimeError("Prepared paper-v2 inputs require a run path.")
        prepared_location = Path(prepared_run)
        (run_dir / "frozen_protocol.json").write_text(
            json.dumps(paper_protocol, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "paper_run_link.json").write_text(
            json.dumps(
                {
                    "paper_run_id": prepared["paper_run_id"],
                    "prepared_run_sha256": sha256_file(
                        prepared_location / "prepared_run.json"
                        if prepared_location.is_dir()
                        else prepared_location
                    ),
                    "llm_cohort_rows": int(len(cohort)),
                    "llm_cohort_sha256": sha256_file(
                        prepared_location / "llm_cohort.csv"
                        if prepared_location.is_dir()
                        else prepared_location.parent / "llm_cohort.csv"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    joblib.dump(vectorizer, run_dir / "tfidf_vectorizer.joblib")
    joblib.dump(ml_model, run_dir / "ml_model.joblib")
    if fitted_models.get("ML_plus_Matrix") is not None:
        joblib.dump(fitted_models["ML_plus_Matrix"], run_dir / "fallback_fusion.joblib")
    if fitted_models.get("Full_ML_Matrix_LLM") is not None:
        joblib.dump(fitted_models["Full_ML_Matrix_LLM"], run_dir / "full_fusion.joblib")

    summary = {
        "schema_version": 2,
        "protocol": (
            "hybrid-paper-v2-frozen-matched-cohort"
            if prepared is not None
            else "hybrid-paper-v1-four-part-grouped"
        ),
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seed": run_seed,
        "task": "phishing vs non-phishing; spam remains a hard negative",
        "selected_ml_variant": selected_variant,
        "partitions": protocol_gate["partition_rows"],
        "protocol_gate": protocol_gate,
        "deployment_gate": deployment_gate,
        "deployment_status": "promoted" if promote_if_passed and deployment_gate["passed"] else "research_only_not_promoted",
        "llm_cache": llm_manifest,
        "llm_cache_amendment": llm_cache_amendment,
        "paper_gate": paper_gate,
        "paper_status": "FROZEN_RESULTS" if paper_gate["passed"] else "BLOCKED_RESULT_INTEGRITY",
        "llm_cohort": cohort_gate if prepared is not None else None,
        "configurations": statuses,
        "data_manifest": data_manifest,
        "metrics": metrics_rows,
        "paired_deltas": paired_deltas,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (run_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )

    artifact_names = [
        "hybrid_summary.json",
        "tfidf_vectorizer.joblib",
        "ml_model.joblib",
        "fallback_fusion.joblib",
        "full_fusion.joblib",
        "benchmark_metrics.csv",
        "predictions.csv",
        "partition_manifest.csv",
        "frozen_registry.json",
        "partition_integrity.json",
        "frozen_protocol.json",
        "paper_run_link.json",
        "paired_deltas.json",
    ]
    artifact_manifest = sign_manifest({
        "schema_version": 1,
        "run_id": run_id,
        "artifacts": {
            name: sha256_file(run_dir / name)
            for name in artifact_names
            if (run_dir / name).exists()
        },
    })
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if promote_if_passed and deployment_gate["passed"]:
        pointer = sign_manifest({"run_id": run_id, "path": str(run_dir.resolve().relative_to(ROOT.resolve()))})
        pointer_path = Path(output_root) / "latest_hybrid.json"
        temporary = pointer_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, pointer_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-part ML + matrix + Local-LLM hybrid benchmark")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--llm-cache", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ml-variants", default=",".join(ML_VARIANTS))
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--promote-if-passed", action="store_true")
    parser.add_argument("--prepared-run", default=None)
    parser.add_argument("--protocol", default=str(DEFAULT_PAPER_PROTOCOL))
    args = parser.parse_args()
    summary = run_hybrid_benchmark(
        registry_path=args.registry,
        llm_cache_path=args.llm_cache,
        output_root=args.output,
        ml_variants=tuple(value.strip() for value in args.ml_variants.split(",") if value.strip()),
        selection_folds=args.selection_folds,
        bootstrap=args.bootstrap,
        promote_if_passed=args.promote_if_passed,
        prepared_run=args.prepared_run,
        protocol_path=args.protocol,
    )
    print(json.dumps({"run_id": summary["run_id"], "deployment_status": summary["deployment_status"]}))


if __name__ == "__main__":
    main()
