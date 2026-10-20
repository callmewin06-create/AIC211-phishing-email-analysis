"""Matched-subset pilot ablation for an expensive Local-LLM module.

The classical benchmark and technical-matrix ablation use the complete
evaluation sets.  Local inference can be much more expensive, so this module
supports a deterministic pilot in which *every* configuration is evaluated on
the exact same partition/source/label-stratified subset.  The output is named
as a pilot and must not be presented as a full-coverage confirmatory result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .ablation_study import (
        _configuration_scores,
        _fit_configuration,
        _latest_run,
        _load_llm_cache,
        _module_frame,
        _split_three_ways,
    )
    from .llm_benchmark import PROMPT_VERSION, SEED as PILOT_SEED
    from .research_benchmark import SEED
    from .research_data import DEFAULT_REGISTRY, split_primary_corpora
    from .research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold
except ImportError:  # pragma: no cover
    from ablation_study import (
        _configuration_scores,
        _fit_configuration,
        _latest_run,
        _load_llm_cache,
        _module_frame,
        _split_three_ways,
    )
    from llm_benchmark import PROMPT_VERSION, SEED as PILOT_SEED
    from research_benchmark import SEED
    from research_data import DEFAULT_REGISTRY, split_primary_corpora
    from research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_evaluation_frames(corpus, external_sets):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, calibration_index = next(
        splitter.split(corpus, corpus["binary_label"], groups=corpus["template_group"])
    )
    calibration = corpus.iloc[calibration_index].reset_index(drop=True)
    meta_train, threshold_train, internal_holdout = _split_three_ways(calibration)
    return {
        "meta_train": meta_train,
        "threshold_train": threshold_train,
        "internal_ablation_holdout": internal_holdout,
        **external_sets,
    }


def _deterministic_pilot(named_frames, sample_per_stratum: int):
    tagged_parts = []
    for partition, frame in named_frames.items():
        tagged = frame.copy()
        tagged["evaluation_partition"] = partition
        tagged_parts.append(tagged)
    candidates = pd.concat(tagged_parts, ignore_index=True).drop_duplicates(
        "content_hash", keep="first"
    )
    sampled = []
    for _, group in candidates.groupby(
        ["evaluation_partition", "source", "binary_label"], observed=True, sort=True
    ):
        sampled.append(
            group.sample(
                n=min(len(group), int(sample_per_stratum)), random_state=PILOT_SEED
            )
        )
    pilot = pd.concat(sampled, ignore_index=True)
    return {
        partition: pilot.loc[pilot["evaluation_partition"].eq(partition)].reset_index(drop=True)
        for partition in named_frames
    }, pilot


def run_pilot_ablation(
    llm_cache: str | Path,
    benchmark_run: str | Path | None = None,
    output_root: str | Path = ROOT / "research_results",
    registry_path: str | Path = DEFAULT_REGISTRY,
    sample_per_stratum: int = 10,
    bootstrap: int = 200,
) -> dict:
    if sample_per_stratum < 1:
        raise ValueError("sample_per_stratum must be positive")
    run_dir = Path(benchmark_run) if benchmark_run else _latest_run(Path(output_root))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    benchmark_summary = json.loads(
        (run_dir / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    vectorizer = joblib.load(run_dir / "tfidf_vectorizer.joblib")
    ml_model = joblib.load(run_dir / "logistic_regression.joblib")

    corpus, external_sets, data_manifest = split_primary_corpora(registry_path)
    named_frames = _named_evaluation_frames(corpus, external_sets)
    pilot_frames, pilot_manifest = _deterministic_pilot(named_frames, sample_per_stratum)

    cache_path = Path(llm_cache)
    llm_scores, cache_manifest = _load_llm_cache(cache_path)
    if cache_manifest.get("status") != "AVAILABLE":
        raise ValueError(cache_manifest.get("reason", "Local-LLM cache is unavailable"))
    if cache_manifest.get("prompt_versions") != [PROMPT_VERSION]:
        raise ValueError(
            f"Pilot requires prompt {PROMPT_VERSION}; found {cache_manifest.get('prompt_versions')}"
        )
    missing = {
        partition: int((~frame["content_hash"].isin(llm_scores)).sum())
        for partition, frame in pilot_frames.items()
    }
    missing = {partition: count for partition, count in missing.items() if count}
    if missing:
        raise ValueError(f"Incomplete deterministic pilot cache: {missing}")

    frames = {
        name: _module_frame(frame, vectorizer, ml_model, llm_scores)
        for name, frame in pilot_frames.items()
    }
    for required in ("meta_train", "threshold_train", "internal_ablation_holdout"):
        if frames[required]["binary_label"].nunique() != 2:
            raise ValueError(f"Pilot partition {required} does not contain both classes")

    configurations = {
        "ML_only": ["ml_score"],
        "URL_Risk_only": ["url_score"],
        "ML_plus_URL_Risk": ["ml_score", "url_score"],
        "Technical_Matrix_only_layers_1_10": ["matrix_score"],
        "ML_plus_Technical_Matrix": ["ml_score", "matrix_score"],
        "LLM_only": ["llm_score"],
        "ML_plus_LLM": ["ml_score", "llm_score"],
        "Full_ML_plus_Matrix_plus_LLM": ["ml_score", "matrix_score", "llm_score"],
    }
    metrics_rows = []
    prediction_rows = []
    statuses = {}
    for name, columns in configurations.items():
        fusion = _fit_configuration(name, columns, frames["meta_train"])
        threshold_scores = _configuration_scores(frames["threshold_train"], columns, fusion)
        threshold, threshold_f2 = tune_threshold(
            frames["threshold_train"]["binary_label"], threshold_scores, beta=2.0
        )
        statuses[name] = {
            "status": "COMPLETED_PILOT",
            "columns": columns,
            "fusion": "identity" if fusion is None else "logistic_stacking_on_pilot_meta_train",
            "threshold": threshold,
            "threshold_train_f2": threshold_f2,
        }
        for dataset in ["internal_ablation_holdout", *external_sets]:
            source_frame = frames[dataset]
            variants = [("pilot_all", source_frame)]
            if dataset in external_sets:
                variants.append(
                    (
                        "pilot_unique_template",
                        source_frame.drop_duplicates("template_group", keep="first"),
                    )
                )
            for variant, frame in variants:
                scores = _configuration_scores(frame, columns, fusion)
                metrics = classification_metrics(frame["binary_label"], scores, threshold)
                metrics["bootstrap_95_ci"] = grouped_bootstrap_ci(
                    frame["binary_label"],
                    scores,
                    frame["template_group"],
                    threshold,
                    bootstrap,
                    PILOT_SEED,
                )
                metrics.update(
                    {
                        "configuration": name,
                        "dataset": dataset,
                        "variant": variant,
                        "evaluation_scope": "deterministic_llm_matched_pilot",
                    }
                )
                metrics_rows.append(metrics)
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "configuration": name,
                            "dataset": dataset,
                            "variant": variant,
                            "message_id": frame["message_id"],
                            "template_group": frame["template_group"],
                            "y_true": frame["binary_label"].astype(int),
                            "score": scores,
                            "threshold": threshold,
                            "y_pred": (scores >= threshold).astype(int),
                        }
                    )
                )

    metrics_frame = pd.DataFrame(metrics_rows)
    baseline = metrics_frame.loc[
        metrics_frame["configuration"].eq("ML_only"),
        ["dataset", "variant", "average_precision", "f1", "recall"],
    ].rename(
        columns={name: f"baseline_{name}" for name in ("average_precision", "f1", "recall")}
    )
    metrics_frame = metrics_frame.merge(baseline, on=["dataset", "variant"], how="left")
    for metric in ("average_precision", "f1", "recall"):
        metrics_frame[f"delta_{metric}_points_vs_ml"] = (
            metrics_frame[metric] - metrics_frame[f"baseline_{metric}"]
        )

    metrics_path = run_dir / "llm_pilot_ablation_metrics.csv"
    predictions_path = run_dir / "llm_pilot_ablation_predictions.csv"
    summary_path = run_dir / "llm_pilot_ablation_summary.json"
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8")
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        predictions_path, index=False, encoding="utf-8"
    )
    stratum_counts = (
        pilot_manifest.groupby(["evaluation_partition", "source", "binary_label"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
        .to_dict("records")
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_run": benchmark_summary["run_id"],
        "status": "COMPLETED_PILOT",
        "claim_scope": (
            "Exploratory deterministic matched-subset Local-LLM ablation. "
            "It is not a substitute for the full-coverage confirmatory run."
        ),
        "sampling": {
            "seed": PILOT_SEED,
            "sample_per_partition_source_label_stratum": sample_per_stratum,
            "target_rows": int(len(pilot_manifest)),
            "partition_rows": {name: int(len(frame)) for name, frame in pilot_frames.items()},
            "strata": stratum_counts,
        },
        "llm_cache": {
            **cache_manifest,
            "sha256": _sha256(cache_path),
            "target_rows_present": int(len(pilot_manifest)),
        },
        "data_manifest": data_manifest,
        "configurations": statuses,
        "metrics": metrics_frame.to_dict("records"),
    }
    summary_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, (np.integer, np.floating))
            else str(value),
        ),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Matched-subset Local-LLM pilot ablation.")
    parser.add_argument("--llm-cache", required=True)
    parser.add_argument("--benchmark-run", default=None)
    parser.add_argument("--output", default=str(ROOT / "research_results"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--sample-per-stratum", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    report = run_pilot_ablation(
        args.llm_cache,
        args.benchmark_run,
        args.output,
        args.registry,
        args.sample_per_stratum,
        args.bootstrap,
    )
    print(
        json.dumps(
            {
                "benchmark_run": report["benchmark_run"],
                "status": report["status"],
                "sampling": report["sampling"],
                "configurations": report["configurations"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
