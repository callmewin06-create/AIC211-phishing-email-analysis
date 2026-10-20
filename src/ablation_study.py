"""Module-level ablation study for ML, technical matrix, and Local LLM.

The script never substitutes one module for another.  If a complete Local-LLM
cache is unavailable, LLM-dependent configurations are recorded as NOT_RUN,
which is more defensible than fabricating a "full system" metric.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .research_benchmark import SEED
    from .research_data import DEFAULT_REGISTRY, split_primary_corpora
    from .research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold
    from .security_matrix import technical_matrix_score, url_risk_score
except ImportError:  # pragma: no cover
    from research_benchmark import SEED
    from research_data import DEFAULT_REGISTRY, split_primary_corpora
    from research_metrics import classification_metrics, grouped_bootstrap_ci, tune_threshold
    from security_matrix import technical_matrix_score, url_risk_score


ROOT = Path(__file__).resolve().parents[1]


def _split_three_ways(frame: pd.DataFrame):
    outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    remaining_index, holdout_index = next(
        outer.split(frame, frame["binary_label"], groups=frame["template_group"])
    )
    remaining = frame.iloc[remaining_index].reset_index(drop=True)
    holdout = frame.iloc[holdout_index].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED + 1)
    meta_index, threshold_index = next(
        inner.split(remaining, remaining["binary_label"], groups=remaining["template_group"])
    )
    return (
        remaining.iloc[meta_index].reset_index(drop=True),
        remaining.iloc[threshold_index].reset_index(drop=True),
        holdout,
    )


def _latest_run(output_root: Path) -> Path:
    pointer = json.loads((output_root / "latest_benchmark.json").read_text(encoding="utf-8"))
    path = Path(pointer["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not (path / "benchmark_summary.json").exists() and pointer.get("run_id"):
        portable_path = output_root / str(pointer["run_id"])
        if (portable_path / "benchmark_summary.json").exists():
            return portable_path
    return path


def _load_llm_cache(path: str | Path | None) -> tuple[dict[str, float], dict]:
    if not path:
        return {}, {"status": "NOT_RUN", "reason": "No --llm-cache was supplied."}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}, {"status": "NOT_RUN", "reason": f"LLM cache does not exist: {cache_path}"}
    frame = pd.read_csv(cache_path)
    required = {"content_hash", "llm_probability", "model", "prompt_version"}
    if not required.issubset(frame.columns):
        return {}, {"status": "NOT_RUN", "reason": f"LLM cache missing columns: {sorted(required - set(frame.columns))}"}
    clean = frame.dropna(subset=["content_hash", "llm_probability"]).drop_duplicates("content_hash", keep="last")
    scores = clean.set_index("content_hash")["llm_probability"].astype(float).clip(0, 1).to_dict()
    return scores, {
        "status": "AVAILABLE",
        "rows": int(len(clean)),
        "models": sorted(clean["model"].astype(str).unique()),
        "prompt_versions": sorted(clean["prompt_version"].astype(str).unique()),
        "path": str(cache_path),
    }


def _module_frame(frame, vectorizer, ml_model, llm_scores):
    result = frame.copy()
    result["ml_score"] = ml_model.predict_proba(vectorizer.transform(result["research_text"]))[:, 1]
    result["url_score"] = [url_risk_score(text) for text in result["raw_text"]]
    result["matrix_score"] = [technical_matrix_score(text) for text in result["raw_text"]]
    result["llm_score"] = result["content_hash"].map(llm_scores)
    return result


def _fusion_input(frame, columns):
    values = []
    for column in columns:
        score = frame[column].astype(float).clip(1e-5, 1 - 1e-5).to_numpy()
        values.append(logit(score))
    return np.column_stack(values)


def _fit_configuration(name, columns, meta_train):
    if len(columns) == 1:
        return None
    model = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=500)
    model.fit(_fusion_input(meta_train, columns), meta_train["binary_label"])
    return model


def _configuration_scores(frame, columns, fusion_model):
    if len(columns) == 1:
        return frame[columns[0]].astype(float).to_numpy()
    return fusion_model.predict_proba(_fusion_input(frame, columns))[:, 1]


def run_ablation(
    benchmark_run: str | Path | None = None,
    output_root: str | Path = ROOT / "research_results",
    registry_path: str | Path = DEFAULT_REGISTRY,
    llm_cache: str | Path | None = None,
    bootstrap: int = 200,
) -> dict:
    output_root = Path(output_root)
    run_dir = Path(benchmark_run) if benchmark_run else _latest_run(output_root)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary = json.loads((run_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    if "logistic_regression" not in summary["models"]:
        raise ValueError("Ablation requires a saved logistic_regression benchmark model.")
    vectorizer_path = run_dir / "tfidf_vectorizer.joblib"
    model_path = run_dir / "logistic_regression.joblib"
    if not vectorizer_path.exists() or not model_path.exists():
        raise FileNotFoundError("Rerun benchmark with --save-models before ablation.")
    vectorizer = joblib.load(vectorizer_path)
    ml_model = joblib.load(model_path)

    corpus, external_sets, data_manifest = split_primary_corpora(registry_path)
    # Recreate the same calibration partition used by research_benchmark.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, calibration_index = next(
        splitter.split(corpus, corpus["binary_label"], groups=corpus["template_group"])
    )
    calibration = corpus.iloc[calibration_index].reset_index(drop=True)
    meta_train, threshold_train, internal_holdout = _split_three_ways(calibration)
    llm_scores, llm_manifest = _load_llm_cache(llm_cache)

    frames = {
        "meta_train": _module_frame(meta_train, vectorizer, ml_model, llm_scores),
        "threshold_train": _module_frame(threshold_train, vectorizer, ml_model, llm_scores),
        "internal_ablation_holdout": _module_frame(internal_holdout, vectorizer, ml_model, llm_scores),
    }
    for dataset, frame in external_sets.items():
        frames[dataset] = _module_frame(frame, vectorizer, ml_model, llm_scores)

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
        needs_llm = "llm_score" in columns
        required_frames = ["meta_train", "threshold_train", "internal_ablation_holdout", *external_sets]
        missing = {
            frame_name: int(frames[frame_name]["llm_score"].isna().sum())
            for frame_name in required_frames
            if needs_llm and frames[frame_name]["llm_score"].isna().any()
        }
        if missing:
            statuses[name] = {
                "status": "NOT_RUN",
                "reason": "Incomplete Local-LLM cache; scores are never imputed from ML or rules.",
                "missing_scores": missing,
            }
            continue
        fusion = _fit_configuration(name, columns, frames["meta_train"])
        threshold_scores = _configuration_scores(frames["threshold_train"], columns, fusion)
        threshold, calibration_f2 = tune_threshold(
            frames["threshold_train"]["binary_label"], threshold_scores, beta=2.0
        )
        statuses[name] = {
            "status": "COMPLETED",
            "columns": columns,
            "fusion": "identity" if fusion is None else "logistic_stacking_on_meta_train",
            "threshold": threshold,
            "threshold_train_f2": calibration_f2,
        }
        for dataset in ["internal_ablation_holdout", *external_sets]:
            source_frame = frames[dataset]
            variants = [("all", source_frame)]
            if dataset in external_sets:
                variants.append(("unique_template", source_frame.drop_duplicates("template_group", keep="first")))
            for variant, frame in variants:
                scores = _configuration_scores(frame, columns, fusion)
                metrics = classification_metrics(frame["binary_label"], scores, threshold)
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
                            "template_group": frame["template_group"],
                            "y_true": frame["binary_label"].astype(int),
                            "score": scores,
                            "threshold": threshold,
                            "y_pred": (scores >= threshold).astype(int),
                        }
                    )
                )

    completed = pd.DataFrame(metrics_rows)
    if not completed.empty:
        baseline = completed.loc[completed["configuration"].eq("ML_only"), ["dataset", "variant", "average_precision", "f1", "recall"]]
        baseline = baseline.rename(columns={metric: f"baseline_{metric}" for metric in ("average_precision", "f1", "recall")})
        completed = completed.merge(baseline, on=["dataset", "variant"], how="left")
        for metric in ("average_precision", "f1", "recall"):
            completed[f"delta_{metric}_points_vs_ml"] = completed[metric] - completed[f"baseline_{metric}"]
        completed.to_csv(run_dir / "ablation_metrics.csv", index=False, encoding="utf-8")
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            run_dir / "ablation_predictions.csv", index=False, encoding="utf-8"
        )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_run": summary["run_id"],
        "matrix_contract": "Layers 1-10 are independent evidence; layer 11=ML; layer 12=Local LLM.",
        "comparison_contract": "ML + technical matrix therefore means ML plus layers 1-10, without double-counting model layers.",
        "data_manifest": data_manifest,
        "partitions": {name: int(len(frame)) for name, frame in frames.items()},
        "llm_cache": llm_manifest,
        "configurations": statuses,
        "metrics": completed.to_dict("records") if not completed.empty else [],
    }
    (run_dir / "ablation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, (np.integer, np.floating)) else str(value)),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for ML, technical matrix, Local LLM and full system.")
    parser.add_argument("--benchmark-run", default=None)
    parser.add_argument("--output", default=str(ROOT / "research_results"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--llm-cache", default=None)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    report = run_ablation(args.benchmark_run, args.output, args.registry, args.llm_cache, args.bootstrap)
    print(json.dumps({"benchmark_run": report["benchmark_run"], "configurations": report["configurations"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
