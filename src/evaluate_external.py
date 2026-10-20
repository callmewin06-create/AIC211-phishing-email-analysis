"""Evaluate the deployed research detector on selected external datasets.

This is intentionally evaluation-only: it loads the frozen model from
``research_results/latest_benchmark.json`` and never fits a new vectorizer or
classifier.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .research_data import DEFAULT_REGISTRY, load_research_corpus
    from .research_deployment import latest_research_run, load_research_detector
    from .research_metrics import classification_metrics
except ImportError:  # pragma: no cover - direct script compatibility
    from research_data import DEFAULT_REGISTRY, load_research_corpus
    from research_deployment import latest_research_run, load_research_detector
    from research_metrics import classification_metrics


def _predict_scores(model, matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        from scipy.special import expit

        return expit(np.asarray(model.decision_function(matrix), dtype=float))
    raise TypeError("Loaded model does not expose predict_proba or decision_function.")


def evaluate_external(
    dataset_ids: list[str],
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_dir: str | Path | None = None,
) -> dict:
    vectorizer, model, manifest = load_research_detector()
    threshold = float(manifest["threshold"])
    external, data_manifest = load_research_corpus(
        registry_path,
        roles=("external_primary", "external_secondary"),
        deduplicate=False,
    )
    external = external.loc[external["binary_label"].notna()].copy()
    external["binary_label"] = external["binary_label"].astype(int)
    if dataset_ids:
        external = external.loc[external["source"].isin(dataset_ids)].copy()
    if external.empty:
        raise ValueError(f"No external rows found for dataset ids: {dataset_ids}")

    started = time.perf_counter()
    matrix = vectorizer.transform(external["research_text"])
    scores = _predict_scores(model, matrix)
    predict_seconds = time.perf_counter() - started

    metrics_rows = []
    prediction_rows = []
    for dataset_id, frame in external.assign(score=scores).groupby("source", sort=True):
        for variant, variant_frame in (
            ("all", frame),
            ("unique_template", frame.drop_duplicates("template_group", keep="first")),
        ):
            metrics = classification_metrics(
                variant_frame["binary_label"].to_numpy(),
                variant_frame["score"].to_numpy(),
                threshold,
            )
            metrics.update(
                {
                    "dataset": dataset_id,
                    "variant": variant,
                    "model": manifest["model"],
                    "run_id": manifest["run_id"],
                    "predict_seconds_total": predict_seconds,
                    "milliseconds_per_email_total": 1000.0 * predict_seconds / max(1, len(external)),
                }
            )
            metrics_rows.append(metrics)
        prediction_rows.append(
            frame[
                ["source", "message_id", "template_group", "threat_type", "binary_label", "score"]
            ].rename(columns={"source": "dataset", "binary_label": "y_true"})
        )

    run_dir = latest_research_run()
    out = Path(output_dir) if output_dir else run_dir / "external_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(dataset_ids) if dataset_ids else "all_external"
    metrics_path = out / f"{suffix}_metrics.csv"
    predictions_path = out / f"{suffix}_predictions.csv"
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8")
    pd.concat(prediction_rows, ignore_index=True).to_csv(predictions_path, index=False, encoding="utf-8")
    report = {
        "run_id": manifest["run_id"],
        "model": manifest["model"],
        "threshold": threshold,
        "data_manifest": data_manifest,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics_rows,
    }
    (out / f"{suffix}_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen research detector on external datasets.")
    parser.add_argument("--datasets", default="major_corpus_2026", help="Comma-separated external dataset ids.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    dataset_ids = [item.strip() for item in args.datasets.split(",") if item.strip()]
    report = evaluate_external(dataset_ids, args.registry, args.output_dir)
    print(json.dumps({"run_id": report["run_id"], "metrics_path": report["metrics_path"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
