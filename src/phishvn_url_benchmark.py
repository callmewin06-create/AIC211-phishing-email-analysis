"""Reproducible, non-promoted benchmark for the PhishVN URL-IOC model.

The benchmark consumes only the PhishVN v2 gold/silver URL core.  It validates
the release manifest, trains only on the official train partition, calibrates a
review threshold from group-disjoint folds within that partition, and reports
official validation/test results without using either set to choose a model or
threshold.  Bronze rows are intentionally excluded from every promotion
metric: the bronze test partition contains phishing-only rows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from .artifact_signing import sign_manifest
    from .deployment_validator import sha256_file
    from .url_ioc_model import (
        FEATURE_CONTRACT_VERSION,
        PHISHVN_VN_SECOND_LEVEL_SUFFIXES,
        URL_FEATURE_COLUMNS,
        URL_IOC_SIGNING_ENV,
        extract_url_features,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from artifact_signing import sign_manifest  # type: ignore[import-not-found]
    from deployment_validator import sha256_file  # type: ignore[import-not-found]
    from url_ioc_model import (  # type: ignore[import-not-found]
        FEATURE_CONTRACT_VERSION,
        PHISHVN_VN_SECOND_LEVEL_SUFFIXES,
        URL_FEATURE_COLUMNS,
        URL_IOC_SIGNING_ENV,
        extract_url_features,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHISHVN_ROOT = ROOT / "data" / "external" / "phishvn_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "research_results"
CORE_TIERS = ("gold", "silver")
EXPECTED_CORE_COUNTS = {
    "train": {"samples": 12_466, "phishing": 2_077, "benign": 10_389},
    "val": {"samples": 2_157, "phishing": 23, "benign": 2_134},
    "test": {"samples": 2_456, "phishing": 488, "benign": 1_968},
}


class PhishVNValidationError(ValueError):
    """Raised for an incomplete, corrupted, or leakage-prone PhishVN release."""


def _read_manifest(dataset_root: Path) -> dict[str, dict[str, str | int]]:
    manifest_path = dataset_root / "MANIFEST.txt"
    if not manifest_path.is_file():
        raise PhishVNValidationError("PhishVN MANIFEST.txt is missing")
    records: dict[str, dict[str, str | int]] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PhishVN"):
            continue
        fields = line.split()
        if len(fields) < 3 or len(fields[0]) != 64:
            continue
        records[fields[1]] = {"sha256": fields[0].casefold(), "bytes": int(fields[2].strip("()"))}
    if "data/vn_compphish.csv" not in records:
        raise PhishVNValidationError("PhishVN manifest lacks vn_compphish.csv")
    return records


def verify_phishvn_source(dataset_root: str | Path = DEFAULT_PHISHVN_ROOT) -> dict[str, Any]:
    """Verify every listed v2 open-tier file before deriving model input."""

    root = Path(dataset_root).resolve()
    records = _read_manifest(root)
    checked: dict[str, dict[str, Any]] = {}
    for relative, expected in records.items():
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise PhishVNValidationError("Dataset manifest path escapes dataset root") from error
        if not target.is_file():
            raise PhishVNValidationError(f"PhishVN source file is missing: {relative}")
        actual_size = target.stat().st_size
        actual_hash = sha256_file(target).casefold()
        if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
            raise PhishVNValidationError(f"PhishVN source file failed integrity check: {relative}")
        checked[relative] = {"sha256": actual_hash, "bytes": actual_size}
    return {"dataset_root": str(root), "files": checked}


def _assert_feature_parity(frame: pd.DataFrame) -> None:
    """Reject feature code drift against a stable sample from the release."""

    parity_sample = pd.concat(
        [frame.loc[frame["split"].eq(name)].head(64) for name in ("train", "val", "test")],
        ignore_index=True,
    )
    for row in parity_sample.itertuples(index=False):
        observed = extract_url_features(row.url)
        for column in URL_FEATURE_COLUMNS:
            if not np.isclose(float(observed[column]), float(getattr(row, column)), atol=1e-10, rtol=0.0):
                raise PhishVNValidationError(f"Runtime feature mismatch for {column}; feature contract drifted")


def load_phishvn_core(dataset_root: str | Path = DEFAULT_PHISHVN_ROOT) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load verified official splits, excluding bronze and checking group isolation."""

    fingerprint = verify_phishvn_source(dataset_root)
    root = Path(dataset_root).resolve()
    path = root / "data" / "vn_compphish.csv"
    frame = pd.read_csv(path)
    required = {"url", "dom", "label", "split", "tier", "is_https", *URL_FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PhishVNValidationError(f"PhishVN feature table is missing: {', '.join(missing)}")
    core = frame.loc[frame["tier"].isin(CORE_TIERS)].copy()
    core["label"] = pd.to_numeric(core["label"], errors="raise").astype(int)
    if not set(core["label"].unique()).issubset({0, 1}):
        raise PhishVNValidationError("PhishVN labels are not binary")
    numeric = core.loc[:, URL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PhishVNValidationError("PhishVN feature values are non-finite")
    core.loc[:, URL_FEATURE_COLUMNS] = numeric
    _assert_feature_parity(core)
    splits = {name: core.loc[core["split"].eq(name)].copy() for name in ("train", "val", "test")}
    for name, expected in EXPECTED_CORE_COUNTS.items():
        split = splits[name]
        actual = {"samples": len(split), "phishing": int(split["label"].sum()), "benign": int((split["label"] == 0).sum())}
        if actual != expected:
            raise PhishVNValidationError(f"Unexpected PhishVN {name} core counts: {actual}")
        if split["label"].nunique() != 2:
            raise PhishVNValidationError(f"PhishVN {name} split lacks a class")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[left]["dom"].str.casefold()).intersection(splits[right]["dom"].str.casefold())
        if overlap:
            raise PhishVNValidationError(f"PhishVN registrable-domain leakage between {left}/{right}")
    return splits, fingerprint


def _f2(precision: float, recall: float) -> float:
    denominator = 4.0 * precision + recall
    return float(5.0 * precision * recall / denominator) if denominator else 0.0


def metrics_at_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(int)
    positive = labels == 1
    true_positive = int(np.sum((predicted == 1) & positive))
    false_positive = int(np.sum((predicted == 1) & ~positive))
    true_negative = int(np.sum((predicted == 0) & ~positive))
    false_negative = int(np.sum((predicted == 0) & positive))
    precision = float(precision_score(labels, predicted, zero_division=0))
    recall = float(recall_score(labels, predicted, zero_division=0))
    return {
        "samples": int(labels.size),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "precision": precision,
        "recall": recall,
        "specificity": float(true_negative / (true_negative + false_positive)),
        "f2": _f2(precision, recall),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "true_negatives": true_negative,
        "false_negatives": false_negative,
        "alerts": int(predicted.sum()),
    }


def select_review_threshold(labels: np.ndarray, probabilities: np.ndarray, *, precision_floor: float = 0.95) -> tuple[float, dict[str, float | int]]:
    """Pick the highest-recall train-only threshold meeting a review precision floor."""

    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    eligible: list[tuple[float, dict[str, float | int]]] = []
    for threshold in candidates:
        metric = metrics_at_threshold(labels, probabilities, float(threshold))
        if metric["precision"] >= precision_floor and metric["alerts"] > 0:
            eligible.append((float(threshold), metric))
    if not eligible:
        raise PhishVNValidationError("No group-calibrated threshold met the required review precision")
    return max(eligible, key=lambda item: (float(item[1]["recall"]), -item[0]))


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=15,
        l2_regularization=0.2,
        random_state=20_260_811,
    )


def _group_oof_probabilities(train: pd.DataFrame) -> np.ndarray:
    features = train.loc[:, URL_FEATURE_COLUMNS]
    labels = train["label"].to_numpy()
    groups = train["dom"].astype(str).str.casefold().to_numpy()
    probabilities = np.full(len(train), np.nan, dtype=float)
    for fit_index, calibration_index in GroupKFold(n_splits=5).split(features, labels, groups):
        model = _new_model()
        model.fit(features.iloc[fit_index], labels[fit_index])
        probabilities[calibration_index] = model.predict_proba(features.iloc[calibration_index])[:, 1]
    if not np.isfinite(probabilities).all():
        raise PhishVNValidationError("Group calibration did not produce a probability for every train row")
    return probabilities


def _json_dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def train_phishvn_url_ioc(
    dataset_root: str | Path = DEFAULT_PHISHVN_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Create a signed-if-keyed research bundle that is deliberately not promoted."""

    splits, fingerprint = load_phishvn_core(dataset_root)
    train, validation, test = splits["train"], splits["val"], splits["test"]
    train_labels = train["label"].to_numpy()
    oof_probabilities = _group_oof_probabilities(train)
    threshold, calibration_metrics = select_review_threshold(train_labels, oof_probabilities)
    model = _new_model()
    model.fit(train.loc[:, URL_FEATURE_COLUMNS], train_labels)
    probabilities = {
        "official_validation": model.predict_proba(validation.loc[:, URL_FEATURE_COLUMNS])[:, 1],
        "official_locked_test": model.predict_proba(test.loc[:, URL_FEATURE_COLUMNS])[:, 1],
    }
    run_id = f"url-phishvn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, run_dir / "url_model.joblib")
    contract = {
        "version": FEATURE_CONTRACT_VERSION,
        "columns": list(URL_FEATURE_COLUMNS),
        "excluded": ["is_https"],
        "normalization": "offline scheme-stripped lexical counts; pinned PSL rules; no DNS/HTTP",
        "vn_second_level_suffixes": sorted(PHISHVN_VN_SECOND_LEVEL_SUFFIXES),
    }
    _json_dump(run_dir / "url_feature_contract.json", contract)

    validation_metrics = metrics_at_threshold(validation["label"].to_numpy(), probabilities["official_validation"], threshold)
    test_metrics = metrics_at_threshold(test["label"].to_numpy(), probabilities["official_locked_test"], threshold)
    gate_reasons = [
        "Official validation has only 23 phishing URLs; minimum is 100 for a promotion decision.",
        "This URL-only corpus cannot validate an email phishing/safe classifier or an LLM decision.",
    ]
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "task": "Vietnamese-targeted URL phishing detection; auxiliary URL IOC evidence only",
        "source": {
            "name": "PhishVN v2.0.0 open tier",
            "version": "2.0.0",
            "doi": "10.17632/b97hxbxtpd.2",
            "license": "CC BY 4.0",
            "tiers": list(CORE_TIERS),
            "source_fingerprint": fingerprint,
        },
        "split_policy": {
            "official": "PhishVN group-aware temporal split by registrable domain",
            "calibration": "5-fold GroupKFold on official train only; group=registrable domain",
            "bronze": "excluded from all calibration, primary metrics, and promotion decisions",
        },
        "feature_contract": contract,
        "model": {"name": "HistGradientBoostingClassifier", "parameters": model.get_params()},
        "threshold": threshold,
        "threshold_policy": {"intent": "high-precision analyst review alert", "minimum_oof_precision": 0.95},
        "metrics": {
            "train_group_oof": calibration_metrics,
            "official_validation": validation_metrics,
            "official_locked_test": test_metrics,
        },
        "deployment_gate": {"passed": False, "reasons": gate_reasons},
        "deployment_status": "research_only_not_promoted",
        "limitations": [
            "A link score is an IOC, not an email verdict and not a safe verdict below threshold.",
            "No DNS, HTTP fetch, live reachability check, or webpage-content feature is used.",
            "No URL-IOC score is fused into the email ML, 12-layer matrix, threat label, or LLM decision.",
        ],
    }
    _json_dump(run_dir / "url_benchmark_summary.json", summary)
    artifact_names = ("url_model.joblib", "url_feature_contract.json", "url_benchmark_summary.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "purpose": "url_ioc_artifact",
        "deployment_status": summary["deployment_status"],
        "artifacts": {name: {"sha256": sha256_file(run_dir / name)} for name in artifact_names},
    }
    _json_dump(run_dir / "artifact_manifest.json", sign_manifest(manifest, signing_env=URL_IOC_SIGNING_ENV))
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PhishVN URL-only IOC model without promotion.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_PHISHVN_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    run_dir, summary = train_phishvn_url_ioc(args.dataset_root, args.output_root)
    print(json.dumps({"run_dir": str(run_dir), "run_id": summary["run_id"], "metrics": summary["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
