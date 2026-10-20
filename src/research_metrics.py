"""Metrics and uncertainty estimates for phishing research experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def probability_like(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        return expit(values)
    return np.clip(values, 0.0, 1.0)


def tune_threshold(y_true, scores, beta: float = 2.0) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = probability_like(scores)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5, 0.0
    beta_squared = beta * beta
    denominator = beta_squared * precision[:-1] + recall[:-1]
    values = np.divide(
        (1.0 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_value = float(values.max())
    # Highest threshold among exact ties avoids adding unnecessary positives.
    best_index = int(np.flatnonzero(np.isclose(values, best_value))[-1])
    return float(thresholds[best_index]), best_value


def expected_calibration_error(y_true, scores, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = probability_like(scores)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper_closed = index == bins - 1
        mask = (scores >= boundaries[index]) & (
            scores <= boundaries[index + 1] if upper_closed else scores < boundaries[index + 1]
        )
        if mask.any():
            error += mask.mean() * abs(float(y_true[mask].mean()) - float(scores[mask].mean()))
    return float(error)


def classification_metrics(y_true, scores, threshold: float) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=int)
    scores = probability_like(scores)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "samples": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f2": float(fbeta_score(y_true, predicted, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "mcc": float(matthews_corrcoef(y_true, predicted)) if both_classes else None,
        "average_precision": float(average_precision_score(y_true, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else None,
        "brier_score": float(brier_score_loss(y_true, scores)),
        "ece_10_bin": expected_calibration_error(y_true, scores, bins=10),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def grouped_bootstrap_ci(
    y_true,
    scores,
    groups,
    threshold: float,
    iterations: int = 200,
    seed: int = 20260715,
) -> dict[str, list[float] | int | dict[str, list[float]]]:
    """Bootstrap template groups and report effective replicate counts.

    One-class replicates cannot support ranking metrics.  Older versions
    silently discarded them, which made the reported interval impossible to
    audit.  The flat metric keys are retained for backwards compatibility,
    while ``intervals`` and the replicate counters form the paper-grade
    contract.
    """

    y_true = np.asarray(y_true, dtype=int)
    scores = probability_like(scores)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    requested = max(0, int(iterations))
    collected: dict[str, list[float]] = {
        "average_precision": [],
        "recall": [],
        "precision": [],
        "f1": [],
        "f2": [],
        "specificity": [],
    }
    skipped_one_class = 0
    for _ in range(requested):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([indices[group] for group in selected])
        if len(np.unique(y_true[sampled])) < 2:
            skipped_one_class += 1
            continue
        metrics = classification_metrics(y_true[sampled], scores[sampled], threshold)
        for name in collected:
            value = metrics.get(name)
            if value is not None:
                collected[name].append(float(value))
    intervals = {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in collected.items()
        if values
    }
    return {
        **intervals,
        "requested_iterations": requested,
        "valid_iterations": requested - skipped_one_class,
        "skipped_one_class": skipped_one_class,
        "intervals": intervals,
    }


def grouped_paired_bootstrap_delta_ci(
    y_true,
    reference_scores,
    candidate_scores,
    groups,
    reference_threshold: float,
    candidate_threshold: float,
    iterations: int = 200,
    seed: int = 20260715,
) -> dict:
    """Return paired candidate-minus-reference metric intervals by group.

    Both systems are resampled on the same template groups.  This is the
    correct uncertainty unit for claims about an ablation delta on a matched
    cohort; overlapping independent intervals are not a paired test.
    """

    y_true = np.asarray(y_true, dtype=int)
    reference_scores = probability_like(reference_scores)
    candidate_scores = probability_like(candidate_scores)
    groups = np.asarray(groups, dtype=str)
    if not (
        len(y_true) == len(reference_scores) == len(candidate_scores) == len(groups)
    ):
        raise ValueError("Paired bootstrap arrays must have equal length.")
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    requested = max(0, int(iterations))
    names = ("average_precision", "roc_auc", "precision", "recall", "f1", "f2", "specificity")
    collected: dict[str, list[float]] = {name: [] for name in names}
    skipped_one_class = 0
    for _ in range(requested):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([indices[group] for group in selected])
        if len(np.unique(y_true[sampled])) < 2:
            skipped_one_class += 1
            continue
        reference = classification_metrics(
            y_true[sampled], reference_scores[sampled], reference_threshold
        )
        candidate = classification_metrics(
            y_true[sampled], candidate_scores[sampled], candidate_threshold
        )
        for name in names:
            reference_value = reference.get(name)
            candidate_value = candidate.get(name)
            if reference_value is not None and candidate_value is not None:
                collected[name].append(float(candidate_value) - float(reference_value))
    intervals = {
        name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for name, values in collected.items()
        if values
    }
    return {
        "requested_iterations": requested,
        "valid_iterations": requested - skipped_one_class,
        "skipped_one_class": skipped_one_class,
        "intervals": intervals,
    }
