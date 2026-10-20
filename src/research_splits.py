"""Leakage-resistant partitions for the hybrid paper protocol.

The hybrid stack needs four genuinely separate roles: fitting the base text
model, fitting the fusion model, selecting operating thresholds, and reporting
the locked internal result.  All partitions are created by normalized template
group, and exact normalized content is checked as a second line of defence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_SEED = 20260715
PARTITION_NAMES = ("ml_train", "fusion_train", "threshold_selection", "locked_internal_test")


@dataclass(frozen=True)
class HybridPartitions:
    ml_train: pd.DataFrame
    fusion_train: pd.DataFrame
    threshold_selection: pd.DataFrame
    locked_internal_test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in PARTITION_NAMES}


def _take_validation_fold(frame: pd.DataFrame, splits: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    train_index, validation_index = next(
        splitter.split(frame, frame["binary_label"], groups=frame["template_group"])
    )
    return (
        frame.iloc[train_index].reset_index(drop=True),
        frame.iloc[validation_index].reset_index(drop=True),
    )


def split_hybrid_partitions(frame: pd.DataFrame, seed: int = DEFAULT_SEED) -> HybridPartitions:
    """Create approximately 55/15/15/15 grouped partitions.

    Nested grouped folds are deliberately used instead of row-level random
    sampling.  Exact percentages can vary with campaign size; isolation takes
    precedence over forcing a row-perfect ratio.
    """

    required = {"binary_label", "template_group", "content_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Hybrid split input is missing columns: {sorted(missing)}")
    if frame.empty or frame["binary_label"].nunique() < 2:
        raise ValueError("Hybrid split requires a non-empty binary-labelled corpus.")

    remaining, locked_test = _take_validation_fold(frame.reset_index(drop=True), 7, seed)
    remaining, threshold = _take_validation_fold(remaining, 6, seed + 1)
    ml_train, fusion_train = _take_validation_fold(remaining, 5, seed + 2)
    partitions = HybridPartitions(ml_train, fusion_train, threshold, locked_test)
    integrity = partition_integrity(partitions.as_dict())
    if not integrity["passed"]:
        raise ValueError(f"Hybrid partition integrity failed: {integrity['violations']}")
    return partitions


def partition_integrity(partitions: Mapping[str, pd.DataFrame]) -> dict:
    """Return pairwise template/content overlap and label coverage checks."""

    names = list(partitions)
    overlaps: dict[str, dict[str, int]] = {}
    violations: list[str] = []
    for index, left_name in enumerate(names):
        left = partitions[left_name]
        if left.empty:
            violations.append(f"{left_name} is empty")
        if "binary_label" not in left or left["binary_label"].nunique() < 2:
            violations.append(f"{left_name} does not contain both labels")
        for right_name in names[index + 1 :]:
            right = partitions[right_name]
            template_overlap = len(set(left["template_group"]) & set(right["template_group"]))
            content_overlap = len(set(left["content_hash"]) & set(right["content_hash"]))
            key = f"{left_name}__{right_name}"
            overlaps[key] = {
                "template_group_overlap": int(template_overlap),
                "content_hash_overlap": int(content_overlap),
            }
            if template_overlap:
                violations.append(f"{key} has {template_overlap} overlapping template groups")
            if content_overlap:
                violations.append(f"{key} has {content_overlap} overlapping content hashes")

    return {
        "passed": not violations,
        "partition_rows": {name: int(len(value)) for name, value in partitions.items()},
        "partition_groups": {
            name: int(value["template_group"].nunique()) for name, value in partitions.items()
        },
        "pairwise_overlaps": overlaps,
        "violations": violations,
    }


def partition_manifest(partitions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the immutable row-to-partition mapping written with each run."""

    frames = []
    for name, frame in partitions.items():
        columns = [column for column in ("message_id", "content_hash", "template_group", "source", "binary_label") if column in frame]
        selected = frame[columns].copy()
        selected.insert(0, "partition", name)
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)
