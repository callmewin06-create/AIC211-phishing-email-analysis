"""Freeze the matched Local-LLM cohort used by paper protocol v2.

The base text model is trained on the complete ``ml_train`` partition.  Every
configuration that contains a Local LLM is fitted, thresholded, and evaluated
only on the same pre-registered cohort.  This makes the comparison feasible
on local hardware without pretending that hundreds of thousands of messages
received an LLM score.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping

import pandas as pd


LLM_PARTITIONS = ("fusion_train", "threshold_selection", "locked_internal_test")


def _length_band(value: object) -> str:
    length = len(str(value or ""))
    if length < 500:
        return "short"
    if length < 2_000:
        return "medium"
    return "long"


def _stable_rank(seed: int, *values: object) -> str:
    payload = "\x1f".join((str(seed), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_matched_llm_cohort(
    partitions: Mapping[str, pd.DataFrame],
    external_sets: Mapping[str, pd.DataFrame],
    *,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic, group-isolated, multi-source LLM cohort.

    One representative is retained per template group.  Rows are then drawn
    round-robin from strata defined before inference: partition, source,
    binary label, detected language, and document-length band.  No model score
    or outcome-dependent error signal participates in selection.
    """

    if int(target_size) < 1:
        raise ValueError("target_size must be positive")
    named: dict[str, pd.DataFrame] = {}
    for name in LLM_PARTITIONS:
        if name not in partitions:
            raise ValueError(f"Missing required LLM partition: {name}")
        named[name] = partitions[name]
    named.update(external_sets)

    candidates = []
    required = {"content_hash", "template_group", "source", "binary_label"}
    for partition, source_frame in named.items():
        missing = required - set(source_frame.columns)
        if missing:
            raise ValueError(f"{partition} is missing cohort fields: {sorted(missing)}")
        frame = source_frame.copy()
        frame.insert(0, "evaluation_partition", partition)
        if "language" not in frame:
            frame["language"] = "unknown"
        text_column = "raw_text" if "raw_text" in frame else "research_text"
        frame["length_band"] = frame[text_column].map(_length_band)
        frame["language"] = frame["language"].fillna("unknown").astype(str)
        frame["cohort_stratum"] = (
            frame[["evaluation_partition", "source", "binary_label", "language", "length_band"]]
            .astype(str)
            .agg("|".join, axis=1)
        )
        frame["cohort_rank"] = [
            _stable_rank(seed, partition, template, content)
            for template, content in zip(frame["template_group"], frame["content_hash"])
        ]
        # Template weighting is removed before sampling.  The selected row is
        # stable even if the source file's physical row order changes.
        frame = frame.sort_values("cohort_rank").drop_duplicates("template_group", keep="first")
        candidates.append(frame)

    available = pd.concat(candidates, ignore_index=True)
    limit = min(int(target_size), len(available))
    queues: dict[str, list[int]] = defaultdict(list)
    for stratum, group in available.groupby("cohort_stratum", sort=True, observed=True):
        queues[str(stratum)] = group.sort_values("cohort_rank").index.tolist()

    selected: list[int] = []
    strata = sorted(queues)
    while len(selected) < limit:
        progressed = False
        for stratum in strata:
            if queues[stratum] and len(selected) < limit:
                selected.append(queues[stratum].pop(0))
                progressed = True
        if not progressed:
            break

    result = available.loc[selected].copy()
    result["cohort_order"] = range(1, len(result) + 1)
    output_columns = [
        "cohort_order",
        "evaluation_partition",
        "message_id",
        "content_hash",
        "template_group",
        "source",
        "binary_label",
        "language",
        "length_band",
        "cohort_stratum",
        "cohort_rank",
    ]
    output_columns = [column for column in output_columns if column in result]
    return result[output_columns].reset_index(drop=True)


def cohort_integrity(
    cohort: pd.DataFrame,
    *,
    target_size: int,
    required_partitions: tuple[str, ...] = LLM_PARTITIONS,
) -> dict:
    """Return an auditable cohort gate without any performance criteria."""

    violations: list[str] = []
    if cohort.empty:
        violations.append("cohort is empty")
    if cohort["content_hash"].duplicated().any():
        violations.append("duplicate content hashes")
    if cohort["template_group"].duplicated().any():
        violations.append("duplicate template groups")
    observed = set(cohort["evaluation_partition"].astype(str))
    for partition in required_partitions:
        subset = cohort.loc[cohort["evaluation_partition"].eq(partition)]
        if partition not in observed:
            violations.append(f"missing partition: {partition}")
        elif subset["binary_label"].nunique() < 2:
            violations.append(f"partition lacks both labels: {partition}")
    if len(cohort) < int(target_size):
        violations.append(f"cohort has {len(cohort)} rows; target is {int(target_size)}")
    return {
        "passed": not violations,
        "rows": int(len(cohort)),
        "target_size": int(target_size),
        "partition_rows": {
            str(name): int(count)
            for name, count in cohort["evaluation_partition"].value_counts().sort_index().items()
        },
        "partition_label_rows": {
            f"{partition}|{int(label)}": int(count)
            for (partition, label), count in cohort.groupby(
                ["evaluation_partition", "binary_label"], observed=True
            ).size().sort_index().items()
        },
        "violations": violations,
    }
