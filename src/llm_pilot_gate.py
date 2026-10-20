"""Evaluate Local-LLM feasibility before a long paper cohort run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .paper_protocol import DEFAULT_PAPER_PROTOCOL, load_paper_protocol
from .research_metrics import classification_metrics


def evaluate_pilot(
    pilot_path: str | Path,
    repeat_path: str | Path,
    *,
    protocol_path: str | Path = DEFAULT_PAPER_PROTOCOL,
) -> dict:
    protocol = load_paper_protocol(protocol_path)
    llm = protocol["llm"]
    gate = llm["pilot_gate"]
    pilot = pd.read_csv(pilot_path)
    repeat = pd.read_csv(repeat_path)
    requested = int(llm["pilot_size"])
    valid = int(len(pilot))
    valid_fraction = valid / requested
    failure_fraction = (requested - valid) / requested
    paired = pilot.merge(repeat, on="content_hash", suffixes=("_first", "_repeat"))
    label_agreement = float(
        (paired["llm_label_first"] == paired["llm_label_repeat"]).mean()
    ) if len(paired) else 0.0
    probability_ranges = (
        paired["llm_probability_first"] - paired["llm_probability_repeat"]
    ).abs()
    maximum_probability_range = float(probability_ranges.max()) if len(paired) else 1.0
    mean_latency = float(pilot["latency_seconds"].mean())
    estimated_full_hours = mean_latency * int(llm["cohort_size"]) / 3600.0
    metric = classification_metrics(
        pilot["binary_label"], pilot["llm_probability"], threshold=0.5
    )
    checks = [
        {
            "name": "schema_valid_fraction",
            "passed": valid_fraction >= float(gate["minimum_schema_valid_fraction"]),
            "value": valid_fraction,
            "threshold": gate["minimum_schema_valid_fraction"],
        },
        {
            "name": "failure_or_abstain_fraction",
            "passed": failure_fraction <= float(gate["maximum_failure_or_abstain_fraction"]),
            "value": failure_fraction,
            "threshold": gate["maximum_failure_or_abstain_fraction"],
        },
        {
            "name": "repeat_label_agreement",
            "passed": label_agreement >= float(gate["required_repeat_label_agreement"]),
            "value": label_agreement,
            "threshold": gate["required_repeat_label_agreement"],
        },
        {
            "name": "repeat_probability_range",
            "passed": maximum_probability_range <= float(gate["maximum_repeat_probability_range"]),
            "value": maximum_probability_range,
            "threshold": gate["maximum_repeat_probability_range"],
        },
        {
            "name": "estimated_full_runtime",
            "passed": estimated_full_hours <= float(gate["maximum_estimated_full_hours"]),
            "value_hours": estimated_full_hours,
            "threshold_hours": gate["maximum_estimated_full_hours"],
        },
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol["protocol"],
        "protocol_sha256": protocol["config_sha256"],
        "pilot_path": str(Path(pilot_path).resolve()),
        "repeat_path": str(Path(repeat_path).resolve()),
        "requested_rows": requested,
        "valid_rows": valid,
        "failed_or_invalid_rows": requested - valid,
        "repeat_rows": int(len(paired)),
        "latency_seconds": {
            "mean": mean_latency,
            "median": float(pilot["latency_seconds"].median()),
            "p95": float(pilot["latency_seconds"].quantile(0.95)),
            "maximum": float(pilot["latency_seconds"].max()),
        },
        "estimated_full_hours": estimated_full_hours,
        "feasibility_metrics_not_confirmatory": metric,
        "gate": {"passed": all(check["passed"] for check in checks), "checks": checks},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pre-registered Local-LLM pilot gate")
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PAPER_PROTOCOL))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_pilot(args.pilot, args.repeat, protocol_path=args.protocol)
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"gate": report["gate"], "estimated_full_hours": report["estimated_full_hours"]}))


if __name__ == "__main__":
    main()
