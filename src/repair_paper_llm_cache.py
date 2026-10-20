"""Repair schema-truncated rows in a frozen paper Local-LLM cache.

This is a transport-level protocol amendment, not a second model or prompt.
It is allowed to operate only on content hashes still absent after the
registered cache run.  The model, model digest, prompt digest, runtime, and
inference identity must match the existing cache exactly.  Raw email text is
never written to the amendment manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import requests

from .deployment_validator import sha256_file
from .llm_benchmark import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    _extract_json,
    _load_prepared_candidates,
    _prompt,
    _score,
    prompt_sha256,
)
from .local_llm import request_local_llm
from .research_data import DEFAULT_REGISTRY


# A second, deterministic prefix schedule is used only after the original
# 1500/750-character amendment has failed.  Bonsai Q1_0 occasionally copies a
# delimiter into an evidence string until the completion budget is exhausted;
# shorter prefixes prevent that transport failure without consulting labels or
# another model.  Every schedule is retained in the amendment event history.
REPAIR_SCHEDULE = ((40, 800),)
IDENTITY_COLUMNS = (
    "paper_run_id",
    "model",
    "model_file_sha256",
    "prompt_version",
    "prompt_sha256",
    "runtime_version",
    "backend_version",
    "inference_config_sha256",
)


def _request_repair(
    endpoint: str,
    model: str,
    email_text: str,
    timeout: float,
) -> tuple[dict, int, int]:
    last_error: Exception | None = None
    for max_chars, max_tokens in REPAIR_SCHEDULE:
        try:
            response = request_local_llm(
                "POST",
                endpoint,
                json={
                    "model": model,
                    "messages": _prompt(email_text, max_chars=max_chars),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    "reasoning_effort": "none",
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "email_phishing_classification",
                            "strict": True,
                            "schema": RESPONSE_SCHEMA,
                        },
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            parsed = _extract_json(payload["choices"][0]["message"].get("content", ""))
            return parsed, max_chars, max_tokens
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError(f"Local-LLM repair failed after registered amendment retries: {last_error}")


def repair(
    prepared_run: str | Path,
    cache_path: str | Path,
    model_file: str | Path,
    *,
    registry: str | Path = DEFAULT_REGISTRY,
    endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
    timeout: float = 120.0,
) -> dict:
    run_dir = Path(prepared_run)
    if run_dir.is_file():
        run_dir = run_dir.parent
    cache_path = Path(cache_path)
    cache = pd.read_csv(cache_path)
    if cache.empty:
        raise ValueError("Repair requires a non-empty identity-bound cache.")

    identity: dict[str, str] = {}
    for column in IDENTITY_COLUMNS:
        if column not in cache:
            raise ValueError(f"Cache lacks identity column: {column}")
        values = set(cache[column].dropna().astype(str))
        if len(values) != 1:
            raise ValueError(f"Cache identity is mixed for {column}: {sorted(values)}")
        identity[column] = next(iter(values))
    if identity["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Repair prompt version differs from the existing cache.")
    if identity["prompt_sha256"] != prompt_sha256():
        raise ValueError("Repair prompt digest differs from the existing cache.")
    if identity["model_file_sha256"] != sha256_file(model_file):
        raise ValueError("Repair model-file digest differs from the existing cache.")

    candidates, prepared = _load_prepared_candidates(run_dir, registry)
    if identity["paper_run_id"] != str(prepared.get("paper_run_id", "")):
        raise ValueError("Cache paper-run identity differs from the prepared run.")
    complete_hashes = set(cache["content_hash"].astype(str))
    missing = candidates.loc[
        ~candidates["content_hash"].astype(str).isin(complete_hashes)
    ].copy()

    cache_before_sha256 = sha256_file(cache_path)
    amendment_path = run_dir / "llm_cache_amendment.json"
    previous_amendment = (
        json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment_path.exists()
        else None
    )
    repaired: list[dict] = []
    failures: list[dict] = []
    for _, row in missing.iterrows():
        started = time.perf_counter()
        try:
            parsed, max_chars, max_tokens = _request_repair(
                endpoint,
                identity["model"],
                row["raw_text"],
                timeout,
            )
            label, probability, evidence = _score(parsed)
            repaired.append(
                {
                    "content_hash": str(row["content_hash"]),
                    "llm_probability": probability,
                    "llm_label": label,
                    "llm_confidence": float(parsed["confidence"]),
                    "abstain": False,
                    "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
                    "evaluation_partition": row["evaluation_partition"],
                    "source": row["source"],
                    "binary_label": int(row["binary_label"]),
                    "latency_seconds": time.perf_counter() - started,
                    **identity,
                    "repair_max_chars": max_chars,
                    "repair_max_tokens": max_tokens,
                }
            )
        except Exception as exc:  # preserve all non-sensitive failure evidence
            failures.append(
                {
                    "content_hash": str(row["content_hash"]),
                    "error": str(exc),
                }
            )
        print(f"repair {len(repaired) + len(failures)}/{len(missing)}", flush=True)

    if repaired:
        cache = pd.concat([cache, pd.DataFrame(repaired)], ignore_index=True)
        cache = cache.drop_duplicates(
            ["content_hash", "model", "prompt_version"], keep="last"
        )
        cache.to_csv(cache_path, index=False, encoding="utf-8")

    final_hashes = set(cache["content_hash"].astype(str))
    cache_after_sha256 = sha256_file(cache_path)
    event = {
        "schedule": [
            {"max_chars": chars, "max_tokens": tokens}
            for chars, tokens in REPAIR_SCHEDULE
        ],
        "cache_before_sha256": cache_before_sha256,
        "missing_before": int(len(missing)),
        "repaired": int(len(repaired)),
        "failures": failures,
        "valid_rows_after": int(len(final_hashes)),
        "cache_after_sha256": cache_after_sha256,
        "repair_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    prior_events: list[dict] = []
    if previous_amendment:
        if previous_amendment.get("events"):
            prior_events = list(previous_amendment["events"])
        else:
            prior_events = [
                {
                    key: previous_amendment[key]
                    for key in (
                        "schedule",
                        "cache_before_sha256",
                        "missing_before",
                        "repaired",
                        "failures",
                        "valid_rows_after",
                        "cache_after_sha256",
                        "repair_source_sha256",
                    )
                    if key in previous_amendment
                }
            ]
        if prior_events and prior_events[-1].get("cache_after_sha256") != cache_before_sha256:
            raise ValueError("Existing amendment history does not match the current cache hash.")
    events = [*prior_events, event]
    missing_initial = int(events[0].get("missing_before", len(missing)))
    repaired_total = sum(int(item.get("repaired", 0)) for item in events)
    amendment = {
        "schema_version": 2,
        "amendment": "schema-truncation-repair-only",
        "paper_run_id": identity["paper_run_id"],
        "model": identity["model"],
        "model_file_sha256": identity["model_file_sha256"],
        "prompt_version": identity["prompt_version"],
        "prompt_sha256": identity["prompt_sha256"],
        "inference_config_sha256": identity["inference_config_sha256"],
        "events": events,
        "schedule": event["schedule"],
        "cache_before_sha256": events[0].get("cache_before_sha256"),
        "missing_before": missing_initial,
        "repaired": repaired_total,
        "failures": failures,
        "valid_rows_after": int(len(final_hashes)),
        "complete_registered_cohort": final_hashes
        == set(candidates["content_hash"].astype(str)),
        "cache_after_sha256": cache_after_sha256,
        "repair_source_sha256": event["repair_source_sha256"],
        "transport_only": True,
        "ground_truth_consulted": False,
        "alternate_model_used": False,
        "raw_text_persisted": False,
    }
    amendment_path.write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return amendment


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing frozen paper LLM rows")
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    result = repair(
        args.prepared_run,
        args.cache,
        args.model_file,
        registry=args.registry,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=True))
    if not result["complete_registered_cohort"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
