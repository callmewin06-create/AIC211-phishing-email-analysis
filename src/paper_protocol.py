"""Executable configuration and hashing helpers for paper experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_PROTOCOL = ROOT / "config" / "hybrid_paper_v2.json"


class PaperProtocolError(ValueError):
    """Raised when the frozen paper protocol is incomplete or inconsistent."""


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_paper_protocol(path: str | Path = DEFAULT_PAPER_PROTOCOL) -> dict[str, Any]:
    protocol_path = Path(path)
    config = json.loads(protocol_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise PaperProtocolError("Unsupported paper protocol schema; expected version 2.")
    if config.get("protocol") != "hybrid-paper-v2-frozen-matched-cohort":
        raise PaperProtocolError("Unexpected paper protocol identifier.")
    partitions = config.get("partition_policy", {}).get("names", [])
    expected = ["ml_train", "fusion_train", "threshold_selection", "locked_internal_test"]
    if partitions != expected:
        raise PaperProtocolError("Paper protocol must preserve the four frozen partition roles.")
    bootstrap = config.get("bootstrap", {})
    if int(bootstrap.get("paper_iterations", 0)) < int(bootstrap.get("minimum_iterations", 200)):
        raise PaperProtocolError("Paper bootstrap iterations are below the declared minimum.")
    llm = config.get("llm", {})
    if llm.get("mode") != "offline_local_only":
        raise PaperProtocolError("Paper protocol permits only an offline Local LLM.")
    if int(llm.get("cohort_size", 0)) < 1 or int(llm.get("pilot_size", 0)) < 1:
        raise PaperProtocolError("LLM cohort and pilot sizes must be positive.")
    required_schema = set(llm.get("response_schema", {}).get("required", []))
    expected_schema = {"label", "confidence", "attack_type", "evidence", "recommended_action"}
    if required_schema != expected_schema:
        raise PaperProtocolError("Configured LLM response schema does not match the executable parser.")
    config["config_path"] = str(protocol_path.resolve())
    config["config_sha256"] = canonical_json_sha256(
        {key: value for key, value in config.items() if key not in {"config_path", "config_sha256"}}
    )
    return config
