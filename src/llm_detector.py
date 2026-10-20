"""Failure-aware Local-LLM result contract for hybrid detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any
import unicodedata


LLM_OUTPUT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LLMDetection:
    label: str
    phishing_probability: float | None
    confidence: float | None
    abstain: bool
    signals: tuple[str, ...]
    reason: str
    error: str | None = None

    @property
    def available(self) -> bool:
        return not self.abstain and self.phishing_probability is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"available": self.available, "schema_version": LLM_OUTPUT_SCHEMA_VERSION}


def abstained_detection(error: object) -> LLMDetection:
    return LLMDetection("UNKNOWN", None, None, True, (), "", str(error))


def _strict_boolean(value: object, default: bool) -> bool:
    """Accept only JSON booleans for safety-critical LLM fields.

    ``bool("false")`` is true in Python.  Treating a model's string output as
    a boolean could therefore silently convert a usable result into an
    abstention (or the opposite in future code).  Invalid schema is safer than
    coercion for a security verdict.
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("LLM abstain must be a JSON boolean")


def _canonical_json_label(value: object) -> str:
    """Accept only exact, schema-level English/Vietnamese label equivalents.

    Local models selected for Vietnamese analysis sometimes localise the JSON
    label even when instructed otherwise.  This accepts that bounded form
    without accepting prose such as ``this looks safe`` as a security verdict.
    """

    label = str(value or "").strip().upper()
    english = {
        "PHISHING": "PHISHING",
        "SAFE": "SAFE",
        "UNKNOWN": "UNKNOWN",
        "NON_PHISHING": "SAFE",
        "NON-PHISHING": "SAFE",
        "LEGITIMATE": "SAFE",
        "BENIGN": "SAFE",
    }
    if label in english:
        return english[label]
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = normalized.replace("Đ", "D").replace("đ", "d")
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii").strip().upper()
    vietnamese = {
        "LUA DAO": "PHISHING",
        "AN TOAN": "SAFE",
        "KHONG XAC DINH": "UNKNOWN",
    }
    return vietnamese.get(ascii_label, "")


def parse_llm_detection(content: object) -> LLMDetection:
    """Validate structured model output; invalid output always abstains."""

    try:
        if isinstance(content, dict):
            payload = content
        else:
            value = str(content or "").strip()
            value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
            match = re.search(r"\{.*\}", value, flags=re.S)
            if not match:
                raise ValueError("LLM response did not contain a JSON object")
            payload = json.loads(match.group(0))
        label = _canonical_json_label(payload.get("label", ""))
        if label not in {"PHISHING", "SAFE", "UNKNOWN"}:
            raise ValueError("LLM label must be PHISHING, SAFE, or UNKNOWN")
        abstain = _strict_boolean(payload.get("abstain"), label == "UNKNOWN")
        probability_value = payload.get("phishing_probability")
        probability = None if probability_value is None else float(probability_value)
        confidence_value = payload.get("confidence")
        confidence = None if confidence_value is None else float(confidence_value)
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError("LLM phishing_probability outside [0,1]")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("LLM confidence outside [0,1]")
        if abstain or label == "UNKNOWN" or probability is None:
            return abstained_detection(payload.get("error") or "LLM abstained")
        expected = "PHISHING" if probability >= 0.5 else "SAFE"
        if label != expected:
            raise ValueError("LLM label conflicts with phishing_probability")
        signals_value = payload.get("signals", ())
        signals = tuple(str(item)[:160] for item in signals_value[:5]) if isinstance(signals_value, list) else ()
        reason = str(payload.get("reason", ""))[:500]
        return LLMDetection(label, probability, confidence, False, signals, reason)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return abstained_detection(error)
