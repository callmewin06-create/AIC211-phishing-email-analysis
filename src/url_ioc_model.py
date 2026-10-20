"""Offline PhishVN URL IOC inference with a fail-closed artifact boundary.

This module intentionally returns *URL evidence*, never an email phishing/safe
verdict.  It has no DNS, HTTP, or reputation-network calls.  A URL model can
become available only through a signed, promoted URL-IOC deployment pointer;
an absent, invalid, or unpromoted artifact remains ``UNAVAILABLE``.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd
import tldextract

try:
    from .artifact_signing import ArtifactSignatureError, verify_manifest_signature
    from .deployment_validator import sha256_file
except ImportError:  # pragma: no cover - direct script compatibility
    from artifact_signing import ArtifactSignatureError, verify_manifest_signature  # type: ignore[import-not-found]
    from deployment_validator import sha256_file  # type: ignore[import-not-found]


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "research_results"
URL_IOC_POINTER_NAME = "latest_url_ioc.json"
URL_IOC_SIGNING_ENV = "AIC211_URL_IOC_SIGNING_KEY"
FEATURE_CONTRACT_VERSION = "phishvn_url_lexical_v1"

# PhishVN's exported CompPhish schema.  ``is_https`` must never be used: the
# datasheet records it as a collection artefact, not a phishing signal.
URL_FEATURE_COLUMNS = (
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len",
)

# The source release treats the following generic Vietnamese suffixes as
# registrable-domain boundaries.  Provincial ``*.vn`` names stay registrable
# domains in the PhishVN feature contract, even if a newer PSL snapshot lists
# them as a delegated suffix.  Pinning this rule prevents a future PSL update
# from silently changing model input features.
PHISHVN_VN_SECOND_LEVEL_SUFFIXES = frozenset(
    {"biz.vn", "com.vn", "edu.vn", "gov.vn", "info.vn", "net.vn", "org.vn", "pro.vn"}
)

# PhishVN v2's feature export was made against an earlier PSL snapshot.  These
# two release-specific rules close the remaining snapshot drift verified
# against every released gold/silver feature row.  They belong to the feature
# contract, not to the learned model or its labels.
PHISHVN_SUFFIX_REWRITES = {"bet.ar": "ar", "notaires.fr": "notaires.fr"}

# ``suffix_list_urls=()`` makes tldextract use its bundled snapshot only; it
# must not fetch a new PSL or create a network dependency at inference time.
_PSL_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(), include_psl_private_domains=False
)


class URLIocArtifactError(RuntimeError):
    """Raised when a URL-IOC artifact does not meet its trust contract."""


@dataclass(frozen=True)
class URLIocDetector:
    pipeline: Any
    threshold: float
    run_id: str
    source: str
    version: str
    run_dir: Path


def normalize_url(value: object) -> str:
    """Return an offline, scheme-normalised URL or ``""`` for unusable input."""

    raw = str(value or "").strip()
    if not raw or any(character.isspace() for character in raw):
        return ""
    candidate = raw if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.IGNORECASE) else f"http://{raw}"
    parsed = urlparse(candidate)
    if not parsed.hostname:
        return ""
    return candidate


def _ascii_host(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii").casefold().rstrip(".")
    except UnicodeError:
        return host.casefold().rstrip(".")


def _domain_parts(host: str) -> tuple[str, str, int]:
    """Match the released PhishVN registrable-domain feature convention."""

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host, "", 0

    extracted = _PSL_EXTRACTOR(host)
    suffix = extracted.suffix.casefold()
    labels = [label for label in host.split(".") if label]
    if host.endswith(".vn") and suffix not in PHISHVN_VN_SECOND_LEVEL_SUFFIXES:
        suffix = "vn"
    for observed_suffix, contract_suffix in PHISHVN_SUFFIX_REWRITES.items():
        if host == observed_suffix or host.endswith(f".{observed_suffix}"):
            suffix = contract_suffix
            break

    # Unknown TLDs are represented by the source table as a host-domain with
    # zero-length suffix, rather than treating the final label as a valid TLD.
    if not suffix:
        return host, "", max(0, len(labels) - 1)

    suffix_labels = len(suffix.split("."))
    domain_labels = suffix_labels + 1
    if len(labels) < domain_labels:
        return host, suffix, 0
    domain = ".".join(labels[-domain_labels:])
    return domain, suffix, max(0, len(labels) - domain_labels)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    length = float(len(value))
    return float(-sum((count / length) * math.log2(count / length) for count in Counter(value).values()))


def extract_url_features(url: object) -> dict[str, float]:
    """Produce the 21 PhishVN lexical fields in their fixed order.

    The function is deterministic and strictly offline.  It deliberately does
    not expose or compute ``is_https`` so the collection artefact cannot enter
    a future model by accident.
    """

    candidate = normalize_url(url)
    if not candidate:
        return {name: 0.0 for name in URL_FEATURE_COLUMNS}
    parsed = urlparse(candidate)
    host = _ascii_host(parsed.hostname or "")
    domain, suffix, subdomain_count = _domain_parts(host)
    try:
        ipaddress.ip_address(host)
        is_ip = 1.0
    except ValueError:
        is_ip = 0.0

    # The dataset strips only scheme, lower-cases the remaining URL, and
    # counts its original punctuation.  Do not URL-decode or query the host.
    scheme_stripped = candidate.split("://", 1)[-1].casefold()
    length = len(scheme_stripped)
    letter_count = sum(character.isalpha() for character in scheme_stripped)
    digit_count = sum(character.isdigit() for character in scheme_stripped)
    special_count = max(0, length - letter_count - digit_count)
    path = parsed.path.lstrip("/")
    return {
        "url_len": float(length),
        "dom_len": float(len(domain)),
        "is_ip": is_ip,
        "tld_len": float(len(suffix)),
        "subdom_cnt": float(subdomain_count),
        "letter_cnt": float(letter_count),
        "digit_cnt": float(digit_count),
        "special_cnt": float(special_count),
        "eq_cnt": float(scheme_stripped.count("=")),
        "qm_cnt": float(scheme_stripped.count("?")),
        "amp_cnt": float(scheme_stripped.count("&")),
        "dot_cnt": float(scheme_stripped.count(".")),
        "dash_cnt": float(scheme_stripped.count("-")),
        "under_cnt": float(scheme_stripped.count("_")),
        "letter_ratio": float(letter_count / length) if length else 0.0,
        "digit_ratio": float(digit_count / length) if length else 0.0,
        "spec_ratio": float(special_count / length) if length else 0.0,
        "slash_cnt": float(scheme_stripped.count("/")),
        "entropy": _entropy(scheme_stripped),
        "path_len": float(len(path)),
        "query_len": float(len(parsed.query)),
    }


def feature_frame(url: object) -> pd.DataFrame:
    """Build one exact-order feature row for a fitted URL pipeline."""

    return pd.DataFrame([[extract_url_features(url)[name] for name in URL_FEATURE_COLUMNS]], columns=URL_FEATURE_COLUMNS)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise URLIocArtifactError(f"Cannot read URL IOC metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise URLIocArtifactError(f"URL IOC metadata is not an object: {path.name}")
    return value


def _safe_run_dir(pointer: dict[str, Any], results_root: Path) -> Path:
    run_id = str(pointer.get("run_id", "")).strip()
    candidate = Path(str(pointer.get("path", run_id)))
    if not candidate.is_absolute():
        candidate = results_root.parent / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(results_root.resolve())
    except ValueError as error:
        raise URLIocArtifactError("URL IOC pointer escapes research_results") from error
    if not run_id or candidate.name != run_id or not candidate.is_dir():
        raise URLIocArtifactError("URL IOC pointer does not name an existing run")
    return candidate


def _verify_signed(metadata: dict[str, Any]) -> None:
    signature = metadata.get("signature")
    if not isinstance(signature, dict):
        raise URLIocArtifactError("URL IOC promoted metadata is unsigned")
    try:
        verify_manifest_signature(metadata, signing_env=URL_IOC_SIGNING_ENV)
    except ArtifactSignatureError as error:
        raise URLIocArtifactError(f"URL IOC signature verification failed: {error}") from error


def _artifact_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    items = manifest.get("artifacts")
    if not isinstance(items, dict):
        raise URLIocArtifactError("URL IOC artifact manifest has no artifact hashes")
    hashes: dict[str, str] = {}
    for name, item in items.items():
        digest = item.get("sha256") if isinstance(item, dict) else item
        if not isinstance(digest, str) or len(digest) != 64:
            raise URLIocArtifactError(f"URL IOC artifact hash is invalid: {name}")
        hashes[str(name)] = digest
    return hashes


def validate_url_ioc_run(run_dir: str | Path, *, require_promoted: bool = True) -> dict[str, Any]:
    """Verify gate status, signature, contract, and hashes before Joblib load."""

    resolved = Path(run_dir).resolve()
    summary = _json(resolved / "url_benchmark_summary.json")
    if require_promoted:
        gate = summary.get("deployment_gate")
        if not isinstance(gate, dict) or not gate.get("passed"):
            raise URLIocArtifactError("URL IOC deployment gate did not pass")
        if summary.get("deployment_status") != "promoted":
            raise URLIocArtifactError("URL IOC run is not promoted")

    required = ("url_model.joblib", "url_feature_contract.json", "url_benchmark_summary.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise URLIocArtifactError(f"URL IOC artifacts are missing: {missing}")
    manifest = _json(resolved / "artifact_manifest.json")
    if require_promoted:
        _verify_signed(manifest)
    expected = _artifact_hashes(manifest)
    for name in required:
        if name not in expected:
            raise URLIocArtifactError(f"URL IOC manifest lacks checksum for {name}")
    checked: dict[str, str] = {}
    for name, digest in expected.items():
        artifact = (resolved / name).resolve()
        try:
            artifact.relative_to(resolved)
        except ValueError as error:
            raise URLIocArtifactError("URL IOC manifest path escapes run directory") from error
        if not artifact.is_file() or sha256_file(artifact).casefold() != digest.casefold():
            raise URLIocArtifactError(f"URL IOC artifact checksum mismatch: {name}")
        checked[name] = digest
    contract = _json(resolved / "url_feature_contract.json")
    if tuple(contract.get("columns", ())) != URL_FEATURE_COLUMNS:
        raise URLIocArtifactError("URL IOC feature contract does not match the runtime")
    if "is_https" in contract.get("columns", ()) or contract.get("version") != FEATURE_CONTRACT_VERSION:
        raise URLIocArtifactError("URL IOC feature contract is unsafe or unsupported")
    if summary.get("run_id") != manifest.get("run_id"):
        raise URLIocArtifactError("URL IOC summary and manifest run IDs differ")
    return {"summary": summary, "contract": contract, "checked_hashes": checked}


def load_url_ioc_detector(
    *, require_promoted: bool = True, results_root: str | Path = RESULTS_ROOT
) -> URLIocDetector:
    """Load a trustworthy URL-IOC model, never a heuristic fallback."""

    root = Path(results_root).resolve()
    pointer_path = root / URL_IOC_POINTER_NAME
    if not pointer_path.is_file():
        raise URLIocArtifactError("No URL IOC deployment pointer is available")
    pointer = _json(pointer_path)
    if require_promoted:
        _verify_signed(pointer)
    run_dir = _safe_run_dir(pointer, root)
    validated = validate_url_ioc_run(run_dir, require_promoted=require_promoted)
    summary = validated["summary"]
    try:
        pipeline = joblib.load(run_dir / "url_model.joblib")
    except Exception as error:  # Joblib errors vary by version and artifact type.
        raise URLIocArtifactError("URL IOC model cannot be loaded") from error
    threshold = float(summary.get("threshold", float("nan")))
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise URLIocArtifactError("URL IOC threshold is invalid")
    return URLIocDetector(
        pipeline=pipeline,
        threshold=threshold,
        run_id=str(summary["run_id"]),
        source=str(summary.get("source", {}).get("name", "PhishVN")),
        version=str(summary.get("source", {}).get("version", "unknown")),
        run_dir=run_dir,
    )


def score_url_ioc(url: object, *, require_promoted: bool = True) -> dict[str, Any]:
    """Return typed advisory URL evidence; no alert is never a safe verdict."""

    if not normalize_url(url):
        return {
            "state": "NOT_APPLICABLE",
            "alert": None,
            "probability": None,
            "threshold": None,
            "scope": "Vietnamese URL lexical IOC only; not an email classifier or safe verdict.",
            "detail": "No valid URL is available for PhishVN URL-IOC scoring.",
        }
    try:
        detector = load_url_ioc_detector(require_promoted=require_promoted)
        probability = float(detector.pipeline.predict_proba(feature_frame(url))[0, 1])
    except (URLIocArtifactError, OSError, ValueError, KeyError) as error:
        return {
            "state": "UNAVAILABLE",
            "alert": None,
            "probability": None,
            "threshold": None,
            "scope": "Vietnamese URL lexical IOC only; not an email classifier or safe verdict.",
            "detail": f"No trusted promoted PhishVN URL-IOC model is available: {error}",
        }
    return {
        "state": "AVAILABLE",
        "alert": probability >= detector.threshold,
        "probability": probability,
        "threshold": detector.threshold,
        "scope": "Vietnamese URL lexical IOC only; not an email classifier or safe verdict.",
        "source": detector.source,
        "version": detector.version,
        "run_id": detector.run_id,
        "detail": (
            "High-confidence URL-model alert: send this link for analyst review."
            if probability >= detector.threshold
            else "No high-confidence URL-model alert; this is not a safe verdict."
        ),
    }
