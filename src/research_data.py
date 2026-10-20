"""Research-grade dataset loading and leakage-resistant text normalization.

The production UI keeps its own active-learning data path.  This module is the
authoritative path for scientific benchmarks: it preserves source provenance,
keeps spam distinct from phishing, and never tunes on external datasets.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .preprocessing import detect_email_language, detect_email_language_details
except ImportError:  # pragma: no cover - direct script compatibility
    from preprocessing import detect_email_language, detect_email_language_details


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "research_datasets.json"
RUNTIME_REGISTRY_DIR = ROOT / "data" / "runtime"
CACHE_SCHEMA_VERSION = 10
RESEARCH_TEXT_MAX_CHARS = 15_000
MODEL_DOCUMENT_MAX_CHARS = 5_000
SOURCE_SAMPLE_SEED = 20260715

# A benchmark can retain unlicensed historical sources for comparative,
# non-deployment research, but a production phishing detector must not be
# promoted from data whose reuse terms are unknown.  Keep this list explicit
# rather than treating any non-empty free-text licence field as approval.
APPROVED_REUSABLE_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC BY 4.0",
        "CC BY-SA 4.0",
        "ODC-BY-1.0",
        "PDDL-1.0",
    }
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
MONEY_RE = re.compile(r"(?:[$€£₫]\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:usd|eur|gbp|vnd)\b)", re.I)
LONG_NUMBER_RE = re.compile(r"\b\d{3,}\b")
WHITESPACE_RE = re.compile(r"\s+")


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported research dataset registry schema.")
    return registry


def registry_project_root(path: str | Path) -> Path:
    """Return the trusted project root used to resolve registered datasets.

    Operator-selected registry snapshots are runtime data and therefore must
    live under ``data/runtime`` rather than the immutable ``config`` tree.
    Their dataset paths remain relative to the project root, not to the
    writable directory.  This avoids both a broken path base and a writable
    configuration file becoming part of the code/configuration trust boundary.
    """

    registry_path = Path(path).resolve()
    try:
        registry_path.relative_to(RUNTIME_REGISTRY_DIR.resolve())
    except ValueError:
        return registry_path.parents[1]
    return ROOT


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_parquet_path(path: Path) -> bool:
    suffixes = "".join(path.suffixes).casefold()
    return ".parquet" in suffixes


def _label_strings(values: pd.Series) -> pd.Series:
    """Return stable registry keys for numeric and named source labels.

    CSV files with missing labels are commonly inferred as floating point,
    turning binary labels into ``0.0``/``1.0``.  Registry mappings use the
    semantically equivalent ``0``/``1`` keys, so normalise only integer-like
    numeric suffixes while preserving named labels such as ``Safe Email``.
    """

    labels = values.astype(str).str.strip()
    return labels.str.replace(r"^([+-]?\d+)\.0$", r"\1", regex=True)


def normalize_research_text(text: object) -> str:
    """Mask volatile artefacts without deleting security-relevant semantics.

    Old years, raw addresses, URLs, IPs, amounts and long identifiers can make
    a same-corpus random split look unrealistically easy.  Placeholders retain
    the fact that the artefact existed while preventing a model from memorising
    e.g. ``2004`` or one campaign URL as the class label.
    """

    value = str(text or "")[:RESEARCH_TEXT_MAX_CHARS].replace("\x00", " ")
    value = URL_RE.sub(" <URL> ", value)
    value = EMAIL_RE.sub(" <EMAIL> ", value)
    value = IP_RE.sub(" <IP> ", value)
    value = MONEY_RE.sub(" <AMOUNT> ", value)
    value = YEAR_RE.sub(" <YEAR> ", value)
    value = LONG_NUMBER_RE.sub(" <NUM> ", value)
    return WHITESPACE_RE.sub(" ", value).strip().casefold()


def build_research_document(text: object) -> str:
    """Create the leakage-resistant, bilingual document used by new models.

    Volatile IOCs remain masked in the lexical content, while bounded indicator
    tokens retain their security meaning.  Character n-grams in the paired
    vectorizer preserve Vietnamese diacritics, URL obfuscation and novel words
    without making a campaign-specific address a learned label.
    """
    raw = str(text or "")[:RESEARCH_TEXT_MAX_CHARS]
    if len(raw) > MODEL_DOCUMENT_MAX_CHARS:
        tail = min(1_000, MODEL_DOCUMENT_MAX_CHARS // 3)
        head = MODEL_DOCUMENT_MAX_CHARS - tail
        model_view = raw[:head] + "\n[TRUNCATED_FOR_MODEL]\n" + raw[-tail:]
    else:
        model_view = raw
    normalized = normalize_research_text(model_view)
    language = detect_email_language_details(raw)
    lowered = raw.casefold()
    feature_tokens = [f"lang_{language['language']}"]
    if float(language["confidence"]) >= 0.75:
        feature_tokens.append(f"lang_{language['language']}_confident")
    urls = URL_RE.findall(raw)
    if urls:
        feature_tokens.extend(("ioc_url", f"ioc_url_count_{min(len(urls), 4)}"))
    if EMAIL_RE.search(raw):
        feature_tokens.append("ioc_email_address")
    if IP_RE.search(raw):
        feature_tokens.append("ioc_ip_literal")
    if re.search(r"\b(?:reply-to|return-path|received|dkim-signature|authentication-results):", lowered):
        feature_tokens.append("mail_transport_metadata")
    if re.search(r"\b(?:\.exe|\.js|\.vbs|\.scr|\.iso|\.img|\.lnk|\.html?)\b", lowered):
        feature_tokens.append("risky_attachment_or_lure")
    if re.search(
        r"\b(?:login|sign[ -]?in|password|verify|credential|otp|mật\s*khẩu|đăng\s*nhập|xác\s*minh|tài\s*khoản|ngân\s*hàng)\b",
        lowered,
    ):
        feature_tokens.append("credential_request")
    if re.search(r"\b(?:urgent|immediately|suspend(?:ed)?|khẩn\s*cấp|ngay\s*lập\s*tức|bị\s*khóa)\b", lowered):
        feature_tokens.append("urgency_signal")
    if re.search(r"\b(?:wire\s*transfer|gift\s*card|crypto|chuyển\s*khoản|thẻ\s*cào|tiền\s*điện\s*tử)\b", lowered):
        feature_tokens.append("payment_request")
    if re.search(r"\.(?:xyz|top|click|shop|site|info|tk|gq|ml|cf)(?:[/:?#\s]|$)", lowered):
        feature_tokens.append("risky_tld")
    return " ".join(part for part in (normalized, " ".join(feature_tokens)) if part).strip()


def _template_from_normalized(value: str) -> str:
    value = re.sub(r"\b(?:dear|hello|hi|estimado|hola)\s+[\w.-]+", "<GREETING>", value)
    value = re.sub(r"\b[a-z]{1,3}\d{1,3}[a-z0-9-]*\b", "<TOKEN>", value)
    value = re.sub(r"[^\w<>]+", " ", value, flags=re.UNICODE)
    return WHITESPACE_RE.sub(" ", value).strip()


def normalized_template(text: object) -> str:
    """Return a conservative template used to group repeated campaigns."""

    return _template_from_normalized(normalize_research_text(text))


def content_hash(text: object) -> str:
    return hashlib.sha256(normalize_research_text(text).encode("utf-8")).hexdigest()


def template_hash(text: object) -> str:
    return hashlib.sha256(normalized_template(text).encode("utf-8")).hexdigest()[:20]


def _callable_source_hash(function: Any) -> str:
    """Hash executable preprocessing source without relying on a Git commit."""

    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = f"{getattr(function, '__module__', '')}:{getattr(function, '__qualname__', repr(function))}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def source_cache_fingerprint(source: dict[str, Any], file_sha256: str) -> tuple[str, dict[str, Any]]:
    """Bind a cached projection to data, registry semantics, and code.

    Changing label mappings, selected columns, normalisation, document
    construction, template grouping, or language detection must create a new
    cache instead of silently reusing stale labels/text.
    """

    registry_fields = {
        key: source.get(key)
        for key in (
            "id",
            "path",
            "role",
            "text_column",
            "subject_column",
            "label_column",
            "id_column",
            "csv_engine",
            "original_labels",
            "research_binary",
        )
    }
    contract = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "file_sha256": file_sha256.casefold(),
        "registry": registry_fields,
        "code": {
            "normalize_research_text": _callable_source_hash(normalize_research_text),
            "build_research_document": _callable_source_hash(build_research_document),
            "template_from_normalized": _callable_source_hash(_template_from_normalized),
            "detect_email_language": _callable_source_hash(detect_email_language),
        },
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), contract


def _read_source(source: dict[str, Any], root: Path, *, include_raw: bool = True) -> pd.DataFrame:
    path = root / source["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing registered dataset: {path}")
    expected_hash = source.get("sha256")
    actual_hash = sha256_file(path)
    if expected_hash and actual_hash.casefold() != expected_hash.casefold():
        raise ValueError(f"Checksum mismatch for {source['id']}: {path}")
    cache_dir = root / "data" / "research_cache"
    cache_fingerprint, cache_contract = source_cache_fingerprint(source, actual_hash)
    cache_key = cache_fingerprint[:24]
    cache_path = cache_dir / f"{source['id']}-v{CACHE_SCHEMA_VERSION}-{cache_key}.parquet"
    cache_metadata_path = cache_path.with_suffix(".meta.json")
    metadata_matches = False
    if cache_metadata_path.exists():
        try:
            cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            metadata_matches = cache_metadata.get("fingerprint") == cache_fingerprint
        except (OSError, json.JSONDecodeError):
            metadata_matches = False
    if cache_path.exists() and metadata_matches:
        cached_columns = None if include_raw else [
            "message_id", "original_label", "threat_type", "binary_label", "source", "role",
            "research_text", "content_hash", "template_group", "language",
        ]
        cached = pd.read_parquet(cache_path, columns=cached_columns)
        required = {
            "message_id", "original_label", "threat_type", "binary_label",
            "source", "role", "research_text", "content_hash", "template_group",
        }
        # Benchmark paths deliberately load a compact projection without the
        # raw email body.  Do not invalidate that projection by requiring a
        # column we explicitly did not read; doing so rebuilt every source
        # cache for each model without changing the training contract.
        if include_raw:
            required.add("raw_text")
        if required.issubset(cached.columns):
            cached["role"] = source["role"]
            return cached

    if source["id"] == "spaphish_v5":
        frame = pd.read_csv(path, engine="python")
        original_label = frame["Label"].astype(str)
        raw_text = frame["subject"].fillna("") + "\n" + frame["body"].fillna("")
        message_id = frame["hash"].astype(str)
    elif source["id"] == "uot_validation_2024":
        frame = pd.read_csv(path)
        original_label = frame["Email Type"].astype(str)
        raw_text = frame["Email Text"].fillna("").astype(str)
        message_id = raw_text.map(content_hash)
    elif "text_column" in source and "label_column" in source:
        requested_columns = [source["text_column"], source["label_column"]]
        for optional_name in (source.get("subject_column"), source.get("id_column")):
            if optional_name and optional_name not in requested_columns:
                requested_columns.append(optional_name)
        if _is_parquet_path(path):
            # Large external parquet files can contain dozens of engineered
            # columns that the text benchmark never consumes.  Reading only
            # the registered contract prevents a 100k-row source from
            # exhausting a 16 GB research workstation before evaluation.
            frame = pd.read_parquet(path, columns=requested_columns)
        else:
            csv_engine = source.get("csv_engine", "c")
            if csv_engine == "python":
                csv.field_size_limit(2**31 - 1)
            frame = pd.read_csv(path, engine=csv_engine, usecols=requested_columns)
        text_column = source["text_column"]
        label_column = source["label_column"]
        if text_column not in frame.columns or label_column not in frame.columns:
            raise ValueError(
                f"{source['id']} missing configured columns: {text_column!r}, {label_column!r}"
            )
        frame = frame.dropna(subset=[label_column]).copy()
        original_label = _label_strings(frame[label_column])
        subject_column = source.get("subject_column")
        if subject_column and subject_column in frame.columns:
            raw_text = frame[subject_column].fillna("").astype(str) + "\n" + frame[text_column].fillna("").astype(str)
        else:
            raw_text = frame[text_column].fillna("").astype(str)
        id_column = source.get("id_column")
        message_id = frame[id_column].astype(str) if id_column and id_column in frame.columns else raw_text.map(content_hash)
    else:
        if _is_parquet_path(path):
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame = frame.dropna(subset=["label"]).copy()
        original_label = frame["label"].astype(int).astype(str)
        subject = frame["subject"].fillna("").astype(str) if "subject" in frame else ""
        body_column = "body" if "body" in frame else "text_combined"
        raw_text = subject + "\n" + frame[body_column].fillna("").astype(str)
        message_id = raw_text.map(content_hash)

    threat_map = source["original_labels"]
    binary_map = source["research_binary"]
    result = pd.DataFrame(
        {
            "message_id": message_id,
            "raw_text": raw_text,
            "original_label": original_label,
            "threat_type": original_label.map(threat_map),
            "binary_label": original_label.map(binary_map),
            "source": source["id"],
            "role": source["role"],
        }
    )
    if result["threat_type"].isna().any():
        unknown = sorted(result.loc[result["threat_type"].isna(), "original_label"].unique())
        raise ValueError(f"Unmapped labels in {source['id']}: {unknown}")
    # Normalisation is deliberately the expensive operation; compute it once
    # per message and derive both hashes from that stable representation.
    result["normalized_text"] = result["raw_text"].map(normalize_research_text)
    result["research_text"] = result["raw_text"].map(build_research_document)
    result["language"] = result["raw_text"].map(detect_email_language)
    result["content_hash"] = result["normalized_text"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    result["template_group"] = result["normalized_text"].map(
        lambda value: hashlib.sha256(_template_from_normalized(value).encode("utf-8")).hexdigest()[:20]
    )
    # ``normalized_text`` is only an intermediate used for stable hashes and
    # template grouping.  Keeping a third near-full text copy in every cached
    # source made a complete train+external load exceed practical RAM on the
    # supported workstation.  The persisted contract needs raw audit text,
    # enriched model text, hashes and language -- not this temporary column.
    result = result.drop(columns=["normalized_text"])
    result = result.loc[result["research_text"].str.len().gt(0)].reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    cache_metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": cache_fingerprint,
                "contract": cache_contract,
                "rows": int(len(result)),
                "columns": list(result.columns),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result if include_raw else result.drop(columns=["raw_text"])


def _sample_source_rows(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Apply a deterministic, label-stratified source cap before concatenation.

    This is a resource-control option for a workstation benchmark, not a
    replacement for the uncapped protocol.  It retains every class whenever
    possible and records the cap in the corpus manifest.
    """
    if max_rows is None or int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame
    maximum = int(max_rows)
    if "binary_label" not in frame.columns or frame["binary_label"].nunique() < 2:
        return frame.sample(n=maximum, random_state=SOURCE_SAMPLE_SEED).reset_index(drop=True)
    groups = [item for _, item in frame.groupby("binary_label", dropna=False, sort=True)]
    total = len(frame)
    target_sizes = [max(1, round(maximum * len(group) / total)) for group in groups]
    # Correct rounding deterministically while keeping every present class.
    while sum(target_sizes) > maximum:
        index = max(range(len(target_sizes)), key=lambda item: (target_sizes[item], -item))
        if target_sizes[index] <= 1:
            break
        target_sizes[index] -= 1
    while sum(target_sizes) < maximum:
        index = max(range(len(groups)), key=lambda item: (len(groups[item]) - target_sizes[item], -item))
        if target_sizes[index] >= len(groups[index]):
            break
        target_sizes[index] += 1
    pieces = [
        group if len(group) <= target else group.sample(n=target, random_state=SOURCE_SAMPLE_SEED + index)
        for index, (group, target) in enumerate(zip(groups, target_sizes))
    ]
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=SOURCE_SAMPLE_SEED).reset_index(drop=True)


def source_governance_record(source: dict[str, Any]) -> dict[str, Any]:
    """Return the provenance facts needed for a deployment decision.

    This is deliberately descriptive, not a downloader or a licence parser:
    the registry is the reviewed source of truth.  Unknown or unsupported
    terms stay visible in research manifests and fail the promotion gate.
    """

    license_name = str(source.get("license", "unknown")).strip()
    return {
        "id": str(source.get("id", "unknown")),
        "license": license_name or "unknown",
        "license_approved": license_name in APPROVED_REUSABLE_LICENSES,
        "provenance_url": str(source.get("provenance_url", "")).strip() or None,
        "sha256_declared": bool(str(source.get("sha256", "")).strip()),
    }


def load_research_corpus(
    registry_path: str | Path = DEFAULT_REGISTRY,
    roles: tuple[str, ...] | None = None,
    deduplicate: bool = True,
    include_raw: bool = True,
    max_rows_per_source: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    root = registry_project_root(registry_path)
    sources = [s for s in registry["sources"] if roles is None or s["role"] in roles]
    frames = [
        _sample_source_rows(
            _read_source(source, root, include_raw=include_raw), max_rows_per_source
        )
        for source in sources
    ]
    corpus = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if corpus.empty:
        return corpus, {"sources": [], "rows": 0}

    before = len(corpus)
    conflicts = corpus.groupby("content_hash")["threat_type"].nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index)
    if conflicting_hashes:
        corpus = corpus.loc[~corpus["content_hash"].isin(conflicting_hashes)].copy()
    duplicate_rows = int(corpus.duplicated(subset=["content_hash", "threat_type"]).sum())
    if deduplicate:
        corpus = corpus.drop_duplicates(
            subset=["content_hash", "threat_type"], keep="first"
        ).reset_index(drop=True)
    manifest = {
        "sources": [s["id"] for s in sources],
        "source_governance": [source_governance_record(s) for s in sources],
        "rows_before_deduplication": before,
        "rows": int(len(corpus)),
        "exact_duplicates_removed": duplicate_rows,
        "deduplication_applied": deduplicate,
        "conflicting_content_groups_removed": len(conflicting_hashes),
        "label_counts": {str(k): int(v) for k, v in corpus["threat_type"].value_counts().items()},
        "language_counts": {str(k): int(v) for k, v in corpus["language"].value_counts().items()},
        "source_counts": {str(k): int(v) for k, v in corpus["source"].value_counts().items()},
        "unique_templates": int(corpus["template_group"].nunique()),
        "max_rows_per_source": int(max_rows_per_source) if max_rows_per_source else None,
    }
    return corpus, manifest


def split_primary_corpora(
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    include_raw: bool = True,
    max_train_rows_per_source: int | None = None,
    max_external_rows_per_source: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    train, train_manifest = load_research_corpus(
        registry_path,
        roles=("train",),
        include_raw=include_raw,
        max_rows_per_source=max_train_rows_per_source,
    )
    train = train.loc[train["binary_label"].notna()].copy()
    train["binary_label"] = train["binary_label"].astype(int)
    external, external_manifest = load_research_corpus(
        registry_path,
        roles=("external_primary", "external_secondary"),
        deduplicate=False,
        include_raw=include_raw,
        max_rows_per_source=max_external_rows_per_source,
    )
    train_hashes = set(train["content_hash"])
    exact_overlap_mask = external["content_hash"].isin(train_hashes)
    overlap_by_source = {
        str(source): int(count)
        for source, count in external.loc[exact_overlap_mask, "source"].value_counts().items()
    }
    external = external.loc[~exact_overlap_mask].reset_index(drop=True)
    train_templates = set(train["template_group"])
    template_overlap_by_source = {
        str(source): int(count)
        for source, count in external.loc[
            external["template_group"].isin(train_templates), "source"
        ].value_counts().items()
    }
    template_overlap_mask = external["template_group"].isin(train_templates)
    # A different content hash is not enough to make an external row
    # independent: campaign templates often differ only in a recipient or
    # identifier.  Keep these rows in neither the external score nor its gate.
    # Their counts remain in the manifest for auditability.
    external = external.loc[~template_overlap_mask].reset_index(drop=True)
    external_manifest["cross_role_exact_duplicates_removed"] = int(exact_overlap_mask.sum())
    external_manifest["cross_role_exact_duplicates_removed_by_source"] = overlap_by_source
    external_manifest["cross_role_template_overlaps_removed"] = int(template_overlap_mask.sum())
    external_manifest["cross_role_template_overlaps_removed_by_source"] = template_overlap_by_source
    external_manifest["cross_role_template_overlaps_remaining_by_source"] = {}
    external_manifest["rows_after_cross_role_deduplication"] = int(len(external))
    external_sets = {
        source: frame.reset_index(drop=True)
        for source, frame in external.groupby("source", sort=True)
    }
    for frame in external_sets.values():
        frame["binary_label"] = frame["binary_label"].astype(int)
    return train, external_sets, {"train": train_manifest, "external": external_manifest}
