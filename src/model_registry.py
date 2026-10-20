"""Versioned candidate-model registry with atomic promotion and rollback.

The active model is selected by one atomic pointer to an immutable version
directory.  Readers therefore never combine a newly copied vectorizer with an
old classifier.  The legacy ``models/*.pkl`` aliases remain only for backwards
compatibility; new code must load through :func:`load_active_artifacts`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

try:
    from .artifact_signing import sign_manifest, signature_is_required, verify_manifest_signature
except ImportError:  # pragma: no cover - supports ``python src/train.py``
    from artifact_signing import sign_manifest, signature_is_required, verify_manifest_signature


MODELS_DIR = Path("models")
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
VERSIONS_DIR = MODELS_DIR / "versions"
CANDIDATES_DIR = MODELS_DIR / "candidates"
ACTIVE_MODEL = MODELS_DIR / "phishing_model.pkl"
ACTIVE_VECTOR = MODELS_DIR / "tfidf_vectorizer.pkl"
ACTIVE_METADATA = MODELS_DIR / "model_metadata.json"
ACTIVE_POINTER_NAME = "active_model.json"
LOCK_NAME = ".registry.lock"
ARTIFACT_MANIFEST = "artifact_manifest.json"
_BUNDLE_FILES = (ACTIVE_MODEL.name, ACTIVE_VECTOR.name, ACTIVE_METADATA.name)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _pointer_path() -> Path:
    return MODELS_DIR / ACTIVE_POINTER_NAME


def _lock_path() -> Path:
    return MODELS_DIR / LOCK_NAME


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def _registry_lock():
    """Serialize promotions/rollbacks across processes on Windows and POSIX."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path()
    with open(lock_file, "a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":  # pragma: no cover - exercised in Windows deployment
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows is the primary supported desktop target
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Model registry must be a JSON object.")
        return payload
    return {"schema_version": 2, "active_version": None, "versions": [], "candidates": []}


def _write_registry(registry: dict[str, Any]) -> None:
    registry = {"schema_version": 2, **registry}
    _atomic_json(REGISTRY_PATH, registry)


def _read_pointer() -> dict[str, Any] | None:
    path = _pointer_path()
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload.get("version_id") or not payload.get("path"):
        raise ValueError("Active model pointer is malformed.")
    return payload


def _write_pointer(version_id: str, path: Path) -> None:
    _atomic_json(
        _pointer_path(),
        {
            "schema_version": 1,
            "version_id": version_id,
            "path": str(path.resolve()),
            "updated_at": utc_stamp(),
        },
    )


def _path_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Registry path escapes its permitted artifact directory.") from error
    return resolved


def _bundle_manifest(path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifacts": {name: _sha256_file(path / name) for name in _BUNDLE_FILES},
    }


def _validate_bundle(path: Path) -> None:
    for filename in _BUNDLE_FILES:
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Model bundle is missing artifact: {filename}")
    manifest_path = path / ARTIFACT_MANIFEST
    if not manifest_path.exists():
        raise ValueError("Model bundle is missing its immutable artifact manifest.")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest_signature(manifest)
    expected = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    for filename in _BUNDLE_FILES:
        if expected.get(filename) != _sha256_file(path / filename):
            raise ValueError(f"Model bundle integrity check failed for {filename}.")


def _write_bundle_manifest(path: Path) -> None:
    _atomic_json(path / ARTIFACT_MANIFEST, sign_manifest(_bundle_manifest(path)))


def _record_for(registry: dict[str, Any], version_id: str | None) -> dict[str, Any] | None:
    return next((item for item in registry.get("versions", []) if item.get("version_id") == version_id), None)


def bootstrap_active_model() -> dict[str, Any] | None:
    """Snapshot legacy aliases once and make them an immutable active bundle."""

    with _registry_lock():
        registry = _read_registry()
        active = _record_for(registry, registry.get("active_version"))
        if active is not None and _pointer_path().exists():
            return active
        if not (ACTIVE_MODEL.exists() and ACTIVE_VECTOR.exists()):
            return active

        version_id = f"legacy-{utc_stamp()}"
        destination = VERSIONS_DIR / version_id
        destination.mkdir(parents=True, exist_ok=False)
        for filename in (ACTIVE_MODEL.name, ACTIVE_VECTOR.name):
            shutil.copy2(MODELS_DIR / filename, destination / filename)
        if ACTIVE_METADATA.exists():
            shutil.copy2(ACTIVE_METADATA, destination / ACTIVE_METADATA.name)
            with open(ACTIVE_METADATA, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        else:
            metadata = {}
            _atomic_json(destination / ACTIVE_METADATA.name, metadata)
        _write_bundle_manifest(destination)
        record = {
            "version_id": version_id,
            "path": str(destination.resolve()),
            "created_at": utc_stamp(),
            "source": "legacy_snapshot",
            "metadata": metadata,
        }
        if active is None:
            registry["versions"].append(record)
            active = record
        registry["active_version"] = active["version_id"]
        _write_pointer(active["version_id"], Path(active["path"]))
        _write_registry(registry)
        return active


def get_active_version() -> dict[str, Any] | None:
    registry = _read_registry()
    pointer = _read_pointer()
    active_id = pointer["version_id"] if pointer else registry.get("active_version")
    return _record_for(registry, active_id)


def list_versions() -> list[dict[str, Any]]:
    return list(reversed(_read_registry().get("versions", [])))


def active_model_version_id() -> str:
    active = bootstrap_active_model() or get_active_version()
    return active["version_id"] if active else "unversioned"


def load_active_artifacts():
    """Load vectorizer/model from one immutable, hash-validated bundle."""

    pointer = _read_pointer()
    if pointer is not None:
        bundle = _path_within(Path(pointer["path"]), VERSIONS_DIR)
        _validate_bundle(bundle)
        return joblib.load(bundle / ACTIVE_VECTOR.name), joblib.load(bundle / ACTIVE_MODEL.name)
    # Read-only compatibility path for an installation not yet bootstrapped.
    if not (ACTIVE_MODEL.exists() and ACTIVE_VECTOR.exists()):
        return None, None
    if signature_is_required():
        raise ValueError("Production refuses unsigned legacy model aliases without an active bundle pointer.")
    return joblib.load(ACTIVE_VECTOR), joblib.load(ACTIVE_MODEL)


def save_candidate(vectorizer, model, metadata: dict[str, Any]) -> dict[str, Any]:
    candidate_id = f"candidate-{utc_stamp()}-{uuid.uuid4().hex[:8]}"
    final_directory = CANDIDATES_DIR / candidate_id
    staging = CANDIDATES_DIR / f".{candidate_id}.staging"
    with _registry_lock():
        staging.mkdir(parents=True, exist_ok=False)
        try:
            joblib.dump(vectorizer, staging / ACTIVE_VECTOR.name)
            joblib.dump(model, staging / ACTIVE_MODEL.name)
            metadata = {**metadata, "candidate_id": candidate_id, "created_at": utc_stamp()}
            _atomic_json(staging / ACTIVE_METADATA.name, metadata)
            _write_bundle_manifest(staging)
            os.replace(staging, final_directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        registry = _read_registry()
        record = {
            "candidate_id": candidate_id,
            "path": str(final_directory.resolve()),
            "created_at": metadata["created_at"],
            "gate": metadata.get("quality_gate", {}),
            "benchmark_id": metadata.get("benchmark_id"),
            "status": "ready" if metadata.get("quality_gate", {}).get("passed") else "rejected",
        }
        registry["candidates"].append(record)
        _write_registry(registry)
    return {**record, "metadata": metadata}


def list_candidates() -> list[dict[str, Any]]:
    return list(reversed(_read_registry().get("candidates", [])))


def _candidate_record(registry: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    record = next((item for item in registry.get("candidates", []) if item.get("candidate_id") == candidate_id), None)
    if record is None:
        raise KeyError("Candidate model was not found.")
    return record


def load_candidate_metadata(candidate_id: str) -> dict[str, Any]:
    record = _candidate_record(_read_registry(), candidate_id)
    path = _path_within(Path(record["path"]), CANDIDATES_DIR)
    _validate_bundle(path)
    with open(path / ACTIVE_METADATA.name, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _publish_legacy_aliases(bundle: Path) -> None:
    """Update old direct-file consumers after the pointer is already atomic."""

    for filename in _BUNDLE_FILES:
        temporary = MODELS_DIR / f".{filename}.{uuid.uuid4().hex}.tmp"
        shutil.copy2(bundle / filename, temporary)
        os.replace(temporary, MODELS_DIR / filename)


def promote_candidate(candidate_id: str) -> dict[str, Any]:
    """Promote only a complete candidate whose persisted quality gate passed."""

    bootstrap_active_model()
    with _registry_lock():
        registry = _read_registry()
        candidate = _candidate_record(registry, candidate_id)
        if candidate.get("status") != "ready" or not candidate.get("gate", {}).get("passed"):
            raise ValueError("Candidate did not pass the persisted quality gate.")
        source = _path_within(Path(candidate["path"]), CANDIDATES_DIR)
        _validate_bundle(source)
        version_id = f"model-{utc_stamp()}-{uuid.uuid4().hex[:8]}"
        version_path = VERSIONS_DIR / version_id
        staging = VERSIONS_DIR / f".{version_id}.staging"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for filename in _BUNDLE_FILES:
                shutil.copy2(source / filename, staging / filename)
            with open(staging / ACTIVE_METADATA.name, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.update(
                {
                    "model_version": version_id,
                    "promotion_time": utc_stamp(),
                    "previous_version": registry.get("active_version"),
                }
            )
            _atomic_json(staging / ACTIVE_METADATA.name, metadata)
            _write_bundle_manifest(staging)
            os.replace(staging, version_path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        record = {
            "version_id": version_id,
            "path": str(version_path.resolve()),
            "created_at": utc_stamp(),
            "source": candidate_id,
            "metadata": metadata,
        }
        _write_pointer(version_id, version_path)
        _publish_legacy_aliases(version_path)
        registry["versions"].append(record)
        registry["active_version"] = version_id
        candidate["status"] = "promoted"
        _write_registry(registry)
        return record


def rollback_active_model(version_id: str) -> dict[str, Any]:
    with _registry_lock():
        registry = _read_registry()
        version = _record_for(registry, version_id)
        if version is None:
            raise KeyError("Model version was not found for rollback.")
        source = _path_within(Path(version["path"]), VERSIONS_DIR)
        _validate_bundle(source)
        _write_pointer(version_id, source)
        _publish_legacy_aliases(source)
        registry["active_version"] = version_id
        _write_registry(registry)
        return version
