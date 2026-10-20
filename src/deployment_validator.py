"""Fail-closed validation for research and hybrid deployment bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .artifact_signing import (
        ArtifactSignatureError,
        signature_is_required,
        verify_manifest_signature,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from artifact_signing import (  # type: ignore[import-not-found, no-redef]
        ArtifactSignatureError,
        signature_is_required,
        verify_manifest_signature,
    )


class DeploymentValidationError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_pointer(pointer_path: str | Path, results_root: str | Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = Path(pointer_path).resolve()
    results_root = Path(results_root).resolve()
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    try:
        verify_manifest_signature(pointer)
    except ArtifactSignatureError as error:
        raise DeploymentValidationError(f"Deployment pointer trust check failed: {error}") from error
    candidate = Path(str(pointer.get("path", "")))
    if not candidate.is_absolute():
        # Portable pointers are normally relative to the project root.
        project_root = results_root.parent
        candidate = project_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(results_root)
    except ValueError as error:
        raise DeploymentValidationError("Deployment pointer escapes research_results") from error
    if pointer.get("run_id") and candidate.name != str(pointer["run_id"]):
        portable = (results_root / str(pointer["run_id"])).resolve()
        if portable.exists():
            candidate = portable
        else:
            raise DeploymentValidationError("Deployment pointer run_id does not match its directory")
    if not candidate.is_dir():
        raise DeploymentValidationError(f"Deployment run directory does not exist: {candidate}")
    return candidate, pointer


def _artifact_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, list):
        return {
            str(item["path"]): str(item["sha256"])
            for item in artifacts
            if isinstance(item, dict) and item.get("path") and item.get("sha256")
        }
    if isinstance(artifacts, dict):
        result = {}
        for name, value in artifacts.items():
            result[str(name)] = str(value.get("sha256")) if isinstance(value, dict) else str(value)
        return result
    return {}


def validate_deployment_run(
    run_dir: str | Path,
    *,
    required_artifacts: tuple[str, ...],
    require_promoted: bool = True,
) -> dict[str, Any]:
    """Validate gate state and artifact checksums before any Joblib load."""

    run_dir = Path(run_dir).resolve()
    summary_candidates = (run_dir / "hybrid_summary.json", run_dir / "benchmark_summary.json")
    summary_path = next((path for path in summary_candidates if path.exists()), None)
    if summary_path is None:
        raise DeploymentValidationError("Deployment summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if require_promoted:
        gate = summary.get("deployment_gate", {})
        if not gate.get("passed"):
            raise DeploymentValidationError("Deployment gate is missing or did not pass")
        if summary.get("deployment_status") != "promoted":
            raise DeploymentValidationError("Run is not marked as promoted")

    missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
    if missing:
        raise DeploymentValidationError(f"Deployment artifacts are missing: {missing}")

    manifest_path = run_dir / "artifact_manifest.json"
    if require_promoted and not manifest_path.exists():
        raise DeploymentValidationError("Promoted run is missing artifact_manifest.json")
    checked_hashes: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            verify_manifest_signature(manifest)
        except ArtifactSignatureError as error:
            raise DeploymentValidationError(str(error)) from error
        expected_hashes = _artifact_hashes(manifest)
        if manifest.get("run_id") and str(manifest["run_id"]) != str(summary.get("run_id", "")):
            raise DeploymentValidationError("Artifact manifest run_id does not match deployment summary")
        if signature_is_required() and summary_path.name not in expected_hashes:
            raise DeploymentValidationError("Production artifact manifest must protect the deployment summary")
        for name in required_artifacts:
            if name not in expected_hashes and require_promoted:
                raise DeploymentValidationError(f"Artifact manifest has no checksum for {name}")
        for name, expected in expected_hashes.items():
            artifact_path = (run_dir / name).resolve()
            try:
                artifact_path.relative_to(run_dir)
            except ValueError as error:
                raise DeploymentValidationError("Artifact manifest path escapes its run directory") from error
            if not artifact_path.is_file():
                raise DeploymentValidationError(f"Artifact manifest references a missing file: {name}")
            actual = sha256_file(artifact_path)
            if actual.casefold() != expected.casefold():
                raise DeploymentValidationError(f"Artifact checksum mismatch: {name}")
            checked_hashes[name] = actual
    return {
        "run_dir": run_dir,
        "summary": summary,
        "summary_path": summary_path,
        "checked_hashes": checked_hashes,
    }
