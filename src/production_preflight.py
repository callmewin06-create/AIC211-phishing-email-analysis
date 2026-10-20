"""Fail-closed production readiness checks for the on-premise deployment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from .artifact_signing import SIGNING_ENV
from .auth import authentication_is_configured
from .deployment_validator import DeploymentValidationError, resolve_run_pointer, validate_deployment_run
from .release_integrity import (
    RELEASE_PUBLIC_VERIFICATION_ENV,
    ReleaseIntegrityError,
    verify_release_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, passed: bool, detail: str, *, critical: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "critical": critical, "detail": detail}


def _deployment_check(root: Path) -> dict[str, Any]:
    results = root / "research_results"
    attempts = (
        ("latest_hybrid.json", ("tfidf_vectorizer.joblib", "ml_model.joblib", "full_fusion.joblib")),
        ("latest_benchmark.json", ("tfidf_vectorizer.joblib", "logistic_regression.joblib")),
    )
    errors: list[str] = []
    for pointer_name, required in attempts:
        pointer = results / pointer_name
        if not pointer.exists():
            errors.append(f"{pointer_name}: missing")
            continue
        try:
            candidate, _pointer = resolve_run_pointer(pointer, results)
            validate_deployment_run(candidate, required_artifacts=required, require_promoted=True)
            return _check("signed_promoted_model", True, f"Validated {pointer_name}.")
        except (OSError, ValueError, KeyError, DeploymentValidationError) as error:
            errors.append(f"{pointer_name}: {error}")
    return _check("signed_promoted_model", False, "; ".join(errors))


def _release_integrity_check(root: Path) -> dict[str, Any]:
    try:
        manifest = verify_release_manifest(root)
    except ReleaseIntegrityError as error:
        return _check("signed_source_release", False, str(error))
    return _check("signed_source_release", True, f"Verified {len(manifest['files'])} runtime files.")


_BROAD_ACL_PRINCIPALS = ("authenticated users:", "builtin\\users:", "everyone:")
_WRITE_MARKERS = ("(m)", "(f)", "(w)", "(wd)")


def _read_windows_acl(path: Path) -> tuple[str | None, str | None]:
    completed = subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=False, timeout=10
    )
    if completed.returncode != 0:
        return None, f"icacls failed for {path.name}."
    return (completed.stdout + completed.stderr).casefold(), None


def _broad_write_principal(output: str) -> str | None:
    for line in output.splitlines():
        if any(principal in line for principal in _BROAD_ACL_PRINCIPALS) and any(
            marker in line for marker in _WRITE_MARKERS
        ):
            return line.strip()
    return None


def _account_acl_line(output: str, account: str) -> str:
    return next(
        (line.strip() for line in output.splitlines() if account.casefold() in line),
        "",
    )


def _windows_acl_checks(root: Path) -> list[dict[str, Any]]:
    """Verify immutable code/config and writable runtime data separately."""

    account = os.environ.get("AIC211_SERVICE_ACCOUNT", "").strip()
    if not account:
        detail = (
            "Set AIC211_SERVICE_ACCOUNT, then use scripts/harden_windows_acl.ps1 "
            "after validating the target identity."
        )
        return [
            _check("windows_code_acl", False, detail),
            _check("windows_data_acl", False, detail),
        ]

    root_acl, root_error = _read_windows_acl(root)
    if root_error:
        code_check = _check("windows_code_acl", False, root_error)
    else:
        broad_write = _broad_write_principal(root_acl or "")
        account_line = _account_acl_line(root_acl or "", account)
        if broad_write:
            code_check = _check(
                "windows_code_acl",
                False,
                "Project root still grants a broad principal write access: " + broad_write,
            )
        elif not account_line or "(rx)" not in account_line:
            code_check = _check(
                "windows_code_acl",
                False,
                f"Project root does not show Read/Execute access for {account}.",
            )
        elif any(marker in account_line for marker in _WRITE_MARKERS):
            code_check = _check(
                "windows_code_acl",
                False,
                f"Project root grants {account} write access; code/config must remain read-only.",
            )
        else:
            code_check = _check("windows_code_acl", True, f"Verified read-only code/config ACL for {account}.")

    targets = [path for path in (root / "data", root / "research_results", root / "models") if path.exists()]
    if not targets:
        data_check = _check("windows_data_acl", False, "No protected data/model directory exists to verify.")
    else:
        data_check = _check("windows_data_acl", True, f"Verified least-privilege runtime ACL for {account}.")
        for target in targets:
            output, error = _read_windows_acl(target)
            if error:
                data_check = _check("windows_data_acl", False, error)
                break
            broad_write = _broad_write_principal(output or "")
            if broad_write:
                data_check = _check(
                    "windows_data_acl",
                    False,
                    f"{target.name} still grants a broad principal write access: {broad_write}",
                )
                break
            account_line = _account_acl_line(output or "", account)
            if not any(marker in account_line for marker in ("(m)", "(f)")):
                data_check = _check(
                    "windows_data_acl",
                    False,
                    f"{target.name} does not show Modify/Full access for {account}.",
                )
                break
    return [code_check, data_check]


def run_preflight(root: str | Path = ROOT) -> dict[str, Any]:
    """Return machine-readable production blockers without starting Streamlit."""

    root = Path(root).resolve()
    environment = os.environ.get("AIC211_ENV", "").strip().lower()
    checks = [
        _check("production_environment", environment == "production", "Set AIC211_ENV=production."),
        _check(
            "authentication",
            authentication_is_configured(),
            "Configure AIC211_AUTH_IDENTITIES_JSON and non-empty token environment variables.",
        ),
    ]
    key = os.environ.get("AIC211_FEEDBACK_ENCRYPTION_KEY", "")
    try:
        if not key:
            raise ValueError("missing")
        Fernet(key.encode("ascii"))
        feedback_key_ok = True
        feedback_detail = "Feedback encryption key is present."
    except (ValueError, TypeError):
        feedback_key_ok = False
        feedback_detail = "Set a valid AIC211_FEEDBACK_ENCRYPTION_KEY from the secret manager."
    checks.append(_check("feedback_encryption", feedback_key_ok, feedback_detail))
    checks.append(
        _check(
            "model_artifact_signing_key",
            bool(os.environ.get(SIGNING_ENV, "").strip()),
            f"Set {SIGNING_ENV} outside the writable workspace.",
        )
    )
    checks.append(
        _check(
            "source_release_public_key",
            bool(os.environ.get(RELEASE_PUBLIC_VERIFICATION_ENV, "").strip()),
            f"Set {RELEASE_PUBLIC_VERIFICATION_ENV}; the release private key must never exist on the runtime host.",
        )
    )
    config_path = root / ".streamlit" / "config.toml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    checks.append(
        _check(
            "streamlit_loopback",
            'address="127.0.0.1"' in config_text.replace(" ", ""),
            "Bind Streamlit to loopback and publish only behind an authenticated TLS reverse proxy.",
        )
    )
    checks.append(_check("dashboard_url_egress", True, "Dashboard live URL probing is disabled by design."))
    checks.append(_release_integrity_check(root))
    checks.append(_deployment_check(root))
    if os.name == "nt":
        checks.extend(_windows_acl_checks(root))
    else:
        checks.append(_check("feedback_file_permissions", True, "SQLite files are set to mode 0600 by FeedbackStore."))
    return {
        "ready": all(item["passed"] for item in checks if item["critical"]),
        "checks": checks,
    }


def main() -> None:
    result = run_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
