"""HMAC signatures for deployment manifests.

Checksums detect accidental corruption, not an attacker who can rewrite both an
artifact and its checksum file. A deployment secret held outside the writable
workspace provides the missing trust boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any


SIGNING_ENV = "AIC211_ARTIFACT_SIGNING_KEY"


class ArtifactSignatureError(RuntimeError):
    """Raised when an artifact manifest is unsigned or has an invalid signature."""


def signature_is_required() -> bool:
    return os.environ.get("AIC211_ENV", "development").strip().lower() == "production"


def _key(signing_env: str = SIGNING_ENV) -> bytes | None:
    value = os.environ.get(signing_env, "")
    return value.encode("utf-8") if value else None


def require_signing_key(signing_env: str = SIGNING_ENV) -> None:
    """Fail before a production operation when its dedicated key is absent."""

    if signature_is_required() and _key(signing_env) is None:
        raise ArtifactSignatureError(f"{signing_env} is required in production.")


def _canonical_bytes(manifest: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest: dict[str, Any], *, signing_env: str = SIGNING_ENV) -> dict[str, Any]:
    """Sign a manifest with its purpose-specific secret.

    ``key_id`` binds a signature to a key domain.  In particular, a model
    builder that can sign artifacts must not be able to sign the immutable
    source release used before the dashboard imports application code.
    """

    key = _key(signing_env)
    if key is None:
        if signature_is_required():
            raise ArtifactSignatureError(f"{signing_env} is required in production.")
        return dict(manifest)
    unsigned = {name: value for name, value in manifest.items() if name != "signature"}
    digest = hmac.new(key, _canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return {
        **unsigned,
        "signature": {"algorithm": "hmac-sha256", "key_id": signing_env, "value": digest},
    }


def verify_manifest_signature(manifest: dict[str, Any], *, signing_env: str = SIGNING_ENV) -> None:
    """Verify a manifest with its expected signing domain; production fails closed."""

    signature = manifest.get("signature") if isinstance(manifest, dict) else None
    key = _key(signing_env)
    if not isinstance(signature, dict):
        if signature_is_required() or key is not None:
            raise ArtifactSignatureError("Production artifact manifest is unsigned.")
        return
    if (
        signature.get("algorithm") != "hmac-sha256"
        or signature.get("key_id") != signing_env
        or not isinstance(signature.get("value"), str)
    ):
        raise ArtifactSignatureError("Artifact signature metadata is invalid.")
    if key is None:
        raise ArtifactSignatureError(f"{signing_env} is required to verify this signed manifest.")
    expected = hmac.new(key, _canonical_bytes(manifest), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(signature["value"]), expected):
        raise ArtifactSignatureError("Artifact manifest signature does not verify.")
