"""Signed integrity manifest for immutable production code and configuration.

Windows ACLs prevent ordinary users from editing the deployment. This module
adds an independent verification layer: a release manifest is signed with a
build-only Ed25519 private key and hashes every runtime source/configuration
file. Runtime receives only the public verification key, and code integrity is
checked before Streamlit imports the dashboard.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "release_manifest.json"
RELEASE_PRIVATE_SIGNING_ENV = "AIC211_RELEASE_PRIVATE_KEY"
RELEASE_PUBLIC_VERIFICATION_ENV = "AIC211_RELEASE_PUBLIC_KEY"
_ROOT_RUNTIME_FILES = ("app.py", "run_tests.py", "download_dataset_huggingface.py", "read_file_parquet.py")
_RUNTIME_DIRECTORIES = ("src", "config", "scripts", ".streamlit")
_ALLOWED_SUFFIXES = {".py", ".ps1", ".json", ".toml", ".in", ".lock", ".txt"}


class ReleaseIntegrityError(RuntimeError):
    """The installed source/configuration differs from the signed release."""


def _decode_key(value: str, env_name: str) -> bytes:
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ReleaseIntegrityError(f"{env_name} must be a base64-encoded Ed25519 key.") from error
    if len(decoded) != 32:
        raise ReleaseIntegrityError(f"{env_name} must decode to exactly 32 bytes.")
    return decoded


def _private_key() -> Ed25519PrivateKey:
    value = os.environ.get(RELEASE_PRIVATE_SIGNING_ENV, "").strip()
    if not value:
        raise ReleaseIntegrityError(
            f"{RELEASE_PRIVATE_SIGNING_ENV} is required on the trusted build/release host."
        )
    return Ed25519PrivateKey.from_private_bytes(_decode_key(value, RELEASE_PRIVATE_SIGNING_ENV))


def _public_key() -> tuple[Ed25519PublicKey, bytes]:
    value = os.environ.get(RELEASE_PUBLIC_VERIFICATION_ENV, "").strip()
    if not value:
        raise ReleaseIntegrityError(
            f"{RELEASE_PUBLIC_VERIFICATION_ENV} is required to verify a release manifest."
        )
    raw = _decode_key(value, RELEASE_PUBLIC_VERIFICATION_ENV)
    return Ed25519PublicKey.from_public_bytes(raw), raw


def _release_signature_payload(manifest: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _key_id(public_key_bytes: bytes) -> str:
    return hashlib.sha256(public_key_bytes).hexdigest()[:24]


def release_public_key_from_private() -> str:
    """Return the public key corresponding to the build-only private key."""

    private_key = _private_key()
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base64.b64encode(public).decode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: str | Path = ROOT) -> list[Path]:
    """List only the runtime code/configuration files that must be immutable."""

    root = Path(root).resolve()
    files: list[Path] = []
    for name in _ROOT_RUNTIME_FILES:
        path = root / name
        if path.is_file():
            files.append(path)
    files.extend(path for path in root.glob("requirements*.*") if path.is_file())
    for directory_name in _RUNTIME_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in _ALLOWED_SUFFIXES
        )
    ordered = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    for path in ordered:
        if path.is_symlink():
            raise ReleaseIntegrityError(f"Runtime file must not be a symlink: {path.relative_to(root)}")
        try:
            path.resolve().relative_to(root)
        except ValueError as error:
            raise ReleaseIntegrityError(f"Runtime file escapes the release root: {path}") from error
    return ordered


def build_release_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    files = release_files(root)
    if not files:
        raise ReleaseIntegrityError("No runtime files were found to sign.")
    return {
        "schema_version": 1,
        "files": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in files
        },
    }


def write_release_manifest(root: str | Path = ROOT) -> Path:
    """Sign a release with a build-only Ed25519 private key.

    Runtime verification uses the separate public key.  The dashboard never
    needs a credential capable of authorising a source/configuration change.
    """

    root = Path(root).resolve()
    private_key = _private_key()
    unsigned = build_release_manifest(root)
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    signed = {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "key_id": _key_id(public_key),
            "value": base64.b64encode(private_key.sign(_release_signature_payload(unsigned))).decode("ascii"),
        },
    }
    destination = root / MANIFEST_NAME
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(signed, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def verify_release_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    """Fail closed if source/config files differ from the signed release."""

    root = Path(root).resolve()
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise ReleaseIntegrityError(f"{MANIFEST_NAME} is missing.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ReleaseIntegrityError(f"{MANIFEST_NAME} cannot be read.") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReleaseIntegrityError("Release manifest schema is invalid.")
    signature = manifest.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "ed25519":
        raise ReleaseIntegrityError("Release signature metadata is invalid.")
    public_key, public_bytes = _public_key()
    if signature.get("key_id") != _key_id(public_bytes) or not isinstance(signature.get("value"), str):
        raise ReleaseIntegrityError("Release signature key identity is invalid.")
    try:
        signature_bytes = base64.b64decode(signature["value"].encode("ascii"), validate=True)
        public_key.verify(signature_bytes, _release_signature_payload(manifest))
    except (InvalidSignature, UnicodeEncodeError, ValueError) as error:
        raise ReleaseIntegrityError("Release signature does not verify.") from error

    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded:
        raise ReleaseIntegrityError("Release manifest has no file inventory.")
    current = build_release_manifest(root)["files"]
    if set(recorded) != set(current):
        missing = sorted(set(recorded).difference(current))
        unexpected = sorted(set(current).difference(recorded))
        raise ReleaseIntegrityError(
            "Release file inventory changed"
            + (f"; missing={missing}" if missing else "")
            + (f"; unexpected={unexpected}" if unexpected else "")
        )
    for name, expected in recorded.items():
        if not isinstance(expected, str) or current[name].casefold() != expected.casefold():
            raise ReleaseIntegrityError(f"Release checksum mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign or verify AIC211 runtime source integrity.")
    parser.add_argument("--root", default=str(ROOT))
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", action="store_true", help="Create a signed release manifest.")
    actions.add_argument("--verify", action="store_true", help="Verify the signed release manifest.")
    actions.add_argument(
        "--print-public-key",
        action="store_true",
        help="Derive the public verification key from the build-only private key.",
    )
    args = parser.parse_args()
    try:
        if args.write:
            destination = write_release_manifest(args.root)
            print(json.dumps({"manifest": str(destination), "status": "signed"}))
        elif args.verify:
            manifest = verify_release_manifest(args.root)
            print(json.dumps({"files": len(manifest["files"]), "status": "verified"}))
        else:
            print(json.dumps({"public_key": release_public_key_from_private(), "status": "derived"}))
    except ReleaseIntegrityError as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
