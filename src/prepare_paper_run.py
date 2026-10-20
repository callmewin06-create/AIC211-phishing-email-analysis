"""Freeze every non-performance input for a paper protocol v2 run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .artifact_signing import require_signing_key, sign_manifest
from .deployment_validator import sha256_file
from .paper_cohort import build_matched_llm_cohort, cohort_integrity
from .paper_protocol import DEFAULT_PAPER_PROTOCOL, load_paper_protocol
from .research_data import DEFAULT_REGISTRY, load_registry, split_primary_corpora
from .research_splits import partition_integrity, partition_manifest, split_hybrid_partitions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research_results"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_snapshot() -> dict:
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "config")
    files: dict[str, str] = {}
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if path.suffix.casefold() not in {".py", ".ps1", ".json", ".md"}:
                continue
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    for name in ("requirements-research.txt", "requirements-runtime.lock.txt", "RESEARCH_PROTOCOL.md"):
        path = ROOT / name
        if path.exists():
            files[name] = sha256_file(path)
    return {"hash_algorithm": "sha256", "files": files}


def _write_source_bundle(path: Path, snapshot: dict) -> None:
    """Persist the exact source/config inputs; hashes alone are not recovery."""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(snapshot["files"]):
            source = ROOT / relative
            info = zipfile.ZipInfo(relative.replace("\\", "/"))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def _governance_gate(data_manifest: dict) -> dict:
    checks = []
    for scope in ("train", "external"):
        records = data_manifest.get(scope, {}).get("source_governance", [])
        checks.append(
            {
                "name": f"{scope}_source_governance",
                "passed": bool(records)
                and all(
                    row.get("license_approved")
                    and row.get("provenance_url")
                    and row.get("sha256_declared")
                    for row in records
                ),
                "non_publishable_sources": [
                    row.get("id")
                    for row in records
                    if not (
                        row.get("license_approved")
                        and row.get("provenance_url")
                        and row.get("sha256_declared")
                    )
                ],
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def prepare_paper_run(
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    protocol_path: str | Path = DEFAULT_PAPER_PROTOCOL,
    output_root: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """Create the immutable input bundle consumed by LLM and hybrid runners."""

    require_signing_key()
    started = datetime.now(timezone.utc)
    protocol = load_paper_protocol(protocol_path)
    registry_path = Path(registry_path).resolve()
    registry = load_registry(registry_path)
    corpus, external_sets, data_manifest = split_primary_corpora(registry_path)
    partitions = split_hybrid_partitions(corpus, seed=int(protocol["seed"]))
    partition_frames = partitions.as_dict()
    split_gate = partition_integrity(partition_frames)
    cohort = build_matched_llm_cohort(
        partition_frames,
        external_sets,
        target_size=int(protocol["llm"]["cohort_size"]),
        seed=int(protocol["seed"]),
    )
    llm_cohort_gate = cohort_integrity(
        cohort, target_size=int(protocol["llm"]["cohort_size"])
    )
    governance_gate = _governance_gate(data_manifest)

    run_id = started.strftime("paper-prep-v2-%Y%m%dT%H%M%SZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_json(run_dir / "frozen_registry.json", registry)
    _write_json(
        run_dir / "frozen_protocol.json",
        {key: value for key, value in protocol.items() if key != "config_path"},
    )
    _write_json(run_dir / "data_manifest.json", data_manifest)
    _write_json(run_dir / "partition_integrity.json", split_gate)
    _write_json(run_dir / "llm_cohort_integrity.json", llm_cohort_gate)
    source_snapshot = _source_snapshot()
    _write_json(run_dir / "source_snapshot.json", source_snapshot)
    _write_source_bundle(run_dir / "source_bundle.zip", source_snapshot)
    partition_manifest(partition_frames).to_csv(
        run_dir / "partition_manifest.csv", index=False, encoding="utf-8"
    )
    cohort.to_csv(run_dir / "llm_cohort.csv", index=False, encoding="utf-8")

    integrity_checks = [
        {"name": "partition_integrity", "passed": bool(split_gate["passed"]), "detail": split_gate["violations"]},
        {"name": "llm_cohort_integrity", "passed": bool(llm_cohort_gate["passed"]), "detail": llm_cohort_gate["violations"]},
        {"name": "source_governance", "passed": bool(governance_gate["passed"]), "detail": governance_gate["checks"]},
        {
            "name": "uncapped_primary_data",
            "passed": not data_manifest["train"].get("max_rows_per_source")
            and not data_manifest["external"].get("max_rows_per_source"),
            "detail": "Paper runs cannot use workstation source caps.",
        },
    ]
    paper_gate = {"passed": all(row["passed"] for row in integrity_checks), "checks": integrity_checks}
    prepared = {
        "schema_version": 2,
        "protocol": protocol["protocol"],
        "paper_run_id": run_id,
        "created_at": started.isoformat(),
        "run_directory": str(run_dir.resolve()),
        "registry_source_path": str(registry_path),
        "registry_source_sha256": sha256_file(registry_path),
        "protocol_source_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": protocol["config_sha256"],
        "seed": int(protocol["seed"]),
        "paper_integrity_gate": paper_gate,
        "paper_status": "FROZEN_INPUTS" if paper_gate["passed"] else "BLOCKED_INPUT_INTEGRITY",
        "llm_cohort_rows": int(len(cohort)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    _write_json(run_dir / "prepared_run.json", prepared)

    artifact_names = (
        "prepared_run.json",
        "frozen_registry.json",
        "frozen_protocol.json",
        "data_manifest.json",
        "partition_manifest.csv",
        "partition_integrity.json",
        "llm_cohort.csv",
        "llm_cohort_integrity.json",
        "source_snapshot.json",
        "source_bundle.zip",
    )
    manifest = sign_manifest(
        {
            "schema_version": 2,
            "paper_run_id": run_id,
            "purpose": "frozen paper inputs; not a deployment promotion",
            "artifacts": {name: sha256_file(run_dir / name) for name in artifact_names},
        }
    )
    _write_json(run_dir / "artifact_manifest.json", manifest)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze inputs for hybrid paper protocol v2")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--protocol", default=str(DEFAULT_PAPER_PROTOCOL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = prepare_paper_run(
        registry_path=args.registry, protocol_path=args.protocol, output_root=args.output
    )
    # Windows PowerShell may expose a legacy CP1252 stdout even though all
    # persisted artifacts are UTF-8.  Escaping non-ASCII keeps the CLI usable
    # for project paths containing Vietnamese characters.
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
