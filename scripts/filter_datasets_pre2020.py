"""Create a strict, reproducible pre-2020 view of registered research datasets.

Only rows with an explicit, parseable time field whose year is lower than the
cutoff are retained. Sources without a row-level time field are reported in the
manifest and excluded instead of inferring a send year from free-form email
text, filenames, publication dates, or corpus reputation.

Original datasets are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "research_datasets.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "filtered_pre2020"

DATE_COLUMN_CANDIDATES = (
    "date",
    "timestamp",
    "datetime",
    "sent_date",
    "sent_at",
    "created_at",
    "year",
)


@dataclass
class AuditRow:
    source_id: str
    input_path: str
    output_path: str | None
    status: str
    time_column: str | None
    cutoff_rule: str
    input_rows: int | None
    valid_time_rows: int | None
    retained_rows: int | None
    excluded_at_or_after_cutoff: int | None
    excluded_unparseable_time: int | None
    min_year: int | None
    max_year: int | None
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_parquet(path: Path) -> bool:
    return ".parquet" in "".join(path.suffixes).casefold()


def read_columns(path: Path) -> list[str]:
    if is_parquet(path):
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if path.suffix.casefold() == ".csv":
        return list(pd.read_csv(path, nrows=0, encoding_errors="replace").columns)
    raise ValueError(f"Unsupported tabular format: {path}")


def select_time_column(source: dict[str, Any], columns: list[str]) -> str | None:
    configured = source.get("time_column")
    if configured:
        return configured if configured in columns else None

    casefolded = {column.casefold(): column for column in columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in casefolded:
            return casefolded[candidate]
    return None


def read_frame(path: Path) -> pd.DataFrame:
    if is_parquet(path):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding_errors="replace", low_memory=False)


def parse_years(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        plausible = numeric.where(numeric.between(1900, 2200))
        return plausible.astype("Int64")

    parsed = pd.to_datetime(
        values.astype("string").str.strip(),
        errors="coerce",
        utc=True,
        format="mixed",
    )
    return parsed.dt.year.astype("Int64")


def write_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_parquet(output_path):
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False, encoding="utf-8")


def relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def build_filtered_registry(
    registry: dict[str, Any],
    retained_sources: list[dict[str, Any]],
    cutoff_year: int,
) -> dict[str, Any]:
    result = {
        "schema_version": registry["schema_version"],
        "temporal_filter": {
            "rule": f"row-level year < {cutoff_year}",
            "cutoff_year": cutoff_year,
            "strict": True,
            "unparseable_time_policy": "exclude",
            "no_time_column_policy": "exclude_source",
        },
        "task_definition": registry.get("task_definition", {}),
        "sources": retained_sources,
        "excluded_derived_files": registry.get("excluded_derived_files", []),
        "exclusion_reason": registry.get("exclusion_reason", ""),
    }
    return result


def verify_filtered_outputs(
    audit_rows: list[AuditRow],
    filtered_registry_path: Path,
    cutoff_year: int,
) -> dict[str, Any]:
    with filtered_registry_path.open("r", encoding="utf-8") as handle:
        filtered_registry = json.load(handle)
    registered = {
        str(source["id"]): source for source in filtered_registry.get("sources", [])
    }

    verified_sources = 0
    verified_rows = 0
    for row in audit_rows:
        if row.status != "filtered":
            continue
        if not row.output_path or not row.time_column:
            raise ValueError(f"Incomplete audit metadata for filtered source {row.source_id}")

        output_path = ROOT / row.output_path
        input_path = ROOT / row.input_path
        if not output_path.exists():
            raise FileNotFoundError(f"Missing filtered output: {output_path}")
        if read_columns(input_path) != read_columns(output_path):
            raise ValueError(f"Schema changed while filtering {row.source_id}")

        frame = read_frame(output_path)
        if len(frame) != row.retained_rows:
            raise ValueError(
                f"Row-count mismatch for {row.source_id}: "
                f"expected {row.retained_rows}, got {len(frame)}"
            )
        years = parse_years(frame[row.time_column])
        if years.isna().any():
            raise ValueError(f"Unparseable time survived in {row.source_id}")
        if years.ge(cutoff_year).any():
            raise ValueError(f"Year >= {cutoff_year} survived in {row.source_id}")

        registry_source = registered.get(row.source_id)
        if registry_source is None:
            raise ValueError(f"Filtered registry is missing {row.source_id}")
        if sha256_file(output_path) != registry_source.get("sha256"):
            raise ValueError(f"Checksum mismatch for filtered output {row.source_id}")

        verified_sources += 1
        verified_rows += len(frame)

    if verified_sources != len(registered):
        raise ValueError(
            "Filtered registry/source count mismatch: "
            f"verified={verified_sources}, registered={len(registered)}"
        )
    return {
        "status": "passed",
        "verified_sources": verified_sources,
        "verified_rows": verified_rows,
        "checks": [
            "output exists",
            "schema matches input",
            "row count matches manifest",
            f"all retained years < {cutoff_year}",
            "no retained unparseable time",
            "SHA-256 matches filtered registry",
        ],
    }


def filter_registry(
    registry_path: Path,
    output_dir: Path,
    cutoff_year: int,
) -> tuple[list[AuditRow], Path, Path]:
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[AuditRow] = []
    retained_sources: list[dict[str, Any]] = []
    cutoff_rule = f"year < {cutoff_year}"

    for source in registry.get("sources", []):
        source_id = str(source["id"])
        input_path = ROOT / source["path"]
        if not input_path.exists():
            audit_rows.append(
                AuditRow(
                    source_id=source_id,
                    input_path=source["path"],
                    output_path=None,
                    status="missing_input",
                    time_column=None,
                    cutoff_rule=cutoff_rule,
                    input_rows=None,
                    valid_time_rows=None,
                    retained_rows=None,
                    excluded_at_or_after_cutoff=None,
                    excluded_unparseable_time=None,
                    min_year=None,
                    max_year=None,
                    note="Registered input file does not exist.",
                )
            )
            continue

        try:
            columns = read_columns(input_path)
        except Exception as exc:
            audit_rows.append(
                AuditRow(
                    source_id=source_id,
                    input_path=source["path"],
                    output_path=None,
                    status="unsupported_or_unreadable",
                    time_column=None,
                    cutoff_rule=cutoff_rule,
                    input_rows=None,
                    valid_time_rows=None,
                    retained_rows=None,
                    excluded_at_or_after_cutoff=None,
                    excluded_unparseable_time=None,
                    min_year=None,
                    max_year=None,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        time_column = select_time_column(source, columns)
        if time_column is None:
            audit_rows.append(
                AuditRow(
                    source_id=source_id,
                    input_path=source["path"],
                    output_path=None,
                    status="excluded_no_row_time",
                    time_column=None,
                    cutoff_rule=cutoff_rule,
                    input_rows=None,
                    valid_time_rows=None,
                    retained_rows=0,
                    excluded_at_or_after_cutoff=None,
                    excluded_unparseable_time=None,
                    min_year=None,
                    max_year=None,
                    note=(
                        "No explicit row-level time column. Publication date, "
                        "filename, and years inside email text were not used."
                    ),
                )
            )
            continue

        try:
            frame = read_frame(input_path)
            years = parse_years(frame[time_column])
        except Exception as exc:
            audit_rows.append(
                AuditRow(
                    source_id=source_id,
                    input_path=source["path"],
                    output_path=None,
                    status="time_parse_failed",
                    time_column=time_column,
                    cutoff_rule=cutoff_rule,
                    input_rows=None,
                    valid_time_rows=None,
                    retained_rows=0,
                    excluded_at_or_after_cutoff=None,
                    excluded_unparseable_time=None,
                    min_year=None,
                    max_year=None,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        valid_mask = years.notna()
        retained_mask = valid_mask & years.lt(cutoff_year)
        excluded_newer_mask = valid_mask & years.ge(cutoff_year)
        filtered = frame.loc[retained_mask].copy()

        source_relative = Path(source["path"]).relative_to("data")
        output_path = output_dir / source_relative
        write_frame(filtered, output_path)

        filtered_source = dict(source)
        filtered_source["path"] = relative_project_path(output_path)
        filtered_source["sha256"] = sha256_file(output_path)
        filtered_source["temporal_filter"] = {
            "column": time_column,
            "rule": cutoff_rule,
            "input_rows": int(len(frame)),
            "retained_rows": int(retained_mask.sum()),
            "excluded_at_or_after_cutoff": int(excluded_newer_mask.sum()),
            "excluded_unparseable_time": int((~valid_mask).sum()),
        }
        retained_sources.append(filtered_source)

        valid_years = years.loc[valid_mask]
        audit_rows.append(
            AuditRow(
                source_id=source_id,
                input_path=source["path"],
                output_path=relative_project_path(output_path),
                status="filtered",
                time_column=time_column,
                cutoff_rule=cutoff_rule,
                input_rows=int(len(frame)),
                valid_time_rows=int(valid_mask.sum()),
                retained_rows=int(retained_mask.sum()),
                excluded_at_or_after_cutoff=int(excluded_newer_mask.sum()),
                excluded_unparseable_time=int((~valid_mask).sum()),
                min_year=int(valid_years.min()) if not valid_years.empty else None,
                max_year=int(valid_years.max()) if not valid_years.empty else None,
                note="Original file preserved; filtered copy created.",
            )
        )

    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame([asdict(row) for row in audit_rows]).to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    filtered_registry_path = ROOT / "config" / f"research_datasets_pre{cutoff_year}.json"
    filtered_registry = build_filtered_registry(
        registry,
        retained_sources,
        cutoff_year,
    )
    with filtered_registry_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered_registry, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    verification = verify_filtered_outputs(
        audit_rows,
        filtered_registry_path,
        cutoff_year,
    )
    summary_path = output_dir / "summary.json"
    summary = {
        "cutoff_rule": cutoff_rule,
        "source_registry": relative_project_path(registry_path),
        "filtered_registry": relative_project_path(filtered_registry_path),
        "output_directory": relative_project_path(output_dir),
        "sources_total": len(audit_rows),
        "sources_filtered": sum(row.status == "filtered" for row in audit_rows),
        "sources_excluded_no_row_time": sum(
            row.status == "excluded_no_row_time" for row in audit_rows
        ),
        "total_input_rows_for_filterable_sources": sum(
            row.input_rows or 0 for row in audit_rows if row.status == "filtered"
        ),
        "total_retained_rows": sum(
            row.retained_rows or 0 for row in audit_rows if row.status == "filtered"
        ),
        "total_excluded_at_or_after_cutoff": sum(
            row.excluded_at_or_after_cutoff or 0
            for row in audit_rows
            if row.status == "filtered"
        ),
        "total_excluded_unparseable_time": sum(
            row.excluded_unparseable_time or 0
            for row in audit_rows
            if row.status == "filtered"
        ),
        "verification": verification,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return audit_rows, manifest_path, filtered_registry_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter registered research datasets to rows before a cutoff year."
    )
    parser.add_argument("--cutoff-year", type=int, default=2020)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace an existing output directory before filtering.",
    )
    args = parser.parse_args()

    registry_path = args.registry.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.replace_output:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Use --replace-output to rebuild it."
            )
        if output_dir == ROOT or ROOT not in output_dir.parents:
            raise SystemExit(f"Refusing to replace unsafe output path: {output_dir}")
        shutil.rmtree(output_dir)

    rows, manifest_path, filtered_registry_path = filter_registry(
        registry_path=registry_path,
        output_dir=output_dir,
        cutoff_year=args.cutoff_year,
    )
    print(f"Rule: year < {args.cutoff_year}")
    print(f"Filtered sources: {sum(row.status == 'filtered' for row in rows)}")
    print(f"Retained rows: {sum(row.retained_rows or 0 for row in rows)}")
    print(f"Manifest: {relative_project_path(manifest_path)}")
    print(f"Filtered registry: {relative_project_path(filtered_registry_path)}")


if __name__ == "__main__":
    main()
