"""Candidate-only ML training with immutable evaluation and automatic gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

# ``python src/train.py`` puts ``src/`` rather than the project root first on
# sys.path.  Pin the root so all persisted classes have stable ``src.*``
# module names and can later be loaded by Streamlit or another process.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

try:
    from src.feedback_store import FeedbackStore, content_hash
    from src.classifiers import HybridPhishingClassifier
    from src.model_registry import (
        active_model_version_id,
        get_active_version,
        load_active_artifacts,
        promote_candidate,
        save_candidate,
    )
    from src.preprocessing import build_ml_document, detect_email_language, escape_csv_formula, parse_raw_email
except ImportError:  # Defensive fallback for copied standalone source files.
    from feedback_store import FeedbackStore, content_hash
    from src.classifiers import HybridPhishingClassifier
    from model_registry import active_model_version_id, get_active_version, load_active_artifacts, promote_candidate, save_candidate
    from preprocessing import build_ml_document, detect_email_language, escape_csv_formula, parse_raw_email


DATA_DIR = Path("data")
BENCHMARK_DIR = DATA_DIR / "benchmarks"
STANDARD_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssassin.csv",
]
MAX_FEEDBACK_WEIGHT = 5.0
BENCHMARK_SEED = 20260713
FEATURE_PIPELINE_VERSION = "bilingual-hybrid-v2"
VIETNAMESE_SOURCE_WEIGHT = 3.0
CURATED_EML_WEIGHT = 18.0
CURATED_EML_LABELS = DATA_DIR / "curated_eml_labels.csv"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def select_dataset_interactively(available_csvs: list[str]) -> str:
    print("\n[?] Các dataset có sẵn:")
    for index, filename in enumerate(available_csvs, start=1):
        print(f"    [{index}] {filename}")
    print(f"    [{len(available_csvs) + 1}] Gộp các dataset tiêu chuẩn")
    choice = input("Chọn dataset: ").strip()
    if not choice:
        return "combined_data_with_vn.csv" if "combined_data_with_vn.csv" in available_csvs else available_csvs[0]
    position = int(choice) - 1
    return "COMBINE_ALL" if position == len(available_csvs) else available_csvs[position]


def _extract_text_column(frame: pd.DataFrame) -> str | None:
    if "body" in frame.columns:
        return "body"
    if "text_combined" in frame.columns:
        return "text_combined"
    for column in frame.columns:
        if column not in {"label", "source"} and frame[column].dtype == object:
            return column
    return None


def _read_dataset(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
    frame = pd.read_csv(path)
    text_column = _extract_text_column(frame)
    if not text_column or "label" not in frame.columns:
        raise ValueError(f"Dataset {filename} phải có cột nội dung và cột label.")
    result = frame[[text_column, "label"]].rename(columns={text_column: "body"}).copy()
    # Subjects are high-signal in phishing while the old loader silently
    # discarded them whenever a separate body column was available.
    if text_column != "text_combined" and "subject" in frame.columns:
        subject = frame["subject"].fillna("").astype(str).str.strip()
        body = result["body"].fillna("").astype(str)
        result["body"] = np.where(subject.ne(""), "subject " + subject + " body " + body, body)
    result["source"] = frame["source"] if "source" in frame.columns else Path(filename).stem
    return result


def load_and_combine_standards() -> pd.DataFrame:
    frames = [_read_dataset(filename) for filename in STANDARD_FILES if (DATA_DIR / filename).exists()]
    if not frames:
        raise FileNotFoundError("Không tìm thấy dataset tiêu chuẩn để gộp.")
    return pd.concat(frames, ignore_index=True)


def _load_curated_eml_examples() -> pd.DataFrame:
    """Load only explicitly reviewed local EML examples.

    The label file keys records by a canonical raw-content hash.  New files in
    the folder are never silently treated as safe or phishing; they must first
    be reviewed and added to ``curated_eml_labels.csv``.
    """
    if not CURATED_EML_LABELS.exists():
        return pd.DataFrame(columns=["body", "label", "source"])
    labels = pd.read_csv(CURATED_EML_LABELS)
    required = {"raw_content_hash", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{CURATED_EML_LABELS} thiếu cột label/hash bắt buộc.")
    mapping = {
        str(row.raw_content_hash): int(row.label)
        for row in labels.itertuples()
        if int(row.label) in {0, 1}
    }
    mail_directory = next(
        (path for path in PROJECT_ROOT.iterdir() if path.is_dir() and path.name.casefold().startswith("mail")),
        None,
    )
    if mail_directory is None:
        return pd.DataFrame(columns=["body", "label", "source"])
    rows: list[dict[str, Any]] = []
    for path in sorted(mail_directory.glob("*.eml")):
        raw_content = path.read_text(encoding="utf-8", errors="replace")
        label = mapping.get(content_hash(raw_content))
        if label is None:
            continue
        parsed = parse_raw_email(raw_content)
        text = parsed.get("clean_text") or parsed.get("body") or raw_content
        rows.append({"body": text, "label": label, "source": "curated_eml_reviewed"})
    return pd.DataFrame(rows, columns=["body", "label", "source"])


def _uses_bilingual_production_dataset(dataset_name: str | list[str]) -> bool:
    selected = _selected_dataset_files(dataset_name)
    return "combined_data_with_vn.csv" in selected


def _load_training_source(dataset_name: str | list[str]) -> pd.DataFrame:
    if dataset_name == "COMBINE_ALL":
        data = load_and_combine_standards()
    elif isinstance(dataset_name, list):
        if not dataset_name:
            raise ValueError("Cần chọn ít nhất một dataset.")
        data = pd.concat([_read_dataset(filename) for filename in dataset_name], ignore_index=True)
    else:
        data = _read_dataset(str(dataset_name))
    if _uses_bilingual_production_dataset(dataset_name):
        curated = _load_curated_eml_examples()
        if not curated.empty:
            data = pd.concat([data, curated], ignore_index=True)
    return data


def _selected_dataset_files(dataset_name: str | list[str]) -> list[str]:
    """Return the exact CSV files selected for one independent training run."""
    if dataset_name == "COMBINE_ALL":
        return [filename for filename in STANDARD_FILES if (DATA_DIR / filename).exists()]
    if isinstance(dataset_name, list):
        return [str(filename) for filename in dataset_name]
    return [str(dataset_name)]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest(dataset_name: str | list[str]) -> dict[str, Any]:
    """Record immutable provenance for the files supplied to a training click."""
    selected_files = _selected_dataset_files(dataset_name)
    files: list[dict[str, Any]] = []
    for filename in selected_files:
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy dataset: {path}")
        files.append(
            {
                "filename": filename,
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        )
    manifest = {
        "selection": "COMBINE_ALL" if dataset_name == "COMBINE_ALL" else selected_files,
        "files": files,
    }
    if _uses_bilingual_production_dataset(dataset_name) and CURATED_EML_LABELS.exists():
        manifest["curated_eml_labels"] = {
            "filename": CURATED_EML_LABELS.name,
            "bytes": int(CURATED_EML_LABELS.stat().st_size),
            "sha256": _file_sha256(CURATED_EML_LABELS),
        }
    return manifest


def _prepare_base_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    data = frame.dropna(subset=["body", "label"]).copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[data["label"].isin([0, 1])].copy()
    data["label"] = data["label"].astype(int)
    data["body"] = data["body"].astype(str)
    data["body_cleaned"] = data["body"].map(build_ml_document)
    data = data[data["body_cleaned"].str.strip().ne("")].copy()
    data["content_hash"] = data["body"].map(content_hash)
    data["language"] = data["body"].map(detect_email_language)

    conflicting = data.groupby("content_hash")["label"].nunique()
    conflict_hashes = set(conflicting[conflicting > 1].index)
    conflict_count = len(conflict_hashes)
    if conflict_hashes:
        data = data.loc[~data["content_hash"].isin(conflict_hashes)].copy()
    duplicate_count = int(data.duplicated(subset=["content_hash", "label"]).sum())
    data = data.drop_duplicates(subset=["content_hash", "label"], keep="first").reset_index(drop=True)
    if data["label"].nunique() != 2:
        raise ValueError("Dữ liệu sau làm sạch phải còn đủ hai nhãn Safe và Phishing.")
    return data, duplicate_count, conflict_count


def _dataset_fingerprint(data: pd.DataFrame) -> str:
    values = sorted(f"{row.content_hash}:{int(row.label)}" for row in data.itertuples())
    payload = FEATURE_PIPELINE_VERSION + "\n" + "\n".join(values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_or_create_benchmark(data: pd.DataFrame, dataset_fingerprint: str) -> tuple[pd.DataFrame, str]:
    """Create a fixed benchmark once; future training never modifies it."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_id = f"benchmark-{dataset_fingerprint}"
    path = BENCHMARK_DIR / f"{benchmark_id}.csv"
    if path.exists():
        benchmark = pd.read_csv(path)
        required = {"body", "body_cleaned", "label", "content_hash"}
        if not required.issubset(benchmark.columns):
            raise ValueError(f"Benchmark {path} không đúng schema v2.")
        return benchmark, benchmark_id

    _, benchmark = train_test_split(
        data,
        test_size=0.20,
        random_state=BENCHMARK_SEED,
        stratify=data["label"],
    )
    benchmark = benchmark[["body", "body_cleaned", "label", "content_hash", "language"]].copy()
    benchmark.to_csv(path, index=False, encoding="utf-8")
    manifest = {
        "benchmark_id": benchmark_id,
        "dataset_fingerprint": dataset_fingerprint,
        "created_at": _now(),
        "seed": BENCHMARK_SEED,
        "samples": int(len(benchmark)),
        "label_counts": {str(key): int(value) for key, value in benchmark["label"].value_counts().items()},
        "language_counts": {str(key): int(value) for key, value in benchmark["language"].value_counts().items()},
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
    }
    with open(BENCHMARK_DIR / f"{benchmark_id}.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return benchmark, benchmark_id


def _metrics(y_true, probabilities) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    false_positives = int(((y_true == 0) & (predicted == 1)).sum())
    false_negatives = int(((y_true == 1) & (predicted == 0)).sum())
    result: dict[str, float | int | None] = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
    }
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    result["pr_auc"] = float(auc(recall, precision))
    return result


def _evaluate(vectorizer, model, data: pd.DataFrame) -> tuple[dict[str, float | int | None], np.ndarray]:
    probabilities = model.predict_proba(vectorizer.transform(data["body_cleaned"]))[:, 1]
    return _metrics(data["label"], probabilities), probabilities


def _evaluate_by_language(data: pd.DataFrame, probabilities: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    languages = data.get("language")
    if languages is None:
        languages = data["body"].astype(str).map(detect_email_language)
    results: dict[str, dict[str, float | int | None]] = {}
    for language in ("vi", "en", "mixed", "unknown"):
        mask = languages.astype(str).eq(language).to_numpy()
        if not mask.any():
            continue
        labels = data.loc[mask, "label"]
        item: dict[str, float | int | None] = {
            "samples": int(mask.sum()),
            "phishing_samples": int((labels == 1).sum()),
            "safe_samples": int((labels == 0).sum()),
        }
        if labels.nunique() == 2:
            item.update(_metrics(labels, probabilities[mask]))
        results[language] = item
    return results


def _quality_gate(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    language_metrics: dict[str, dict[str, float | int | None]] | None = None,
) -> dict[str, Any]:
    """Safety-oriented gate: candidate may not miss more phishing emails."""
    checks: list[dict[str, Any]] = []
    if candidate["samples"] < 100 or candidate["recall"] is None:
        checks.append({"name": "benchmark đủ lớn", "passed": False, "detail": "Benchmark cần ít nhất 100 mẫu."})
    if baseline is None:
        # First installation has no active model to compare against.  It may be
        # bootstrapped only if it clears conservative absolute safety metrics.
        checks.extend(
            [
                {
                    "name": "bootstrap recall tối thiểu 90%",
                    "passed": candidate["recall"] >= 0.90,
                    "detail": f"recall candidate={candidate['recall']:.4f}",
                },
                {
                    "name": "bootstrap precision tối thiểu 80%",
                    "passed": candidate["precision"] >= 0.80,
                    "detail": f"precision candidate={candidate['precision']:.4f}",
                },
                {
                    "name": "bootstrap F1 tối thiểu 85%",
                    "passed": candidate["f1_score"] >= 0.85,
                    "detail": f"F1 candidate={candidate['f1_score']:.4f}",
                },
            ]
        )
    else:
        checks.extend(
            [
                {
                    "name": "không tăng bỏ lọt phishing",
                    "passed": candidate["false_negatives"] <= baseline["false_negatives"],
                    "detail": f"FN candidate={candidate['false_negatives']}, baseline={baseline['false_negatives']}",
                },
                {
                    "name": "recall không giảm quá 0.2 điểm %",
                    "passed": candidate["recall"] + 0.002 >= baseline["recall"],
                    "detail": f"recall candidate={candidate['recall']:.4f}, baseline={baseline['recall']:.4f}",
                },
                {
                    "name": "precision không giảm quá 1 điểm %",
                    "passed": candidate["precision"] + 0.01 >= baseline["precision"],
                    "detail": f"precision candidate={candidate['precision']:.4f}, baseline={baseline['precision']:.4f}",
                },
                {
                    "name": "F1 không giảm quá 0.2 điểm %",
                    "passed": candidate["f1_score"] + 0.002 >= baseline["f1_score"],
                    "detail": f"F1 candidate={candidate['f1_score']:.4f}, baseline={baseline['f1_score']:.4f}",
                },
            ]
        )
    # Only enforce a bilingual floor when the immutable benchmark actually
    # contains enough labeled Vietnamese and English examples.  Tiny custom
    # datasets used by analysts remain valid training inputs.
    for language, title in (("vi", "Vietnamese"), ("en", "English")):
        metrics = (language_metrics or {}).get(language, {})
        if metrics.get("samples", 0) >= 30 and metrics.get("phishing_samples", 0) >= 10 and metrics.get("safe_samples", 0) >= 10:
            recall = metrics.get("recall")
            precision = metrics.get("precision")
            passed = recall is not None and precision is not None and recall >= 0.90 and precision >= 0.80
            checks.append(
                {
                    "name": f"bilingual floor {title}",
                    "passed": passed,
                    "detail": f"recall={recall:.4f}, precision={precision:.4f}" if recall is not None and precision is not None else "insufficient class coverage",
                }
            )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _prepare_feedback(
    base_hashes: set[str], benchmark_hashes: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    store = FeedbackStore()
    train_feedback, holdout_feedback = store.accepted_for_training()
    report = {
        "accepted_trainable": int(len(train_feedback) + len(holdout_feedback)),
        "feedback_train_used": 0,
        "feedback_holdout": 0,
        "feedback_excluded_base_overlap": 0,
        "feedback_excluded_benchmark_overlap": 0,
    }

    def clean_feedback(frame: pd.DataFrame, bucket: str) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        result = frame.copy()
        benchmark_overlap = result["content_hash"].isin(benchmark_hashes)
        base_overlap = result["content_hash"].isin(base_hashes)
        report["feedback_excluded_benchmark_overlap"] += int(benchmark_overlap.sum())
        report["feedback_excluded_base_overlap"] += int((base_overlap & ~benchmark_overlap).sum())
        result = result.loc[~(benchmark_overlap | base_overlap)].copy()
        result["body_cleaned"] = result["body"].astype(str).map(build_ml_document)
        result = result[result["body_cleaned"].str.strip().ne("")].copy()
        report[bucket] = int(len(result))
        return result

    return clean_feedback(train_feedback, "feedback_train_used"), clean_feedback(holdout_feedback, "feedback_holdout"), report


def _write_benchmark_errors(
    candidate_path: str, benchmark: pd.DataFrame, probabilities: np.ndarray, benchmark_id: str
) -> None:
    predicted = (probabilities >= 0.5).astype(int)
    errors = benchmark.loc[predicted != benchmark["label"].to_numpy(), ["body", "label"]].copy()
    if errors.empty:
        errors = pd.DataFrame(columns=["body", "label", "predicted", "error_type", "benchmark_id"])
    else:
        errors["predicted"] = predicted[predicted != benchmark["label"].to_numpy()]
        errors["error_type"] = np.where(errors["label"] == 1, "False Negative", "False Positive")
        errors["benchmark_id"] = benchmark_id
        errors["body"] = errors["body"].astype(str).map(escape_csv_formula)
    Path(candidate_path).mkdir(parents=True, exist_ok=True)
    errors.to_csv(Path(candidate_path) / "benchmark_missed_cases.csv", index=False, encoding="utf-8")


def run_training_pipeline(
    dataset_name: str | list[str],
    feedback_weight: float = 2.0,
    progress_callback=None,
    auto_promote: bool = True,
) -> dict[str, Any]:
    """Train a candidate, evaluate it on immutable data, then optionally promote it.

    The active artifacts are untouched when the gate fails.  This is the only
    entry point used by the Streamlit buttons and the command-line script.
    """
    def report(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    try:
        report(3, "Đang tải và kiểm tra dữ liệu nguồn…")
        source_manifest = _dataset_manifest(dataset_name)
        base, duplicates_removed, base_conflicts = _prepare_base_data(_load_training_source(dataset_name))
        fingerprint = _dataset_fingerprint(base)
        benchmark_path = BENCHMARK_DIR / f"benchmark-{fingerprint}.csv"
        benchmark_previously_existed = benchmark_path.exists()
        benchmark, benchmark_id = _get_or_create_benchmark(base, fingerprint)
        benchmark_hashes = set(benchmark["content_hash"])
        train_base = base.loc[~base["content_hash"].isin(benchmark_hashes)].copy()
        if train_base["label"].nunique() != 2:
            raise ValueError("Tập train sau khi tách benchmark không còn đủ hai nhãn.")

        report(22, "Đang nạp feedback đã xác minh và loại bỏ leakage…")
        feedback_train, feedback_holdout, feedback_report = _prepare_feedback(
            set(base["content_hash"]), benchmark_hashes
        )
        bounded_weight = min(MAX_FEEDBACK_WEIGHT, max(1.0, float(feedback_weight)))

        train_text = train_base["body_cleaned"].reset_index(drop=True)
        train_label = train_base["label"].reset_index(drop=True)
        source_weights = np.ones(len(train_base), dtype=float)
        if "source" in train_base.columns:
            vietnamese_source = train_base["source"].astype(str).str.casefold().eq("synthetic_vn").to_numpy()
            source_weights[vietnamese_source] = VIETNAMESE_SOURCE_WEIGHT
            curated_source = train_base["source"].astype(str).str.casefold().eq("curated_eml_reviewed").to_numpy()
            source_weights[curated_source] = CURATED_EML_WEIGHT
        if not feedback_train.empty:
            train_text = pd.concat([train_text, feedback_train["body_cleaned"]], ignore_index=True)
            train_label = pd.concat([train_label, feedback_train["label"].astype(int)], ignore_index=True)
        weights = source_weights
        if not feedback_train.empty:
            confidence = feedback_train["confidence"].astype(float).clip(0, 1).to_numpy()
            feedback_weights = 1.0 + (bounded_weight - 1.0) * confidence
            weights = np.concatenate([weights, feedback_weights])

        report(42, "Đang vector hoá TF-IDF và huấn luyện candidate…")
        vietnamese_stop_words = [
            "và", "của", "là", "trong", "để", "tự", "cho", "này", "các", "bạn", "có", "một", "được",
            "với", "những", "ra", "thì", "khi", "vào", "đã", "đang", "sẽ", "tại", "như", "nhưng", "cũng",
            "đến", "ở", "đó", "nếu", "qua", "bằng", "chỉ", "từ", "lên", "đi", "về", "lại", "mới", "cả",
            "ngày", "năm", "sau", "trước", "tôi", "anh", "chị", "em", "không", "ai", "gì",
        ]
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        stop_words=list(ENGLISH_STOP_WORDS) + vietnamese_stop_words,
                        max_features=22_000,
                        min_df=2,
                        sublinear_tf=True,
                        norm="l2",
                        ngram_range=(1, 2),
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        max_features=22_000,
                        sublinear_tf=True,
                        norm="l2",
                    ),
                ),
            ]
        )
        train_matrix = vectorizer.fit_transform(train_text)
        # The discriminative model receives the primary vote.  NB remains a
        # small smoothing vote for short/sparse emails without masking clear
        # phishing phrases captured by Logistic Regression.
        model = HybridPhishingClassifier(logistic_weight=0.90, nb_alpha=0.35)
        model.fit(train_matrix, train_label, sample_weight=weights)
        baseline_model_version = active_model_version_id()
        baseline_version = get_active_version()
        baseline_vectorizer, baseline_model = load_active_artifacts()

        report(68, "Đang đánh giá candidate trên benchmark bất biến…")
        candidate_metrics, candidate_probabilities = _evaluate(vectorizer, model, benchmark)
        baseline_metrics = None
        if baseline_vectorizer is not None and baseline_model is not None:
            baseline_metrics, _ = _evaluate(baseline_vectorizer, baseline_model, benchmark)
        feedback_holdout_metrics = None
        if not feedback_holdout.empty:
            feedback_holdout_metrics, _ = _evaluate(vectorizer, model, feedback_holdout)
        language_performance = _evaluate_by_language(benchmark, candidate_probabilities)
        gate = _quality_gate(candidate_metrics, baseline_metrics, language_performance)

        base_label_counts = {
            str(label): int(count)
            for label, count in base["label"].value_counts().sort_index().items()
        }
        dataset_manifest = {
            **source_manifest,
            "semantic_fingerprint": fingerprint,
            "after_validation": {
                "samples": int(len(base)),
                "label_counts": base_label_counts,
                "language_counts": {str(key): int(value) for key, value in base["language"].value_counts().items()},
                "duplicates_removed": duplicates_removed,
                "conflicting_labels_quarantined": base_conflicts,
            },
        }
        baseline_fingerprint = (baseline_version or {}).get("metadata", {}).get("dataset_fingerprint")
        comparison = {
            "baseline_model_version": baseline_model_version,
            "baseline_dataset_fingerprint": baseline_fingerprint,
            "same_source_fingerprint": baseline_fingerprint == fingerprint if baseline_fingerprint else None,
            "evaluation_basis": (
                "Baseline và candidate đều được chấm trên benchmark của lần chạy này. "
                "Nếu fingerprint khác, đây chỉ là so sánh an toàn để quyết định triển khai, "
                "không phải xu hướng hiệu năng giữa hai dataset khác nhau."
            ),
        }

        metadata: dict[str, Any] = {
            "schema_version": 4,
            "dataset_name": dataset_name,
            "dataset_fingerprint": fingerprint,
            "dataset_manifest": dataset_manifest,
            "benchmark_id": benchmark_id,
            "benchmark_samples": int(len(benchmark)),
            "benchmark_lifecycle": {
                "immutable": True,
                "reused": benchmark_previously_existed,
                "created_for_this_run": not benchmark_previously_existed,
                "split_seed": BENCHMARK_SEED,
            },
            "total_samples": int(len(base)),
            "train_samples": int(len(train_text)),
            "feedback_weight": bounded_weight,
            "feature_pipeline": {
                "version": FEATURE_PIPELINE_VERSION,
                "representation": "word bigrams + character 3-5 grams + bounded security tokens",
                "language_detector": "deterministic vi/en/mixed/unknown",
            },
            "classifier": {
                "name": "HybridPhishingClassifier",
                "members": ["LogisticRegression(C=8.0, balanced)", "MultinomialNB(alpha=0.35)"],
                "probability_weights": {"logistic_regression": 0.90, "multinomial_nb": 0.10},
            },
            "vietnamese_source_weight": VIETNAMESE_SOURCE_WEIGHT,
            "curated_eml_weight": CURATED_EML_WEIGHT,
            "feedback": feedback_report,
            "training_policy": {
                "mode": "full_rebuild_from_selected_dataset",
                "model_warm_start": False,
                "what_carries_between_runs": (
                    "Chỉ feedback đã được analyst chấp nhận và đủ điều kiện; "
                    "trọng số/TF-IDF của model cũ không được tái sử dụng."
                ),
                "llm_feedback_policy": (
                    "Feedback chỉ sửa Local LLM được dùng làm bài học RAG, "
                    "không được dùng để huấn luyện classifier ML."
                ),
            },
            "data_quality": {
                "base_duplicates_removed": duplicates_removed,
                "base_conflicts_quarantined": base_conflicts,
            },
            "baseline_model_version": baseline_model_version,
            "baseline_metrics": baseline_metrics,
            "baseline_comparison": comparison,
            "immutable_benchmark_metrics": candidate_metrics,
            "language_performance": language_performance,
            "feedback_holdout_metrics": feedback_holdout_metrics,
            "quality_gate": gate,
            # Compatibility fields for the existing dashboard.
            **candidate_metrics,
            "training_time": _now(),
        }

        report(86, "Đang lưu candidate và kiểm tra quality gate…")
        candidate = save_candidate(vectorizer, model, metadata)
        _write_benchmark_errors(candidate["path"], benchmark, candidate_probabilities, benchmark_id)
        promoted = False
        active_version = None
        if auto_promote and gate["passed"]:
            active = promote_candidate(candidate["candidate_id"])
            promoted = True
            active_version = active["version_id"]
        report(100, "Đã promote candidate qua gate." if promoted else "Candidate được lưu; gate không cho phép triển khai.")
        return {
            "success": True,
            "candidate_id": candidate["candidate_id"],
            "candidate_path": candidate["path"],
            "promoted": promoted,
            "active_version": active_version,
            "quality_gate": gate,
            "metadata": metadata,
        }
    except Exception as error:
        report(100, "Huấn luyện candidate bị dừng.")
        return {"success": False, "error": str(error), "promoted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện candidate phishing classifier có quality gate.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--feedback-weight", type=float, default=2.0)
    parser.add_argument("--no-auto-promote", action="store_true")
    args = parser.parse_args()
    available = [path.name for path in DATA_DIR.glob("*.csv") if path.name not in {"user_feedback.csv"}]
    if not available:
        raise SystemExit("Thư mục data chưa có dataset CSV.")
    dataset = args.dataset or (
        select_dataset_interactively(available)
        if sys.stdin.isatty()
        else ("combined_data_with_vn.csv" if "combined_data_with_vn.csv" in available else available[0])
    )
    result = run_training_pipeline(dataset, args.feedback_weight, auto_promote=not args.no_auto_promote)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
