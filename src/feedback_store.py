"""Governed storage for analyst feedback and LLM error observations.

The old CSV feedback loop mixed unreviewed labels, LLM prose and benchmark
errors directly into model training.  This module is deliberately small and
uses SQLite from the standard library so every feedback item has a durable
state before it can influence either ML training or RAG.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken

try:
    from .auth import Actor, ROLE_ANALYST, ROLE_SYSTEM, require_role
except ImportError:  # pragma: no cover - supports ``python src/train.py``
    from auth import Actor, ROLE_ANALYST, ROLE_SYSTEM, require_role


DEFAULT_DB_PATH = Path("data") / "feedback_registry.sqlite3"
# Feedback categories are deliberately separated: an LLM-only mistake is a
# useful retrieval lesson, but it must never alter the ML classifier because
# the ML classifier was already correct in that case.
ML_FEEDBACK_SOURCE = "user_confirmed"
LLM_FEEDBACK_SOURCE = "llm_user_confirmed"
ML_AND_LLM_FEEDBACK_SOURCE = "ml_and_llm_user_confirmed"

# Only corrections to a wrong ML prediction are trainable.  All explicitly
# confirmed corrections may become RAG context for the local LLM.
TRAINABLE_SOURCES = {ML_FEEDBACK_SOURCE, ML_AND_LLM_FEEDBACK_SOURCE}
RAG_SOURCES = {
    ML_FEEDBACK_SOURCE,
    LLM_FEEDBACK_SOURCE,
    ML_AND_LLM_FEEDBACK_SOURCE,
}
VALID_STATUSES = {"pending", "accepted", "rejected", "conflict", "observed"}
VALID_SOURCES = TRAINABLE_SOURCES | RAG_SOURCES | {"legacy_import"}
MAX_FEEDBACK_BODY_CHARS = 15_000
MAX_FEEDBACK_ANALYSIS_CHARS = 4_000
MAX_REVIEW_NOTE_CHARS = 1_200
ENCRYPTED_VALUE_PREFIX = "enc:v1:"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonicalize_body(body: str) -> str:
    """Return a stable representation used only for dedupe and split routing."""
    text = unicodedata.normalize("NFKC", str(body or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def content_hash(body: str) -> str:
    return hashlib.sha256(canonicalize_body(body).encode("utf-8")).hexdigest()


def stable_holdout_bucket(content_digest: str, buckets: int = 5) -> int:
    """Stable hash bucket; bucket 0 is an immutable feedback holdout."""
    return int(content_digest[:12], 16) % buckets


def merge_feedback_source(existing_source: str, incoming_source: str) -> str:
    """Preserve whether a confirmed case corrects ML, LLM, or both.

    A user may submit the same confirmed label again after adding a reason.  A
    historical ML-only record must then be upgraded to a combined record when
    the LLM was wrong too, rather than discarded as a duplicate.
    """
    existing_source = str(existing_source or "")
    incoming_source = str(incoming_source or "")
    known_sources = {
        ML_FEEDBACK_SOURCE,
        LLM_FEEDBACK_SOURCE,
        ML_AND_LLM_FEEDBACK_SOURCE,
    }
    if existing_source not in known_sources:
        return incoming_source
    if incoming_source not in known_sources:
        return existing_source

    corrects_ml = (
        existing_source in TRAINABLE_SOURCES
        or incoming_source in TRAINABLE_SOURCES
    )
    corrects_llm = (
        existing_source in {LLM_FEEDBACK_SOURCE, ML_AND_LLM_FEEDBACK_SOURCE}
        or incoming_source in {LLM_FEEDBACK_SOURCE, ML_AND_LLM_FEEDBACK_SOURCE}
    )
    if corrects_ml and corrects_llm:
        return ML_AND_LLM_FEEDBACK_SOURCE
    if corrects_ml:
        return ML_FEEDBACK_SOURCE
    return LLM_FEEDBACK_SOURCE


class FeedbackStore:
    """SQLite-backed feedback registry with dedupe and human-review gates."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, encryption_key: str | bytes | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = self._configure_cipher(encryption_key)
        self.initialize()

    @staticmethod
    def _encryption_is_required() -> bool:
        """Production feedback must never silently fall back to plaintext."""

        return os.environ.get("AIC211_ENV", "development").strip().lower() == "production"

    def _configure_cipher(self, encryption_key: str | bytes | None) -> Fernet | None:
        key = encryption_key if encryption_key is not None else os.environ.get("AIC211_FEEDBACK_ENCRYPTION_KEY", "")
        if not key:
            if self._encryption_is_required():
                raise RuntimeError(
                    "AIC211_FEEDBACK_ENCRYPTION_KEY is required when AIC211_ENV=production. "
                    "Generate a Fernet key and inject it through the secret manager."
                )
            return None
        try:
            return Fernet(key.encode("ascii") if isinstance(key, str) else bytes(key))
        except (TypeError, ValueError) as error:
            raise RuntimeError("AIC211_FEEDBACK_ENCRYPTION_KEY is not a valid Fernet key.") from error

    def _protect(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        if self._cipher is None:
            return text
        return ENCRYPTED_VALUE_PREFIX + self._cipher.encrypt(text.encode("utf-8")).decode("ascii")

    def _reveal(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        if not text.startswith(ENCRYPTED_VALUE_PREFIX):
            return text
        if self._cipher is None:
            raise RuntimeError("Encrypted feedback cannot be read without AIC211_FEEDBACK_ENCRYPTION_KEY.")
        try:
            return self._cipher.decrypt(text[len(ENCRYPTED_VALUE_PREFIX):].encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as error:
            raise RuntimeError("Feedback encryption integrity check failed.") from error

    def _decode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        for field in ("body", "analyst_id", "review_note", "llm_analysis"):
            if field in record:
                record[field] = self._reveal(record[field])
        return record

    def _harden_file_permissions(self) -> None:
        """Restrict files on POSIX; Windows ACLs remain a deployment requirement."""

        if os.name == "nt":
            return
        for path in (self.db_path, self.db_path.with_name(self.db_path.name + "-wal"), self.db_path.with_name(self.db_path.name + "-shm")):
            if path.exists():
                path.chmod(0o600)

    def _writer_subject(self, actor: Actor | None) -> str:
        """Return an authorised, non-forgeable reviewer identity in production."""

        if actor is None:
            if self._encryption_is_required():
                raise PermissionError("Production feedback writes require an authenticated analyst or system actor.")
            return ""
        return require_role(actor, ROLE_ANALYST, ROLE_SYSTEM).subject

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()
            self._harden_file_permissions()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    body TEXT NOT NULL,
                    label INTEGER NOT NULL CHECK (label IN (0, 1)),
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected', 'conflict', 'observed')),
                    analyst_id TEXT,
                    review_note TEXT,
                    model_version TEXT,
                    ml_probability REAL,
                    llm_model TEXT,
                    prompt_version TEXT,
                    llm_analysis TEXT,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback_records(content_hash);
                CREATE INDEX IF NOT EXISTS idx_feedback_status_source ON feedback_records(status, source);
                """
            )
            if self._cipher is not None:
                rows = connection.execute(
                    "SELECT feedback_id, body, analyst_id, review_note, llm_analysis FROM feedback_records"
                ).fetchall()
                for row in rows:
                    values = {
                        field: row[field]
                        for field in ("body", "analyst_id", "review_note", "llm_analysis")
                    }
                    if not any(value is not None and not str(value).startswith(ENCRYPTED_VALUE_PREFIX) for value in values.values()):
                        continue
                    connection.execute(
                        """
                        UPDATE feedback_records
                        SET body = ?, analyst_id = ?, review_note = ?, llm_analysis = ?
                        WHERE feedback_id = ?
                        """,
                        (
                            self._protect(values["body"]),
                            self._protect(values["analyst_id"]),
                            self._protect(values["review_note"]),
                            self._protect(values["llm_analysis"]),
                            row["feedback_id"],
                        ),
                    )

    @staticmethod
    def _validate(body: str, label: int, confidence: float, status: str) -> None:
        if len(str(body)) > MAX_FEEDBACK_BODY_CHARS:
            raise ValueError(f"Feedback body exceeds the {MAX_FEEDBACK_BODY_CHARS:,}-character limit.")
        if not canonicalize_body(body):
            raise ValueError("Nội dung email phản hồi không được để trống.")
        if label not in (0, 1):
            raise ValueError("Nhãn phản hồi phải là 0 (Safe) hoặc 1 (Phishing).")
        if not 0 <= float(confidence) <= 1:
            raise ValueError("Độ tin cậy của phản hồi phải nằm trong khoảng 0 đến 1.")
        if status not in VALID_STATUSES:
            raise ValueError(f"Trạng thái phản hồi không hợp lệ: {status}")

    def submit(
        self,
        *,
        body: str,
        label: int,
        source: str,
        confidence: float = 1.0,
        status: str = "pending",
        analyst_id: str = "",
        review_note: str = "",
        model_version: str = "",
        ml_probability: float | None = None,
        llm_model: str = "",
        prompt_version: str = "",
        llm_analysis: str = "",
        actor: Actor | None = None,
    ) -> dict[str, Any]:
        """Store one feedback item without allowing it to train immediately.

        Same normalized content and label is idempotent.  A different label for
        the same content quarantines all active versions of that item as a
        conflict so neither side can reach ML or RAG until an analyst resolves
        it.
        """
        label = int(label)
        confidence = float(confidence)
        self._validate(body, label, confidence, status)
        if source not in VALID_SOURCES:
            raise ValueError("Feedback source is not permitted.")
        if len(str(review_note)) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError("Review note exceeds the permitted length.")
        authenticated_subject = self._writer_subject(actor)
        if authenticated_subject:
            analyst_id = authenticated_subject
        digest = content_hash(body)
        feedback_id = f"fb_{digest[:20]}_{label}"
        now = utc_now()
        compact_analysis = str(llm_analysis or "").strip()[:MAX_FEEDBACK_ANALYSIS_CHARS]

        with self._connection() as connection:
            # Serialize the decision for this database.  This prevents concurrent
            # submitters from surfacing raw SQLite uniqueness errors.
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT feedback_id, label, source, status, confidence, analyst_id,
                       review_note, llm_analysis
                FROM feedback_records
                WHERE content_hash = ?
                """,
                (digest,),
            ).fetchall()

            same_label = next((row for row in rows if row["label"] == label), None)
            if same_label is not None:
                # Never revive or overwrite a quarantined conflict.  It still
                # requires an explicit analyst resolution.
                if same_label["status"] == "conflict":
                    return {
                        "action": "duplicate",
                        "feedback_id": same_label["feedback_id"],
                        "status": same_label["status"],
                        "content_hash": digest,
                    }

                # A new unreviewed submission must not be able to enrich or
                # widen the scope of an already approved lesson.  Without
                # this guard, a pending UI submission could turn an ML-only
                # accepted record into ML+LLM feedback (and alter its RAG
                # lesson) without another analyst decision.
                if same_label["status"] in {"accepted", "rejected"} and status == "pending":
                    return {
                        "action": "duplicate",
                        "feedback_id": same_label["feedback_id"],
                        "status": same_label["status"],
                        "content_hash": digest,
                    }

                merged_source = merge_feedback_source(same_label["source"], source)
                existing_analysis = str(self._reveal(same_label["llm_analysis"]) or "").strip()
                merged_analysis = compact_analysis or existing_analysis
                # An explicit user confirmation can promote a legacy pending
                # row with the same label, but a rejected record stays rejected.
                merged_status = same_label["status"]
                if (
                    merged_status == "pending"
                    and status == "accepted"
                    and source in RAG_SOURCES
                ):
                    merged_status = "accepted"

                needs_update = (
                    merged_source != same_label["source"]
                    or merged_analysis != existing_analysis
                    or merged_status != same_label["status"]
                )
                if needs_update:
                    connection.execute(
                        """
                        UPDATE feedback_records
                        SET source = ?, status = ?, confidence = ?,
                            analyst_id = CASE WHEN ? = '' THEN analyst_id ELSE ? END,
                            review_note = CASE WHEN ? = '' THEN review_note ELSE ? END,
                            ml_probability = ?, llm_model = ?, prompt_version = ?,
                            llm_analysis = ?, reviewed_at = ?
                        WHERE feedback_id = ?
                        """,
                        (
                            merged_source,
                            merged_status,
                            confidence,
                            str(analyst_id),
                            self._protect(analyst_id),
                            str(review_note),
                            self._protect(review_note),
                            ml_probability,
                            str(llm_model),
                            str(prompt_version),
                            self._protect(merged_analysis),
                            now if merged_status == "accepted" else None,
                            same_label["feedback_id"],
                        ),
                    )
                    return {
                        "action": "updated",
                        "feedback_id": same_label["feedback_id"],
                        "status": merged_status,
                        "content_hash": digest,
                    }
                return {
                    "action": "duplicate",
                    "feedback_id": same_label["feedback_id"],
                    "status": same_label["status"],
                    "content_hash": digest,
                }

            has_conflict = bool(rows)
            final_status = "conflict" if has_conflict else status
            if has_conflict:
                connection.execute(
                    """
                    UPDATE feedback_records
                    SET status = 'conflict', reviewed_at = ?,
                        review_note = CASE
                            WHEN review_note IS NULL OR review_note = ''
                            THEN 'Tự động cách ly: cùng nội dung có nhãn trái ngược.'
                            ELSE review_note
                        END
                    WHERE content_hash = ? AND status IN ('pending', 'accepted', 'observed')
                    """,
                    (now, digest),
                )

            connection.execute(
                """
                INSERT INTO feedback_records (
                    feedback_id, content_hash, body, label, source, confidence,
                    status, analyst_id, review_note, model_version, ml_probability,
                    llm_model, prompt_version, llm_analysis, created_at, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    digest,
                    self._protect(body),
                    label,
                    str(source),
                    confidence,
                    final_status,
                    self._protect(analyst_id),
                    self._protect(review_note),
                    str(model_version),
                    ml_probability,
                    str(llm_model),
                    str(prompt_version),
                    self._protect(compact_analysis),
                    now,
                    now if final_status in {"observed", "conflict"} else None,
                ),
            )

        return {
            "action": "conflict" if has_conflict else "created",
            "feedback_id": feedback_id,
            "status": final_status,
            "content_hash": digest,
        }

    def review(
        self,
        feedback_id: str,
        decision: str,
        reviewer: str = "analyst",
        note: str = "",
        actor: Actor | None = None,
    ) -> None:
        """Accept or reject a pending feedback record after analyst review."""
        if not str(reviewer).strip():
            raise ValueError("Reviewer identity is required.")
        if len(str(note)) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError("Review note exceeds the permitted length.")
        authenticated_subject = self._writer_subject(actor)
        if authenticated_subject:
            reviewer = authenticated_subject
        if decision not in {"accepted", "rejected"}:
            raise ValueError("Quyết định review chỉ có thể là accepted hoặc rejected.")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            record = connection.execute(
                "SELECT status FROM feedback_records WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
            if record is None:
                raise KeyError("Không tìm thấy feedback cần review.")
            if record["status"] != "pending":
                raise ValueError("Chỉ feedback ở trạng thái pending mới có thể được duyệt.")
            connection.execute(
                """
                UPDATE feedback_records
                SET status = ?, analyst_id = ?, review_note = ?, reviewed_at = ?
                WHERE feedback_id = ?
                """,
                (decision, self._protect(reviewer), self._protect(note), utc_now(), feedback_id),
            )

    def resolve_conflict(
        self,
        content_digest: str,
        accepted_label: int,
        reviewer: str = "analyst",
        note: str = "",
        actor: Actor | None = None,
    ) -> None:
        """Resolve all conflicting labels for one normalized email explicitly."""
        if not str(reviewer).strip():
            raise ValueError("Reviewer identity is required.")
        if len(str(note)) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError("Review note exceeds the permitted length.")
        authenticated_subject = self._writer_subject(actor)
        if authenticated_subject:
            reviewer = authenticated_subject
        if accepted_label not in (0, 1):
            raise ValueError("Nhãn được chọn để giải quyết xung đột phải là 0 hoặc 1.")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT feedback_id FROM feedback_records WHERE content_hash = ?", (content_digest,)
            ).fetchall()
            if not rows:
                raise KeyError("Không tìm thấy nhóm feedback xung đột.")
            now = utc_now()
            connection.execute(
                """
                UPDATE feedback_records
                SET status = CASE WHEN label = ? THEN 'accepted' ELSE 'rejected' END,
                    analyst_id = ?, review_note = ?, reviewed_at = ?
                WHERE content_hash = ?
                """,
                (accepted_label, self._protect(reviewer), self._protect(note), now, content_digest),
            )

    def stats(self) -> dict[str, int]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM feedback_records GROUP BY status"
            ).fetchall()
        result = {status: 0 for status in VALID_STATUSES}
        result.update({row["status"]: int(row["count"]) for row in rows})
        result["total"] = sum(result.values())
        return result

    def governance_summary(self) -> dict[str, int]:
        """Return the counts that explain exactly where feedback may be used.

        This deliberately reports *eligible* records rather than merely the
        number of submissions.  In particular, feedback held out for a shadow
        evaluation is not silently counted as ML training data, and an
        LLM-only correction is not counted as a trainable ML example.
        """
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT content_hash, source
                FROM feedback_records
                WHERE status = 'accepted'
                """
            ).fetchall()

        accepted = [dict(row) for row in rows]
        trainable = [row for row in accepted if row["source"] in TRAINABLE_SOURCES]
        rag_ready = [row for row in accepted if row["source"] in RAG_SOURCES]
        trainable_holdout = [
            row for row in trainable if stable_holdout_bucket(row["content_hash"]) == 0
        ]
        summary = self.stats()
        summary.update(
            {
                "accepted_trainable": len(trainable),
                "accepted_ml_train": len(trainable) - len(trainable_holdout),
                "accepted_ml_holdout": len(trainable_holdout),
                "accepted_rag": len(rag_ready),
                "accepted_llm_only": sum(
                    row["source"] == LLM_FEEDBACK_SOURCE for row in accepted
                ),
            }
        )
        return summary

    def review_queue(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT feedback_id, content_hash, body, label, source, confidence,
                       model_version, ml_probability, llm_analysis, review_note, created_at
                FROM feedback_records
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._decode_record(dict(row)) for row in rows]

    def conflict_queue(self, limit: int = 25) -> list[dict[str, Any]]:
        """List quarantined feedback groups so an analyst can resolve them."""
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT feedback_id, content_hash, body, label, source, confidence,
                       model_version, ml_probability, llm_model, llm_analysis,
                       review_note, created_at
                FROM feedback_records
                WHERE status = 'conflict'
                ORDER BY created_at ASC
                """
            ).fetchall()

        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            record = self._decode_record(dict(row))
            group = groups.setdefault(
                record["content_hash"],
                {
                    "content_hash": record["content_hash"],
                    "body": record["body"],
                    "records": [],
                    "created_at": record["created_at"],
                },
            )
            group["records"].append(record)
        return list(groups.values())[: int(limit)]

    def _accepted_rows(self, sources: Iterable[str]) -> pd.DataFrame:
        source_values = tuple(sources)
        if not source_values:
            return pd.DataFrame()
        if len(source_values) > 3:
            raise ValueError("Feedback source query exceeds the fixed governance source set.")
        padded_sources = (*source_values, *("" for _ in range(3 - len(source_values))))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT feedback_id, content_hash, body, label, source, confidence,
                       llm_analysis, created_at
                FROM feedback_records
                WHERE status = 'accepted' AND source IN (?, ?, ?)
                ORDER BY created_at ASC
                """,
                padded_sources,
            ).fetchall()
        return pd.DataFrame([self._decode_record(dict(row)) for row in rows])

    def accepted_for_training(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return accepted field feedback split deterministically into train/holdout."""
        records = self._accepted_rows(TRAINABLE_SOURCES)
        if records.empty:
            return records.copy(), records.copy()
        holdout_mask = records["content_hash"].map(stable_holdout_bucket).eq(0)
        return records.loc[~holdout_mask].copy(), records.loc[holdout_mask].copy()

    def accepted_for_rag(self) -> pd.DataFrame:
        """Only reviewed, non-benchmark records may become LLM retrieval context."""
        records = self._accepted_rows(RAG_SOURCES)
        if records.empty:
            return records
        return records.drop_duplicates(subset=["content_hash", "label"], keep="last").copy()

    def import_legacy_csv(self, csv_path: str | Path = Path("data") / "user_feedback.csv") -> dict[str, int]:
        """Safely migrate legacy CSV to a review queue; it never auto-accepts rows."""
        path = Path(csv_path)
        result = {"created": 0, "duplicate": 0, "conflict": 0, "failed": 0}
        if not path.exists():
            return result
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            try:
                outcome = self.submit(
                    body=str(row["body"]),
                    label=int(row["label"]),
                    source="legacy_import",
                    confidence=0.5,
                    status="pending",
                    llm_analysis=str(row.get("ai_analysis", "")),
                    review_note="Nhập từ user_feedback.csv cũ; tự cách ly, không dùng cho ML/RAG v2.",
                    actor=Actor("legacy-import", frozenset({ROLE_SYSTEM})),
                )
                result[outcome["action"]] += 1
            except Exception:
                result["failed"] += 1
        return result
