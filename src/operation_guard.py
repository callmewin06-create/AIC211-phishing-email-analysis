"""Durable per-identity operation budgets for the Streamlit control plane."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


DEFAULT_GUARD_PATH = Path("data") / "operation_guard.sqlite3"


class OperationGuard:
    """A small cross-session rate limiter backed by SQLite transactions."""

    def __init__(self, path: str | Path = DEFAULT_GUARD_PATH, now=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_events (
                    actor_hash TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    occurred_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_operation_events_window ON operation_events(actor_hash, operation, occurred_at)"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _actor_hash(subject: str) -> str:
        return hashlib.sha256(str(subject).encode("utf-8")).hexdigest()

    def reserve(self, subject: str, operation: str, *, limit: int, window_seconds: float) -> tuple[bool, int]:
        """Atomically reserve one operation slot; return ``(allowed, retry_seconds)``."""

        if not str(subject).strip() or not str(operation).strip() or limit < 1 or window_seconds <= 0:
            raise ValueError("Operation guard requires a subject, operation, positive limit and window.")
        now = float(self._now())
        cutoff = now - float(window_seconds)
        actor_hash = self._actor_hash(subject)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM operation_events WHERE occurred_at < ?", (cutoff,))
            rows = connection.execute(
                """
                SELECT occurred_at FROM operation_events
                WHERE actor_hash = ? AND operation = ? AND occurred_at >= ?
                ORDER BY occurred_at ASC
                """,
                (actor_hash, str(operation), cutoff),
            ).fetchall()
            if len(rows) >= limit:
                retry = max(1, int(rows[0][0] + window_seconds - now) + 1)
                connection.execute("COMMIT")
                return False, retry
            connection.execute(
                "INSERT INTO operation_events(actor_hash, operation, occurred_at) VALUES (?, ?, ?)",
                (actor_hash, str(operation), now),
            )
            connection.execute("COMMIT")
        return True, 0
