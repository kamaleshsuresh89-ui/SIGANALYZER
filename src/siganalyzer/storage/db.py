"""Local SQLite database for analysis history and user settings persistence."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any

from siganalyzer.common.logger import logger


@dataclass
class SessionRecord:
    """Historical signal analysis session record."""

    id: int
    timestamp: str
    file_path: str
    file_name: str
    file_type: str
    file_size_bytes: int
    sample_rate: float | None
    center_freq: float | None
    duration_s: float | None
    modulation_scheme: str
    modulation_confidence: float
    snr_db: float | None
    obw_hz: float | None
    symbol_rate: float | None
    recovered_bits_count: int
    bitstream_entropy: float | None
    fec_scheme: str
    processing_time_s: float


class DatabaseManager:
    """Embedded SQLite persistence manager."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            # Default to workspace .siganalyzer directory or fallback
            workspace_dir = Path.cwd() / ".siganalyzer"
            try:
                workspace_dir.mkdir(parents=True, exist_ok=True)
                self.db_path = str(workspace_dir / "history.db")
            except Exception:
                self.db_path = ":memory:"
        else:
            self.db_path = str(db_path)

        self._conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    sample_rate REAL,
                    center_freq REAL,
                    duration_s REAL,
                    modulation_scheme TEXT NOT NULL,
                    modulation_confidence REAL NOT NULL,
                    snr_db REAL,
                    obw_hz REAL,
                    symbol_rate REAL,
                    recovered_bits_count INTEGER DEFAULT 0,
                    bitstream_entropy REAL,
                    fec_scheme TEXT DEFAULT 'None',
                    processing_time_s REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_history_timestamp ON analysis_history(timestamp DESC);
                """
            )
            conn.commit()

    def save_session(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        file_size_bytes: int,
        modulation_scheme: str,
        modulation_confidence: float,
        processing_time_s: float,
        sample_rate: float | None = None,
        center_freq: float | None = None,
        duration_s: float | None = None,
        snr_db: float | None = None,
        obw_hz: float | None = None,
        symbol_rate: float | None = None,
        recovered_bits_count: int = 0,
        bitstream_entropy: float | None = None,
        fec_scheme: str = "None",
    ) -> int:
        """Record an analysis run in the database."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history (
                    timestamp, file_path, file_name, file_type, file_size_bytes,
                    sample_rate, center_freq, duration_s, modulation_scheme,
                    modulation_confidence, snr_db, obw_hz, symbol_rate,
                    recovered_bits_count, bitstream_entropy, fec_scheme, processing_time_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_str,
                    file_path,
                    file_name,
                    file_type,
                    file_size_bytes,
                    sample_rate,
                    center_freq,
                    duration_s,
                    modulation_scheme,
                    modulation_confidence,
                    snr_db,
                    obw_hz,
                    symbol_rate,
                    recovered_bits_count,
                    bitstream_entropy,
                    fec_scheme,
                    processing_time_s,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_recent_sessions(self, limit: int = 50) -> list[SessionRecord]:
        """Fetch list of most recent analysis sessions."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM analysis_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                SessionRecord(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    file_path=row["file_path"],
                    file_name=row["file_name"],
                    file_type=row["file_type"],
                    file_size_bytes=row["file_size_bytes"],
                    sample_rate=row["sample_rate"],
                    center_freq=row["center_freq"],
                    duration_s=row["duration_s"],
                    modulation_scheme=row["modulation_scheme"],
                    modulation_confidence=row["modulation_confidence"],
                    snr_db=row["snr_db"],
                    obw_hz=row["obw_hz"],
                    symbol_rate=row["symbol_rate"],
                    recovered_bits_count=row["recovered_bits_count"],
                    bitstream_entropy=row["bitstream_entropy"],
                    fec_scheme=row["fec_scheme"],
                    processing_time_s=row["processing_time_s"],
                )
                for row in rows
            ]

    def delete_session(self, session_id: int) -> bool:
        """Delete an analysis session from history."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM analysis_history WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def set_setting(self, key: str, value: str) -> None:
        """Store or update a key-value setting."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now_str),
            )
            conn.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Retrieve a setting value."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return str(row["value"])
            return default
