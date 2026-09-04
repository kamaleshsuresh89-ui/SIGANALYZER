"""Unit tests for SQLite storage persistence and history tracking."""

import pytest

from siganalyzer.storage.db import DatabaseManager


def test_sqlite_session_crud() -> None:
    # Use in-memory SQLite database
    db = DatabaseManager(db_path=":memory:")

    # 1. Save session
    session_id = db.save_session(
        file_path="/tmp/test_signal.wav",
        file_name="test_signal.wav",
        file_type="WAV (PCM 16-bit)",
        file_size_bytes=10240,
        modulation_scheme="QPSK",
        modulation_confidence=0.95,
        processing_time_s=0.25,
        sample_rate=1_000_000.0,
        center_freq=433_920_000.0,
        duration_s=0.01,
        snr_db=22.5,
        obw_hz=120_000.0,
        symbol_rate=50_000.0,
        recovered_bits_count=1000,
        bitstream_entropy=0.99,
        fec_scheme="Uncoded",
    )
    assert session_id > 0

    # 2. Retrieve recent sessions
    sessions = db.get_recent_sessions(limit=10)
    assert len(sessions) == 1
    s0 = sessions[0]
    assert s0.id == session_id
    assert s0.file_name == "test_signal.wav"
    assert s0.modulation_scheme == "QPSK"
    assert s0.recovered_bits_count == 1000

    # 3. Settings Key-Value
    db.set_setting("favorite_colormap", "inferno")
    val = db.get_setting("favorite_colormap")
    assert val == "inferno"

    # 4. Delete session
    deleted = db.delete_session(session_id)
    assert deleted is True
    assert len(db.get_recent_sessions()) == 0
