"""Unit tests for automatic file format identification engine."""

import pytest
from pathlib import Path
import numpy as np

from siganalyzer.common.types import SignalFileType
from siganalyzer.io.detector import FileTypeDetector
from tests.fixtures.generate_fixtures import (
    generate_bpsk,
    generate_qpsk,
    save_as_raw_iq,
    save_as_wav,
)


def test_detect_wav_pcm16(tmp_path: Path):
    """Test auto-identification of 16-bit PCM WAV recording."""
    samples, _ = generate_bpsk(num_symbols=500, sps=8, sample_rate=1e6)
    wav_file = tmp_path / "signal_16bit.wav"
    save_as_wav(wav_file, samples, sample_rate=1_000_000, bit_depth=16)

    res = FileTypeDetector.detect(wav_file)
    assert res.file_type == SignalFileType.WAV_PCM16
    assert res.confidence == 1.0
    assert res.suggested_reader == "WavSignalReader"
    assert any("RIFF WAVE" in ev for ev in res.evidence)


def test_detect_wav_float32(tmp_path: Path):
    """Test auto-identification of 32-bit float WAV recording."""
    samples, _ = generate_qpsk(num_symbols=500, sps=8, sample_rate=1e6)
    wav_file = tmp_path / "signal_f32.wav"
    save_as_wav(wav_file, samples, sample_rate=1_000_000, bit_depth=32)

    res = FileTypeDetector.detect(wav_file)
    assert res.file_type == SignalFileType.WAV_FLOAT32
    assert res.confidence == 1.0
    assert res.suggested_reader == "WavSignalReader"


def test_detect_raw_iq_float32(tmp_path: Path):
    """Test statistical probe on raw float32 IQ file."""
    samples, _ = generate_qpsk(num_symbols=1000, sps=8, sample_rate=1e6)
    iq_file = tmp_path / "capture.iq"
    save_as_raw_iq(iq_file, samples, file_type=SignalFileType.RAW_IQ_FLOAT32)

    res = FileTypeDetector.detect(iq_file)
    assert res.file_type == SignalFileType.RAW_IQ_FLOAT32
    assert res.confidence >= 0.8
    assert res.suggested_reader == "RawIQSignalReader"


def test_detect_raw_iq_int16(tmp_path: Path):
    """Test statistical probe on raw int16 IQ file."""
    samples, _ = generate_bpsk(num_symbols=1000, sps=8, sample_rate=1e6)
    iq_file = tmp_path / "capture_16.raw"
    save_as_raw_iq(iq_file, samples, file_type=SignalFileType.RAW_IQ_INT16)

    res = FileTypeDetector.detect(iq_file)
    assert res.file_type == SignalFileType.RAW_IQ_INT16
    assert res.confidence >= 0.8


def test_detect_raw_iq_rtl_sdr(tmp_path: Path):
    """Test statistical probe on RTL-SDR uint8 IQ file."""
    samples, _ = generate_bpsk(num_symbols=1000, sps=8, sample_rate=1e6)
    iq_file = tmp_path / "capture_rtl.dat"
    save_as_raw_iq(iq_file, samples, file_type=SignalFileType.RAW_IQ_UINT8_RTL)

    res = FileTypeDetector.detect(iq_file)
    assert res.file_type == SignalFileType.RAW_IQ_UINT8_RTL
    assert res.confidence >= 0.85
    assert any("RTL-SDR" in ev for ev in res.evidence)


def test_detect_empty_file(tmp_path: Path):
    """Test handling of 0-byte file."""
    empty_file = tmp_path / "empty.bin"
    empty_file.touch()

    res = FileTypeDetector.detect(empty_file)
    assert res.file_type == SignalFileType.UNKNOWN
    assert res.confidence == 0.0


def test_detect_nonexistent_file(tmp_path: Path):
    """Test exception on missing file."""
    with pytest.raises(FileNotFoundError):
        FileTypeDetector.detect(tmp_path / "nonexistent.wav")
