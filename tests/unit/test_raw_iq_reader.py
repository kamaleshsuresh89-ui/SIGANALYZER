"""Unit tests for RawIQSignalReader."""

from pathlib import Path
import numpy as np
import pytest

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import SignalFileType
from siganalyzer.io.readers.raw_iq_reader import RawIQSignalReader
from tests.fixtures.generate_fixtures import generate_bpsk, generate_qpsk, save_as_raw_iq


def test_raw_iq_float32(tmp_path: Path):
    """Test reading raw float32 IQ data."""
    samples, _ = generate_qpsk(num_symbols=500, sps=8, sample_rate=1e6)
    file_path = tmp_path / "signal.iq"
    save_as_raw_iq(file_path, samples, file_type=SignalFileType.RAW_IQ_FLOAT32)

    reader = RawIQSignalReader(file_path, file_type=SignalFileType.RAW_IQ_FLOAT32, sample_rate=1e6)
    try:
        assert reader.total_samples == len(samples)
        meta = reader.metadata
        assert meta.sample_rate.value == 1e6
        assert meta.sample_rate.confidence == ConfidenceLevel.ESTIMATED
        read_samples = reader.read_samples(0, 100)
        assert len(read_samples) == 100
        assert read_samples.dtype == np.complex64
    finally:
        reader.close()


def test_raw_iq_int16(tmp_path: Path):
    """Test reading raw int16 IQ data."""
    samples, _ = generate_bpsk(num_symbols=500, sps=8, sample_rate=2e6)
    file_path = tmp_path / "signal.raw"
    save_as_raw_iq(file_path, samples, file_type=SignalFileType.RAW_IQ_INT16)

    reader = RawIQSignalReader(file_path, file_type=SignalFileType.RAW_IQ_INT16)
    try:
        assert reader.total_samples == len(samples)
        read_samples = reader.read_samples(10, 50)
        assert len(read_samples) == 50
        assert np.max(np.abs(read_samples)) <= 1.05
    finally:
        reader.close()


def test_raw_iq_rtl_sdr(tmp_path: Path):
    """Test reading RTL-SDR uint8 IQ data with 127.5 DC offset removal."""
    samples, _ = generate_bpsk(num_symbols=500, sps=8, sample_rate=2.4e6)
    file_path = tmp_path / "signal_rtl.dat"
    save_as_raw_iq(file_path, samples, file_type=SignalFileType.RAW_IQ_UINT8_RTL)

    reader = RawIQSignalReader(file_path, file_type=SignalFileType.RAW_IQ_UINT8_RTL)
    try:
        assert reader.total_samples == len(samples)
        read_samples = reader.read_samples(0, 200)
        # Mean of I and Q should be approximately 0.0 after offset removal
        assert abs(np.mean(np.real(read_samples))) < 0.1
        assert abs(np.mean(np.imag(read_samples))) < 0.1
    finally:
        reader.close()
