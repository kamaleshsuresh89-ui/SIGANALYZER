"""Unit tests for WavSignalReader."""

from pathlib import Path
import numpy as np
import pytest

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import DataRepresentation, SignalFileType
from siganalyzer.io.readers.wav_reader import WavSignalReader
from tests.fixtures.generate_fixtures import generate_bpsk, generate_qpsk, save_as_wav


def test_wav_reader_basic_metadata(tmp_path: Path):
    """Test reading basic metadata from standard WAV."""
    sample_rate = 500_000
    samples, gt = generate_bpsk(num_symbols=1000, sps=8, sample_rate=sample_rate)
    wav_path = tmp_path / "test_bpsk.wav"
    save_as_wav(wav_path, samples, sample_rate=sample_rate, bit_depth=16, center_freq=433_920_000)

    with WavSignalReader(wav_path) as reader:
        meta = reader.metadata
        assert meta.file_type == SignalFileType.WAV_PCM16
        assert meta.channels == 2
        assert meta.bit_depth == 16
        assert meta.representation == DataRepresentation.COMPLEX_INTERLEAVED
        assert meta.sample_rate.value == sample_rate
        assert meta.sample_rate.confidence == ConfidenceLevel.DIRECT
        assert meta.center_frequency.value == 433_920_000
        assert meta.center_frequency.confidence == ConfidenceLevel.DIRECT
        assert reader.total_samples == len(samples)


def test_wav_reader_sample_slicing(tmp_path: Path):
    """Test random access reading of slices."""
    sample_rate = 250_000
    samples, _ = generate_qpsk(num_symbols=500, sps=4, sample_rate=sample_rate)
    wav_path = tmp_path / "test_qpsk.wav"
    save_as_wav(wav_path, samples, sample_rate=sample_rate, bit_depth=32)

    with WavSignalReader(wav_path) as reader:
        # Read entire buffer
        all_read = reader.read_samples()
        assert len(all_read) == len(samples)
        assert np.all(np.abs(all_read) <= 1.05)

        # Read middle slice
        slice_100 = reader.read_samples(start_sample=100, count=200)
        assert len(slice_100) == 200

        # Correlation between written and read samples should be ~1.0
        corr = np.abs(np.corrcoef(all_read[100:300], slice_100)[0, 1])
        assert corr > 0.999


def test_wav_reader_chunk_streaming(tmp_path: Path):
    """Test chunked reading with overlap."""
    sample_rate = 100_000
    samples, _ = generate_bpsk(num_symbols=1000, sps=4, sample_rate=sample_rate)
    wav_path = tmp_path / "test_chunks.wav"
    save_as_wav(wav_path, samples, sample_rate=sample_rate, bit_depth=16)

    chunk_size = 512
    overlap = 128
    chunks = []

    with WavSignalReader(wav_path) as reader:
        for idx, chunk in reader.read_chunks(chunk_size=chunk_size, overlap=overlap):
            chunks.append((idx, chunk))

    assert len(chunks) > 0
    # First chunk starts at 0
    assert chunks[0][0] == 0
    assert len(chunks[0][1]) == chunk_size
    # Second chunk starts at chunk_size - overlap
    assert chunks[1][0] == chunk_size - overlap
