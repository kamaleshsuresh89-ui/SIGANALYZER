"""Unit tests for ModulationClassifier."""

import numpy as np
import pytest

from siganalyzer.classification.classifier import ModulationClassifier
from tests.fixtures.generate_fixtures import generate_2fsk, generate_bpsk, generate_qpsk


def test_classify_bpsk():
    """Test classification of synthetic BPSK signal."""
    sample_rate = 1_000_000.0
    samples, _ = generate_bpsk(num_symbols=3000, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=25.0)

    result = ModulationClassifier.classify(samples, sample_rate, snr_db=25.0)
    assert result.scheme == "BPSK"
    assert result.confidence_score >= 0.70
    assert len(result.evidence) >= 1
    assert result.constellation_points is not None
    assert len(result.constellation_points) > 0


def test_classify_qpsk():
    """Test classification of synthetic QPSK signal."""
    sample_rate = 1_000_000.0
    samples, _ = generate_qpsk(num_symbols=3000, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=25.0)

    result = ModulationClassifier.classify(samples, sample_rate, snr_db=25.0)
    assert result.scheme == "QPSK"
    assert result.confidence_score >= 0.70
    assert any("QPSK" in ev for ev in result.evidence)


def test_classify_2fsk():
    """Test classification of synthetic 2FSK signal."""
    sample_rate = 1_000_000.0
    samples, _ = generate_2fsk(num_symbols=2000, sps=16, sample_rate=sample_rate, carrier_freq=100e3, freq_deviation=40e3, snr_db=25.0)

    result = ModulationClassifier.classify(samples, sample_rate, snr_db=25.0)
    assert result.scheme in ("2FSK", "MSK")
    assert result.confidence_score >= 0.65
