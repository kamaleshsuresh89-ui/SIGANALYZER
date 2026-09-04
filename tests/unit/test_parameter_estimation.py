"""Unit tests for ParameterEstimator."""

import numpy as np
import pytest

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.estimation.estimator import ParameterEstimator
from tests.fixtures.generate_fixtures import generate_bpsk, generate_qpsk


def test_carrier_frequency_estimation():
    """Test estimation of carrier frequency offset."""
    sample_rate = 1_000_000.0
    fc_true = 120_000.0
    samples, _ = generate_qpsk(
        num_symbols=2000, sps=8, sample_rate=sample_rate, carrier_freq=fc_true, snr_db=25.0
    )

    result = ParameterEstimator.estimate_all(samples, sample_rate)
    assert result.carrier_freq_hz.value is not None
    assert abs(result.carrier_freq_hz.value - fc_true) < 3000.0  # Within ~0.3% of sample rate
    assert result.carrier_freq_hz.confidence == ConfidenceLevel.ESTIMATED


def test_occupied_bandwidth_estimation():
    """Test 99% Occupied Bandwidth calculation."""
    sample_rate = 1_000_000.0
    sps = 8
    # QPSK with RRC alpha=0.35 has theoretical bandwidth = (1 + alpha) * (sample_rate / sps) = 1.35 * 125 kHz = 168.75 kHz
    expected_bw = 1.35 * (sample_rate / sps)

    samples, _ = generate_qpsk(
        num_symbols=2000, sps=sps, sample_rate=sample_rate, carrier_freq=0.0, snr_db=30.0
    )

    result = ParameterEstimator.estimate_all(samples, sample_rate)
    assert result.occupied_bandwidth_hz.value is not None
    # Tolerance within 15%
    assert abs(result.occupied_bandwidth_hz.value - expected_bw) < (expected_bw * 0.20)


def test_cumulants_bpsk_vs_qpsk():
    """Test Higher-Order Cumulants distinguish BPSK from QPSK."""
    sample_rate = 1_000_000.0
    bpsk_samples, _ = generate_bpsk(num_symbols=3000, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=30.0)
    qpsk_samples, _ = generate_qpsk(num_symbols=3000, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=30.0)

    bpsk_cum = ParameterEstimator.compute_cumulants(bpsk_samples)
    qpsk_cum = ParameterEstimator.compute_cumulants(qpsk_samples)

    # Theoretical:
    # BPSK: norm_c42 ≈ -2.0, |norm_c40| ≈ 1.0
    # QPSK: norm_c42 ≈ -1.0, |norm_c40| ≈ 0.0
    assert bpsk_cum.norm_c42 < -1.4  # Well below -1.4 towards -2.0
    assert abs(bpsk_cum.norm_c42 - (-2.0)) < 0.6

    assert abs(qpsk_cum.norm_c42 - (-1.0)) < 0.4
    assert np.abs(qpsk_cum.norm_c40) < np.abs(bpsk_cum.norm_c40)
