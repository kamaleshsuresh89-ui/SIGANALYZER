"""Unit tests for carrier recovery, clock synchronization, and demodulation."""

from pathlib import Path
import numpy as np
import pytest

from siganalyzer.demodulation.constellation import ConstellationAnalyzer
from siganalyzer.demodulation.manager import DemodulationManager
from siganalyzer.demodulation.schemes.ask import ASKDemodulator
from siganalyzer.demodulation.schemes.fsk import FSKDemodulator
from siganalyzer.demodulation.schemes.psk import PSKDemodulator
from siganalyzer.demodulation.sync.costas import CostasLoop
from siganalyzer.demodulation.sync.gardner import GardnerTimingRecovery
from tests.fixtures.generate_fixtures import generate_2fsk, generate_bpsk, generate_qpsk


def test_costas_loop_qpsk_lock():
    """Test Costas loop acquires carrier phase on QPSK."""
    sample_rate = 1_000_000.0
    # Generate QPSK at small carrier offset
    samples, _ = generate_qpsk(num_symbols=1500, sps=8, sample_rate=sample_rate, carrier_freq=2000.0, snr_db=30.0)

    res = CostasLoop.recover_carrier(samples, sample_rate, order=4, loop_bandwidth=0.03)
    assert len(res.synchronized_signal) == len(samples)
    assert len(res.phase_history) == len(samples)
    # Check that residual frequency was detected
    assert abs(res.residual_freq_hz) < 5000.0


def test_gardner_timing_recovery():
    """Test Gardner timing recovery extracts 1 symbol per symbol period."""
    sample_rate = 1_000_000.0
    sps = 8
    num_symbols = 1000
    samples, _ = generate_bpsk(num_symbols=num_symbols, sps=sps, sample_rate=sample_rate, carrier_freq=0.0, snr_db=30.0)

    res = GardnerTimingRecovery.recover_timing(samples, nominal_sps=float(sps))
    # Number of recovered symbols should be close to num_symbols
    assert abs(len(res.symbols) - num_symbols) < 50
    assert len(res.timing_errors) > 0


def test_constellation_analyzer_evm():
    """Test EVM calculation on clean vs noisy constellation."""
    clean_qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    # Repeat clean symbols
    symbols = np.tile(clean_qpsk, 250)

    metrics_clean = ConstellationAnalyzer.analyze_constellation(symbols, clean_qpsk)
    assert metrics_clean.evm_rms_percent.value is not None
    assert metrics_clean.evm_rms_percent.value < 1.0  # Clean EVM near 0%

    # Add noise
    noisy_symbols = symbols + (np.random.normal(0, 0.1, len(symbols)) + 1j * np.random.normal(0, 0.1, len(symbols)))
    metrics_noisy = ConstellationAnalyzer.analyze_constellation(noisy_symbols, clean_qpsk)
    assert metrics_noisy.evm_rms_percent.value is not None
    assert metrics_noisy.evm_rms_percent.value > 5.0


def test_psk_demodulator():
    """Test end-to-end PSK demodulator on QPSK and BPSK."""
    sample_rate = 1_000_000.0
    qpsk_samples, q_gt = generate_qpsk(num_symbols=500, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=30.0)

    res = PSKDemodulator.demodulate(qpsk_samples, sample_rate, nominal_symbol_rate=125_000.0, scheme="QPSK")
    assert res.scheme == "QPSK"
    assert len(res.symbols) > 400
    assert len(res.bits) == len(res.symbols) * 2  # 2 bits per QPSK symbol
    assert set(np.unique(res.bits)).issubset({0, 1})


def test_fsk_demodulator():
    """Test end-to-end 2FSK demodulator."""
    sample_rate = 1_000_000.0
    fsk_samples, f_gt = generate_2fsk(num_symbols=500, sps=16, sample_rate=sample_rate, carrier_freq=0.0, freq_deviation=40e3)

    res = FSKDemodulator.demodulate(fsk_samples, sample_rate, nominal_symbol_rate=62_500.0, scheme="2FSK")
    assert res.scheme == "2FSK"
    assert len(res.symbols) > 400
    assert len(res.bits) == len(res.symbols)
    assert set(np.unique(res.bits)).issubset({0, 1})


def test_demodulation_manager_dispatch():
    """Test DemodulationManager dispatches to correct demodulator and computes metrics."""
    sample_rate = 1_000_000.0
    samples, _ = generate_bpsk(num_symbols=400, sps=8, sample_rate=sample_rate, carrier_freq=0.0, snr_db=25.0)

    res, metrics = DemodulationManager.demodulate_signal(
        samples, sample_rate, detected_scheme="BPSK", symbol_rate_hz=125_000.0
    )
    assert res.scheme == "BPSK"
    assert len(res.bits) > 300
    assert metrics.evm_rms_percent.value is not None
