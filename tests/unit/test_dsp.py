"""Unit tests for DSP preprocessing, spectral analysis, spectrogram, and burst detection."""

import numpy as np
import pytest

from siganalyzer.dsp.detection import SignalDetector
from siganalyzer.dsp.preprocessing import SignalPreprocessor
from siganalyzer.dsp.spectrogram import SpectrogramGenerator
from siganalyzer.dsp.spectrum import SpectralAnalyzer
from tests.fixtures.generate_fixtures import (
    add_iq_imbalance,
    generate_2fsk,
    generate_bpsk,
    generate_qpsk,
)


def test_remove_dc_offset():
    """Test mean subtraction and IIR DC removal."""
    clean = np.ones(1000, dtype=np.complex64) * (0.5 + 0.5j)
    tone = np.exp(1j * 2 * np.pi * 0.1 * np.arange(1000)).astype(np.complex64)
    biased = tone + clean

    # Mean subtraction
    unbiased = SignalPreprocessor.remove_dc_offset(biased, use_filter=False)
    assert abs(np.mean(np.real(unbiased))) < 1e-5
    assert abs(np.mean(np.imag(unbiased))) < 1e-5

    # Filtered subtraction (with settling period)
    clean_long = np.ones(5000, dtype=np.complex64) * (0.5 + 0.5j)
    tone_long = np.exp(1j * 2 * np.pi * 0.1 * np.arange(5000)).astype(np.complex64)
    biased_long = tone_long + clean_long
    filtered = SignalPreprocessor.remove_dc_offset(biased_long, use_filter=True)
    assert abs(np.mean(filtered[3000:])) < 0.05


def test_iq_imbalance_correction():
    """Test blind estimation and correction of IQ gain/phase imbalance."""
    samples, _ = generate_qpsk(num_symbols=5000, sps=8, sample_rate=1e6)
    imbalanced = add_iq_imbalance(samples, gain_imbalance_db=2.0, phase_imbalance_deg=10.0)

    est_metrics = SignalPreprocessor.estimate_iq_imbalance(imbalanced)
    assert abs(est_metrics.gain_imbalance_db - 2.0) < 0.5
    assert abs(est_metrics.phase_imbalance_deg - 10.0) < 2.0

    corrected, _ = SignalPreprocessor.correct_iq_imbalance(imbalanced, est_metrics)
    post_metrics = SignalPreprocessor.estimate_iq_imbalance(corrected)
    assert abs(post_metrics.gain_imbalance_db) < 0.3
    assert abs(post_metrics.phase_imbalance_deg) < 1.0


def test_spectral_analyzer():
    """Test Welch PSD and peak finding on a known carrier tone."""
    sample_rate = 1_000_000.0
    freq_carrier = 150_000.0
    t = np.arange(10000) / sample_rate
    tone = np.exp(1j * 2 * np.pi * freq_carrier * t).astype(np.complex64)

    res = SpectralAnalyzer.analyze_spectrum(tone, sample_rate, nfft=2048)
    assert abs(res.peak_freq_hz - freq_carrier) < 1000.0  # Within FFT bin resolution
    assert res.peak_power_db > res.noise_floor_db + 20.0
    assert len(res.detected_peaks) >= 1
    assert abs(res.detected_peaks[0].freq_hz - freq_carrier) < 1000.0


def test_spectrogram_generator():
    """Test STFT spectrogram dimensioning and adaptive decimation."""
    sample_rate = 500_000.0
    samples, _ = generate_2fsk(num_symbols=500, sps=16, sample_rate=sample_rate)

    spec_data = SpectrogramGenerator.generate(samples, sample_rate, nfft=512, overlap=256, max_time_bins=100)
    assert spec_data.power_matrix_db.ndim == 2
    assert spec_data.power_matrix_db.shape[1] == 512
    assert spec_data.power_matrix_db.shape[0] <= 100
    assert len(spec_data.time_axis) == spec_data.power_matrix_db.shape[0]
    assert len(spec_data.freq_axis) == 512


def test_signal_detector_burst_segmentation():
    """Test automatic detection and segmentation of a burst signal surrounded by noise."""
    sample_rate = 1_000_000.0
    burst, _ = generate_bpsk(num_symbols=500, sps=8, sample_rate=sample_rate, snr_db=30.0)

    # Place burst in the middle of silence/noise
    noise_pre = np.random.normal(0, 0.01, 2000) + 1j * np.random.normal(0, 0.01, 2000)
    noise_post = np.random.normal(0, 0.01, 2000) + 1j * np.random.normal(0, 0.01, 2000)
    composite = np.concatenate([noise_pre, burst, noise_post]).astype(np.complex64)

    regions = SignalDetector.detect_regions(composite, sample_rate)
    assert len(regions) >= 1
    main_region = max(regions, key=lambda r: r.duration_s)
    # Burst started at sample 2000 and ended at 2000 + 4000 = 6000
    assert 1500 <= main_region.start_sample <= 2500
    assert 5500 <= main_region.end_sample <= 6800
    assert main_region.is_signal
    assert main_region.snr_db.value > 10.0
