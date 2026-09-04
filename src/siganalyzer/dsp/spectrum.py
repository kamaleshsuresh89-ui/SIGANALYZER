"""Spectral analysis: FFT, Welch PSD, noise floor estimation, and peak detection."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SpectralPeak:
    """A detected peak in the power spectrum."""

    freq_hz: float
    power_db: float
    prominence_db: float


@dataclass
class SpectrumResult:
    """Results from power spectral density analysis."""

    frequencies_hz: np.ndarray
    psd_db: np.ndarray
    noise_floor_db: float
    peak_freq_hz: float
    peak_power_db: float
    detected_peaks: list[SpectralPeak]


class SpectralAnalyzer:
    """Computes Welch PSD, estimates noise floor, and identifies spectral peaks."""

    @staticmethod
    def compute_welch_psd(
        signal: np.ndarray,
        sample_rate: float,
        nfft: int = 2048,
        overlap_pct: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Welch Power Spectral Density for complex or real signal.

        Returns:
            frequencies_hz: 1D array from -fs/2 to +fs/2 (complex) or 0 to fs/2 (real)
            psd_db: 1D array of power in dB/Hz
        """
        if len(signal) < nfft:
            # Zero pad up to nfft
            padded = np.zeros(nfft, dtype=np.complex64)
            padded[: len(signal)] = signal
            signal = padded

        step = int(nfft * (1.0 - overlap_pct))
        window = np.blackman(nfft)
        window_power = np.sum(window ** 2)

        num_segments = (len(signal) - nfft) // step + 1
        psd_accum = np.zeros(nfft, dtype=np.float64)

        for i in range(num_segments):
            seg = signal[i * step : i * step + nfft] * window
            fft_seg = np.fft.fft(seg)
            psd_accum += np.abs(fft_seg) ** 2

        psd_linear = (psd_accum / (num_segments * sample_rate * window_power))
        psd_linear = np.fft.fftshift(psd_linear)
        psd_db = 10.0 * np.log10(np.maximum(psd_linear, 1e-18))

        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
        return freqs, psd_db

    @classmethod
    def estimate_noise_floor(cls, psd_db: np.ndarray, percentile: float = 20.0) -> float:
        """Estimate the baseline noise floor in dB using lower percentile statistics."""
        if len(psd_db) == 0:
            return -100.0
        # Robust against strong signals by taking the 20th percentile
        return float(np.percentile(psd_db, percentile))

    @classmethod
    def find_peaks(
        cls,
        freqs_hz: np.ndarray,
        psd_db: np.ndarray,
        noise_floor_db: float,
        min_snr_db: float = 6.0,
        max_peaks: int = 10,
    ) -> list[SpectralPeak]:
        """Detect prominent peaks rising above the noise floor."""
        threshold_db = noise_floor_db + min_snr_db
        peaks: list[SpectralPeak] = []

        # Local maxima detection
        for i in range(1, len(psd_db) - 1):
            if psd_db[i] > threshold_db and psd_db[i] > psd_db[i - 1] and psd_db[i] > psd_db[i + 1]:
                prominence = psd_db[i] - noise_floor_db
                peaks.append(
                    SpectralPeak(
                        freq_hz=float(freqs_hz[i]),
                        power_db=float(psd_db[i]),
                        prominence_db=float(prominence),
                    )
                )

        # Sort by power descending
        peaks.sort(key=lambda p: p.power_db, reverse=True)
        return peaks[:max_peaks]

    @classmethod
    def analyze_spectrum(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        nfft: int = 2048,
    ) -> SpectrumResult:
        """Full spectral analysis pipeline."""
        freqs, psd_db = cls.compute_welch_psd(signal, sample_rate, nfft=nfft)
        noise_floor = cls.estimate_noise_floor(psd_db)
        peaks = cls.find_peaks(freqs, psd_db, noise_floor)

        if peaks:
            peak_freq = peaks[0].freq_hz
            peak_power = peaks[0].power_db
        else:
            max_idx = np.argmax(psd_db)
            peak_freq = float(freqs[max_idx])
            peak_power = float(psd_db[max_idx])

        return SpectrumResult(
            frequencies_hz=freqs,
            psd_db=psd_db,
            noise_floor_db=noise_floor,
            peak_freq_hz=peak_freq,
            peak_power_db=peak_power,
            detected_peaks=peaks,
        )
