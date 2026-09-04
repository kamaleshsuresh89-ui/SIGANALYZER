"""Spectrogram and time-frequency waterfall generation with adaptive downsampling."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SpectrogramData:
    """2D time-frequency matrix ready for rendering in pyqtgraph."""

    time_axis: np.ndarray  # 1D array of timestamps in seconds
    freq_axis: np.ndarray  # 1D array of frequency bins in Hz
    power_matrix_db: np.ndarray  # 2D array of shape (time_bins, freq_bins) in dB
    min_power_db: float
    max_power_db: float


class SpectrogramGenerator:
    """Generates 2D spectrograms with adaptive decimation for fast interactive display."""

    @staticmethod
    def generate(
        signal: np.ndarray,
        sample_rate: float,
        nfft: int = 1024,
        overlap: int = 512,
        max_time_bins: int = 1000,
    ) -> SpectrogramData:
        """Compute STFT spectrogram decimated to a maximum number of time columns for fast rendering."""
        if len(signal) == 0:
            return SpectrogramData(
                time_axis=np.empty(0),
                freq_axis=np.empty(0),
                power_matrix_db=np.empty((0, 0)),
                min_power_db=-100.0,
                max_power_db=0.0,
            )

        step = nfft - overlap
        total_time_slices = (len(signal) - nfft) // step + 1

        if total_time_slices <= 0:
            # Buffer shorter than nfft, pad
            padded = np.zeros(nfft, dtype=np.complex64)
            padded[: len(signal)] = signal
            signal = padded
            total_time_slices = 1
            step = nfft

        # Adaptive decimation if too many slices
        stride = 1
        if total_time_slices > max_time_bins:
            stride = int(np.ceil(total_time_slices / max_time_bins))

        window = np.hanning(nfft)
        window_pow = np.sum(window ** 2)

        indices = list(range(0, total_time_slices, stride))
        num_kept = len(indices)

        matrix = np.zeros((num_kept, nfft), dtype=np.float32)
        time_axis = np.zeros(num_kept, dtype=np.float64)

        for out_idx, slice_idx in enumerate(indices):
            start = slice_idx * step
            chunk = signal[start : start + nfft] * window
            fft_res = np.fft.fftshift(np.fft.fft(chunk))
            pwr = (np.abs(fft_res) ** 2) / (sample_rate * window_pow)
            matrix[out_idx, :] = 10.0 * np.log10(np.maximum(pwr, 1e-15))
            time_axis[out_idx] = (start + nfft / 2.0) / sample_rate

        freq_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))

        min_db = float(np.min(matrix))
        max_db = float(np.max(matrix))

        return SpectrogramData(
            time_axis=time_axis,
            freq_axis=freq_axis,
            power_matrix_db=matrix,
            min_power_db=min_db,
            max_power_db=max_db,
        )
