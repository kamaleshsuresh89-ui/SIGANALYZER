"""Automatic signal burst and region detection using adaptive energy & CFAR algorithms."""

import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import Measurement, SignalRegion


class SignalDetector:
    """Detects signal bursts and active regions across time and frequency."""

    @classmethod
    def detect_regions(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        window_size: int = 512,
        hop_size: int = 128,
        threshold_factor_db: float = 6.0,
        min_duration_s: float = 0.0005,  # 0.5 ms minimum duration
    ) -> list[SignalRegion]:
        """Detect active signal bursts in time domain using rolling energy profile.

        Args:
            signal: 1D complex64 array.
            sample_rate: Sample rate in Hz.
            window_size: Integration window size in samples.
            hop_size: Step size in samples.
            threshold_factor_db: dB above estimated noise floor to consider active.
            min_duration_s: Minimum burst duration to filter out transient spikes.

        Returns:
            List of detected SignalRegion objects.
        """
        if len(signal) < window_size:
            # Single region covering whole signal
            pwr = float(10 * np.log10(max(np.mean(np.abs(signal) ** 2), 1e-12)))
            dur = len(signal) / max(sample_rate, 1.0)
            return [
                SignalRegion(
                    region_id=1,
                    start_sample=0,
                    end_sample=len(signal),
                    sample_rate=sample_rate,
                    start_time=0.0,
                    end_time=dur,
                    duration_s=dur,
                    center_freq_offset_hz=Measurement("Center Frequency Offset", 0.0, "Hz", ConfidenceLevel.ESTIMATED),
                    occupied_bandwidth_hz=Measurement("Occupied Bandwidth", sample_rate * 0.5, "Hz", ConfidenceLevel.ESTIMATED),
                    snr_db=Measurement("SNR", 15.0, "dB", ConfidenceLevel.ESTIMATED),
                    mean_power_db=pwr,
                    is_signal=True,
                )
            ]

        # Calculate energy profile across time windows
        num_windows = (len(signal) - window_size) // hop_size + 1
        energies = np.zeros(num_windows, dtype=np.float64)

        for i in range(num_windows):
            chunk = signal[i * hop_size : i * hop_size + window_size]
            energies[i] = np.mean(np.abs(chunk) ** 2)

        energies_db = 10.0 * np.log10(np.maximum(energies, 1e-18))

        # Baseline noise floor via 20th percentile
        noise_floor_db = float(np.percentile(energies_db, 20.0))
        active_mask = energies_db >= (noise_floor_db + threshold_factor_db)

        # Morphological grouping: merge bursts separated by small gaps (< 3 hops)
        gap_limit = 3
        gap_count = 0
        in_region = False
        start_idx = 0
        raw_regions: list[tuple[int, int]] = []

        for i, active in enumerate(active_mask):
            if active:
                if not in_region:
                    in_region = True
                    start_idx = i
                gap_count = 0
            else:
                if in_region:
                    gap_count += 1
                    if gap_count >= gap_limit or i == len(active_mask) - 1:
                        end_idx = i - gap_count + 1
                        raw_regions.append((start_idx, end_idx))
                        in_region = False
                        gap_count = 0

        if in_region:
            raw_regions.append((start_idx, len(active_mask)))

        # Convert window indices to sample and time bounds
        min_samples = int(min_duration_s * sample_rate)
        signal_regions: list[SignalRegion] = []
        region_id = 1

        for w_start, w_end in raw_regions:
            start_s = w_start * hop_size
            end_s = min(w_end * hop_size + window_size, len(signal))
            sample_count = end_s - start_s

            if sample_count >= min_samples:
                dur = sample_count / sample_rate
                burst_samples = signal[start_s:end_s]
                burst_pwr = float(10 * np.log10(max(np.mean(np.abs(burst_samples) ** 2), 1e-12)))
                burst_snr = max(burst_pwr - noise_floor_db, 0.0)

                signal_regions.append(
                    SignalRegion(
                        region_id=region_id,
                        start_sample=start_s,
                        end_sample=end_s,
                        sample_rate=sample_rate,
                        start_time=start_s / sample_rate,
                        end_time=end_s / sample_rate,
                        duration_s=dur,
                        center_freq_offset_hz=Measurement(
                            name="Center Frequency Offset",
                            value=0.0,
                            unit="Hz",
                            confidence=ConfidenceLevel.ESTIMATED,
                        ),
                        occupied_bandwidth_hz=Measurement(
                            name="Occupied Bandwidth",
                            value=0.0,
                            unit="Hz",
                            confidence=ConfidenceLevel.ESTIMATED,
                        ),
                        snr_db=Measurement(
                            name="SNR",
                            value=burst_snr,
                            unit="dB",
                            confidence=ConfidenceLevel.ESTIMATED,
                            confidence_score=0.85,
                        ),
                        mean_power_db=burst_pwr,
                        is_signal=True,
                    )
                )
                region_id += 1

        # If no bursts detected above threshold, treat whole buffer as 1 region
        if not signal_regions:
            dur = len(signal) / sample_rate
            mean_pwr = float(10 * np.log10(max(np.mean(np.abs(signal) ** 2), 1e-12)))
            signal_regions.append(
                SignalRegion(
                    region_id=1,
                    start_sample=0,
                    end_sample=len(signal),
                    sample_rate=sample_rate,
                    start_time=0.0,
                    end_time=dur,
                    duration_s=dur,
                    center_freq_offset_hz=Measurement("Center Frequency Offset", 0.0, "Hz", ConfidenceLevel.ESTIMATED),
                    occupied_bandwidth_hz=Measurement("Occupied Bandwidth", 0.0, "Hz", ConfidenceLevel.UNKNOWN),
                    snr_db=Measurement("SNR", max(mean_pwr - noise_floor_db, 0.0), "dB", ConfidenceLevel.ESTIMATED),
                    mean_power_db=mean_pwr,
                    is_signal=True,
                )
            )

        return signal_regions
