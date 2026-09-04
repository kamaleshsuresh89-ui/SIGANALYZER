"""Anomaly and signal impairment detector."""

from dataclasses import dataclass
import numpy as np

from siganalyzer.common.types import AnomalyFinding


class AnomalyDetector:
    """Detects signal abnormalities such as clipping, dropouts, power steps, and frequency jumps."""

    @classmethod
    def analyze_anomalies(
        cls, signal: np.ndarray, sample_rate: float
    ) -> list[AnomalyFinding]:
        """Scan signal buffer for physical anomalies and impairments."""
        findings: list[AnomalyFinding] = []
        if len(signal) == 0:
            return findings

        magnitudes = np.abs(signal)

        # 1. Clipping detection
        clipped_indices = np.where(magnitudes >= 0.999)[0]
        if len(clipped_indices) > 0:
            pct = (len(clipped_indices) / len(signal)) * 100
            findings.append(
                AnomalyFinding(
                    anomaly_type="Clipping",
                    severity="CRITICAL" if pct > 1.0 else "WARNING",
                    sample_index=int(clipped_indices[0]),
                    time_seconds=float(clipped_indices[0] / sample_rate),
                    description=f"{len(clipped_indices)} samples ({pct:.2f}%) clipped at or near ADC full scale.",
                    recommended_action="Reduce SDR receiver gain (RF/IF/LNA gain) to prevent non-linear distortion.",
                )
            )

        # 2. Sudden Signal Dropout / Gap Detection
        rms_total = np.sqrt(np.mean(magnitudes ** 2))
        if rms_total > 1e-4:
            window_size = min(512, len(signal))
            hop = window_size // 2
            num_windows = (len(signal) - window_size) // hop + 1
            if num_windows > 10:
                window_rms = [
                    np.sqrt(np.mean(magnitudes[i * hop : i * hop + window_size] ** 2))
                    for i in range(num_windows)
                ]
                dropout_threshold = rms_total * 0.05
                for w_idx, w_rms in enumerate(window_rms):
                    if w_rms < dropout_threshold:
                        s_idx = w_idx * hop
                        findings.append(
                            AnomalyFinding(
                                anomaly_type="Signal Dropout",
                                severity="WARNING",
                                sample_index=s_idx,
                                time_seconds=float(s_idx / sample_rate),
                                description=f"Signal power dropped by >26 dB at t={s_idx / sample_rate:.4f} s.",
                                recommended_action="Check for transmitter silence, squelch cutoff, or cable disconnection.",
                            )
                        )
                        break

        # 3. Sudden Power Step
        if len(signal) > 2048:
            seg_len = 1024
            half1_pwr = np.mean(magnitudes[:seg_len] ** 2)
            half2_pwr = np.mean(magnitudes[-seg_len:] ** 2)
            pwr_ratio_db = abs(10 * np.log10(max(half1_pwr / max(half2_pwr, 1e-12), 1e-12)))
            if pwr_ratio_db > 12.0:
                findings.append(
                    AnomalyFinding(
                        anomaly_type="Power Step",
                        severity="INFO",
                        sample_index=len(signal) // 2,
                        time_seconds=float((len(signal) // 2) / sample_rate),
                        description=f"Significant average power change ({pwr_ratio_db:.1f} dB) between recording segments.",
                        recommended_action="Verify if transmitter was keyed or AGC adjusted during recording.",
                    )
                )

        return findings
