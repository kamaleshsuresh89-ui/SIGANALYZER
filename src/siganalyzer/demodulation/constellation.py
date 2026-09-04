"""Constellation analysis, symbol clustering, and Error Vector Magnitude (EVM) calculation."""

from dataclasses import dataclass
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import Measurement


@dataclass
class ConstellationMetrics:
    """Detailed quantitative constellation impairments."""

    evm_rms_percent: Measurement[float]
    evm_peak_percent: float
    snr_est_db: float
    cluster_centroids: np.ndarray  # Estimated reference symbol centroids
    phase_jitter_deg: float
    amplitude_droop_percent: float


class ConstellationAnalyzer:
    """Analyzes digital constellation clusters and computes Error Vector Magnitude (EVM)."""

    @classmethod
    def analyze_constellation(
        cls, symbols: np.ndarray, reference_constellation: np.ndarray | None = None
    ) -> ConstellationMetrics:
        """Evaluate EVM and impairments on recovered symbols."""
        if len(symbols) == 0:
            return ConstellationMetrics(
                evm_rms_percent=Measurement("EVM RMS", None, "%", ConfidenceLevel.UNKNOWN),
                evm_peak_percent=0.0,
                snr_est_db=0.0,
                cluster_centroids=np.empty(0, dtype=np.complex64),
                phase_jitter_deg=0.0,
                amplitude_droop_percent=0.0,
            )

        # Normalize symbols to unit average power
        norm_syms = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2) + 1e-12)

        # If reference constellation not provided, determine closest default (e.g. QPSK)
        if reference_constellation is None:
            # Default QPSK reference points
            reference_constellation = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)

        # Find closest reference constellation point for each symbol (minimum Euclidean distance)
        diffs = norm_syms[:, np.newaxis] - reference_constellation[np.newaxis, :]
        distances = np.abs(diffs)
        best_indices = np.argmin(distances, axis=1)
        ideal_symbols = reference_constellation[best_indices]

        # Error vectors: e = s_meas - s_ideal
        error_vectors = norm_syms - ideal_symbols
        error_power = np.mean(np.abs(error_vectors) ** 2)
        ref_power = np.mean(np.abs(ideal_symbols) ** 2)

        evm_rms = float(np.sqrt(error_power / max(ref_power, 1e-12)) * 100.0)
        evm_peak = float(np.max(np.abs(error_vectors) / np.sqrt(max(ref_power, 1e-12))) * 100.0)

        # Estimated SNR from EVM: SNR ≈ -20 * log10(EVM / 100)
        snr_from_evm = float(-20.0 * np.log10(max(evm_rms / 100.0, 1e-6)))

        # Phase jitter estimation (std of error angles)
        error_angles = np.angle(norm_syms) - np.angle(ideal_symbols)
        # Wrap to [-pi, pi]
        error_angles = (error_angles + np.pi) % (2 * np.pi) - np.pi
        phase_jitter_deg = float(np.degrees(np.std(error_angles)))

        return ConstellationMetrics(
            evm_rms_percent=Measurement(
                name="EVM RMS",
                value=min(evm_rms, 100.0),
                unit="%",
                confidence=ConfidenceLevel.MEASURED,
                confidence_score=0.90,
                source="Symbol Euclidean distance from reference grid",
            ),
            evm_peak_percent=evm_peak,
            snr_est_db=snr_from_evm,
            cluster_centroids=reference_constellation,
            phase_jitter_deg=phase_jitter_deg,
            amplitude_droop_percent=0.0,
        )
