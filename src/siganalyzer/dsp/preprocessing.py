"""Signal conditioning and preprocessing: DC removal, IQ imbalance correction, and AGC."""

from dataclasses import dataclass
import numpy as np


@dataclass
class IQImbalanceMetrics:
    """Estimated hardware IQ imbalance parameters."""

    gain_imbalance_db: float
    phase_imbalance_deg: float
    dc_offset_i: float
    dc_offset_q: float


class SignalPreprocessor:
    """Modular preprocessing algorithms for complex baseband I/Q signals."""

    @staticmethod
    def remove_dc_offset(signal: np.ndarray, use_filter: bool = False) -> np.ndarray:
        """Remove DC offset from complex I/Q signal.

        Args:
            signal: 1D complex64 array.
            use_filter: If True, uses a 1-pole high-pass IIR filter. If False, subtracts mean.
        """
        if len(signal) == 0:
            return signal

        if not use_filter:
            # Mean subtraction
            dc_i = np.mean(np.real(signal))
            dc_q = np.mean(np.imag(signal))
            return (np.real(signal) - dc_i) + 1j * (np.imag(signal) - dc_q)
        else:
            # 1-pole high-pass IIR: y[n] = x[n] - x[n-1] + R * y[n-1], with R = 0.999
            # Vectorized or fast iterative filter
            r = 0.999
            out = np.zeros_like(signal)
            out_i = np.zeros(len(signal), dtype=np.float32)
            out_q = np.zeros(len(signal), dtype=np.float32)
            x_i = np.real(signal).astype(np.float32)
            x_q = np.imag(signal).astype(np.float32)

            prev_xi, prev_xq = 0.0, 0.0
            prev_yi, prev_yq = 0.0, 0.0

            for n in range(len(signal)):
                yi = x_i[n] - prev_xi + r * prev_yi
                yq = x_q[n] - prev_xq + r * prev_yq
                out_i[n] = yi
                out_q[n] = yq
                prev_xi, prev_xq = x_i[n], x_q[n]
                prev_yi, prev_yq = yi, yq

            return (out_i + 1j * out_q).astype(np.complex64)

    @staticmethod
    def estimate_iq_imbalance(signal: np.ndarray) -> IQImbalanceMetrics:
        """Estimate IQ gain and phase imbalance using blind statistical moments."""
        if len(signal) < 128:
            return IQImbalanceMetrics(0.0, 0.0, 0.0, 0.0)

        i = np.real(signal)
        q = np.imag(signal)

        dc_i = float(np.mean(i))
        dc_q = float(np.mean(q))

        i_zero = i - dc_i
        q_zero = q - dc_q

        p_i = float(np.mean(i_zero ** 2))
        p_q = float(np.mean(q_zero ** 2))

        # Gain imbalance in dB
        if p_q > 0:
            gain_ratio = np.sqrt(max(p_i / p_q, 1e-12))
            gain_db = float(20 * np.log10(gain_ratio))
        else:
            gain_db = 0.0

        # Phase imbalance
        if p_i > 0 and p_q > 0:
            cross_corr = float(np.mean(i_zero * q_zero))
            sin_phi = cross_corr / np.sqrt(p_i * p_q)
            sin_phi = np.clip(sin_phi, -1.0, 1.0)
            phase_deg = float(np.degrees(np.arcsin(sin_phi)))
        else:
            phase_deg = 0.0

        return IQImbalanceMetrics(
            gain_imbalance_db=gain_db,
            phase_imbalance_deg=phase_deg,
            dc_offset_i=dc_i,
            dc_offset_q=dc_q,
        )

    @classmethod
    def correct_iq_imbalance(
        cls, signal: np.ndarray, metrics: IQImbalanceMetrics | None = None
    ) -> tuple[np.ndarray, IQImbalanceMetrics]:
        """Correct IQ gain and phase imbalance via Gram-Schmidt orthogonalization."""
        if metrics is None:
            metrics = cls.estimate_iq_imbalance(signal)

        if len(signal) == 0:
            return signal, metrics

        i = np.real(signal) - metrics.dc_offset_i
        q = np.imag(signal) - metrics.dc_offset_q

        gain_linear = 10 ** (metrics.gain_imbalance_db / 20.0)
        phase_rad = np.radians(metrics.phase_imbalance_deg)

        # Correct gain
        i_corr = i / max(gain_linear, 1e-6)

        # Correct phase
        cos_phi = np.cos(phase_rad)
        sin_phi = np.sin(phase_rad)
        if abs(cos_phi) > 1e-4:
            q_corr = (q - i_corr * sin_phi) / cos_phi
        else:
            q_corr = q

        corrected = (i_corr + 1j * q_corr).astype(np.complex64)
        return corrected, metrics

    @staticmethod
    def normalize_power(signal: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
        """Normalize signal to achieve a target RMS power."""
        if len(signal) == 0:
            return signal
        rms = np.sqrt(np.mean(np.abs(signal) ** 2))
        if rms > 1e-9:
            return (signal * (target_rms / rms)).astype(np.complex64)
        return signal
