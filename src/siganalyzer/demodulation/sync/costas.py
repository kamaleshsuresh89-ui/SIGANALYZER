"""Carrier frequency and phase recovery via Costas loop and PLL."""

from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass
class CostasLoopResult:
    """Result of carrier synchronization."""

    synchronized_signal: np.ndarray  # Complex baseband with carrier removed
    phase_history: np.ndarray  # Estimated phase trajectory in radians
    residual_freq_hz: float  # Estimated carrier frequency offset in Hz
    is_locked: bool  # True if carrier phase acquisition succeeded


@njit(fastmath=True)
def _costas_loop_core(
    signal: np.ndarray,
    order: int,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """JIT-compiled inner loop for Costas carrier tracking.

    Args:
        signal: 1D complex array of samples.
        order: 2 for BPSK, 4 for QPSK, 8 for 8PSK.
        alpha: Proportional loop filter gain.
        beta: Integral loop filter gain.

    Returns:
        Tuple of (synchronized_signal, phase_history, final_frequency_step).
    """
    n_samples = len(signal)
    out = np.zeros(n_samples, dtype=np.complex64)
    phase_hist = np.zeros(n_samples, dtype=np.float32)

    phase = 0.0
    freq = 0.0

    for i in range(n_samples):
        # Rotate input by current phase estimate: s * exp(-j * phase)
        rot_cos = np.cos(-phase)
        rot_sin = np.sin(-phase)
        s_i = np.real(signal[i])
        s_q = np.imag(signal[i])

        out_i = s_i * rot_cos - s_q * rot_sin
        out_q = s_i * rot_sin + s_q * rot_cos
        out[i] = out_i + 1j * out_q
        phase_hist[i] = phase

        # Phase error detection (PED)
        error = 0.0
        if order == 2:
            # BPSK PED: sign(I) * Q
            sign_i = 1.0 if out_i >= 0.0 else -1.0
            error = sign_i * out_q
        elif order == 4:
            # QPSK PED: sign(I) * Q - sign(Q) * I
            sign_i = 1.0 if out_i >= 0.0 else -1.0
            sign_q = 1.0 if out_q >= 0.0 else -1.0
            error = sign_i * out_q - sign_q * out_i
        else:
            # General M-PSK PED: sin(M * theta)
            theta = np.arctan2(out_q, out_i)
            error = np.sin(order * theta) / order

        # PI Loop Filter
        freq += beta * error
        phase += freq + alpha * error

        # Wrap phase to [-pi, pi]
        if phase > np.pi:
            phase -= 2.0 * np.pi
        elif phase < -np.pi:
            phase += 2.0 * np.pi

    return out, phase_hist, freq


class CostasLoop:
    """Costas loop carrier recovery for PSK and QAM modulations."""

    @classmethod
    def recover_carrier(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        order: int = 4,
        loop_bandwidth: float = 0.01,
        damping_factor: float = 0.707,
    ) -> CostasLoopResult:
        """Run Costas loop on complex I/Q samples.

        Args:
            signal: 1D complex64 array.
            sample_rate: Sample rate in Hz.
            order: Modulation order (2=BPSK, 4=QPSK, 8=8PSK).
            loop_bandwidth: Normalized loop bandwidth (typically 0.005 - 0.05).
            damping_factor: Damping ratio (0.707 for critical damping).
        """
        if len(signal) == 0:
            return CostasLoopResult(np.empty(0, dtype=np.complex64), np.empty(0), 0.0, False)

        # Standard 2nd-order loop filter coefficients
        denom = 1.0 + 2.0 * damping_factor * loop_bandwidth + loop_bandwidth * loop_bandwidth
        alpha = (4.0 * damping_factor * loop_bandwidth) / denom
        beta = (4.0 * loop_bandwidth * loop_bandwidth) / denom

        out, phase_hist, final_freq = _costas_loop_core(
            signal.astype(np.complex64), int(order), float(alpha), float(beta)
        )

        residual_freq_hz = float((final_freq / (2.0 * np.pi)) * sample_rate)

        # Determine lock status by evaluating phase variance over the second half
        half = len(phase_hist) // 2
        steady_state = phase_hist[half:] if len(phase_hist) > 100 else phase_hist
        phase_var = float(np.var(np.diff(steady_state))) if len(steady_state) > 1 else 1.0
        is_locked = phase_var < 0.15

        return CostasLoopResult(
            synchronized_signal=out,
            phase_history=phase_hist,
            residual_freq_hz=residual_freq_hz,
            is_locked=is_locked,
        )
