"""Gardner Timing Error Detector (TED) and clock recovery for digital communications."""

from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass
class TimingRecoveryResult:
    """Result of symbol clock synchronization."""

    symbols: np.ndarray  # Recovered symbols at 1 sample per symbol (SPS=1)
    timing_errors: np.ndarray  # Timing error detector history
    is_locked: bool  # True if timing recovery converged


@njit(fastmath=True)
def _gardner_recovery_core(
    signal: np.ndarray,
    sps: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """JIT-compiled Gardner timing error detector and sample interpolator.

    Args:
        signal: 1D complex64 samples.
        sps: Nominal samples per symbol (e.g. 4.0, 8.0).
        alpha: Proportional loop filter gain.
        beta: Integral loop filter gain.

    Returns:
        Tuple of (symbols, error_history).
    """
    n_samples = len(signal)
    max_symbols = int(n_samples / sps * 1.5) + 100
    symbols = np.zeros(max_symbols, dtype=np.complex64)
    errors = np.zeros(max_symbols, dtype=np.float32)

    sym_count = 0
    sample_idx = 0.0
    step = sps
    half_step = sps / 2.0

    prev_sym_i = 0.0
    prev_sym_q = 0.0

    while sample_idx + step + 2.0 < n_samples:
        # Linear/cubic interpolation for symbol sample at sample_idx
        base_idx = int(sample_idx)
        frac = sample_idx - base_idx
        curr_sym = signal[base_idx] * (1.0 - frac) + signal[base_idx + 1] * frac

        # Midpoint sample (at sample_idx - half_step)
        mid_idx_f = sample_idx - half_step
        if mid_idx_f >= 0.0:
            m_base = int(mid_idx_f)
            m_frac = mid_idx_f - m_base
            mid_sym = signal[m_base] * (1.0 - m_frac) + signal[m_base + 1] * m_frac
        else:
            mid_sym = curr_sym

        curr_i = np.real(curr_sym)
        curr_q = np.imag(curr_sym)
        mid_i = np.real(mid_sym)
        mid_q = np.imag(mid_sym)

        # Gardner TED: mid * (curr - prev)
        error = mid_i * (curr_i - prev_sym_i) + mid_q * (curr_q - prev_sym_q)

        # Update symbol history
        prev_sym_i = curr_i
        prev_sym_q = curr_q

        symbols[sym_count] = curr_sym
        errors[sym_count] = error
        sym_count += 1

        # Loop filter updates timing step
        step += beta * error
        sample_idx += step + alpha * error

    return symbols[:sym_count], errors[:sym_count]


class GardnerTimingRecovery:
    """Symbol clock synchronizer based on the Gardner algorithm."""

    @classmethod
    def recover_timing(
        cls,
        signal: np.ndarray,
        nominal_sps: float,
        loop_bandwidth: float = 0.02,
        damping_factor: float = 0.707,
    ) -> TimingRecoveryResult:
        """Run Gardner clock recovery to decimate signal from SPS > 1 to SPS = 1.

        Args:
            signal: 1D complex64 samples.
            nominal_sps: Estimated samples per symbol (e.g. 4.0, 8.0, 16.0).
            loop_bandwidth: Normalized loop bandwidth.
            damping_factor: Damping ratio.
        """
        if len(signal) < int(nominal_sps * 4):
            return TimingRecoveryResult(signal, np.empty(0), False)

        # Calculate loop filter gains
        denom = 1.0 + 2.0 * damping_factor * loop_bandwidth + loop_bandwidth * loop_bandwidth
        alpha = (4.0 * damping_factor * loop_bandwidth) / denom
        beta = (4.0 * loop_bandwidth * loop_bandwidth) / denom

        symbols, errors = _gardner_recovery_core(
            signal.astype(np.complex64), float(nominal_sps), float(alpha), float(beta)
        )

        # Check convergence on second half of error history
        if len(errors) > 50:
            half = len(errors) // 2
            err_var = float(np.var(errors[half:]))
            is_locked = err_var < 0.25
        else:
            is_locked = False

        return TimingRecoveryResult(
            symbols=symbols,
            timing_errors=errors,
            is_locked=is_locked,
        )
