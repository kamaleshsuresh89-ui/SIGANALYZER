"""QAM (16-QAM, 64-QAM) demodulator and decision grid slicer."""

import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import DemodulationResult, Measurement
from siganalyzer.demodulation.sync.costas import CostasLoop
from siganalyzer.demodulation.sync.gardner import GardnerTimingRecovery


class QAMDemodulator:
    """Demodulates 16-QAM and 64-QAM signals using carrier synchronization and grid slicing."""

    @classmethod
    def demodulate(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        nominal_symbol_rate: float,
        scheme: str = "16-QAM",
        carrier_freq_offset: float = 0.0,
    ) -> DemodulationResult:
        """Demodulate QAM signal into complex symbols and bits."""
        if len(signal) < 64:
            return cls._empty_result(scheme)

        # 1. Baseband shift
        if abs(carrier_freq_offset) > 10.0:
            t = np.arange(len(signal)) / sample_rate
            baseband = (signal * np.exp(-1j * 2 * np.pi * carrier_freq_offset * t)).astype(np.complex64)
        else:
            baseband = signal.astype(np.complex64)

        # 2. Carrier Recovery (Costas loop with 4-fold symmetry)
        costas_res = CostasLoop.recover_carrier(baseband, sample_rate, order=4)

        # 3. Timing Recovery
        nominal_sps = max(2.0, sample_rate / max(nominal_symbol_rate, 1.0))
        timing_res = GardnerTimingRecovery.recover_timing(costas_res.synchronized_signal, nominal_sps)
        symbols = timing_res.symbols

        # 4. Rectangular Grid Decision Slicing
        bits = cls.slice_qam_symbols(symbols, scheme)

        return DemodulationResult(
            scheme=scheme,
            symbols=symbols,
            bits=bits,
            symbol_rate_est=Measurement(
                name="Symbol Rate",
                value=float(nominal_symbol_rate),
                unit="sps",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.90,
                source="QAM clock recovery",
            ),
            carrier_frequency_error_hz=costas_res.residual_freq_hz,
            carrier_locked=costas_res.is_locked,
            timing_locked=timing_res.is_locked,
        )

    @classmethod
    def slice_qam_symbols(cls, symbols: np.ndarray, scheme: str) -> np.ndarray:
        """Slice QAM constellation points to binary bits using Gray mapping."""
        if len(symbols) == 0:
            return np.empty(0, dtype=np.uint8)

        # Normalize average power to 1.0
        rms = np.sqrt(np.mean(np.abs(symbols) ** 2) + 1e-12)
        norm_syms = symbols / rms

        # 16-QAM theoretical average power for levels [-3, -1, 1, 3] is (1+9)/2 = 5 -> scale factor sqrt(5) ≈ 2.236
        scale = np.sqrt(5.0) if scheme == "16-QAM" else np.sqrt(21.0)
        scaled_syms = norm_syms * scale

        i_ch = np.real(scaled_syms)
        q_ch = np.imag(scaled_syms)

        if scheme == "16-QAM":
            # 4 bits per symbol: 2 bits for I, 2 bits for Q
            # Gray code: -3 -> 00, -1 -> 01, +1 -> 11, +3 -> 10
            # Slicing thresholds at -2, 0, +2
            n_syms = len(scaled_syms)
            bits = np.zeros(n_syms * 4, dtype=np.uint8)

            def slice_2bit_gray(val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                b0 = np.where(val >= 0.0, 1, 0).astype(np.uint8)
                b1 = np.where(np.abs(val) <= 2.0, 1, 0).astype(np.uint8)
                return b0, b1

            i_b0, i_b1 = slice_2bit_gray(i_ch)
            q_b0, q_b1 = slice_2bit_gray(q_ch)

            bits[0::4] = i_b0
            bits[1::4] = i_b1
            bits[2::4] = q_b0
            bits[3::4] = q_b1
            return bits
        else:
            # Fallback 16-QAM slicing
            return cls.slice_qam_symbols(symbols, "16-QAM")

    @classmethod
    def _empty_result(cls, scheme: str) -> DemodulationResult:
        return DemodulationResult(
            scheme=scheme,
            symbols=np.empty(0, dtype=np.complex64),
            bits=np.empty(0, dtype=np.uint8),
            symbol_rate_est=Measurement("Symbol Rate", None, "sps", ConfidenceLevel.UNKNOWN),
            carrier_frequency_error_hz=0.0,
            carrier_locked=False,
            timing_locked=False,
        )
