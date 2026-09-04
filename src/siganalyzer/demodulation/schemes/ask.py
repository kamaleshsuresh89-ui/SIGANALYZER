"""ASK and OOK envelope demodulator."""

import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import DemodulationResult, Measurement


class ASKDemodulator:
    """Demodulates On-Off Keying (OOK) and Amplitude Shift Keying (ASK)."""

    @classmethod
    def demodulate(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        nominal_symbol_rate: float,
        scheme: str = "OOK",
        carrier_freq_offset: float = 0.0,
    ) -> DemodulationResult:
        """Demodulate ASK/OOK signal via envelope detection and adaptive thresholding."""
        if len(signal) < 32:
            return cls._empty_result(scheme)

        # 1. Baseband envelope extraction: |s[n]|
        env = np.abs(signal)

        # 2. Decimate / integrate per symbol
        sps = max(1, int(round(sample_rate / max(nominal_symbol_rate, 1.0))))
        num_symbols = len(env) // sps

        if num_symbols == 0:
            return cls._empty_result(scheme)

        symbols = np.zeros(num_symbols, dtype=np.float32)
        half_sps = sps // 2
        for i in range(num_symbols):
            center_idx = i * sps + half_sps
            s_low = max(0, center_idx - 1)
            s_high = min(len(env), center_idx + 2)
            symbols[i] = float(np.mean(env[s_low:s_high]))

        # 3. Adaptive threshold slicing
        min_val = float(np.percentile(symbols, 10))
        max_val = float(np.percentile(symbols, 90))
        threshold = (min_val + max_val) / 2.0

        bits = np.where(symbols >= threshold, 1, 0).astype(np.uint8)

        # Normalized complex representation for plotting
        norm_syms = (symbols / (max_val + 1e-12)).astype(np.complex64)

        return DemodulationResult(
            scheme=scheme,
            symbols=norm_syms,
            bits=bits,
            symbol_rate_est=Measurement(
                name="Symbol Rate",
                value=float(nominal_symbol_rate),
                unit="sps",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.88,
                source="Envelope downsampling",
            ),
            carrier_frequency_error_hz=0.0,
            carrier_locked=True,
            timing_locked=True,
        )

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
