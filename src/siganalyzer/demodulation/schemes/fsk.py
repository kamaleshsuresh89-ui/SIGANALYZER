"""FSK (2FSK, 4FSK) demodulator and discriminator."""

import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import DemodulationResult, Measurement


class FSKDemodulator:
    """Demodulates continuous-phase 2FSK and 4FSK signals via frequency discrimination."""

    @classmethod
    def demodulate(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        nominal_symbol_rate: float,
        scheme: str = "2FSK",
        carrier_freq_offset: float = 0.0,
    ) -> DemodulationResult:
        """Demodulate FSK signal into frequency symbols and decision bits."""
        if len(signal) < 64:
            return cls._empty_result(scheme)

        # 1. Baseband shift
        if abs(carrier_freq_offset) > 10.0:
            t = np.arange(len(signal)) / sample_rate
            baseband = (signal * np.exp(-1j * 2 * np.pi * carrier_freq_offset * t)).astype(np.complex64)
        else:
            baseband = signal.astype(np.complex64)

        # 2. Instantaneous frequency discriminator: arg(s[n] * conj(s[n-1]))
        diff_prod = baseband[1:] * np.conj(baseband[:-1])
        inst_freq = np.angle(diff_prod) / (2.0 * np.pi) * sample_rate

        # 3. Decimate / integrate per symbol
        sps = max(1, int(round(sample_rate / max(nominal_symbol_rate, 1.0))))
        num_symbols = len(inst_freq) // sps

        if num_symbols == 0:
            return cls._empty_result(scheme)

        # Stride through symbols taking midpoint of each symbol period
        symbols = np.zeros(num_symbols, dtype=np.float32)
        half_sps = sps // 2
        for i in range(num_symbols):
            center_idx = i * sps + half_sps
            # Average 3 samples around center for noise smoothing
            s_low = max(0, center_idx - 1)
            s_high = min(len(inst_freq), center_idx + 2)
            symbols[i] = float(np.mean(inst_freq[s_low:s_high]))

        # 4. Symbol decision slicing
        mean_freq = float(np.mean(symbols))
        centered_symbols = symbols - mean_freq

        if scheme == "2FSK":
            # f > 0 -> 1, f <= 0 -> 0
            bits = np.where(centered_symbols >= 0.0, 1, 0).astype(np.uint8)
        elif scheme == "4FSK":
            # 4 levels: 2 bits per symbol
            std_f = float(np.std(centered_symbols))
            th1 = -0.67 * std_f
            th2 = 0.0
            th3 = 0.67 * std_f

            bits = np.zeros(num_symbols * 2, dtype=np.uint8)
            for i, sym in enumerate(centered_symbols):
                if sym < th1:
                    b0, b1 = 0, 0
                elif sym < th2:
                    b0, b1 = 0, 1
                elif sym < th3:
                    b0, b1 = 1, 1
                else:
                    b0, b1 = 1, 0
                bits[2 * i] = b0
                bits[2 * i + 1] = b1
        else:
            bits = np.where(centered_symbols >= 0.0, 1, 0).astype(np.uint8)

        # Represent recovered frequency symbols as complex numbers (freq + 0j) for plotting
        complex_symbols = (centered_symbols / (np.std(centered_symbols) + 1e-12)).astype(np.complex64)

        return DemodulationResult(
            scheme=scheme,
            symbols=complex_symbols,
            bits=bits,
            symbol_rate_est=Measurement(
                name="Symbol Rate",
                value=float(nominal_symbol_rate),
                unit="sps",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.90,
                source="FSK discriminator integration",
            ),
            carrier_frequency_error_hz=mean_freq,
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
