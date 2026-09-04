"""Central Demodulation Manager coordinating synchronization and scheme decoders."""

import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import DemodulationResult, Measurement
from siganalyzer.demodulation.constellation import ConstellationAnalyzer, ConstellationMetrics
from siganalyzer.demodulation.schemes.ask import ASKDemodulator
from siganalyzer.demodulation.schemes.fsk import FSKDemodulator
from siganalyzer.demodulation.schemes.psk import PSKDemodulator
from siganalyzer.demodulation.schemes.qam import QAMDemodulator


class DemodulationManager:
    """Coordinates signal synchronization, symbol extraction, and decision slicing."""

    @classmethod
    def demodulate_signal(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        detected_scheme: str,
        symbol_rate_hz: float | None = None,
        carrier_freq_offset_hz: float = 0.0,
    ) -> tuple[DemodulationResult, ConstellationMetrics]:
        """Run modular demodulator matching the detected modulation scheme.

        Returns:
            Tuple of (DemodulationResult, ConstellationMetrics).
        """
        # Default nominal symbol rate if not detected (e.g. 1/8 of sample rate)
        sym_rate = symbol_rate_hz if symbol_rate_hz and symbol_rate_hz > 0 else (sample_rate / 8.0)

        # Dispatch to scheme demodulator
        scheme_upper = detected_scheme.upper()

        if scheme_upper in ("BPSK", "QPSK", "8PSK"):
            result = PSKDemodulator.demodulate(
                signal, sample_rate, sym_rate, scheme=scheme_upper, carrier_freq_offset=carrier_freq_offset_hz
            )
        elif scheme_upper in ("2FSK", "4FSK", "MSK", "GMSK"):
            # Map MSK to 2FSK discriminator
            fsk_type = "4FSK" if scheme_upper == "4FSK" else "2FSK"
            result = FSKDemodulator.demodulate(
                signal, sample_rate, sym_rate, scheme=fsk_type, carrier_freq_offset=carrier_freq_offset_hz
            )
        elif scheme_upper in ("OOK", "ASK"):
            result = ASKDemodulator.demodulate(
                signal, sample_rate, sym_rate, scheme=scheme_upper, carrier_freq_offset=carrier_freq_offset_hz
            )
        elif "QAM" in scheme_upper:
            qam_type = "64-QAM" if "64" in scheme_upper else "16-QAM"
            result = QAMDemodulator.demodulate(
                signal, sample_rate, sym_rate, scheme=qam_type, carrier_freq_offset=carrier_freq_offset_hz
            )
        else:
            # Fallback to QPSK default
            result = PSKDemodulator.demodulate(
                signal, sample_rate, sym_rate, scheme="QPSK", carrier_freq_offset=carrier_freq_offset_hz
            )

        # Compute constellation metrics on recovered symbols
        const_metrics = ConstellationAnalyzer.analyze_constellation(result.symbols)

        return result, const_metrics
