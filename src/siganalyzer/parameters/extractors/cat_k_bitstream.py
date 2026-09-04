"""Category K: Bitstream Analysis parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class BitstreamAnalysisExtractor(BaseParameterExtractor):
    """Extracts bitstream statistics, 0/1 balance, run lengths, Shannon entropy, and autocorrelation periodicity."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.BITSTREAM

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("bitstream.total_bits_count", "Total Demodulated Bits", cat, "bits", int, "Length of recovered binary bit sequence", ConfidenceLevel.MEASURED, export_priority=1),
            ParameterDefinition("bitstream.total_bytes_count", "Total Decoded Bytes", cat, "bytes", int, "Byte count floor(bits / 8)", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("bitstream.ones_ratio_percent", "Ones (1) Ratio", cat, "%", float, "Percentage of binary '1' bits", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("bitstream.zeros_ratio_percent", "Zeros (0) Ratio", cat, "%", float, "Percentage of binary '0' bits", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("bitstream.zero_one_bias", "Zero/One Bias Factor", cat, "", float, "Normalized balance metric (N_1 - N_0) / N", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("bitstream.transition_density_flips_per_bit", "Transition Density", cat, "flips/bit", float, "Rate of bit reversals (0->1 or 1->0)", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("bitstream.max_consecutive_zeros", "Max Consecutive Zeros", cat, "bits", int, "Longest contiguous sequence of 0s", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("bitstream.max_consecutive_ones", "Max Consecutive Ones", cat, "bits", int, "Longest contiguous sequence of 1s", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("bitstream.shannon_bit_entropy", "Shannon Bit Entropy", cat, "bits", float, "Information entropy per bit (max 1.0)", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("bitstream.shannon_byte_entropy", "Shannon Byte Entropy", cat, "bits/byte", float, "Information entropy over 8-bit symbols (max 8.0)", ConfidenceLevel.MEASURED, export_priority=10),
            ParameterDefinition("bitstream.autocorrelation_peak_lag_bits", "Autocorrelation Peak Lag", cat, "bits", int, "Bit lag of strongest periodic repetition peak", ConfidenceLevel.MEASURED, export_priority=11),
            ParameterDefinition("bitstream.periodicity_detected_flag", "Bitstream Periodicity Detected", cat, "", bool, "Indicates repeating frame or PRBS sequence", ConfidenceLevel.INFERRED, export_priority=12),
            ParameterDefinition("bitstream.hex_preview_snippet", "Hex Data Preview", cat, "hex", str, "Hexadecimal representation of leading payload bytes", ConfidenceLevel.DIRECT, export_priority=13),
            ParameterDefinition("bitstream.ascii_preview_snippet", "ASCII Preview", cat, "", str, "Printable text interpretation of leading payload bytes", ConfidenceLevel.DIRECT, export_priority=14),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        bitstream = context.bitstream
        results: list[ParameterResult] = []

        bits = bitstream.raw_bits if (bitstream and len(bitstream.raw_bits) > 0) else None

        if bits is not None and len(bits) > 0:
            total_bits = len(bits)
            total_bytes = total_bits // 8

            ones = int(np.sum(bits == 1))
            zeros = total_bits - ones

            p1 = ones / total_bits
            p0 = zeros / total_bits

            pct_ones = float(p1 * 100.0)
            pct_zeros = float(p0 * 100.0)
            bias = float(p1 - p0)

            # Transition density
            if total_bits > 1:
                flips = int(np.sum(bits[1:] != bits[:-1]))
                t_density = float(flips / (total_bits - 1))
            else:
                t_density = 0.0

            # Max consecutive runs
            max_zeros = 0
            max_ones = 0
            curr_zeros = 0
            curr_ones = 0
            for b in bits:
                if b == 0:
                    curr_zeros += 1
                    curr_ones = 0
                    if curr_zeros > max_zeros:
                        max_zeros = curr_zeros
                else:
                    curr_ones += 1
                    curr_zeros = 0
                    if curr_ones > max_ones:
                        max_ones = curr_ones

            # Shannon bit entropy: H = - sum(p * log2(p))
            h_bit = 0.0
            if p0 > 0:
                h_bit -= p0 * np.log2(p0)
            if p1 > 0:
                h_bit -= p1 * np.log2(p1)
            h_bit = float(min(1.0, max(0.0, h_bit)))

            # Shannon byte entropy
            if total_bytes >= 4:
                packed_bytes = np.packbits(bits[: total_bytes * 8])
                _, counts = np.unique(packed_bytes, return_counts=True)
                probs = counts / float(total_bytes)
                h_byte = float(-np.sum(probs * np.log2(probs)))
            else:
                h_byte = float(h_bit * 8.0)

            # Autocorrelation periodicity
            period_detected = False
            best_lag = 0
            if total_bits >= 64:
                test_len = min(total_bits, 4096)
                b_norm = bits[:test_len].astype(np.float32) - 0.5
                autocorr = np.correlate(b_norm, b_norm, mode='full')
                half = len(autocorr) // 2
                pos_corr = autocorr[half + 1 : half + min(512, len(b_norm) // 2)]
                if len(pos_corr) > 16:
                    peak_lag = int(np.argmax(pos_corr) + 1)
                    peak_val = pos_corr[peak_lag - 1] / (autocorr[half] + 1e-12)
                    if peak_val > 0.40:
                        period_detected = True
                        best_lag = peak_lag

            # Previews
            hex_str = bitstream.hex_preview if bitstream.hex_preview else "N/A"
            ascii_str = bitstream.ascii_preview if bitstream.ascii_preview else "N/A"

            results.append(self._make_res("bitstream.total_bits_count", total_bits, "len(bits)", f"{total_bits:,} bits."))
            results.append(self._make_res("bitstream.total_bytes_count", total_bytes, "floor(len / 8)", f"{total_bytes:,} bytes."))
            results.append(self._make_res("bitstream.ones_ratio_percent", pct_ones, "count(1) / N * 100", f"{pct_ones:.2f}%."))
            results.append(self._make_res("bitstream.zeros_ratio_percent", pct_zeros, "count(0) / N * 100", f"{pct_zeros:.2f}%."))
            results.append(self._make_res("bitstream.zero_one_bias", bias, "(N_1 - N_0) / N", f"Balance bias: {bias:+.4f}."))
            results.append(self._make_res("bitstream.transition_density_flips_per_bit", t_density, "sum(b[n] != b[n-1]) / (N-1)", f"{t_density:.4f} flips/bit (ideal random: 0.5000)."))
            results.append(self._make_res("bitstream.max_consecutive_zeros", max_zeros, "Max run of zeros", f"{max_zeros} bits."))
            results.append(self._make_res("bitstream.max_consecutive_ones", max_ones, "Max run of ones", f"{max_ones} bits."))
            results.append(self._make_res("bitstream.shannon_bit_entropy", h_bit, "-sum(p*log2(p))", f"{h_bit:.4f} bits/bit."))
            results.append(self._make_res("bitstream.shannon_byte_entropy", h_byte, "-sum(p_byte*log2(p_byte))", f"{h_byte:.3f} bits/byte (max 8.000)."))
            results.append(self._make_res("bitstream.autocorrelation_peak_lag_bits", best_lag, "argmax(R_xx[lag > 0])", f"Dominant lag: {best_lag} bits." if period_detected else "No dominant periodic lag."))
            results.append(self._make_res("bitstream.periodicity_detected_flag", period_detected, "Normalized autocorrelation threshold > 0.40", "Periodic framing / pattern confirmed." if period_detected else "No significant bit periodicity."))
            results.append(self._make_res("bitstream.hex_preview_snippet", hex_str, "Bitstream formatting", hex_str, ConfidenceLevel.DIRECT))
            results.append(self._make_res("bitstream.ascii_preview_snippet", ascii_str, "ASCII character filter", ascii_str, ConfidenceLevel.DIRECT))

        else:
            for pid, desc in [
                ("bitstream.total_bits_count", "Demodulation did not produce bitstream"),
                ("bitstream.total_bytes_count", "Demodulation did not produce bitstream"),
                ("bitstream.ones_ratio_percent", "Demodulation did not produce bitstream"),
                ("bitstream.zeros_ratio_percent", "Demodulation did not produce bitstream"),
                ("bitstream.zero_one_bias", "Demodulation did not produce bitstream"),
                ("bitstream.transition_density_flips_per_bit", "Demodulation did not produce bitstream"),
                ("bitstream.max_consecutive_zeros", "Demodulation did not produce bitstream"),
                ("bitstream.max_consecutive_ones", "Demodulation did not produce bitstream"),
                ("bitstream.shannon_bit_entropy", "Demodulation did not produce bitstream"),
                ("bitstream.shannon_byte_entropy", "Demodulation did not produce bitstream"),
                ("bitstream.autocorrelation_peak_lag_bits", "Demodulation did not produce bitstream"),
                ("bitstream.periodicity_detected_flag", "Demodulation did not produce bitstream"),
                ("bitstream.hex_preview_snippet", "Demodulation did not produce bitstream"),
                ("bitstream.ascii_preview_snippet", "Demodulation did not produce bitstream"),
            ]:
                results.append(ParameterResult(
                    definition=self._get_def(pid),
                    value=None,
                    status=ConfidenceLevel.UNKNOWN,
                    confidence_score=0.0,
                    source="Bitstream Analyzer",
                    method="N/A",
                    explanation=f"UNKNOWN: {desc}.",
                ))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.98 if status == ConfidenceLevel.DIRECT else 0.95,
            source="Bitstream Subsystem",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
