"""Statistical bitstream and entropy analyzer."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import BitstreamAnalysis, Measurement


class BitstreamAnalyzer:
    """Computes bit-level statistics, entropy, transition metrics, and data previews."""

    @classmethod
    def analyze(
        cls,
        bits: np.ndarray,
        symbol_rate_hz: float | None = None,
        bits_per_symbol: int = 1,
        detected_sync_words: list[dict[str, Any]] | None = None,
        inferred_frame_length_bits: int | None = None,
    ) -> BitstreamAnalysis:
        """Perform comprehensive statistical and information-theoretic analysis on raw bitstream."""
        arr = np.asarray(bits, dtype=np.uint8)
        total_bits = len(arr)

        if total_bits == 0:
            return BitstreamAnalysis(
                raw_bits=np.empty(0, dtype=np.uint8),
                total_bits=0,
                bit_rate_est=Measurement("Bit Rate", 0.0, "bps", ConfidenceLevel.UNKNOWN),
                transition_density=0.0,
                entropy_per_bit=0.0,
                detected_sync_words=[],
                preamble_detected=False,
                hex_preview="",
                ascii_preview="",
            )

        # 1. Bit Rate Estimation
        if symbol_rate_hz is not None and symbol_rate_hz > 0:
            bit_rate_val = symbol_rate_hz * bits_per_symbol
            bit_rate = Measurement(
                name="Bit Rate",
                value=float(bit_rate_val),
                unit="bps",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.90,
                source="Symbol Rate * Bits/Symbol",
            )
        else:
            bit_rate = Measurement(
                name="Bit Rate",
                value=None,
                unit="bps",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        # 2. Transition Density
        if total_bits > 1:
            transitions = int(np.sum(arr[1:] != arr[:-1]))
            transition_density = float(transitions / (total_bits - 1))
        else:
            transition_density = 0.0

        # 3. Shannon Entropy per bit
        p1 = float(np.mean(arr))
        p0 = 1.0 - p1
        eps = 1e-12
        if p0 > eps and p1 > eps:
            entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            entropy = 0.0

        # 4. Hex & ASCII Preview (First 256 bytes / 2048 bits)
        preview_bits = arr[: min(2048, total_bits)]
        pad_len = (8 - len(preview_bits) % 8) % 8
        if pad_len > 0:
            padded = np.pad(preview_bits, (0, pad_len), constant_values=0)
        else:
            padded = preview_bits
        packed_bytes = np.packbits(padded)

        hex_preview = " ".join(f"{b:02X}" for b in packed_bytes[:64])
        if len(packed_bytes) > 64:
            hex_preview += " ..."

        ascii_chars = [chr(b) if 32 <= b <= 126 else "." for b in packed_bytes[:64]]
        ascii_preview = "".join(ascii_chars)
        if len(packed_bytes) > 64:
            ascii_preview += " ..."

        sync_words = detected_sync_words or []
        has_preamble = any(s.get("type") == "Preamble" or s.get("name", "").startswith("Preamble") for s in sync_words)

        return BitstreamAnalysis(
            raw_bits=arr,
            total_bits=total_bits,
            bit_rate_est=bit_rate,
            transition_density=transition_density,
            entropy_per_bit=float(entropy),
            detected_sync_words=sync_words,
            preamble_detected=has_preamble,
            inferred_frame_length_bits=inferred_frame_length_bits,
            hex_preview=hex_preview,
            ascii_preview=ascii_preview,
        )
