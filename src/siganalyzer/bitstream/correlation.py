"""Fast preamble and sync-word cross-correlation search."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import signal


@dataclass
class SyncWordMatch:
    """A detected sync-word or preamble occurrence."""

    name: str
    bit_index: int
    pattern_hex: str
    pattern_length: int
    hamming_distance: int
    correlation_score: float  # 0.0 to 1.0 (1.0 = exact match)
    inverted: bool  # True if matched with inverted bit polarity (180 deg phase ambiguity)


class SyncWordCorrelator:
    """Detects standard and custom preambles/sync-words in a binary bitstream."""

    # Standard RF sync sequences (represented as 0/1 lists)
    PRESETS: dict[str, list[int]] = {
        "Barker-7": [1, 1, 1, 0, 0, 1, 0],
        "Barker-11": [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "Barker-13": [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        "Preamble 0xAA (16-bit)": [1, 0] * 8,
        "Preamble 0xAA (32-bit)": [1, 0] * 16,
        "Preamble 0x55 (16-bit)": [0, 1] * 8,
        "CCSDS ASM (0x1ACFFC1D)": [
            0, 0, 0, 1, 1, 0, 1, 0,
            1, 1, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 1,
        ],
        "GSM TSC0": [
            0, 0, 1, 0, 0, 1, 0, 1,
            1, 1, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1, 0, 1,
            1, 1,
        ],
    }

    @classmethod
    def search_pattern(
        cls,
        bits: np.ndarray,
        pattern: list[int] | np.ndarray,
        name: str = "Custom",
        max_hamming_distance: int = 1,
        check_inverted: bool = True,
    ) -> list[SyncWordMatch]:
        """Search for a specific bit pattern across bitstream using cross-correlation."""
        arr = np.asarray(bits, dtype=np.int8)
        pat = np.asarray(pattern, dtype=np.int8)
        L = len(pat)

        if len(arr) < L:
            return []

        # Convert {0, 1} to bipolar {-1, +1}
        # 0 -> -1, 1 -> +1
        arr_bip = 2 * arr.astype(np.float32) - 1.0
        pat_bip = 2 * pat.astype(np.float32) - 1.0

        # Cross-correlation via valid convolution with flipped pattern
        corr = signal.convolve(arr_bip, pat_bip[::-1], mode="valid")

        matches: list[SyncWordMatch] = []
        pat_hex = cls._pattern_to_hex(pat)

        for idx, val in enumerate(corr):
            # Normal polarity: val in [-L, L]
            # Hamming distance d = (L - val) / 2
            d = int(round((L - val) / 2.0))
            if 0 <= d <= max_hamming_distance:
                score = (L - d) / L
                matches.append(
                    SyncWordMatch(
                        name=name,
                        bit_index=idx,
                        pattern_hex=pat_hex,
                        pattern_length=L,
                        hamming_distance=d,
                        correlation_score=float(score),
                        inverted=False,
                    )
                )

            # Inverted polarity (180 deg phase shift): val -> -L is 0 bit flips
            if check_inverted:
                d_inv = int(round((L + val) / 2.0))
                if 0 <= d_inv <= max_hamming_distance:
                    score_inv = (L - d_inv) / L
                    matches.append(
                        SyncWordMatch(
                            name=f"{name} (Inverted)",
                            bit_index=idx,
                            pattern_hex=pat_hex,
                            pattern_length=L,
                            hamming_distance=d_inv,
                            correlation_score=float(score_inv),
                            inverted=True,
                        )
                    )

        # De-duplicate close adjacent matches within L // 2 bits
        return cls._filter_redundant_matches(matches, min_spacing=max(1, L // 2))

    @classmethod
    def scan_standard_presets(
        cls,
        bits: np.ndarray,
        max_hamming_distance: int = 1,
    ) -> list[SyncWordMatch]:
        """Scan bitstream for all standard library sync-words and preambles."""
        all_matches: list[SyncWordMatch] = []
        for name, pattern in cls.PRESETS.items():
            matches = cls.search_pattern(
                bits,
                pattern,
                name=name,
                max_hamming_distance=max_hamming_distance,
                check_inverted=True,
            )
            all_matches.extend(matches)

        all_matches.sort(key=lambda m: m.bit_index)
        return all_matches

    @classmethod
    def pattern_from_hex(cls, hex_str: str) -> list[int]:
        """Convert hex string (e.g. '1ACFFC1D' or '0xAA') to bit list."""
        clean = hex_str.strip().replace("0x", "").replace(" ", "")
        data = bytes.fromhex(clean)
        bits = []
        for b in data:
            for shift in range(7, -1, -1):
                bits.append((b >> shift) & 1)
        return bits

    @staticmethod
    def _pattern_to_hex(pattern: np.ndarray) -> str:
        pad_len = (8 - len(pattern) % 8) % 8
        if pad_len > 0:
            padded = np.pad(pattern, (0, pad_len), constant_values=0)
        else:
            padded = pattern
        packed = np.packbits(padded)
        return "0x" + bytes(packed).hex().upper()

    @staticmethod
    def _filter_redundant_matches(matches: list[SyncWordMatch], min_spacing: int) -> list[SyncWordMatch]:
        if not matches:
            return []
        matches.sort(key=lambda m: (m.bit_index, m.hamming_distance))
        filtered = [matches[0]]
        for m in matches[1:]:
            last = filtered[-1]
            if (m.bit_index - last.bit_index) >= min_spacing:
                filtered.append(m)
            elif m.hamming_distance < last.hamming_distance:
                filtered[-1] = m
        return filtered
