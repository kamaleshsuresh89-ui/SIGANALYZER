"""Frame boundary detection, protocol partitioning, and CRC validation."""

from dataclasses import dataclass, field
import numpy as np

from siganalyzer.bitstream.correlation import SyncWordMatch
from siganalyzer.coding.crc import CrcAlgorithm, CrcEngine


@dataclass
class BitstreamFrame:
    """A partitioned protocol frame extracted from the bitstream."""

    frame_index: int
    start_bit: int
    end_bit: int
    length_bits: int
    sync_word: str
    payload_bytes: bytes
    payload_hex: str
    crc_algorithm: str = "None"
    crc_valid: bool = False
    expected_crc: int | None = None
    computed_crc: int | None = None


class ProtocolFramer:
    """Partitions continuous bitstream into frames based on detected sync markers."""

    @classmethod
    def infer_frame_length(cls, matches: list[SyncWordMatch]) -> int | None:
        """Infer nominal frame length by examining delta spacings between identical sync words."""
        if len(matches) < 2:
            return None

        # Group by sync word name
        by_name: dict[str, list[int]] = {}
        for m in matches:
            by_name.setdefault(m.name, []).append(m.bit_index)

        candidate_lengths: list[int] = []
        for name, indices in by_name.items():
            if len(indices) >= 2:
                deltas = np.diff(indices)
                # Find most common delta
                vals, counts = np.unique(deltas, return_counts=True)
                top_idx = int(np.argmax(counts))
                if counts[top_idx] >= 2 or len(indices) == 2:
                    candidate_lengths.append(int(vals[top_idx]))

        if candidate_lengths:
            return candidate_lengths[0]
        return None

    @classmethod
    def extract_frames(
        cls,
        bits: np.ndarray,
        sync_matches: list[SyncWordMatch],
        nominal_frame_length_bits: int | None = None,
        crc_algo: CrcAlgorithm = CrcAlgorithm.CRC16_CCITT,
    ) -> list[BitstreamFrame]:
        """Slice bitstream into discrete frames following sync-word occurrences."""
        arr = np.asarray(bits, dtype=np.uint8)
        if not sync_matches:
            return []

        # If nominal length not specified, infer or use inter-match distance
        frame_len = nominal_frame_length_bits or cls.infer_frame_length(sync_matches)

        frames: list[BitstreamFrame] = []
        for i, match in enumerate(sync_matches):
            start = match.bit_index
            if frame_len is not None:
                end = min(len(arr), start + frame_len)
            elif i + 1 < len(sync_matches):
                end = sync_matches[i + 1].bit_index
            else:
                end = min(len(arr), start + 256)  # Default fallback 256 bits

            frame_bits = arr[start:end]
            # Strip sync word for payload
            payload_bits = frame_bits[match.pattern_length :]
            if len(payload_bits) == 0:
                continue

            # Pack payload bits to bytes
            pad_len = (8 - len(payload_bits) % 8) % 8
            if pad_len > 0:
                padded = np.pad(payload_bits, (0, pad_len), constant_values=0)
            else:
                padded = payload_bits
            payload_bytes = bytes(np.packbits(padded))
            payload_hex = payload_bytes.hex().upper()

            # CRC check
            valid, clean_p, exp_c, comp_c = CrcEngine.check_trailing_crc(payload_bytes, crc_algo)

            frames.append(
                BitstreamFrame(
                    frame_index=i + 1,
                    start_bit=start,
                    end_bit=end,
                    length_bits=len(frame_bits),
                    sync_word=match.name,
                    payload_bytes=payload_bytes,
                    payload_hex=payload_hex,
                    crc_algorithm=crc_algo.value if valid else "None",
                    crc_valid=valid,
                    expected_crc=exp_c if valid else None,
                    computed_crc=comp_c if valid else None,
                )
            )

        return frames
