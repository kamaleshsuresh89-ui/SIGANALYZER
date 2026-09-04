"""High-performance Hex dump, Binary, and ASCII text formatters."""

import numpy as np


class BitstreamFormatter:
    """Formats raw bitstreams into classic hex dumps and structured text representations."""

    @classmethod
    def format_hexdump(
        cls,
        data: bytes | np.ndarray,
        bytes_per_row: int = 16,
        max_rows: int = 2000,
    ) -> str:
        """Format data into standard canonical hex dump format:

        00000000: 48 65 6c 6c 6f 20 57 6f  72 6c 64 21 00 00 00 00  |Hello World!....|
        """
        raw = cls._to_bytes(data)
        if not raw:
            return "(Empty bitstream)"

        lines: list[str] = []
        total_len = len(raw)
        row_count = min(max_rows, (total_len + bytes_per_row - 1) // bytes_per_row)

        for r in range(row_count):
            offset = r * bytes_per_row
            chunk = raw[offset : offset + bytes_per_row]

            # Hex bytes formatted into two groups of 8
            hex_parts = [f"{b:02x}" for b in chunk]
            if len(hex_parts) > 8:
                hex_str = " ".join(hex_parts[:8]) + "  " + " ".join(hex_parts[8:])
            else:
                hex_str = " ".join(hex_parts)

            # Pad hex string if last line is short
            expected_hex_len = bytes_per_row * 3
            if len(chunk) < bytes_per_row:
                hex_str = hex_str.ljust(expected_hex_len)

            # ASCII representation
            ascii_str = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)

            lines.append(f"{offset:08x}: {hex_str}  |{ascii_str}|")

        if total_len > max_rows * bytes_per_row:
            remaining = total_len - max_rows * bytes_per_row
            lines.append(f"... ({remaining:,} additional bytes truncated for display)")

        return "\n".join(lines)

    @classmethod
    def format_binary_dump(
        cls,
        bits: np.ndarray,
        bits_per_group: int = 8,
        groups_per_row: int = 4,
        max_rows: int = 1000,
    ) -> str:
        """Format bitstream into grouped binary representation with line numbers."""
        arr = np.asarray(bits, dtype=np.uint8)
        if len(arr) == 0:
            return "(Empty bitstream)"

        bits_per_row = bits_per_group * groups_per_row
        total_bits = len(arr)
        row_count = min(max_rows, (total_bits + bits_per_row - 1) // bits_per_row)

        lines: list[str] = []
        for r in range(row_count):
            offset = r * bits_per_row
            chunk = arr[offset : offset + bits_per_row]

            groups = []
            for g in range((len(chunk) + bits_per_group - 1) // bits_per_group):
                grp = chunk[g * bits_per_group : (g + 1) * bits_per_group]
                groups.append("".join(str(int(b)) for b in grp))

            row_str = " ".join(groups)
            lines.append(f"bit {offset:06d}: {row_str}")

        if total_bits > max_rows * bits_per_row:
            remaining = total_bits - max_rows * bits_per_row
            lines.append(f"... ({remaining:,} additional bits truncated for display)")

        return "\n".join(lines)

    @staticmethod
    def _to_bytes(data: bytes | np.ndarray) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, np.ndarray):
            if data.dtype == np.uint8 and np.all((data == 0) | (data == 1)):
                pad_len = (8 - len(data) % 8) % 8
                if pad_len > 0:
                    padded = np.pad(data, (0, pad_len), constant_values=0)
                else:
                    padded = data
                return bytes(np.packbits(padded))
            return bytes(data.astype(np.uint8))
        return bytes(data)
