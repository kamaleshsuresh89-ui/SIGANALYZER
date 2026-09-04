"""Cyclic Redundancy Check (CRC) verification and calculation engine."""

from enum import Enum
import numpy as np


class CrcAlgorithm(str, Enum):
    """Supported standard CRC algorithms."""

    CRC8_ATM = "CRC-8 (ATM)"
    CRC16_CCITT = "CRC-16-CCITT"
    CRC16_IBM = "CRC-16-IBM"
    CRC32 = "CRC-32 (IEEE 802.3)"


class CrcEngine:
    """Calculates and verifies standard CRCs on bit arrays and byte buffers."""

    # CRC-32 (IEEE 802.3) precomputed table
    _CRC32_TABLE: list[int] = []
    # CRC-16-CCITT precomputed table
    _CRC16_CCITT_TABLE: list[int] = []

    @classmethod
    def _init_tables(cls) -> None:
        if not cls._CRC32_TABLE:
            for i in range(256):
                c = i
                for _ in range(8):
                    if c & 1:
                        c = 0xEDB88320 ^ (c >> 1)
                    else:
                        c = c >> 1
                cls._CRC32_TABLE.append(c)

        if not cls._CRC16_CCITT_TABLE:
            for i in range(256):
                curr = i << 8
                for _ in range(8):
                    if curr & 0x8000:
                        curr = ((curr << 1) ^ 0x1021) & 0xFFFF
                    else:
                        curr = (curr << 1) & 0xFFFF
                cls._CRC16_CCITT_TABLE.append(curr)

    @classmethod
    def compute_crc8_atm(cls, data: bytes | np.ndarray) -> int:
        """Compute CRC-8 ATM with polynomial x^8 + x^2 + x + 1 (0x07), init 0x00."""
        byte_data = cls._to_bytes(data)
        crc = 0x00
        for b in byte_data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    @classmethod
    def compute_crc16_ccitt(cls, data: bytes | np.ndarray, init: int = 0xFFFF) -> int:
        """Compute CRC-16-CCITT with polynomial x^16 + x^12 + x^5 + 1 (0x1021)."""
        cls._init_tables()
        byte_data = cls._to_bytes(data)
        crc = init
        for b in byte_data:
            tbl_idx = ((crc >> 8) ^ b) & 0xFF
            crc = ((crc << 8) ^ cls._CRC16_CCITT_TABLE[tbl_idx]) & 0xFFFF
        return crc

    @classmethod
    def compute_crc16_ibm(cls, data: bytes | np.ndarray, init: int = 0x0000) -> int:
        """Compute CRC-16-IBM / ANSI with polynomial x^16 + x^15 + x^2 + 1 (0x8005), reflected."""
        byte_data = cls._to_bytes(data)
        crc = init
        for b in byte_data:
            crc ^= b
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001  # Reflected 0x8005
                else:
                    crc = crc >> 1
        return crc & 0xFFFF

    @classmethod
    def compute_crc32(cls, data: bytes | np.ndarray) -> int:
        """Compute standard CRC-32 (IEEE 802.3) with init 0xFFFFFFFF and final XOR 0xFFFFFFFF."""
        cls._init_tables()
        byte_data = cls._to_bytes(data)
        crc = 0xFFFFFFFF
        for b in byte_data:
            tbl_idx = (crc ^ b) & 0xFF
            crc = (crc >> 8) ^ cls._CRC32_TABLE[tbl_idx]
        return crc ^ 0xFFFFFFFF

    @classmethod
    def verify(
        cls,
        payload: bytes | np.ndarray,
        expected_crc: int,
        algorithm: CrcAlgorithm = CrcAlgorithm.CRC16_CCITT,
    ) -> bool:
        """Verify if payload matches expected CRC checksum."""
        computed = cls.compute(payload, algorithm)
        return computed == expected_crc

    @classmethod
    def compute(
        cls,
        data: bytes | np.ndarray,
        algorithm: CrcAlgorithm = CrcAlgorithm.CRC16_CCITT,
    ) -> int:
        """Dispatch CRC calculation based on selected algorithm."""
        if algorithm == CrcAlgorithm.CRC8_ATM:
            return cls.compute_crc8_atm(data)
        elif algorithm == CrcAlgorithm.CRC16_CCITT:
            return cls.compute_crc16_ccitt(data)
        elif algorithm == CrcAlgorithm.CRC16_IBM:
            return cls.compute_crc16_ibm(data)
        elif algorithm == CrcAlgorithm.CRC32:
            return cls.compute_crc32(data)
        else:
            raise ValueError(f"Unsupported CRC algorithm: {algorithm}")

    @classmethod
    def check_trailing_crc(
        cls,
        frame_bytes: bytes | np.ndarray,
        algorithm: CrcAlgorithm = CrcAlgorithm.CRC16_CCITT,
    ) -> tuple[bool, bytes, int, int]:
        """Check if the last bytes of a frame contain a valid CRC checksum.

        Returns:
            Tuple of (is_valid, payload_bytes, expected_crc, computed_crc).
        """
        raw = cls._to_bytes(frame_bytes)
        crc_len = 1 if algorithm == CrcAlgorithm.CRC8_ATM else (4 if algorithm == CrcAlgorithm.CRC32 else 2)

        if len(raw) <= crc_len:
            return False, raw, 0, 0

        payload = raw[:-crc_len]
        crc_bytes = raw[-crc_len:]

        # Big-endian unpack
        expected_crc = int.from_bytes(crc_bytes, byteorder="big")
        computed_crc = cls.compute(payload, algorithm)

        if computed_crc == expected_crc:
            return True, payload, expected_crc, computed_crc

        # Try little-endian unpack
        expected_crc_le = int.from_bytes(crc_bytes, byteorder="little")
        if computed_crc == expected_crc_le:
            return True, payload, expected_crc_le, computed_crc

        return False, payload, expected_crc, computed_crc

    @staticmethod
    def _to_bytes(data: bytes | np.ndarray) -> bytes:
        if isinstance(data, bytes):
            return data
        if isinstance(data, np.ndarray):
            # Check if bits (0/1) or packed bytes (uint8)
            if data.dtype == np.uint8 and np.all((data == 0) | (data == 1)):
                # Pack bits (MSB first)
                pad_len = (8 - len(data) % 8) % 8
                if pad_len > 0:
                    padded = np.pad(data, (0, pad_len), constant_values=0)
                else:
                    padded = data
                packed = np.packbits(padded)
                return bytes(packed)
            return bytes(data.astype(np.uint8))
        return bytes(data)
