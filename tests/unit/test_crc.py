"""Unit tests for CRC calculation and verification engine."""

import numpy as np
import pytest

from siganalyzer.coding.crc import CrcAlgorithm, CrcEngine


def test_crc32_standard_vector() -> None:
    # Standard ASCII test vector "123456789"
    test_bytes = b"123456789"
    crc = CrcEngine.compute_crc32(test_bytes)
    assert crc == 0xCBF43926


def test_crc16_ccitt_standard_vector() -> None:
    test_bytes = b"123456789"
    crc = CrcEngine.compute_crc16_ccitt(test_bytes, init=0xFFFF)
    assert crc == 0x29B1


def test_crc16_ibm_standard_vector() -> None:
    test_bytes = b"123456789"
    crc = CrcEngine.compute_crc16_ibm(test_bytes, init=0x0000)
    assert crc == 0xBB3D


def test_crc8_atm() -> None:
    test_bytes = b"123456789"
    crc = CrcEngine.compute_crc8_atm(test_bytes)
    assert isinstance(crc, int)
    assert 0 <= crc <= 255


def test_trailing_crc_check() -> None:
    payload = b"Telemetry_Frame_Data_Payload_123"
    crc16 = CrcEngine.compute_crc16_ccitt(payload)
    frame = payload + crc16.to_bytes(2, byteorder="big")

    is_valid, extracted_payload, exp_c, comp_c = CrcEngine.check_trailing_crc(
        frame, CrcAlgorithm.CRC16_CCITT
    )
    assert is_valid is True
    assert extracted_payload == payload
    assert exp_c == crc16
    assert comp_c == crc16

    # Corrupt payload
    corrupted_frame = b"X" + frame[1:]
    is_valid_bad, _, _, _ = CrcEngine.check_trailing_crc(corrupted_frame, CrcAlgorithm.CRC16_CCITT)
    assert is_valid_bad is False
