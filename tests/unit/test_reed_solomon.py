"""Unit tests for Galois Field arithmetic and Reed-Solomon codec."""

import pytest

from siganalyzer.coding.reed_solomon import GaloisField256, ReedSolomonCodec


def test_galois_field_arithmetic() -> None:
    gf = GaloisField256()

    # Addition & Subtraction (XOR)
    assert gf.add(5, 3) == 6
    assert gf.sub(6, 3) == 5

    # Multiplication & Division
    assert gf.mul(0, 42) == 0
    assert gf.mul(1, 42) == 42
    p = gf.mul(7, 11)
    assert gf.div(p, 11) == 7
    assert gf.div(p, 7) == 11

    # Inversion
    inv7 = gf.inv(7)
    assert gf.mul(7, inv7) == 1


def test_reed_solomon_clean_codeword() -> None:
    # Small test RS(15, 9) with 6 parity symbols (can correct up to 3 errors)
    rs = ReedSolomonCodec(n=15, k=9)
    msg = b"Hello RF!"
    codeword = rs.encode(msg)

    assert len(codeword) == 15
    res = rs.decode(codeword)
    assert res.success is True
    assert res.error_count == 0
    assert res.corrected_data == msg


def test_reed_solomon_error_correction() -> None:
    rs = ReedSolomonCodec(n=30, k=20)  # 10 parity symbols -> can correct up to 5 errors
    msg = b"SIGANALYZER_TEST1234"
    codeword = rs.encode(msg)

    # Introduce 3 corrupted bytes
    corrupted = bytearray(codeword)
    corrupted[2] ^= 0x55
    corrupted[7] ^= 0xAA
    corrupted[14] ^= 0xFF

    res = rs.decode(bytes(corrupted))
    assert res.success is True
    assert res.error_count == 3
    assert res.corrected_data == msg


def test_reed_solomon_uncorrectable() -> None:
    rs = ReedSolomonCodec(n=20, k=16)  # 4 parity symbols -> can correct at most 2 errors
    msg = b"1234567890ABCDEF"
    codeword = rs.encode(msg)

    # Introduce 4 corrupted bytes (exceeds t = 2)
    corrupted = bytearray(codeword)
    corrupted[0] ^= 0x01
    corrupted[1] ^= 0x02
    corrupted[2] ^= 0x03
    corrupted[3] ^= 0x04

    res = rs.decode(bytes(corrupted))
    assert res.success is False
