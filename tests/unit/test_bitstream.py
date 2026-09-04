"""Unit tests for bitstream analysis, sync-word correlation, and framing."""

import numpy as np
import pytest

from siganalyzer.bitstream.analyzer import BitstreamAnalyzer
from siganalyzer.bitstream.correlation import SyncWordCorrelator
from siganalyzer.bitstream.formatter import BitstreamFormatter
from siganalyzer.bitstream.framing import ProtocolFramer
from siganalyzer.coding.crc import CrcAlgorithm, CrcEngine


def test_bitstream_analyzer_metrics() -> None:
    # Alternating bits 0 1 0 1 ...
    alt_bits = np.array([0, 1] * 500, dtype=np.uint8)
    res_alt = BitstreamAnalyzer.analyze(alt_bits, symbol_rate_hz=1000.0)

    assert res_alt.total_bits == 1000
    assert abs(res_alt.transition_density - 1.0) < 1e-3
    assert abs(res_alt.entropy_per_bit - 1.0) < 1e-3

    # Constant bits (all 0s)
    const_bits = np.zeros(500, dtype=np.uint8)
    res_const = BitstreamAnalyzer.analyze(const_bits)
    assert res_const.transition_density == 0.0
    assert res_const.entropy_per_bit == 0.0


def test_sync_word_correlator_barker() -> None:
    # Barker-11: 1 1 1 0 0 0 1 0 0 1 0
    barker11 = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    lead_bits = [0] * 50
    trail_bits = [1] * 50
    stream = np.array(lead_bits + barker11 + trail_bits, dtype=np.uint8)

    matches = SyncWordCorrelator.search_pattern(stream, barker11, name="Barker-11", max_hamming_distance=0)
    assert len(matches) == 1
    assert matches[0].bit_index == 50
    assert matches[0].hamming_distance == 0
    assert matches[0].correlation_score == 1.0
    assert matches[0].inverted is False


def test_sync_word_correlator_inverted() -> None:
    barker7 = [1, 1, 1, 0, 0, 1, 0]
    # Invert bits: 0 0 0 1 1 0 1
    inverted_barker = [1 - b for b in barker7]
    stream = np.array([0] * 20 + inverted_barker + [0] * 20, dtype=np.uint8)

    matches = SyncWordCorrelator.search_pattern(stream, barker7, name="Barker-7", max_hamming_distance=0, check_inverted=True)
    inv_matches = [m for m in matches if m.inverted]
    assert len(inv_matches) == 1
    assert inv_matches[0].bit_index == 20
    assert inv_matches[0].hamming_distance == 0


def test_protocol_framer_extraction() -> None:
    # Construct 2 frames with Barker-11 sync words and CRC16 checksums
    barker11 = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
    payload1 = b"Frame_Payload_One!"
    crc1 = CrcEngine.compute_crc16_ccitt(payload1)
    bytes1 = payload1 + crc1.to_bytes(2, "big")
    bits1 = list(np.unpackbits(np.frombuffer(bytes1, dtype=np.uint8)))

    payload2 = b"Frame_Payload_Two!"
    crc2 = CrcEngine.compute_crc16_ccitt(payload2)
    bytes2 = payload2 + crc2.to_bytes(2, "big")
    bits2 = list(np.unpackbits(np.frombuffer(bytes2, dtype=np.uint8)))

    stream = np.array(barker11 + bits1 + barker11 + bits2, dtype=np.uint8)

    matches = SyncWordCorrelator.search_pattern(stream, barker11, name="Barker-11", max_hamming_distance=0)
    assert len(matches) == 2

    frames = ProtocolFramer.extract_frames(stream, matches, crc_algo=CrcAlgorithm.CRC16_CCITT)
    assert len(frames) == 2
    assert frames[0].crc_valid is True
    assert frames[1].crc_valid is True


def test_bitstream_formatter() -> None:
    data = b"SIGANALYZER Hex Dumper 123456!"
    hexdump = BitstreamFormatter.format_hexdump(data)
    assert "00000000:" in hexdump
    assert "SIGANALYZER" in hexdump

    bits = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)
    bindump = BitstreamFormatter.format_binary_dump(bits, bits_per_group=4)
    assert "bit 000000:" in bindump
    assert "1010" in bindump
