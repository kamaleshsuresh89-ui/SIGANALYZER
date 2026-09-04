"""Unit tests for Viterbi convolutional encoder and decoder."""

import numpy as np
import pytest

from siganalyzer.coding.viterbi import ConvolutionalCodec, ViterbiConfig


def test_viterbi_clean_encode_decode() -> None:
    codec = ConvolutionalCodec()
    rng = np.random.default_rng(99)
    info_bits = rng.integers(0, 2, size=64, dtype=np.uint8)

    encoded = codec.encode(info_bits, flush=True)
    assert len(encoded) == (len(info_bits) + codec.k - 1) * 2

    decoded, metric = codec.decode_hard(encoded, truncate_tail=True)
    assert metric == 0
    np.testing.assert_array_equal(info_bits, decoded)


def test_viterbi_error_correction() -> None:
    codec = ConvolutionalCodec()
    rng = np.random.default_rng(101)
    info_bits = rng.integers(0, 2, size=64, dtype=np.uint8)

    encoded = codec.encode(info_bits, flush=True)

    # Corrupt 3 isolated bits in the encoded stream
    corrupted = np.copy(encoded)
    corrupted[5] ^= 1
    corrupted[25] ^= 1
    corrupted[55] ^= 1

    decoded, metric = codec.decode_hard(corrupted, truncate_tail=True)
    # Viterbi should correct isolated bit flips and recover the exact payload
    assert metric > 0
    np.testing.assert_array_equal(info_bits, decoded)


def test_viterbi_soft_decision() -> None:
    codec = ConvolutionalCodec()
    info_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    encoded = codec.encode(info_bits, flush=True)

    # Convert encoded bits to soft symbols: 0 -> +1.0, 1 -> -1.0
    soft_symbols = (1.0 - 2.0 * encoded.astype(np.float32))

    # Add small Gaussian noise
    rng = np.random.default_rng(42)
    noisy = soft_symbols + rng.normal(0, 0.2, size=len(soft_symbols)).astype(np.float32)

    decoded, metric = codec.decode_soft(noisy, truncate_tail=True)
    np.testing.assert_array_equal(info_bits, decoded)
