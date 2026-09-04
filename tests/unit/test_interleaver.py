"""Unit tests for block and convolutional interleavers."""

import numpy as np
import pytest

from siganalyzer.coding.interleaver import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    detect_interleaving_depth,
)


def test_block_interleaver_roundtrip() -> None:
    rng = np.random.default_rng(42)
    original = rng.integers(0, 2, size=120, dtype=np.uint8)

    interleaver = BlockInterleaver(nrows=10, ncols=12)
    interleaved = interleaver.interleave(original)

    # Must have same length
    assert len(interleaved) == len(original)
    # Shouldn't be identical to original in general
    assert not np.array_equal(original, interleaved)

    deinterleaved = interleaver.deinterleave(interleaved, original_length=len(original))
    np.testing.assert_array_equal(original, deinterleaved)


def test_block_interleaver_padding() -> None:
    # Length 50 with 8x8 = 64 block size -> requires 14 padding bits
    data = np.ones(50, dtype=np.uint8)
    interleaver = BlockInterleaver(nrows=8, ncols=8)
    interleaved = interleaver.interleave(data)
    assert len(interleaved) == 64

    recovered = interleaver.deinterleave(interleaved, original_length=50)
    assert len(recovered) == 50
    np.testing.assert_array_equal(data, recovered)


def test_convolutional_interleaver() -> None:
    interleaver = ConvolutionalInterleaver(num_branches=4, delay_step=2)
    data = np.arange(40, dtype=np.int32)
    interleaved = interleaver.interleave(data)
    deinterleaved = interleaver.deinterleave(interleaved)

    # After full flushing delay (num_branches * (num_branches - 1) * delay_step = 4 * 3 * 2 = 24)
    # the signal is restored with a delay
    delay = 4 * 3 * 2
    np.testing.assert_array_equal(data[: 40 - delay], deinterleaved[delay:])


def test_detect_interleaving_depth() -> None:
    # Synthesize periodic repeating pattern of period 16
    rng = np.random.default_rng(123)
    base_pattern = rng.integers(0, 2, size=16, dtype=np.uint8)
    repeats = np.tile(base_pattern, 50)  # 800 bits

    candidates = detect_interleaving_depth(repeats, min_depth=4, max_depth=64)
    assert len(candidates) > 0
    # Depth 16 (or its harmonic) should be found
    depths = [c.depth for c in candidates]
    assert 16 in depths or 32 in depths
