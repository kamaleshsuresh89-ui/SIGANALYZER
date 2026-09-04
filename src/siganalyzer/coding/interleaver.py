"""Matrix and convolutional de-interleaving with autocorrelation depth detection."""

from dataclasses import dataclass
import numpy as np


@dataclass
class InterleaverCandidate:
    """Candidate interleaving depth detected from autocorrelation."""

    depth: int
    score: float  # Normalized peak-to-average ratio
    confidence: str  # "HIGH", "MEDIUM", "LOW"


class BlockInterleaver:
    """Matrix block interleaver and de-interleaver.

    Standard block interleaver arranges data in an (nrows x ncols) matrix:
    - Write row by row, read column by column (Interleaving)
    - Write column by column, read row by row (De-interleaving)
    """

    def __init__(self, nrows: int, ncols: int, mode: str = "row_write_col_read") -> None:
        if nrows <= 0 or ncols <= 0:
            raise ValueError(f"Matrix dimensions must be positive, got {nrows}x{ncols}")
        self.nrows = nrows
        self.ncols = ncols
        self.block_size = nrows * ncols
        self.mode = mode

    def interleave(self, data: np.ndarray) -> np.ndarray:
        """Interleave 1D array by writing rows and reading columns."""
        arr = np.asarray(data)
        pad_len = (self.block_size - len(arr) % self.block_size) % self.block_size
        if pad_len > 0:
            padded = np.pad(arr, (0, pad_len), constant_values=0)
        else:
            padded = arr

        num_blocks = len(padded) // self.block_size
        output = np.empty_like(padded)

        for b in range(num_blocks):
            blk = padded[b * self.block_size : (b + 1) * self.block_size]
            matrix = blk.reshape((self.nrows, self.ncols))
            if self.mode == "row_write_col_read":
                interleaved_blk = matrix.T.flatten()
            else:
                interleaved_blk = matrix.flatten()
            output[b * self.block_size : (b + 1) * self.block_size] = interleaved_blk

        return output

    def deinterleave(self, data: np.ndarray, original_length: int | None = None) -> np.ndarray:
        """De-interleave 1D array by inverting the matrix permutation."""
        arr = np.asarray(data)
        pad_len = (self.block_size - len(arr) % self.block_size) % self.block_size
        if pad_len > 0:
            padded = np.pad(arr, (0, pad_len), constant_values=0)
        else:
            padded = arr

        num_blocks = len(padded) // self.block_size
        output = np.empty_like(padded)

        for b in range(num_blocks):
            blk = padded[b * self.block_size : (b + 1) * self.block_size]
            if self.mode == "row_write_col_read":
                matrix = blk.reshape((self.ncols, self.nrows))
                deinterleaved_blk = matrix.T.flatten()
            else:
                matrix = blk.reshape((self.nrows, self.ncols))
                deinterleaved_blk = matrix.flatten()
            output[b * self.block_size : (b + 1) * self.block_size] = deinterleaved_blk

        if original_length is not None and original_length < len(output):
            return output[:original_length]
        return output


class ConvolutionalInterleaver:
    """Convolutional (Forney) interleaver and de-interleaver.

    Consists of B parallel shift register branches.
    Branch i has a delay of i * M symbols in the interleaver,
    and (B - 1 - i) * M symbols in the de-interleaver.
    """

    def __init__(self, num_branches: int = 12, delay_step: int = 17) -> None:
        self.num_branches = num_branches
        self.delay_step = delay_step

    def interleave(self, data: np.ndarray) -> np.ndarray:
        """Interleave data through convolutional delay lines."""
        arr = np.asarray(data)
        out = np.empty_like(arr)
        # Shift register states for each branch
        registers = [np.zeros(i * self.delay_step, dtype=arr.dtype) for i in range(self.num_branches)]

        for idx, val in enumerate(arr):
            branch_idx = idx % self.num_branches
            delay_len = branch_idx * self.delay_step
            if delay_len == 0:
                out[idx] = val
            else:
                reg = registers[branch_idx]
                out[idx] = reg[-1]
                reg[1:] = reg[:-1]
                reg[0] = val

        return out

    def deinterleave(self, data: np.ndarray) -> np.ndarray:
        """De-interleave data through complementary convolutional delay lines."""
        arr = np.asarray(data)
        out = np.empty_like(arr)
        # Complementary delay registers
        registers = [
            np.zeros((self.num_branches - 1 - i) * self.delay_step, dtype=arr.dtype)
            for i in range(self.num_branches)
        ]

        for idx, val in enumerate(arr):
            branch_idx = idx % self.num_branches
            delay_len = (self.num_branches - 1 - branch_idx) * self.delay_step
            if delay_len == 0:
                out[idx] = val
            else:
                reg = registers[branch_idx]
                out[idx] = reg[-1]
                reg[1:] = reg[:-1]
                reg[0] = val

        return out


def detect_interleaving_depth(
    bits: np.ndarray,
    min_depth: int = 4,
    max_depth: int = 128,
) -> list[InterleaverCandidate]:
    """Detect candidate interleaving depths via autocorrelation of bit transitions.

    Interleaved signals often have periodic transition structures or periodic error bursts.
    """
    arr = np.asarray(bits, dtype=np.float32)
    if len(arr) < max_depth * 4:
        return []

    # Map {0, 1} bits to {-1, +1}
    bipolar = 2.0 * arr - 1.0

    # Transitions
    diffs = np.diff(bipolar)
    n = len(diffs)
    if n < max_depth * 2:
        return []

    # Autocorrelation for lags
    autocorr = np.zeros(max_depth + 1, dtype=np.float32)
    for lag in range(min_depth, max_depth + 1):
        c = np.sum(diffs[: n - lag] * diffs[lag:]) / (n - lag)
        autocorr[lag] = abs(c)

    # Baseline noise level
    mean_val = float(np.mean(autocorr[min_depth:]))
    std_val = float(np.std(autocorr[min_depth:]))
    threshold = mean_val + 2.0 * std_val

    candidates: list[InterleaverCandidate] = []
    # Check peaks that exceed threshold
    for lag in range(min_depth, max_depth + 1):
        val = autocorr[lag]
        is_peak = True
        if lag > min_depth and val <= autocorr[lag - 1]:
            is_peak = False
        if lag < max_depth and val <= autocorr[lag + 1]:
            is_peak = False

        if is_peak and val > threshold:
            snr_peak = (val - mean_val) / (std_val + 1e-6)
            conf = "HIGH" if snr_peak > 4.0 else ("MEDIUM" if snr_peak > 2.5 else "LOW")
            candidates.append(InterleaverCandidate(depth=lag, score=float(snr_peak), confidence=conf))

    # Sort descending by score
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
