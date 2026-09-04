"""Viterbi convolutional decoder and encoder for Forward Error Correction (FEC)."""

from dataclasses import dataclass
import numpy as np


def _parity(val: int) -> int:
    """Compute parity (XOR of all bits) of an integer."""
    p = 0
    while val:
        p ^= 1
        val &= val - 1
    return p


@dataclass
class ViterbiConfig:
    """Configuration for rate 1/2 convolutional code."""

    constraint_length: int = 7  # K
    g1: int = 0o171  # Octal 171 (NASA/CCSDS standard)
    g2: int = 0o133  # Octal 133 (NASA/CCSDS standard)


class ConvolutionalCodec:
    """Rate 1/2 Convolutional Encoder and Viterbi Trellis Decoder."""

    def __init__(self, config: ViterbiConfig | None = None) -> None:
        self.config = config or ViterbiConfig()
        self.k = self.config.constraint_length
        self.num_states = 1 << (self.k - 1)
        self.g1 = self.config.g1
        self.g2 = self.config.g2

        # Precompute state transitions:
        # predecessors[next_state] = [(prev_state, input_bit, out0, out1), ...]
        self.predecessors: list[list[tuple[int, int, int, int]]] = [[] for _ in range(self.num_states)]
        # transitions[state][input_bit] = (next_state, out0, out1)
        self.transitions: list[list[tuple[int, int, int]]] = [[(0, 0, 0), (0, 0, 0)] for _ in range(self.num_states)]

        for state in range(self.num_states):
            for bit in (0, 1):
                reg = (bit << (self.k - 1)) | state
                next_state = reg >> 1
                out0 = _parity(reg & self.g1)
                out1 = _parity(reg & self.g2)
                self.transitions[state][bit] = (next_state, out0, out1)
                self.predecessors[next_state].append((state, bit, out0, out1))

    def encode(self, bits: np.ndarray, flush: bool = True) -> np.ndarray:
        """Encode input bit stream (0/1) using rate 1/2 convolutional code."""
        arr = np.asarray(bits, dtype=np.uint8)
        if flush:
            arr = np.concatenate([arr, np.zeros(self.k - 1, dtype=np.uint8)])

        encoded = np.empty(len(arr) * 2, dtype=np.uint8)
        state = 0
        out_idx = 0

        for b in arr:
            next_state, out0, out1 = self.transitions[state][int(b)]
            encoded[out_idx] = out0
            encoded[out_idx + 1] = out1
            out_idx += 2
            state = next_state

        return encoded

    def decode_hard(self, encoded_bits: np.ndarray, truncate_tail: bool = True) -> tuple[np.ndarray, int]:
        """Decode rate 1/2 convolutional bitstream using hard-decision Viterbi algorithm.

        Args:
            encoded_bits: Array of received bits (0 and 1).
            truncate_tail: If True, strips (K - 1) flush bits from output.

        Returns:
            Tuple of (decoded_bits, total_path_metric).
        """
        arr = np.asarray(encoded_bits, dtype=np.uint8)
        n_pairs = len(arr) // 2
        if n_pairs == 0:
            return np.empty(0, dtype=np.uint8), 0

        # Metrics for states
        INF = 10_000_000
        path_metrics = np.full(self.num_states, INF, dtype=np.int32)
        path_metrics[0] = 0

        # Traceback matrix: [time_step, state] -> (prev_state, bit)
        traceback_state = np.zeros((n_pairs, self.num_states), dtype=np.int32)
        traceback_bit = np.zeros((n_pairs, self.num_states), dtype=np.uint8)

        new_metrics = np.empty(self.num_states, dtype=np.int32)

        for t in range(n_pairs):
            r0 = arr[2 * t]
            r1 = arr[2 * t + 1]

            for s_next in range(self.num_states):
                best_m = INF
                best_s_prev = 0
                best_b = 0

                preds = self.predecessors[s_next]
                for s_prev, bit, o0, o1 in preds:
                    m = path_metrics[s_prev]
                    if m < INF:
                        # Hamming branch distance
                        dist = (r0 ^ o0) + (r1 ^ o1)
                        total_m = m + dist
                        if total_m < best_m:
                            best_m = total_m
                            best_s_prev = s_prev
                            best_b = bit

                new_metrics[s_next] = best_m
                traceback_state[t, s_next] = best_s_prev
                traceback_bit[t, s_next] = best_b

            path_metrics[:] = new_metrics

        # Select state with lowest metric at end
        curr_state = int(np.argmin(path_metrics))
        total_metric = int(path_metrics[curr_state])

        # Traceback
        decoded = np.empty(n_pairs, dtype=np.uint8)
        for t in range(n_pairs - 1, -1, -1):
            decoded[t] = traceback_bit[t, curr_state]
            curr_state = traceback_state[t, curr_state]

        if truncate_tail and len(decoded) > (self.k - 1):
            decoded = decoded[: -(self.k - 1)]

        return decoded, total_metric

    def decode_soft(self, soft_symbols: np.ndarray, truncate_tail: bool = True) -> tuple[np.ndarray, float]:
        """Decode rate 1/2 convolutional code using soft-decision Euclidean distances.

        Args:
            soft_symbols: Real-valued soft samples where bit 0 is mapped to ~ +1.0 and bit 1 to ~ -1.0.
        """
        arr = np.asarray(soft_symbols, dtype=np.float32)
        n_pairs = len(arr) // 2
        if n_pairs == 0:
            return np.empty(0, dtype=np.uint8), 0.0

        INF = 1e9
        path_metrics = np.full(self.num_states, INF, dtype=np.float32)
        path_metrics[0] = 0.0

        traceback_state = np.zeros((n_pairs, self.num_states), dtype=np.int32)
        traceback_bit = np.zeros((n_pairs, self.num_states), dtype=np.uint8)
        new_metrics = np.empty(self.num_states, dtype=np.float32)

        for t in range(n_pairs):
            r0 = arr[2 * t]
            r1 = arr[2 * t + 1]

            for s_next in range(self.num_states):
                best_m = INF
                best_s_prev = 0
                best_b = 0

                preds = self.predecessors[s_next]
                for s_prev, bit, o0, o1 in preds:
                    m = path_metrics[s_prev]
                    if m < INF:
                        # Soft Euclidean distance: bit 0 -> +1.0, bit 1 -> -1.0
                        exp0 = 1.0 - 2.0 * o0
                        exp1 = 1.0 - 2.0 * o1
                        dist = (r0 - exp0) ** 2 + (r1 - exp1) ** 2
                        total_m = m + dist
                        if total_m < best_m:
                            best_m = total_m
                            best_s_prev = s_prev
                            best_b = bit

                new_metrics[s_next] = best_m
                traceback_state[t, s_next] = best_s_prev
                traceback_bit[t, s_next] = best_b

            path_metrics[:] = new_metrics

        curr_state = int(np.argmin(path_metrics))
        total_metric = float(path_metrics[curr_state])

        decoded = np.empty(n_pairs, dtype=np.uint8)
        for t in range(n_pairs - 1, -1, -1):
            decoded[t] = traceback_bit[t, curr_state]
            curr_state = traceback_state[t, curr_state]

        if truncate_tail and len(decoded) > (self.k - 1):
            decoded = decoded[: -(self.k - 1)]

        return decoded, total_metric
