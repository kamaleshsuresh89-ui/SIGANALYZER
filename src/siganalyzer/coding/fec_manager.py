"""Forward Error Correction (FEC) manager and coordinator."""

import numpy as np

from siganalyzer.coding.crc import CrcAlgorithm, CrcEngine
from siganalyzer.coding.interleaver import BlockInterleaver, detect_interleaving_depth
from siganalyzer.coding.reed_solomon import ReedSolomonCodec, ReedSolomonResult
from siganalyzer.coding.viterbi import ConvolutionalCodec, ViterbiConfig
from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import FecResult


class FecManager:
    """Coordinates detection and execution of FEC decoding algorithms."""

    @classmethod
    def probe_and_decode(cls, bits: np.ndarray) -> FecResult:
        """Probe bitstream for probable FEC schemes and attempt decoding.

        Evaluates:
        1. NASA/CCSDS Rate 1/2, K=7 Viterbi Convolutional Code
        2. Returns FecResult
        """
        arr = np.asarray(bits, dtype=np.uint8)
        if len(arr) < 64:
            return FecResult(
                scheme="None detected",
                detected=False,
                confidence=ConfidenceLevel.UNKNOWN,
                notes="Bitstream too short for FEC detection.",
            )

        # 1. Probe NASA K=7 Rate 1/2 Viterbi
        viterbi = ConvolutionalCodec(ViterbiConfig(constraint_length=7, g1=0o171, g2=0o133))
        # Test on first 1024 bit pairs (2048 bits)
        probe_len = min(2048, len(arr) - (len(arr) % 2))
        decoded, metric = viterbi.decode_hard(arr[:probe_len], truncate_tail=True)

        # Theoretical metric for random non-convolutional data is ~25% to 50% of bits
        metric_ratio = metric / (probe_len or 1)
        if metric_ratio < 0.08 and len(decoded) > 0:
            # High probability of convolutional code
            conf = ConfidenceLevel.MEASURED if metric_ratio < 0.03 else ConfidenceLevel.ESTIMATED
            # Full decode
            full_decoded, full_metric = viterbi.decode_hard(arr, truncate_tail=True)
            return FecResult(
                scheme="Viterbi (K=7, r=1/2, G1=171, G2=133)",
                detected=True,
                confidence=conf,
                corrected_errors=full_metric,
                decoded_bits=full_decoded,
                notes=f"Convolutional code detected with normalized path metric error rate {metric_ratio:.2%}.",
            )

        return FecResult(
            scheme="Uncoded / None",
            detected=False,
            confidence=ConfidenceLevel.MEASURED,
            notes="No standard FEC scheme detected with high confidence.",
        )

    @classmethod
    def decode_viterbi(
        cls,
        bits: np.ndarray,
        k: int = 7,
        g1: int = 0o171,
        g2: int = 0o133,
    ) -> tuple[np.ndarray, int]:
        """Perform Viterbi decoding with specified polynomial parameters."""
        codec = ConvolutionalCodec(ViterbiConfig(constraint_length=k, g1=g1, g2=g2))
        return codec.decode_hard(bits)

    @classmethod
    def decode_reed_solomon(
        cls,
        data: bytes | np.ndarray,
        n: int = 255,
        k: int = 223,
    ) -> ReedSolomonResult:
        """Perform Reed-Solomon decoding with specified codeword parameters."""
        codec = ReedSolomonCodec(n=n, k=k)
        return codec.decode(data)

    @classmethod
    def deinterleave_matrix(
        cls,
        data: np.ndarray,
        nrows: int,
        ncols: int,
    ) -> np.ndarray:
        """De-interleave a matrix-interleaved bit array."""
        interleaver = BlockInterleaver(nrows=nrows, ncols=ncols)
        return interleaver.deinterleave(data)
