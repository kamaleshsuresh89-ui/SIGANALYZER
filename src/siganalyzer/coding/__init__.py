"""Coding and Forward Error Correction (FEC) subsystem."""

from siganalyzer.coding.crc import CrcAlgorithm, CrcEngine
from siganalyzer.coding.fec_manager import FecManager
from siganalyzer.coding.interleaver import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    InterleaverCandidate,
    detect_interleaving_depth,
)
from siganalyzer.coding.reed_solomon import ReedSolomonCodec, ReedSolomonResult
from siganalyzer.coding.viterbi import ConvolutionalCodec, ViterbiConfig

__all__ = [
    "CrcAlgorithm",
    "CrcEngine",
    "BlockInterleaver",
    "ConvolutionalInterleaver",
    "InterleaverCandidate",
    "detect_interleaving_depth",
    "ConvolutionalCodec",
    "ViterbiConfig",
    "ReedSolomonCodec",
    "ReedSolomonResult",
    "FecManager",
]
