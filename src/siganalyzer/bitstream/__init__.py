"""Bitstream extraction, analysis, protocol framing, and formatting subsystem."""

from siganalyzer.bitstream.analyzer import BitstreamAnalyzer
from siganalyzer.bitstream.correlation import SyncWordCorrelator, SyncWordMatch
from siganalyzer.bitstream.formatter import BitstreamFormatter
from siganalyzer.bitstream.framing import BitstreamFrame, ProtocolFramer

__all__ = [
    "BitstreamAnalyzer",
    "SyncWordCorrelator",
    "SyncWordMatch",
    "BitstreamFrame",
    "ProtocolFramer",
    "BitstreamFormatter",
]
