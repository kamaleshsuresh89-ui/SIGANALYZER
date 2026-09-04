"""Common utilities, types, confidence levels, and exceptions for SIGANALYZER."""

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.errors import (
    ConfigurationError,
    CorruptedFileError,
    DecodingError,
    DemodulationError,
    FileFormatError,
    InvalidSampleError,
    SigAnalyzerError,
    SignalProcessingError,
    SynchronizationError,
)
from siganalyzer.common.logger import logger, setup_logger
from siganalyzer.common.types import (
    AnomalyFinding,
    BitstreamAnalysis,
    DataRepresentation,
    DemodulationResult,
    FecResult,
    Measurement,
    ModulationResult,
    SignalFileType,
    SignalMetadata,
    SignalRegion,
)

__all__ = [
    "ConfidenceLevel",
    "Measurement",
    "SignalFileType",
    "DataRepresentation",
    "SignalMetadata",
    "SignalRegion",
    "ModulationResult",
    "DemodulationResult",
    "FecResult",
    "BitstreamAnalysis",
    "AnomalyFinding",
    "SigAnalyzerError",
    "FileFormatError",
    "CorruptedFileError",
    "InvalidSampleError",
    "SignalProcessingError",
    "SynchronizationError",
    "DemodulationError",
    "DecodingError",
    "ConfigurationError",
    "logger",
    "setup_logger",
]
