"""I/O subsystem for SIGANALYZER: file detection, readers, and validation."""

from siganalyzer.io.detector import DetectionResult, FileTypeDetector
from siganalyzer.io.metadata_extractor import SignalLoader
from siganalyzer.io.readers.base import BaseSignalReader
from siganalyzer.io.readers.raw_iq_reader import RawIQSignalReader
from siganalyzer.io.readers.wav_reader import WavSignalReader
from siganalyzer.io.validator import FileValidator, ValidationResult

__all__ = [
    "FileTypeDetector",
    "DetectionResult",
    "FileValidator",
    "ValidationResult",
    "SignalLoader",
    "BaseSignalReader",
    "WavSignalReader",
    "RawIQSignalReader",
]
