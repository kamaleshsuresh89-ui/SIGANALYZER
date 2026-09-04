"""Unified signal loader and metadata extraction coordinator."""

from pathlib import Path
from typing import Tuple

from siganalyzer.common.errors import FileFormatError
from siganalyzer.common.types import SignalFileType, SignalMetadata
from siganalyzer.io.detector import DetectionResult, FileTypeDetector
from siganalyzer.io.readers.base import BaseSignalReader
from siganalyzer.io.readers.raw_iq_reader import RawIQSignalReader
from siganalyzer.io.readers.wav_reader import WavSignalReader
from siganalyzer.io.validator import FileValidator, ValidationResult


class SignalLoader:
    """Unified entrypoint to automatically identify, validate, and load signal recordings."""

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        sample_rate_override: float | None = None,
        center_freq_override: float | None = None,
    ) -> Tuple[BaseSignalReader, DetectionResult, ValidationResult]:
        """Automatically detect format, validate file structure, and return appropriate reader.

        Args:
            file_path: Path to the recorded signal file.
            sample_rate_override: Optional user-supplied sample rate in Hz.
            center_freq_override: Optional user-supplied center frequency in Hz.

        Returns:
            Tuple of (reader_instance, detection_result, validation_result).
        """
        path = Path(file_path).resolve()

        # Step 1: Validate file structure
        validation = FileValidator.validate_file_structure(path)
        if not validation.is_valid:
            raise FileFormatError(
                f"File validation failed: {', '.join(validation.errors)}",
                str(path),
            )

        # Step 2: Automatically detect file type
        detection = FileTypeDetector.detect(path)

        # Step 3: Instantiate appropriate reader
        match detection.file_type:
            case (
                SignalFileType.WAV_PCM16
                | SignalFileType.WAV_FLOAT32
                | SignalFileType.WAV_PCM24
                | SignalFileType.WAV_PCM32
                | SignalFileType.RF64_WAV
            ):
                reader = WavSignalReader(path)
                if center_freq_override is not None:
                    reader.metadata.center_frequency.value = center_freq_override
                    reader.metadata.center_frequency.source = "User specified override"
                if sample_rate_override is not None:
                    reader.metadata.sample_rate.value = sample_rate_override
                    reader.metadata.sample_rate.source = "User specified override"

            case (
                SignalFileType.RAW_IQ_FLOAT32
                | SignalFileType.RAW_IQ_INT16
                | SignalFileType.RAW_IQ_UINT8_RTL
                | SignalFileType.RAW_IQ_INT8_HACKRF
            ):
                reader = RawIQSignalReader(
                    path,
                    file_type=detection.file_type,
                    sample_rate=sample_rate_override,
                    center_frequency=center_freq_override,
                )

            case _:
                # If unknown, try RawIQ Float32 as a fallback with low confidence
                reader = RawIQSignalReader(
                    path,
                    file_type=SignalFileType.RAW_IQ_FLOAT32,
                    sample_rate=sample_rate_override,
                    center_frequency=center_freq_override,
                )

        return reader, detection, validation
