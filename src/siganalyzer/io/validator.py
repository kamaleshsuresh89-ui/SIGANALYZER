"""File integrity and sample validation engine."""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from siganalyzer.common.errors import CorruptedFileError, InvalidSampleError


@dataclass
class ValidationResult:
    """Outcome of file and sample validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_nans: bool = False
    has_infs: bool = False
    has_clipping: bool = False
    has_excessive_silence: bool = False


class FileValidator:
    """Validates file structure and sample integrity."""

    CLIPPING_THRESHOLD = 0.999
    SILENCE_THRESHOLD_DB = -60.0

    @classmethod
    def validate_file_structure(cls, file_path: str | Path) -> ValidationResult:
        """Check file existence, permissions, and minimum size."""
        path = Path(file_path).resolve()
        result = ValidationResult(is_valid=True)

        if not path.exists():
            result.is_valid = False
            result.errors.append(f"File does not exist: {path}")
            return result

        if not path.is_file():
            result.is_valid = False
            result.errors.append(f"Target path is not a file: {path}")
            return result

        file_size = path.stat().st_size
        if file_size == 0:
            result.is_valid = False
            result.errors.append("File is empty (0 bytes).")
            return result

        if file_size < 16:
            result.is_valid = False
            result.errors.append("File is too small to contain valid signal data (<16 bytes).")
            return result

        # Test readability
        try:
            with open(path, "rb") as f:
                f.read(16)
        except PermissionError:
            result.is_valid = False
            result.errors.append("Permission denied when opening file.")
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Failed to read file: {e}")

        return result

    @classmethod
    def validate_samples(cls, samples: np.ndarray) -> ValidationResult:
        """Inspect a sample buffer for numerical issues, clipping, or dropouts."""
        result = ValidationResult(is_valid=True)

        if len(samples) == 0:
            result.warnings.append("Sample buffer is empty.")
            return result

        # Check for NaNs
        if np.any(np.isnan(samples)):
            result.is_valid = False
            result.has_nans = True
            result.errors.append("Samples contain NaN (Not-a-Number) values.")

        # Check for Infs
        if np.any(np.isinf(samples)):
            result.is_valid = False
            result.has_infs = True
            result.errors.append("Samples contain infinite (Inf) values.")

        if not result.is_valid:
            return result

        # Check clipping
        magnitudes = np.abs(samples)
        clipped_count = np.count_nonzero(magnitudes >= cls.CLIPPING_THRESHOLD)
        if clipped_count > 0:
            result.has_clipping = True
            pct = (clipped_count / len(samples)) * 100
            result.warnings.append(f"Signal contains {clipped_count} clipped samples ({pct:.2f}% of buffer).")

        # Check excessive silence / zero dropouts
        rms = np.sqrt(np.mean(magnitudes ** 2))
        rms_db = 20 * np.log10(max(rms, 1e-12))
        if rms_db < cls.SILENCE_THRESHOLD_DB:
            result.has_excessive_silence = True
            result.warnings.append(f"Low signal power ({rms_db:.1f} dBFS); may contain pure noise or silence.")

        return result
