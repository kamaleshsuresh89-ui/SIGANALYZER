"""Confidence level and source classification for signal analysis parameters."""

from enum import Enum


class ConfidenceLevel(str, Enum):
    """Classification of how a value or metadata field was determined.

    Strict rule: Never fabricate missing metadata. Always classify clearly:
    - DIRECT: Read directly from headers or companion metadata.
    - MEASURED: Deterministically calculated from the raw samples (e.g. sample count, peak amplitude).
    - ESTIMATED: Derived via mathematical DSP algorithms (e.g. SNR, bandwidth, symbol rate).
    - INFERRED: Deduced through pattern matching or modulation classification (e.g. modulation scheme).
    - UNKNOWN: Cannot be determined reliably from available data.
    """

    DIRECT = "Direct"
    MEASURED = "Measured"
    ESTIMATED = "Estimated"
    INFERRED = "Inferred"
    UNKNOWN = "Unknown"

    @property
    def badge_color(self) -> str:
        """Hex color code for UI badges."""
        match self:
            case ConfidenceLevel.DIRECT:
                return "#22c55e"  # Green
            case ConfidenceLevel.MEASURED:
                return "#3b82f6"  # Blue
            case ConfidenceLevel.ESTIMATED:
                return "#f59e0b"  # Amber
            case ConfidenceLevel.INFERRED:
                return "#8b5cf6"  # Purple
            case ConfidenceLevel.UNKNOWN:
                return "#6b7280"  # Gray

    @property
    def description(self) -> str:
        """Human-readable explanation of the confidence level."""
        match self:
            case ConfidenceLevel.DIRECT:
                return "Obtained directly from file header or metadata"
            case ConfidenceLevel.MEASURED:
                return "Calculated deterministically from the signal samples"
            case ConfidenceLevel.ESTIMATED:
                return "Estimated via digital signal processing algorithms"
            case ConfidenceLevel.INFERRED:
                return "Inferred via statistical classification or heuristic models"
            case ConfidenceLevel.UNKNOWN:
                return "Not available or could not be determined reliably"
