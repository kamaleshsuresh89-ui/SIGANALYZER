"""Parameter definition and result data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from siganalyzer.common.confidence import ConfidenceLevel


class ParameterCategory(str, Enum):
    """The 16 formal signal parameter categories (SIH Specification)."""

    FILE_CAPTURE = "A. File & Capture Information"
    TIME_DOMAIN = "B. Time-Domain Parameters"
    FREQUENCY_DOMAIN = "C. Frequency-Domain Parameters"
    POWER_QUALITY = "D. Power & Signal Quality"
    MODULATION = "E. Modulation Analysis"
    CONSTELLATION = "F. Constellation & Modulation Quality"
    SYNCHRONIZATION = "G. Synchronization Parameters"
    BURST_SEGMENTATION = "H. Burst / Signal Segmentation"
    DIGITAL_COMM = "I. Digital Communication Parameters"
    FEC_CODING = "J. FEC / Coding Analysis"
    BITSTREAM = "K. Bitstream Analysis"
    FRAME_PAYLOAD = "L. Header / Payload / Frame Analysis"
    NOISE_INTERFERENCE = "M. Noise & Interference Analysis"
    STATISTICAL_HOC = "N. Statistical & Higher-Order Features"
    HARDWARE_ADC = "O. Hardware / ADC / Capture Quality"
    ANOMALY_DETECTION = "P. Signal Anomaly Detection"


@dataclass(frozen=True)
class ParameterDefinition:
    """Static schema definition for an extractable signal parameter."""

    id: str
    name: str
    category: ParameterCategory
    unit: str
    data_type: type
    description: str
    default_confidence: ConfidenceLevel
    valid_range: tuple[float, float] | None = None
    export_priority: int = 100


@dataclass
class ParameterResult:
    """A quantified parameter result with provenance, confidence, and mathematical justification."""

    definition: ParameterDefinition
    value: Any
    status: ConfidenceLevel
    confidence_score: float  # Normalized 0.0 to 1.0 (1.0 = absolute certainty)
    source: str  # e.g., "WAV fmt chunk", "Welch PSD Integral", "Hilbert Transform"
    method: str  # Mathematical equation or derivation method
    explanation: str  # Human-readable explanation of why this value was produced
    validity_notes: str = ""  # Technical caveats, constraints, or boundary conditions

    @property
    def formatted_value(self) -> str:
        """Format the parameter value with appropriate precision and engineering units."""
        if self.value is None or self.status == ConfidenceLevel.UNKNOWN:
            return "UNKNOWN"
        if isinstance(self.value, bool):
            return "True" if self.value else "False"
        if isinstance(self.value, float):
            u = f" {self.definition.unit}" if self.definition.unit else ""
            val = self.value
            if abs(val) >= 1e9:
                return f"{val / 1e9:.4f} G{self.definition.unit}".strip()
            elif abs(val) >= 1e6:
                return f"{val / 1e6:.4f} M{self.definition.unit}".strip()
            elif abs(val) >= 1e3:
                return f"{val / 1e3:.3f} k{self.definition.unit}".strip()
            elif 0 < abs(val) < 1e-3:
                return f"{val * 1e6:.2f} µ{self.definition.unit}".strip()
            else:
                return f"{val:.4g}{u}".strip()
        if isinstance(self.value, int):
            u = f" {self.definition.unit}" if self.definition.unit else ""
            return f"{self.value:,}{u}".strip()
        if isinstance(self.value, list):
            if len(self.value) == 0:
                return "[]"
            if len(self.value) <= 4:
                return ", ".join(str(v) for v in self.value)
            return f"{len(self.value)} items [{self.value[0]}, ...]"
        return f"{self.value} {self.definition.unit}".strip()

    def to_dict(self) -> dict[str, Any]:
        """Convert parameter result to dictionary for export and serialization."""
        return {
            "id": self.definition.id,
            "name": self.definition.name,
            "category": self.definition.category.value,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "unit": self.definition.unit,
            "status": self.status.value,
            "confidence_score": round(self.confidence_score, 4),
            "source": self.source,
            "method": self.method,
            "explanation": self.explanation,
            "validity_notes": self.validity_notes,
        }
