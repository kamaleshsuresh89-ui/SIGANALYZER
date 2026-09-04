"""Core domain data types, models, and measurement containers."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel

T = TypeVar("T")


class SignalFileType(str, Enum):
    """Identified signal recording file types."""

    WAV_PCM16 = "WAV (PCM 16-bit)"
    WAV_FLOAT32 = "WAV (Float 32-bit)"
    WAV_PCM24 = "WAV (PCM 24-bit)"
    WAV_PCM32 = "WAV (PCM 32-bit)"
    RF64_WAV = "RF64 Large-File WAV"
    RAW_IQ_FLOAT32 = "Raw IQ (Float32 Little-Endian)"
    RAW_IQ_INT16 = "Raw IQ (Int16 Little-Endian)"
    RAW_IQ_UINT8_RTL = "Raw IQ (RTL-SDR Unsigned 8-bit)"
    RAW_IQ_INT8_HACKRF = "Raw IQ (HackRF Signed 8-bit)"
    SIGMF = "SigMF Dataset"
    UNKNOWN = "Unknown / Unsupported"


class DataRepresentation(str, Enum):
    """How complex I/Q samples are structured in memory or on disk."""

    COMPLEX_INTERLEAVED = "Interleaved IQ (I0, Q0, I1, Q1, ...)"
    COMPLEX_PLANAR = "Planar IQ (I-channel, then Q-channel)"
    REAL_ONLY = "Real-valued Signal"


@dataclass
class Measurement(Generic[T]):
    """A quantified metric tagged with strict confidence and provenance metadata.

    NEVER fabricate values. If a value is estimated, mark as ESTIMATED with score.
    """

    name: str
    value: T | None
    unit: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    confidence_score: float = 0.0  # Normalized 0.0 to 1.0
    source: str = ""  # e.g. "Welch PSD", "Cyclostationary Non-linearity", "WAV fmt chunk"
    notes: str = ""

    @property
    def formatted_value(self) -> str:
        """Format the value with unit and appropriate precision."""
        if self.value is None or self.confidence == ConfidenceLevel.UNKNOWN:
            return "Unknown"
        if isinstance(self.value, float):
            if abs(self.value) >= 1e6:
                return f"{self.value / 1e6:.4f} M{self.unit}".strip()
            elif abs(self.value) >= 1e3:
                return f"{self.value / 1e3:.3f} k{self.unit}".strip()
            elif 0 < abs(self.value) < 1e-3:
                return f"{self.value * 1e6:.2f} µ{self.unit}".strip()
            else:
                return f"{self.value:.4g} {self.unit}".strip()
        return f"{self.value} {self.unit}".strip()


@dataclass
class SignalMetadata:
    """Metadata extracted or measured from the input signal file."""

    file_path: str
    file_name: str
    file_type: SignalFileType
    file_size_bytes: int
    total_samples: int
    channels: int
    bit_depth: int
    representation: DataRepresentation
    sample_rate: Measurement[float] = field(
        default_factory=lambda: Measurement("Sample Rate", None, "Hz", ConfidenceLevel.UNKNOWN)
    )
    center_frequency: Measurement[float] = field(
        default_factory=lambda: Measurement("Center Frequency", None, "Hz", ConfidenceLevel.UNKNOWN)
    )
    duration_seconds: Measurement[float] = field(
        default_factory=lambda: Measurement("Duration", None, "s", ConfidenceLevel.UNKNOWN)
    )
    peak_amplitude: Measurement[float] = field(
        default_factory=lambda: Measurement("Peak Amplitude", None, "", ConfidenceLevel.UNKNOWN)
    )
    rms_power: Measurement[float] = field(
        default_factory=lambda: Measurement("RMS Power", None, "dBFS", ConfidenceLevel.UNKNOWN)
    )
    dynamic_range_db: Measurement[float] = field(
        default_factory=lambda: Measurement("Dynamic Range", None, "dB", ConfidenceLevel.UNKNOWN)
    )
    has_clipping: Measurement[bool] = field(
        default_factory=lambda: Measurement("Clipping Detected", False, "", ConfidenceLevel.UNKNOWN)
    )
    raw_header_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRegion:
    """An automatically detected signal burst or region of interest."""

    region_id: int
    start_sample: int
    end_sample: int
    sample_rate: float
    start_time: float
    end_time: float
    duration_s: float
    center_freq_offset_hz: Measurement[float]
    occupied_bandwidth_hz: Measurement[float]
    snr_db: Measurement[float]
    mean_power_db: float
    is_signal: bool = True  # False indicates noise / silence interval


@dataclass
class ModulationResult:
    """Results from automatic modulation classification."""

    scheme: str  # e.g. "QPSK", "16-QAM", "2FSK", "BPSK"
    confidence_score: float  # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)
    secondary_candidates: list[tuple[str, float]] = field(default_factory=list)
    constellation_points: np.ndarray | None = None  # Complex symbols for plotting
    evm_percent: Measurement[float] = field(
        default_factory=lambda: Measurement("EVM", None, "%", ConfidenceLevel.UNKNOWN)
    )


@dataclass
class DemodulationResult:
    """Results from symbol recovery and demodulation."""

    scheme: str
    symbols: np.ndarray  # Complex or real symbols
    bits: np.ndarray  # 1D uint8 array of 0s and 1s
    symbol_rate_est: Measurement[float]
    carrier_frequency_error_hz: float = 0.0
    carrier_locked: bool = False
    timing_locked: bool = False
    eye_diagram_snr_db: float = 0.0


@dataclass
class FecResult:
    """Results from Forward Error Correction detection and decoding."""

    scheme: str  # "Viterbi (k=7, r=1/2)", "Reed-Solomon (255, 223)", "None detected"
    detected: bool
    confidence: ConfidenceLevel
    corrected_errors: int = 0
    uncorrectable_errors: int = 0
    decoded_bits: np.ndarray | None = None
    notes: str = ""


@dataclass
class BitstreamAnalysis:
    """Bit-level statistical and protocol analysis."""

    raw_bits: np.ndarray
    total_bits: int
    bit_rate_est: Measurement[float]
    transition_density: float  # Fraction of bit flips (ideal ~0.5 for random data)
    entropy_per_bit: float  # Shannon entropy (ideal 1.0 for random data)
    detected_sync_words: list[dict[str, Any]] = field(default_factory=list)
    preamble_detected: bool = False
    inferred_frame_length_bits: int | None = None
    hex_preview: str = ""
    ascii_preview: str = ""


@dataclass
class AnomalyFinding:
    """Signal anomaly or impairment warning."""

    anomaly_type: str  # "Clipping", "Dropout", "Frequency Jump", "IQ Imbalance", "Sudden Power Drop"
    severity: str  # "INFO", "WARNING", "CRITICAL"
    sample_index: int
    time_seconds: float
    description: str
    recommended_action: str = ""
