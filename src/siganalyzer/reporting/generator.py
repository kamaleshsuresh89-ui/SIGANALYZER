"""Analysis report data aggregator."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from siganalyzer.common.types import (
    AnomalyFinding,
    BitstreamAnalysis,
    DemodulationResult,
    FecResult,
    ModulationResult,
    SignalMetadata,
    SignalRegion,
)
from siganalyzer.demodulation.constellation import ConstellationMetrics
from siganalyzer.dsp.spectrum import SpectrumResult
from siganalyzer.estimation.estimator import ParameterEstimateResult
from siganalyzer.io.validator import ValidationResult


@dataclass
class CompleteAnalysisReport:
    """Full analysis report model ready for offline export and GUI rendering."""

    app_version: str
    generated_at: str
    processing_duration_s: float
    metadata: SignalMetadata
    validation: ValidationResult
    spectrum: SpectrumResult
    regions: list[SignalRegion]
    primary_parameters: ParameterEstimateResult
    primary_modulation: ModulationResult
    anomalies: list[AnomalyFinding]
    demodulation: DemodulationResult | None = None
    constellation_metrics: ConstellationMetrics | None = None
    fec: FecResult | None = None
    bitstream: BitstreamAnalysis | None = None
    signal_profile: Any = None
    raw_summary: dict[str, Any] = field(default_factory=dict)
