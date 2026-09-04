"""Signal context aggregating raw data and intermediate DSP states for parameter extractors."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np

from siganalyzer.bitstream.framing import BitstreamFrame
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
from siganalyzer.dsp.spectrogram import SpectrogramData
from siganalyzer.dsp.spectrum import SpectrumResult
from siganalyzer.estimation.estimator import ParameterEstimateResult
from siganalyzer.io.readers.base import BaseSignalReader
from siganalyzer.io.validator import ValidationResult


@dataclass
class SignalContext:
    """Comprehensive analysis context provided to parameter extraction plugins."""

    file_path: Path
    reader: BaseSignalReader
    metadata: SignalMetadata
    validation: ValidationResult
    raw_samples: np.ndarray
    conditioned_samples: np.ndarray
    sample_rate: float
    duration_s: float
    center_frequency_rf: float | None = None

    # Analytical signal components (cached upon access)
    _envelope: np.ndarray | None = None
    _phase: np.ndarray | None = None
    _unwrapped_phase: np.ndarray | None = None
    _inst_freq: np.ndarray | None = None

    # Subsystem artifacts
    spectrum: SpectrumResult | None = None
    spectrogram: SpectrogramData | None = None
    regions: list[SignalRegion] = field(default_factory=list)
    primary_parameters: ParameterEstimateResult | None = None
    modulation: ModulationResult | None = None
    demodulation: DemodulationResult | None = None
    constellation_metrics: ConstellationMetrics | None = None
    fec: FecResult | None = None
    bitstream: BitstreamAnalysis | None = None
    frames: list[BitstreamFrame] = field(default_factory=list)
    anomalies: list[AnomalyFinding] = field(default_factory=list)

    # Scratchpad for sharing intermediate extractor computations
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def envelope(self) -> np.ndarray:
        """Instantaneous amplitude envelope |s[n]|."""
        if self._envelope is None:
            self._envelope = np.abs(self.conditioned_samples)
        return self._envelope

    @property
    def phase(self) -> np.ndarray:
        """Instantaneous wrapped phase angle in radians [-pi, +pi]."""
        if self._phase is None:
            self._phase = np.angle(self.conditioned_samples)
        return self._phase

    @property
    def unwrapped_phase(self) -> np.ndarray:
        """Continuous unwrapped phase in radians."""
        if self._unwrapped_phase is None:
            self._unwrapped_phase = np.unwrap(self.phase)
        return self._unwrapped_phase

    @property
    def instantaneous_frequency(self) -> np.ndarray:
        """Instantaneous frequency in Hertz: f[n] = (1 / 2pi) * d(theta)/dt."""
        if self._inst_freq is None:
            u_phase = self.unwrapped_phase
            if len(u_phase) > 1:
                diff_phase = np.diff(u_phase)
                self._inst_freq = diff_phase * (self.sample_rate / (2.0 * np.pi))
            else:
                self._inst_freq = np.zeros(0, dtype=np.float32)
        return self._inst_freq
