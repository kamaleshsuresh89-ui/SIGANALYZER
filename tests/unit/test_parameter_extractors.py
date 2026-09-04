"""Comprehensive unit tests for the 16 Parameter Extractors, Registry, and Engine."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import (
    BitstreamAnalysis,
    DataRepresentation,
    DemodulationResult,
    FecResult,
    ModulationResult,
    SignalFileType,
    SignalMetadata,
    SignalRegion,
)
from siganalyzer.demodulation.constellation import ConstellationMetrics
from siganalyzer.dsp.spectrum import SpectralAnalyzer
from siganalyzer.estimation.estimator import ParameterEstimator
from siganalyzer.io.readers.base import BaseSignalReader
from siganalyzer.io.validator import ValidationResult
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.engine import ExtractionEngine, SignalProfile
from siganalyzer.parameters.model import ParameterCategory, ParameterResult
from siganalyzer.parameters.registry import ParameterRegistry
from tests.fixtures.generate_fixtures import generate_qpsk, save_as_wav


class DummyReader(BaseSignalReader):
    """Minimal mock reader for testing SignalContext."""

    def __init__(self, samples: np.ndarray, sample_rate: float = 1_000_000.0, file_path: str = "mock.wav") -> None:
        self.file_path = Path(file_path)
        self._samples = samples
        self._metadata = SignalMetadata(
            file_path=str(self.file_path),
            file_name=self.file_path.name,
            file_type=SignalFileType.WAV_PCM16,
            file_size_bytes=len(samples) * 4,
            total_samples=len(samples),
            channels=2,
            bit_depth=16,
            representation=DataRepresentation.COMPLEX_INTERLEAVED,
        )
        self._metadata.sample_rate.value = sample_rate

    @property
    def metadata(self) -> SignalMetadata:
        return self._metadata

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    def read_samples(self, start_sample: int = 0, count: int | None = None) -> np.ndarray:
        if count is None:
            return self._samples[start_sample:]
        return self._samples[start_sample : start_sample + count]

    def close(self) -> None:
        pass


@pytest.fixture
def mock_context() -> SignalContext:
    """Creates a fully populated realistic SignalContext for extractor unit testing."""
    sr = 1_000_000.0
    samples, _ = generate_qpsk(num_symbols=1000, sps=8, sample_rate=sr, carrier_freq=50_000.0, snr_db=25.0)
    reader = DummyReader(samples, sample_rate=sr)

    # Compute minimal realistic dependencies
    spectrum = SpectralAnalyzer.analyze_spectrum(samples, sr, nfft=1024)
    params = ParameterEstimator.estimate_all(samples, sr, None)
    mod = ModulationResult(scheme="QPSK", confidence_score=0.92, evidence=["HOC match"])
    demod = DemodulationResult(
        scheme="QPSK",
        symbols=samples[::8],
        bits=np.random.randint(0, 2, size=2000, dtype=np.uint8),
        symbol_rate_est=None,
        carrier_locked=True,
        timing_locked=True,
    )
    metrics = ConstellationMetrics(
        evm_rms_percent=None,
        evm_peak_percent=12.5,
        snr_est_db=24.0,
        cluster_centroids=np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
        phase_jitter_deg=3.2,
        amplitude_droop_percent=0.0,
    )
    bitstream = BitstreamAnalysis(
        raw_bits=demod.bits,
        total_bits=len(demod.bits),
        bit_rate_est=None,
        transition_density=0.498,
        entropy_per_bit=0.999,
        hex_preview="A1B2C3D4",
        ascii_preview="....",
        preamble_detected=True,
    )
    fec = FecResult(scheme="None detected / Uncoded", detected=False, confidence=ConfidenceLevel.INFERRED)

    return SignalContext(
        file_path=Path("mock.wav"),
        reader=reader,
        metadata=reader.metadata,
        validation=ValidationResult(is_valid=True),
        raw_samples=samples,
        conditioned_samples=samples,
        sample_rate=sr,
        duration_s=len(samples) / sr,
        spectrum=spectrum,
        primary_parameters=params,
        modulation=mod,
        demodulation=demod,
        constellation_metrics=metrics,
        bitstream=bitstream,
        fec=fec,
    )


def test_registry_discovery():
    """Verify dynamic discovery loads all 16 categories and 200+ parameter definitions."""
    definitions = ParameterRegistry.get_all_parameter_definitions()
    assert len(definitions) >= 240

    categories = set(d.category for d in definitions)
    assert len(categories) == 16
    for cat in ParameterCategory:
        assert cat in categories


def test_extraction_engine_all_categories(mock_context: SignalContext):
    """Test full extraction across all 16 categories."""
    profile = ExtractionEngine.extract_all(mock_context)
    assert profile.total_count >= 240
    assert len(profile._by_category) == 16

    # Verify each category has extracted results
    for cat in ParameterCategory:
        cat_results = profile.get_category_results(cat)
        assert len(cat_results) > 0, f"Category {cat.value} returned 0 results"


def test_signal_profile_filtering_and_search(mock_context: SignalContext):
    """Test SignalProfile filtering, querying, and counting."""
    profile = ExtractionEngine.extract_all(mock_context)

    # 1. Category filter
    time_results = profile.filter(category=ParameterCategory.TIME_DOMAIN)
    assert len(time_results) > 0
    assert all(r.definition.category == ParameterCategory.TIME_DOMAIN for r in time_results)

    # 2. Status filter
    measured = profile.filter(status=ConfidenceLevel.MEASURED)
    assert len(measured) > 0
    assert all(r.status == ConfidenceLevel.MEASURED for r in measured)

    # 3. Text search
    evm_search = profile.filter(search_query="evm")
    assert len(evm_search) > 0
    assert any("evm" in r.definition.name.lower() or "evm" in r.definition.id.lower() for r in evm_search)

    # 4. CSV & JSON export
    csv_text = profile.to_csv()
    assert "Category,ID,Name,Value" in csv_text
    assert "const.evm_rms_percent" in csv_text

    json_text = profile.to_json()
    assert '"id": "const.evm_rms_percent"' in json_text

    # 5. Status counts
    counts = profile.counts_by_status
    assert counts["Measured"] > 0
    assert counts["Inferred"] > 0


def test_extractor_safe_error_handling():
    """Verify that an extractor encountering incomplete or empty data never crashes."""
    empty_samples = np.empty(0, dtype=np.complex64)
    reader = DummyReader(empty_samples, 1_000_000.0)
    empty_context = SignalContext(
        file_path=Path("empty.wav"),
        reader=reader,
        metadata=reader.metadata,
        validation=ValidationResult(is_valid=False),
        raw_samples=empty_samples,
        conditioned_samples=empty_samples,
        sample_rate=1_000_000.0,
        duration_s=0.0,
    )

    profile = ExtractionEngine.extract_all(empty_context)
    assert profile.total_count >= 240
    # Values should gracefully degrade to UNKNOWN without raising exceptions
    unknown_count = profile.counts_by_status.get("Unknown", 0)
    assert unknown_count > 0


def test_individual_categories_presence(mock_context: SignalContext):
    """Verify key signature parameters from categories A through P."""
    profile = ExtractionEngine.extract_all(mock_context)

    # Cat A
    assert profile.get("file.name") is not None
    # Cat B
    assert profile.get("time.crest_factor_db") is not None
    # Cat C
    assert profile.get("freq.spectral_centroid_hz") is not None
    # Cat D
    assert profile.get("power.snr_db") is not None
    # Cat E
    assert profile.get("mod.detected_scheme") is not None
    # Cat F
    assert profile.get("const.evm_rms_percent") is not None
    # Cat G
    assert profile.get("sync.carrier_lock_status") is not None
    # Cat H
    assert profile.get("burst.duty_cycle_percent") is not None
    # Cat I
    assert profile.get("digcomm.gross_bit_rate_bps") is not None
    # Cat J
    assert profile.get("fec.detected_scheme") is not None
    # Cat K
    assert profile.get("bitstream.shannon_bit_entropy") is not None
    # Cat L
    assert profile.get("frame.crc_pass_rate_percent") is not None
    # Cat M
    assert profile.get("noise.spurious_tones_count") is not None
    # Cat N
    assert profile.get("stat.cumulant_c42_real") is not None
    # Cat O
    assert profile.get("hw.estimated_enob_bits") is not None
    # Cat P
    assert profile.get("anomaly.total_anomalies_detected") is not None
