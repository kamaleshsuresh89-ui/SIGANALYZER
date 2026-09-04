"""Integration test for end-to-end signal analysis pipeline and reporting."""

from pathlib import Path
import pytest

from siganalyzer.dsp.pipeline import AnalysisPipeline
from siganalyzer.reporting.html_exporter import export_html
from siganalyzer.reporting.json_exporter import export_json
from tests.fixtures.generate_fixtures import generate_qpsk, save_as_wav


def test_end_to_end_pipeline(tmp_path: Path):
    """Test full analysis pipeline on a synthetic QPSK WAV file and export reports."""
    sample_rate = 1_000_000.0
    fc = 150_000.0
    samples, _ = generate_qpsk(num_symbols=2000, sps=8, sample_rate=sample_rate, carrier_freq=fc, snr_db=25.0)
    wav_path = tmp_path / "test_signal.wav"
    save_as_wav(wav_path, samples, sample_rate=int(sample_rate), bit_depth=16, center_freq=433_920_000)

    progress_log: list[tuple[int, str]] = []

    def on_progress(pct: int, msg: str):
        progress_log.append((pct, msg))

    pipeline = AnalysisPipeline(progress_callback=on_progress)
    report, spectrogram_data, raw_samples = pipeline.analyze_file(wav_path)

    # 1. Check pipeline completion
    assert len(progress_log) > 0
    assert progress_log[-1][0] == 100
    assert len(raw_samples) == len(samples)

    # 2. Check metadata
    assert report.metadata.sample_rate.value == sample_rate
    assert report.metadata.center_frequency.value == 433_920_000
    assert report.metadata.total_samples == len(samples)

    # 3. Check parameters
    assert report.primary_parameters.carrier_freq_hz.value is not None
    assert abs(report.primary_parameters.carrier_freq_hz.value - (433_920_000 + fc)) < 15_000.0

    # 4. Check modulation classification
    assert report.primary_modulation.scheme in ("QPSK", "8PSK", "BPSK")
    assert report.primary_modulation.confidence_score > 0.60
    assert len(report.primary_modulation.evidence) >= 1

    # 5. Check spectrogram data
    assert spectrogram_data.power_matrix_db.ndim == 2
    assert len(spectrogram_data.freq_axis) == 1024
    assert len(spectrogram_data.time_axis) == spectrogram_data.power_matrix_db.shape[0]

    # 6. Test Demodulation, Bitstream & FEC results
    assert report.demodulation is not None
    assert len(report.demodulation.bits) > 0
    assert report.bitstream is not None
    assert report.bitstream.total_bits == len(report.demodulation.bits)
    assert report.bitstream.entropy_per_bit > 0.0
    assert report.fec is not None

    # 7. Test offline HTML report generation
    html_file = tmp_path / "report.html"
    export_html(report, html_file)
    assert html_file.exists()
    html_content = html_file.read_text(encoding="utf-8")
    assert "SIGANALYZER Automated Report" in html_content
    assert report.metadata.file_name in html_content
    assert report.primary_modulation.scheme in html_content
    assert "Demodulation & Bitstream Analysis" in html_content

    # 8. Test JSON export
    json_file = tmp_path / "report.json"
    export_json(report, json_file)
    assert json_file.exists()
    json_text = json_file.read_text(encoding="utf-8")
    assert '"detected_scheme"' in json_text
    assert '"bitstream"' in json_text
    assert '"fec"' in json_text


def test_pipeline_cancellation(tmp_path: Path):
    """Test thread-safe cancellation stops pipeline early."""
    samples, _ = generate_qpsk(num_symbols=1000, sps=8, sample_rate=1e6)
    wav_path = tmp_path / "cancel.wav"
    save_as_wav(wav_path, samples, sample_rate=1_000_000, bit_depth=16)

    pipeline = AnalysisPipeline(is_cancelled=lambda: True)
    with pytest.raises(InterruptedError):
        pipeline.analyze_file(wav_path)
