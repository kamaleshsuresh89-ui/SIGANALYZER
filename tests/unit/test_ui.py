"""Headless unit tests for PySide6 UI components."""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from siganalyzer.app import create_app
from siganalyzer.dsp.pipeline import AnalysisPipeline
from siganalyzer.ui.main_window import MainWindow
from siganalyzer.ui.views.constellation_view import ConstellationView
from siganalyzer.ui.views.drop_zone import DropZoneWidget
from siganalyzer.ui.views.parameters_table import ParametersTableWidget
from siganalyzer.ui.views.regions_list import RegionsListWidget
from siganalyzer.ui.views.report_dialog import ReportDialog
from siganalyzer.ui.views.spectrogram_view import SpectrogramView
from siganalyzer.ui.views.spectrum_view import SpectrumView
from siganalyzer.ui.views.time_view import TimeDomainView
from tests.fixtures.generate_fixtures import generate_bpsk, save_as_wav


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication is created once for the test session in offscreen mode."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_main_window_initialization(qapp):
    """Test MainWindow instantiates and has expected initial state."""
    window = MainWindow()
    assert window.windowTitle().startswith("SIGANALYZER")
    assert not window.btn_cancel.isEnabled()
    assert not window.btn_report.isEnabled()
    assert window.btn_analyze.isEnabled()
    assert window.tabs.count() == 5


def test_views_population(qapp, tmp_path: Path):
    """Test populating all views with real analysis results."""
    sample_rate = 500_000.0
    samples, _ = generate_bpsk(num_symbols=1000, sps=8, sample_rate=sample_rate)
    wav_path = tmp_path / "ui_test.wav"
    save_as_wav(wav_path, samples, sample_rate=int(sample_rate), bit_depth=16)

    pipeline = AnalysisPipeline()
    report, spec_data, raw_samples = pipeline.analyze_file(wav_path)

    # Test TimeDomainView
    time_view = TimeDomainView()
    time_view.set_samples(raw_samples, sample_rate)

    # Test SpectrumView
    spectrum_view = SpectrumView()
    spectrum_view.set_spectrum(report.spectrum)

    # Test SpectrogramView
    spectrogram_view = SpectrogramView()
    spectrogram_view.set_spectrogram(spec_data)

    # Test ConstellationView
    constellation_view = ConstellationView()
    constellation_view.set_modulation_result(
        report.primary_modulation, demod_result=report.demodulation
    )
    assert report.primary_modulation.scheme in constellation_view.lbl_info.text()
    assert "Recovered Bits" in constellation_view.lbl_bits_count.text()

    # Test BitstreamView
    from siganalyzer.ui.views.bitstream_view import BitstreamView
    bitstream_view = BitstreamView()
    bitstream_view.populate(
        report.demodulation,
        bitstream_analysis=report.bitstream,
        symbol_rate=report.primary_parameters.symbol_rate_sps.value,
    )
    assert int(bitstream_view.card_bits.value_label.text().replace(",", "")) > 0

    # Test ParametersTableWidget
    param_table = ParametersTableWidget()
    param_table.set_report(report)
    assert param_table.table.rowCount() > 5

    # Test RegionsListWidget
    regions_list = RegionsListWidget()
    regions_list.set_regions(report.regions)
    assert regions_list.table.rowCount() == len(report.regions)

    # Test ReportDialog
    dialog = ReportDialog(report)
    assert "SIGANALYZER" in dialog.browser.toHtml()

    # Test ParameterExplorerWidget
    from siganalyzer.ui.views.parameter_explorer import ParameterExplorerWidget
    explorer = ParameterExplorerWidget()
    explorer.set_profile(report.signal_profile)
    assert explorer.table_model.rowCount() >= 240
    explorer.search_input.setText("evm")
    assert explorer.table_model.rowCount() > 0
    explorer.search_input.setText("")
    assert explorer.table_model.rowCount() >= 240
    explorer.clear()
    assert explorer.table_model.rowCount() == 0
