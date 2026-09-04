"""Main Application Window for SIGANALYZER."""

from pathlib import Path
from PySide6.QtCore import QThreadPool, Qt, Slot
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.reporting.generator import CompleteAnalysisReport
from siganalyzer.reporting.html_exporter import generate_html_report
from siganalyzer.ui.views.bitstream_view import BitstreamView
from siganalyzer.ui.views.constellation_view import ConstellationView
from siganalyzer.ui.views.drop_zone import DropZoneWidget
from siganalyzer.ui.views.parameter_explorer import ParameterExplorerWidget
from siganalyzer.ui.views.parameters_table import ParametersTableWidget
from siganalyzer.ui.views.regions_list import RegionsListWidget
from siganalyzer.ui.views.report_dialog import ReportDialog
from siganalyzer.ui.views.spectrogram_view import SpectrogramView
from siganalyzer.ui.views.spectrum_view import SpectrumView
from siganalyzer.ui.views.time_view import TimeDomainView
from siganalyzer.ui.workers import AnalysisWorker


class MainWindow(QMainWindow):
    """Main window coordinating signal loading, background processing, and visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SIGANALYZER — Offline RF Signal Analysis Suite")
        self.resize(1400, 900)

        self.thread_pool = QThreadPool()
        self.current_worker: AnalysisWorker | None = None
        self.current_report: CompleteAnalysisReport | None = None

        self._init_ui()
        self._apply_dark_theme()

    def _init_ui(self) -> None:
        # Central Widget & Tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Top Control Bar
        top_bar = QHBoxLayout()
        self.drop_zone = DropZoneWidget()
        self.drop_zone.file_selected.connect(self.start_analysis)
        top_bar.addWidget(self.drop_zone, stretch=1)

        self.btn_analyze = QPushButton("⚡ Analyze")
        self.btn_analyze.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; padding: 7px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #047857; }"
        )
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        top_bar.addWidget(self.btn_analyze)

        self.btn_cancel = QPushButton("✖ Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet(
            "QPushButton { background-color: #dc2626; color: white; font-weight: bold; padding: 7px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #b91c1c; }"
            "QPushButton:disabled { background-color: #475569; }"
        )
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        top_bar.addWidget(self.btn_cancel)

        self.btn_report = QPushButton("📊 View Report")
        self.btn_report.setEnabled(False)
        self.btn_report.setStyleSheet(
            "QPushButton { background-color: #7c3aed; color: white; font-weight: bold; padding: 7px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #6d28d9; }"
            "QPushButton:disabled { background-color: #475569; }"
        )
        self.btn_report.clicked.connect(self._on_view_report_clicked)
        top_bar.addWidget(self.btn_report)

        main_layout.addLayout(top_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #1e293b; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #38bdf8; border-radius: 4px; }"
        )
        main_layout.addWidget(self.progress_bar)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #334155; background-color: #0f172a; }"
            "QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 8px 16px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; }"
            "QTabBar::tab:selected { background-color: #0f172a; color: #38bdf8; border: 1px solid #334155; border-bottom: none; }"
        )

        # Tab 1: Signal Visualizations (Time, Spectrum, Spectrogram in vertical splitter)
        tab1_widget = QWidget()
        tab1_layout = QVBoxLayout(tab1_widget)
        tab1_layout.setContentsMargins(4, 4, 4, 4)

        splitter_plots = QSplitter(Qt.Orientation.Vertical)
        self.time_view = TimeDomainView()
        self.spectrum_view = SpectrumView()
        self.spectrogram_view = SpectrogramView()

        splitter_plots.addWidget(self.time_view)
        splitter_plots.addWidget(self.spectrum_view)
        splitter_plots.addWidget(self.spectrogram_view)
        splitter_plots.setSizes([260, 260, 320])

        tab1_layout.addWidget(splitter_plots)
        self.tabs.addTab(tab1_widget, "📈 Signal Overview")

        # Tab 2: Constellation & Demodulation
        self.constellation_view = ConstellationView()
        self.tabs.addTab(self.constellation_view, "🎯 Constellation & Clusters")

        # Tab 3: Bitstream & Protocol Inspector
        self.bitstream_view = BitstreamView()
        self.tabs.addTab(self.bitstream_view, "0101 Bitstream & Protocol")

        # Tab 4: Comprehensive Parameter Explorer (Categories A to P)
        self.parameter_explorer = ParameterExplorerWidget()
        self.tabs.addTab(self.parameter_explorer, "🔬 Parameter Explorer (A–P)")

        # Tab 5: Detailed Report Preview
        self.report_browser = QTextBrowser()
        self.report_browser.setStyleSheet("background-color: #0f172a; border: none;")
        self.tabs.addTab(self.report_browser, "📋 Analysis Report")

        main_layout.addWidget(self.tabs)

        # Left Dock: Parameter Table & Detected Regions
        self._setup_dock()

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("background-color: #0f172a; color: #94a3b8; font-size: 12px;")
        self.status_bar.showMessage("Ready. Drag and drop a WAV or IQ recording to analyze.")

    def _setup_dock(self) -> None:
        dock = QDockWidget("Findings & Parameters", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setStyleSheet(
            "QDockWidget { color: #f8fafc; font-weight: bold; }"
            "QDockWidget::title { background-color: #1e293b; padding: 6px; text-align: left; }"
        )

        dock_contents = QWidget()
        dock_layout = QVBoxLayout(dock_contents)
        dock_layout.setContentsMargins(4, 4, 4, 4)

        splitter_dock = QSplitter(Qt.Orientation.Vertical)

        # Parameters Table
        self.parameters_table = ParametersTableWidget()
        splitter_dock.addWidget(self.parameters_table)

        # Regions List
        self.regions_list = RegionsListWidget()
        splitter_dock.addWidget(self.regions_list)
        splitter_dock.setSizes([450, 200])

        dock_layout.addWidget(splitter_dock)
        dock.setWidget(dock_contents)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            "QMainWindow { background-color: #0f172a; color: #f8fafc; }"
            "QToolTip { color: #ffffff; background-color: #1e293b; border: 1px solid #475569; padding: 4px; }"
        )

    def _on_analyze_clicked(self) -> None:
        path = self.drop_zone.path_input.text().strip()
        if path:
            self.start_analysis(path)
        else:
            QMessageBox.warning(self, "No File", "Please select or drag-and-drop a signal file first.")

    def _on_cancel_clicked(self) -> None:
        if self.current_worker:
            self.current_worker.cancel()
            self.status_bar.showMessage("Cancelling analysis...")

    def _on_view_report_clicked(self) -> None:
        if self.current_report:
            dialog = ReportDialog(self.current_report, self)
            dialog.exec()

    @Slot(str)
    def start_analysis(self, file_path: str) -> None:
        """Start asynchronous background analysis of selected signal file."""
        path = Path(file_path).resolve()
        if not path.exists():
            QMessageBox.critical(self, "File Not Found", f"Cannot find file:\n{file_path}")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_report.setEnabled(False)
        self.progress_bar.setValue(0)

        # Clear existing views
        self.time_view.clear()
        self.spectrum_view.clear()
        self.spectrogram_view.clear()
        self.constellation_view.clear()
        self.bitstream_view.clear()
        self.parameters_table.clear()
        self.regions_list.clear()
        self.parameter_explorer.clear()

        self.current_worker = AnalysisWorker(path)
        self.current_worker.signals.progress.connect(self._on_progress)
        self.current_worker.signals.finished.connect(self._on_finished)
        self.current_worker.signals.error.connect(self._on_error)

        self.thread_pool.start(self.current_worker)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_bar.showMessage(f"[{percent}%] {message}")

    @Slot(object, object, object)
    def _on_finished(self, report: CompleteAnalysisReport, spectrogram_data, samples) -> None:
        self.current_report = report
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_report.setEnabled(True)

        meta = report.metadata
        sr = meta.sample_rate.value or 1_000_000.0
        cf = meta.center_frequency.value

        # Update Visualizations
        self.time_view.set_samples(samples, sr)
        self.spectrum_view.set_spectrum(report.spectrum, center_frequency=cf)
        self.spectrogram_view.set_spectrogram(spectrogram_data, center_frequency=cf)
        self.constellation_view.set_modulation_result(
            report.primary_modulation, demod_result=report.demodulation
        )
        self.bitstream_view.populate(
            report.demodulation,
            bitstream_analysis=report.bitstream,
            symbol_rate=report.primary_parameters.symbol_rate_sps.value,
        )
        self.parameters_table.set_report(report)
        self.regions_list.set_regions(report.regions)
        self.parameter_explorer.set_profile(report.signal_profile)

        # Update HTML tab
        self.report_browser.setHtml(generate_html_report(report))

        mod = report.primary_modulation.scheme
        conf = report.primary_modulation.confidence_score * 100
        self.status_bar.showMessage(
            f"Done. Detected: {mod} ({conf:.1f}%) | {meta.total_samples:,} samples | Duration: {meta.duration_seconds.formatted_value}"
        )

    @Slot(str)
    def _on_error(self, err_msg: str) -> None:
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Error: {err_msg}")
        QMessageBox.critical(self, "Analysis Failed", f"An error occurred during analysis:\n\n{err_msg}")
