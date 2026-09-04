"""Parameters findings and metadata table with color-coded confidence badges."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.reporting.generator import CompleteAnalysisReport


class ParametersTableWidget(QWidget):
    """Table showing estimated parameters with provenance and confidence indicators."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value", "Confidence", "Source / Evidence"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1e293b; color: #f8fafc; gridline-color: #334155; border: none; }"
            "QHeaderView::section { background-color: #0f172a; color: #94a3b8; font-weight: bold; border: 1px solid #334155; padding: 4px; }"
            "QTableWidget::item { padding: 6px; }"
        )

        layout.addWidget(self.table)

    def set_report(self, report: CompleteAnalysisReport) -> None:
        """Populate the table with report findings."""
        meta = report.metadata
        params = report.primary_parameters
        mod = report.primary_modulation

        evidence_str = "; ".join(mod.evidence) if mod.evidence else "Rule-based analysis"

        rows = [
            ("Detected Modulation", f"{mod.scheme} ({mod.confidence_score * 100:.1f}%)", ConfidenceLevel.INFERRED, evidence_str),
            ("Symbol Rate", params.symbol_rate_sps.formatted_value, params.symbol_rate_sps.confidence, params.symbol_rate_sps.source),
            ("Bit Rate", params.bit_rate_bps.formatted_value, params.bit_rate_bps.confidence, params.bit_rate_bps.source),
            ("Carrier Frequency", params.carrier_freq_hz.formatted_value, params.carrier_freq_hz.confidence, params.carrier_freq_hz.source),
            ("Occupied Bandwidth (99%)", params.occupied_bandwidth_hz.formatted_value, params.occupied_bandwidth_hz.confidence, params.occupied_bandwidth_hz.source),
            ("Bandwidth (-3dB)", params.bandwidth_3db_hz.formatted_value, params.bandwidth_3db_hz.confidence, params.bandwidth_3db_hz.source),
            ("Estimated SNR", params.snr_db.formatted_value, params.snr_db.confidence, params.snr_db.source),
            ("Sample Rate", meta.sample_rate.formatted_value, meta.sample_rate.confidence, meta.sample_rate.source),
            ("File Format", meta.file_type.value, ConfidenceLevel.DIRECT, "Format detection engine"),
            ("Duration", meta.duration_seconds.formatted_value, meta.duration_seconds.confidence, f"{meta.total_samples:,} samples"),
            ("Peak Amplitude", meta.peak_amplitude.formatted_value, meta.peak_amplitude.confidence, "Sample scan"),
            ("RMS Power", meta.rms_power.formatted_value, meta.rms_power.confidence, "Sample scan"),
        ]

        self.table.setRowCount(len(rows))

        for row_idx, (name, val, conf, source) in enumerate(rows):
            # Parameter Name
            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row_idx, 0, item_name)

            # Value
            item_val = QTableWidgetItem(val)
            item_val.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item_val.setForeground(Qt.GlobalColor.white)
            self.table.setItem(row_idx, 1, item_val)

            # Confidence Badge
            badge_widget = self._create_badge(conf)
            self.table.setCellWidget(row_idx, 2, badge_widget)

            # Source
            item_src = QTableWidgetItem(source)
            item_src.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item_src.setToolTip(source)
            self.table.setItem(row_idx, 3, item_src)

    def _create_badge(self, conf: ConfidenceLevel) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl = QLabel(conf.value)
        lbl.setStyleSheet(
            f"background-color: {conf.badge_color}; color: white; font-size: 10px; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
        )
        layout.addWidget(lbl)
        return container

    def clear(self) -> None:
        self.table.setRowCount(0)
