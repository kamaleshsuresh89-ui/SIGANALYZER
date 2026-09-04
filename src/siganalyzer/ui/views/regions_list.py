"""Detected signal regions and burst segments list widget."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.common.types import SignalRegion


class RegionsListWidget(QWidget):
    """Table showing detected signal burst segments."""

    region_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._regions: list[SignalRegion] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "Start (s)", "End (s)", "Duration", "SNR"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #1e293b; color: #f8fafc; gridline-color: #334155; border: none; font-size: 12px; }"
            "QHeaderView::section { background-color: #0f172a; color: #94a3b8; font-weight: bold; border: 1px solid #334155; }"
        )
        self.table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self.table)

    def set_regions(self, regions: list[SignalRegion]) -> None:
        self._regions = regions
        self.table.setRowCount(len(regions))

        for row, r in enumerate(regions):
            items = [
                f"#{r.region_id}",
                f"{r.start_time:.3f}",
                f"{r.end_time:.3f}",
                f"{r.duration_s * 1000:.1f} ms",
                r.snr_db.formatted_value,
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(row, col, item)

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if 0 <= row < len(self._regions):
            self.region_selected.emit(self._regions[row].region_id)

    def clear(self) -> None:
        self._regions = []
        self.table.setRowCount(0)
