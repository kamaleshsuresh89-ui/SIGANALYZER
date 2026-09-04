"""Interactive Parameter Explorer widget with virtualized table, filtering, inspection, and export."""

from typing import Any
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.engine import SignalProfile
from siganalyzer.parameters.model import ParameterCategory, ParameterResult


class ParameterTableModel(QAbstractTableModel):
    """Virtualized table model for instant scrolling and rendering of hundreds of parameters."""

    COLUMNS = ["Category", "ID", "Parameter Name", "Value", "Unit", "Status", "Confidence"]

    STATUS_COLORS = {
        ConfidenceLevel.DIRECT: "#10b981",    # Emerald
        ConfidenceLevel.MEASURED: "#38bdf8",  # Sky Blue
        ConfidenceLevel.ESTIMATED: "#f59e0b", # Amber
        ConfidenceLevel.INFERRED: "#a855f7",  # Purple
        ConfidenceLevel.UNKNOWN: "#64748b",   # Slate / Gray
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: list[ParameterResult] = []

    def set_results(self, results: list[ParameterResult]) -> None:
        self.beginResetModel()
        self._results = list(results)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._results)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._results)):
            return None

        param = self._results[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return param.definition.category.value.split(". ", 1)[-1]
            elif col == 1:
                return param.definition.id
            elif col == 2:
                return param.definition.name
            elif col == 3:
                return param.formatted_value
            elif col == 4:
                return param.definition.unit
            elif col == 5:
                return param.status.value
            elif col == 6:
                return f"{param.confidence_score * 100:.0f}%"

        elif role == Qt.ItemDataRole.ForegroundRole:
            if col == 5:
                hex_col = self.STATUS_COLORS.get(param.status, "#f8fafc")
                return QColor(hex_col)
            elif col == 1:
                return QColor("#94a3b8")
            elif col == 3:
                if param.status == ConfidenceLevel.UNKNOWN:
                    return QColor("#64748b")
                return QColor("#38bdf8")

        elif role == Qt.ItemDataRole.FontRole:
            if col in (1, 3):
                font = QFont("Courier New", 10)
                font.setStyleHint(QFont.StyleHint.Monospace)
                return font
            elif col == 5:
                font = QFont()
                font.setBold(True)
                return font

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (3, 4, 6):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            elif col == 5:
                return int(Qt.AlignmentFlag.AlignCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        elif role == Qt.ItemDataRole.UserRole:
            return param

        return None

    def get_result(self, row: int) -> ParameterResult | None:
        if 0 <= row < len(self._results):
            return self._results[row]
        return None


class ParameterDetailDrawer(QScrollArea):
    """Slide-in inspector displaying in-depth provenance, equations, and explanations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("QScrollArea { background-color: #0f172a; border-left: 1px solid #334155; }")

        container = QWidget()
        container.setStyleSheet("background-color: #0f172a;")
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)

        # Header: Name & ID
        self.lbl_name = QLabel("Select a Parameter")
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.lbl_name)

        self.lbl_id = QLabel("Click any row in the table to inspect details")
        self.lbl_id.setStyleSheet("color: #64748b; font-family: monospace; font-size: 11px;")
        self.layout.addWidget(self.lbl_id)

        # Value Box
        self.val_frame = QFrame()
        self.val_frame.setStyleSheet("QFrame { background-color: #1e293b; border-radius: 6px; padding: 12px; border: 1px solid #334155; }")
        val_box = QVBoxLayout(self.val_frame)
        val_box.setContentsMargins(8, 8, 8, 8)
        val_box.setSpacing(4)

        lbl_val_title = QLabel("EXTRACTED VALUE")
        lbl_val_title.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        val_box.addWidget(lbl_val_title)

        self.lbl_val = QLabel("—")
        self.lbl_val.setStyleSheet("color: #38bdf8; font-size: 20px; font-weight: bold; font-family: monospace;")
        val_box.addWidget(self.lbl_val)

        # Badges row
        badge_row = QHBoxLayout()
        self.lbl_status = QLabel("STATUS")
        self.lbl_status.setStyleSheet("background-color: #334155; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;")
        badge_row.addWidget(self.lbl_status)

        self.lbl_confidence = QLabel("Confidence: —")
        self.lbl_confidence.setStyleSheet("color: #94a3b8; font-size: 11px;")
        badge_row.addWidget(self.lbl_confidence)
        badge_row.addStretch()
        val_box.addLayout(badge_row)

        self.layout.addWidget(self.val_frame)

        # Provenance / Source
        self._add_field_section("PROVENANCE / SOURCE", self._create_label("_lbl_source"))
        self._add_field_section("MATHEMATICAL METHOD / FORMULA", self._create_label("_lbl_method", is_code=True))
        self._add_field_section("TECHNICAL EXPLANATION", self._create_label("_lbl_explanation"))
        self._add_field_section("VALIDITY NOTES & CONSTRAINTS", self._create_label("_lbl_notes"))

        self.layout.addStretch()
        self.setWidget(container)

    def _create_label(self, attr_name: str, is_code: bool = False) -> QLabel:
        lbl = QLabel("—")
        lbl.setWordWrap(True)
        if is_code:
            lbl.setStyleSheet("color: #e2e8f0; background-color: #1e293b; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 11px;")
        else:
            lbl.setStyleSheet("color: #cbd5e1; font-size: 12px; line-height: 1.4;")
        setattr(self, attr_name, lbl)
        return lbl

    def _add_field_section(self, title: str, widget: QWidget) -> None:
        sec_lbl = QLabel(title)
        sec_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold; letter-spacing: 0.8px; margin-top: 6px;")
        self.layout.addWidget(sec_lbl)
        self.layout.addWidget(widget)

    def set_parameter(self, param: ParameterResult) -> None:
        self.lbl_name.setText(param.definition.name)
        self.lbl_id.setText(f"{param.definition.category.value}  •  {param.definition.id}")
        self.lbl_val.setText(param.formatted_value)

        # Status badge color
        stat_color = ParameterTableModel.STATUS_COLORS.get(param.status, "#64748b")
        self.lbl_status.setStyleSheet(f"background-color: {stat_color}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold;")
        self.lbl_status.setText(param.status.value.upper())

        self.lbl_confidence.setText(f"Confidence: {param.confidence_score * 100:.0f}%")
        self._lbl_source.setText(param.source or "N/A")
        self._lbl_method.setText(param.method or "N/A")
        self._lbl_explanation.setText(param.explanation or "No explanation available.")
        self._lbl_notes.setText(param.validity_notes or "Verified within nominal signal operating boundaries.")


class ParameterExplorerWidget(QWidget):
    """Complete interactive parameter analysis and exploration tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._profile: SignalProfile | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 1. Top Controls & Filter Bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search 250+ parameters by name, ID, value, source, or formula...")
        self.search_input.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 6px 12px; border-radius: 4px; font-size: 12px; }"
            "QLineEdit:focus { border-color: #38bdf8; }"
        )
        self.search_input.textChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.search_input, stretch=2)

        # Category Filter Dropdown
        self.combo_category = QComboBox()
        self.combo_category.setStyleSheet(
            "QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 200px; }"
            "QComboBox QAbstractItemView { background-color: #1e293b; color: #f8fafc; selection-background-color: #0284c7; }"
        )
        self.combo_category.addItem("All Categories (A–P)", None)
        for cat in ParameterCategory:
            self.combo_category.addItem(cat.value, cat)
        self.combo_category.currentIndexChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.combo_category)

        # Status Filter Dropdown
        self.combo_status = QComboBox()
        self.combo_status.setStyleSheet(
            "QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; padding: 6px 10px; border-radius: 4px; font-size: 12px; min-width: 140px; }"
            "QComboBox QAbstractItemView { background-color: #1e293b; color: #f8fafc; selection-background-color: #0284c7; }"
        )
        self.combo_status.addItem("All Statuses", None)
        for stat in ConfidenceLevel:
            self.combo_status.addItem(stat.value, stat)
        self.combo_status.currentIndexChanged.connect(self._apply_filters)
        filter_bar.addWidget(self.combo_status)

        # Export Buttons
        btn_export_csv = QPushButton("💾 CSV")
        btn_export_csv.setStyleSheet("QPushButton { background-color: #334155; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #475569; }")
        btn_export_csv.clicked.connect(self._export_csv)
        filter_bar.addWidget(btn_export_csv)

        btn_export_json = QPushButton("💾 JSON")
        btn_export_json.setStyleSheet("QPushButton { background-color: #334155; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #475569; }")
        btn_export_json.clicked.connect(self._export_json)
        filter_bar.addWidget(btn_export_json)

        main_layout.addLayout(filter_bar)

        # Status summary counters banner
        self.lbl_summary = QLabel("Awaiting signal analysis...")
        self.lbl_summary.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 4px;")
        main_layout.addWidget(self.lbl_summary)

        # 2. Main Horizontal Splitter (Table on Left, Detail Drawer on Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table setup
        self.table_model = ParameterTableModel(self)
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(False)
        self.table_view.setShowGrid(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setStyleSheet(
            "QTableView { background-color: #0f172a; alternate-background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; selection-background-color: #0369a1; selection-color: white; }"
            "QHeaderView::section { background-color: #1e293b; color: #94a3b8; padding: 6px; border: 1px solid #334155; font-weight: bold; font-size: 11px; }"
        )

        header = self.table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        self.table_view.selectionModel().selectionChanged.connect(self._on_row_selected)
        splitter.addWidget(self.table_view)

        # Detail Drawer on Right
        self.detail_drawer = ParameterDetailDrawer()
        splitter.addWidget(self.detail_drawer)
        splitter.setSizes([850, 450])

        main_layout.addWidget(splitter)

    def set_profile(self, profile: SignalProfile | None) -> None:
        """Populate explorer with complete extracted signal profile."""
        self._profile = profile
        self._apply_filters()

    def clear(self) -> None:
        self._profile = None
        self.table_model.set_results([])
        self.lbl_summary.setText("Awaiting signal analysis...")

    def _apply_filters(self) -> None:
        if self._profile is None:
            self.table_model.set_results([])
            self.lbl_summary.setText("Awaiting signal analysis...")
            return

        cat_filter = self.combo_category.currentData()
        status_filter = self.combo_status.currentData()
        query = self.search_input.text().strip()

        filtered = self._profile.filter(category=cat_filter, status=status_filter, search_query=query)
        self.table_model.set_results(filtered)

        # Update summary counts
        counts = self._profile.counts_by_status
        total = self._profile.total_count
        showing = len(filtered)
        self.lbl_summary.setText(
            f"Showing <b>{showing}</b> of <b>{total}</b> parameters  |  "
            f"<span style='color:#10b981'>Direct: {counts.get('Direct', 0)}</span>  |  "
            f"<span style='color:#38bdf8'>Measured: {counts.get('Measured', 0)}</span>  |  "
            f"<span style='color:#f59e0b'>Estimated: {counts.get('Estimated', 0)}</span>  |  "
            f"<span style='color:#a855f7'>Inferred: {counts.get('Inferred', 0)}</span>  |  "
            f"<span style='color:#64748b'>Unknown: {counts.get('Unknown', 0)}</span>"
        )

        if filtered:
            self.table_view.selectRow(0)

    def _on_row_selected(self) -> None:
        indexes = self.table_view.selectionModel().selectedRows()
        if indexes:
            row = indexes[0].row()
            res = self.table_model.get_result(row)
            if res is not None:
                self.detail_drawer.set_parameter(res)

    def _export_csv(self) -> None:
        if self._profile is None:
            QMessageBox.warning(self, "No Data", "No parameters available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters as CSV", "signal_parameters.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._profile.to_csv())
                QMessageBox.information(self, "Export Successful", f"Saved {self._profile.total_count} parameters to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not write CSV file:\n{e}")

    def _export_json(self) -> None:
        if self._profile is None:
            QMessageBox.warning(self, "No Data", "No parameters available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Parameters as JSON", "signal_parameters.json", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._profile.to_json(indent=2))
                QMessageBox.information(self, "Export Successful", f"Saved {self._profile.total_count} parameters to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not write JSON file:\n{e}")
