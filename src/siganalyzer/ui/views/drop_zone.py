"""Drag-and-drop file loading widget."""

from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DropZoneWidget(QWidget):
    """Interactive drag & drop loading bar."""

    file_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        self.btn_select = QPushButton("📂 Open File")
        self.btn_select.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_select.clicked.connect(self._on_browse_clicked)
        layout.addWidget(self.btn_select)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select or Drag & Drop recorded WAV / IQ signal file here...")
        self.path_input.setReadOnly(True)
        self.path_input.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #f8fafc; border: 1px dashed #475569; padding: 6px 10px; border-radius: 4px; font-size: 13px; }"
        )
        layout.addWidget(self.path_input)

    def _on_browse_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal Recording",
            "",
            "Signal Files (*.wav *.rf64 *.iq *.raw *.dat *.bin *.cfile);;All Files (*)",
        )
        if file_path:
            self.set_file_path(file_path)

    def set_file_path(self, file_path: str) -> None:
        self.path_input.setText(file_path)
        self.file_selected.emit(file_path)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.path_input.setStyleSheet(
                "QLineEdit { background-color: #0f172a; color: #38bdf8; border: 2px dashed #38bdf8; padding: 6px 10px; border-radius: 4px; }"
            )

    def dragLeaveEvent(self, event) -> None:
        self.path_input.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #f8fafc; border: 1px dashed #475569; padding: 6px 10px; border-radius: 4px; }"
        )

    def dropEvent(self, event) -> None:
        self.path_input.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #f8fafc; border: 1px dashed #475569; padding: 6px 10px; border-radius: 4px; }"
        )
        urls = event.mimeData().urls()
        if urls:
            local_path = urls[0].toLocalFile()
            if local_path:
                self.set_file_path(local_path)
