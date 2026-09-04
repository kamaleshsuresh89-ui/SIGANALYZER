"""Local analysis report preview and export dialog."""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.reporting.generator import CompleteAnalysisReport
from siganalyzer.reporting.html_exporter import export_html, generate_html_report
from siganalyzer.reporting.json_exporter import export_json


class ReportDialog(QDialog):
    """Modal dialog to preview and export local analysis reports."""

    def __init__(self, report: CompleteAnalysisReport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.report = report
        self.setWindowTitle(f"Analysis Report — {report.metadata.file_name}")
        self.resize(850, 650)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.browser = QTextBrowser()
        self.browser.setHtml(generate_html_report(self.report))
        self.browser.setStyleSheet("background-color: #0f172a; border: 1px solid #334155;")
        layout.addWidget(self.browser)

        btn_layout = QHBoxLayout()

        btn_export_html = QPushButton("💾 Export HTML Report")
        btn_export_html.setStyleSheet(
            "background-color: #0284c7; color: white; padding: 6px 14px; font-weight: bold; border-radius: 4px;"
        )
        btn_export_html.clicked.connect(self._on_export_html)
        btn_layout.addWidget(btn_export_html)

        btn_export_json = QPushButton("📄 Export JSON Report")
        btn_export_json.setStyleSheet(
            "background-color: #475569; color: white; padding: 6px 14px; font-weight: bold; border-radius: 4px;"
        )
        btn_export_json.clicked.connect(self._on_export_json)
        btn_layout.addWidget(btn_export_json)

        btn_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def _on_export_html(self) -> None:
        default_name = f"report_{Path(self.report.metadata.file_name).stem}.html"
        path, _ = QFileDialog.getSaveFileName(self, "Save HTML Report", default_name, "HTML Files (*.html)")
        if path:
            export_html(self.report, path)
            QMessageBox.information(self, "Export Successful", f"Saved HTML report to:\n{path}")

    def _on_export_json(self) -> None:
        default_name = f"report_{Path(self.report.metadata.file_name).stem}.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON Report", default_name, "JSON Files (*.json)")
        if path:
            export_json(self.report, path)
            QMessageBox.information(self, "Export Successful", f"Saved JSON report to:\n{path}")
