"""Application bootstrap and Qt lifecycle coordinator."""

import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from siganalyzer.ui.main_window import MainWindow


def create_app() -> tuple[QApplication, MainWindow]:
    """Create and configure the Qt application with dark palette."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("SIGANALYZER")
    app.setApplicationDisplayName("SIGANALYZER — RF Signal Analyzer")
    app.setOrganizationName("SIGANALYZER")

    # Clean modern system font
    font = QFont("-apple-system", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Dark Theme Palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(15, 23, 42))  # #0f172a
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(30, 41, 59))  # #1e293b
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(15, 23, 42))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(30, 41, 59))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(30, 41, 59))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(248, 250, 252))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(56, 189, 248))  # #38bdf8
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(15, 23, 42))
    app.setPalette(dark_palette)

    window = MainWindow()

    # If file path was passed via CLI arguments, auto-load it
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        if target_path.exists() and target_path.is_file():
            window.drop_zone.set_file_path(str(target_path.resolve()))

    return app, window


def run() -> int:
    """Run the SIGANALYZER desktop application."""
    app, window = create_app()
    window.show()
    return app.exec()
