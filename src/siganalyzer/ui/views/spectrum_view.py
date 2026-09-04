"""Frequency spectrum visualization widget using pyqtgraph."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg

from siganalyzer.dsp.spectrum import SpectrumResult


class SpectrumView(QWidget):
    """Interactive frequency spectrum / Welch PSD view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Power Spectral Density (Welch PSD)")
        self.plot_widget.setBackground("#0f172a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self.plot_widget.setLabel("left", "Power Spectral Density", units="dB/Hz")
        self.plot_widget.addLegend(offset=(10, 10))

        # Spectrum curve
        self.curve_psd = self.plot_widget.plot(
            pen=pg.mkPen(color="#38bdf8", width=1.5),
            fillLevel=-160,
            fillBrush=pg.mkBrush(color=(56, 189, 248, 40)),
            name="PSD",
        )

        # Noise floor line
        self.line_noise = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(color="#f59e0b", style=pg.QtCore.Qt.PenStyle.DashLine, width=1.5),
            label="Noise Floor: {value:.1f} dB",
            labelOpts={"position": 0.85, "color": "#f59e0b", "fill": (15, 23, 42, 200)},
        )
        self.plot_widget.addItem(self.line_noise)

        # Peak marker
        self.marker_peak = self.plot_widget.plot(
            pen=None,
            symbol="t1",  # Triangle marker
            symbolSize=12,
            symbolBrush=pg.mkBrush("#ef4444"),
            name="Peak",
        )

        layout.addWidget(self.plot_widget)

    def set_spectrum(self, spectrum: SpectrumResult, center_frequency: float | None = None) -> None:
        """Update PSD curve, noise floor line, and peak markers."""
        freqs = spectrum.frequencies_hz
        if center_frequency is not None:
            freqs = freqs + center_frequency
            self.plot_widget.setLabel("bottom", "RF Frequency", units="Hz")
        else:
            self.plot_widget.setLabel("bottom", "Baseband Frequency", units="Hz")

        self.curve_psd.setData(freqs, spectrum.psd_db)
        self.line_noise.setValue(spectrum.noise_floor_db)

        # Update peak marker
        if spectrum.detected_peaks:
            peak_f = (
                spectrum.detected_peaks[0].freq_hz + (center_frequency or 0.0)
            )
            peak_p = spectrum.detected_peaks[0].power_db
            self.marker_peak.setData([peak_f], [peak_p])
        else:
            self.marker_peak.clear()

        self.plot_widget.enableAutoRange()

    def clear(self) -> None:
        self.curve_psd.clear()
        self.marker_peak.clear()
