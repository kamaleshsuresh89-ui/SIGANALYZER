"""Time-domain waveform visualization widget using pyqtgraph."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg


class TimeDomainView(QWidget):
    """Interactive time-domain I and Q waveform view with peak-preserving downsampling."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Time Domain Waveform (I & Q)")
        self.plot_widget.setBackground("#0f172a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.addLegend(offset=(10, 10))

        # Peak-preserving downsampling for smooth zoom/pan
        self.plot_widget.setDownsampling(auto=True, mode="peak")
        self.plot_widget.setClipToView(True)

        self.curve_i = self.plot_widget.plot(
            pen=pg.mkPen(color="#38bdf8", width=1.5), name="I (In-Phase)"
        )
        self.curve_q = self.plot_widget.plot(
            pen=pg.mkPen(color="#fb923c", width=1.5), name="Q (Quadrature)"
        )

        layout.addWidget(self.plot_widget)

    def set_samples(self, samples: np.ndarray, sample_rate: float, max_display_points: int = 10000) -> None:
        """Update the time-domain waveform display."""
        if len(samples) == 0:
            self.clear()
            return

        total_pts = len(samples)
        step = max(1, total_pts // max_display_points)
        sub_samples = samples[::step]

        t_axis = (np.arange(len(sub_samples)) * step) / sample_rate

        self.curve_i.setData(t_axis, np.real(sub_samples))
        self.curve_q.setData(t_axis, np.imag(sub_samples))
        self.plot_widget.enableAutoRange()

    def clear(self) -> None:
        self.curve_i.clear()
        self.curve_q.clear()
