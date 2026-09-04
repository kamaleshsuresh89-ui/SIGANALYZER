"""2D Spectrogram and waterfall visualization widget using pyqtgraph."""

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg

from siganalyzer.dsp.spectrogram import SpectrogramData


class SpectrogramView(QWidget):
    """Interactive 2D time-frequency spectrogram / waterfall view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Spectrogram / Waterfall (Time vs Frequency)")
        self.plot_widget.setBackground("#0f172a")
        self.plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self.plot_widget.setLabel("left", "Time", units="s")

        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        # Apply self-contained Viridis colormap (zero external dependencies)
        pos = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        colors = np.array([
            [68, 1, 84, 255],     # Dark purple
            [59, 82, 139, 255],   # Blue
            [33, 145, 140, 255],  # Teal
            [94, 201, 98, 255],   # Green
            [253, 231, 37, 255],  # Yellow
        ], dtype=np.uint8)
        colormap = pg.ColorMap(pos, colors)
        self.lut = colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(self.lut)

        layout.addWidget(self.plot_widget)

    def set_spectrogram(self, data: SpectrogramData, center_frequency: float | None = None) -> None:
        """Render 2D spectrogram matrix."""
        if data.power_matrix_db.size == 0:
            self.clear()
            return

        matrix = data.power_matrix_db  # shape (time_bins, freq_bins)
        # In pyqtgraph ImageItem, shape is (x, y) where x is freq and y is time
        # Transpose so x is frequency and y is time
        img_data = matrix.T

        f_start = data.freq_axis[0] + (center_frequency or 0.0)
        f_end = data.freq_axis[-1] + (center_frequency or 0.0)
        f_span = f_end - f_start

        t_start = data.time_axis[0] if len(data.time_axis) > 0 else 0.0
        t_end = data.time_axis[-1] if len(data.time_axis) > 0 else 1.0
        t_span = t_end - t_start

        self.img_item.setImage(img_data)
        self.img_item.setRect(pg.QtCore.QRectF(f_start, t_start, f_span, t_span))

        # Dynamic contrast scaling
        min_pwr = max(data.min_power_db, data.max_power_db - 60.0)
        max_pwr = data.max_power_db
        self.img_item.setLevels([min_pwr, max_pwr])

        self.plot_widget.enableAutoRange()

    def clear(self) -> None:
        self.img_item.clear()
