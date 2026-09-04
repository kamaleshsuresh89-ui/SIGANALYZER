"""Constellation and I/Q symbol scatter visualization widget with bitstream preview."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from siganalyzer.common.types import DemodulationResult, ModulationResult


class ConstellationView(QWidget):
    """Interactive I/Q constellation scatter plot and bit stream inspector."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Constellation plot widget
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_info = QLabel("Modulation: None | EVM: --")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: bold; padding: 6px;")
        left_layout.addWidget(self.lbl_info)

        self.plot_widget = pg.PlotWidget(title="I/Q Constellation Diagram (Synchronized Symbols)")
        self.plot_widget.setBackground("#0f172a")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "In-Phase (I)")
        self.plot_widget.setLabel("left", "Quadrature (Q)")
        self.plot_widget.setAspectLocked(True)

        self.axis_h = pg.InfiniteLine(angle=0, pen=pg.mkPen("#475569", width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.axis_v = pg.InfiniteLine(angle=90, pen=pg.mkPen("#475569", width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.axis_h)
        self.plot_widget.addItem(self.axis_v)

        self.scatter_item = pg.ScatterPlotItem(
            size=6,
            pen=None,
            brush=pg.mkBrush(56, 189, 248, 160),
            pxMode=True,
        )
        self.plot_widget.addItem(self.scatter_item)
        left_layout.addWidget(self.plot_widget)
        splitter.addWidget(left_panel)

        # Right: Demodulation status & Bitstream preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)

        group_sync = QGroupBox("Synchronization Status")
        group_sync.setStyleSheet(
            "QGroupBox { color: #f8fafc; font-weight: bold; border: 1px solid #334155; border-radius: 4px; padding-top: 15px; margin-top: 6px; }"
        )
        sync_layout = QVBoxLayout(group_sync)

        self.lbl_carrier_lock = QLabel("Carrier Lock: UNLOCKED")
        self.lbl_carrier_lock.setStyleSheet("color: #94a3b8; font-size: 12px;")
        sync_layout.addWidget(self.lbl_carrier_lock)

        self.lbl_timing_lock = QLabel("Clock Timing: UNLOCKED")
        self.lbl_timing_lock.setStyleSheet("color: #94a3b8; font-size: 12px;")
        sync_layout.addWidget(self.lbl_timing_lock)

        self.lbl_freq_err = QLabel("Residual Carrier Offset: 0.0 Hz")
        self.lbl_freq_err.setStyleSheet("color: #94a3b8; font-size: 12px;")
        sync_layout.addWidget(self.lbl_freq_err)

        right_layout.addWidget(group_sync)

        # Bitstream Previews
        group_bits = QGroupBox("Demodulated Bitstream")
        group_bits.setStyleSheet(
            "QGroupBox { color: #f8fafc; font-weight: bold; border: 1px solid #334155; border-radius: 4px; padding-top: 15px; margin-top: 6px; }"
        )
        bits_layout = QVBoxLayout(group_bits)

        self.lbl_bits_count = QLabel("Total Recovered Bits: 0")
        self.lbl_bits_count.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        bits_layout.addWidget(self.lbl_bits_count)

        lbl_hex = QLabel("Hexadecimal View:")
        lbl_hex.setStyleSheet("color: #94a3b8; font-size: 11px;")
        bits_layout.addWidget(lbl_hex)

        self.txt_hex = QPlainTextEdit()
        self.txt_hex.setReadOnly(True)
        self.txt_hex.setFixedHeight(90)
        self.txt_hex.setStyleSheet("background-color: #0f172a; color: #38bdf8; font-family: monospace; font-size: 12px; border: 1px solid #334155;")
        bits_layout.addWidget(self.txt_hex)

        lbl_bin = QLabel("Binary Stream View:")
        lbl_bin.setStyleSheet("color: #94a3b8; font-size: 11px;")
        bits_layout.addWidget(lbl_bin)

        self.txt_bin = QPlainTextEdit()
        self.txt_bin.setReadOnly(True)
        self.txt_bin.setStyleSheet("background-color: #0f172a; color: #a5f3fc; font-family: monospace; font-size: 11px; border: 1px solid #334155;")
        bits_layout.addWidget(self.txt_bin)

        right_layout.addWidget(group_bits)
        splitter.addWidget(right_panel)

        splitter.setSizes([850, 450])
        layout.addWidget(splitter)

    def set_modulation_result(
        self,
        result: ModulationResult,
        demod_result: DemodulationResult | None = None,
    ) -> None:
        """Update constellation points, sync indicators, and recovered bits."""
        if result.constellation_points is None or len(result.constellation_points) == 0:
            self.clear()
            return

        points = result.constellation_points
        i_pts = np.real(points)
        q_pts = np.imag(points)

        self.scatter_item.setData(x=i_pts, y=q_pts)
        self.plot_widget.setRange(xRange=[-2.2, 2.2], yRange=[-2.2, 2.2])

        evm_str = result.evm_percent.formatted_value
        self.lbl_info.setText(
            f"Modulation: <strong style='color:#38bdf8;'>{result.scheme}</strong> ({result.confidence_score * 100:.1f}%) | EVM: <strong>{evm_str}</strong>"
        )

        if demod_result is not None and len(demod_result.bits) > 0:
            bits = demod_result.bits
            self.lbl_bits_count.setText(f"Total Recovered Bits: {len(bits):,}")

            # Carrier lock
            c_color = "#22c55e" if demod_result.carrier_locked else "#94a3b8"
            c_text = "LOCKED" if demod_result.carrier_locked else "UNLOCKED / OPEN"
            self.lbl_carrier_lock.setText(f"Carrier Lock: <strong style='color:{c_color};'>{c_text}</strong>")

            # Timing lock
            t_color = "#22c55e" if demod_result.timing_locked else "#94a3b8"
            t_text = "LOCKED" if demod_result.timing_locked else "CONVERGED"
            self.lbl_timing_lock.setText(f"Clock Timing: <strong style='color:{t_color};'>{t_text}</strong>")

            self.lbl_freq_err.setText(f"Residual Carrier Offset: {demod_result.carrier_frequency_error_hz:+.1f} Hz")

            # Binary preview (first 512 bits in blocks of 8)
            bin_slice = bits[:512]
            bin_str = "".join(str(b) for b in bin_slice)
            bin_spaced = " ".join(bin_str[i : i + 8] for i in range(0, len(bin_str), 8))
            self.txt_bin.setPlainText(bin_spaced + (" ..." if len(bits) > 512 else ""))

            # Hex preview
            # Pack bits to bytes
            pad_len = (8 - (len(bits) % 8)) % 8
            padded = np.pad(bits, (0, pad_len), constant_values=0)
            byte_arr = np.packbits(padded)
            hex_slice = byte_arr[:64]
            hex_str = " ".join(f"{b:02X}" for b in hex_slice)
            self.txt_hex.setPlainText(hex_str + (" ..." if len(byte_arr) > 64 else ""))

    def clear(self) -> None:
        self.scatter_item.clear()
        self.lbl_info.setText("Modulation: None | EVM: --")
        self.lbl_carrier_lock.setText("Carrier Lock: UNLOCKED")
        self.lbl_timing_lock.setText("Clock Timing: UNLOCKED")
        self.lbl_freq_err.setText("Residual Carrier Offset: 0.0 Hz")
        self.lbl_bits_count.setText("Total Recovered Bits: 0")
        self.txt_hex.clear()
        self.txt_bin.clear()
