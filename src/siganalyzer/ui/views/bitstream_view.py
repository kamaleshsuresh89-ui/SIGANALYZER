"""Interactive Bitstream, Protocol Framing, and FEC Analysis View."""

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from siganalyzer.bitstream.analyzer import BitstreamAnalyzer
from siganalyzer.bitstream.correlation import SyncWordCorrelator, SyncWordMatch
from siganalyzer.bitstream.formatter import BitstreamFormatter
from siganalyzer.bitstream.framing import BitstreamFrame, ProtocolFramer
from siganalyzer.coding.fec_manager import FecManager
from siganalyzer.common.types import BitstreamAnalysis, DemodulationResult, FecResult


class BitstreamView(QWidget):
    """Full-featured interactive bitstream inspector, sync-word correlator, and protocol framer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.raw_bits: np.ndarray = np.empty(0, dtype=np.uint8)
        self.current_bits: np.ndarray = np.empty(0, dtype=np.uint8)
        self.symbol_rate: float | None = None
        self.detected_matches: list[SyncWordMatch] = []
        self.detected_frames: list[BitstreamFrame] = []

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Top Metrics Cards
        metrics_bar = QHBoxLayout()
        metrics_bar.setSpacing(10)

        self.card_bits = self._create_metric_card("Total Recovered Bits", "0")
        self.card_rate = self._create_metric_card("Bit Rate", "--")
        self.card_entropy = self._create_metric_card("Shannon Entropy", "--")
        self.card_trans = self._create_metric_card("Transition Density", "--")
        self.card_bias = self._create_metric_card("0/1 Ratio", "--")

        metrics_bar.addWidget(self.card_bits)
        metrics_bar.addWidget(self.card_rate)
        metrics_bar.addWidget(self.card_entropy)
        metrics_bar.addWidget(self.card_trans)
        metrics_bar.addWidget(self.card_bias)
        main_layout.addLayout(metrics_bar)

        # 2. Sync Word & Framing Toolbar
        tools_group = QGroupBox("Protocol Sync-Word Search & Framing Engine")
        tools_group.setStyleSheet(
            "QGroupBox { color: #f8fafc; font-weight: bold; border: 1px solid #334155; border-radius: 4px; padding-top: 14px; margin-top: 4px; }"
        )
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.setContentsMargins(8, 4, 8, 4)
        tools_layout.setSpacing(8)

        lbl_preset = QLabel("Pattern Preset:")
        lbl_preset.setStyleSheet("color: #94a3b8; font-size: 12px;")
        tools_layout.addWidget(lbl_preset)

        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "All Standard Library Presets",
            "Barker-11 (11100010010)",
            "Barker-13 (1111100110101)",
            "Preamble 0xAA (16-bit)",
            "Preamble 0xAA (32-bit)",
            "CCSDS ASM (0x1ACFFC1D)",
            "GSM TSC0",
            "Custom Hex Pattern...",
        ])
        self.combo_presets.setStyleSheet(
            "QComboBox { background-color: #1e293b; color: #f8fafc; padding: 4px 8px; border: 1px solid #475569; border-radius: 4px; }"
            "QComboBox QAbstractItemView { background-color: #1e293b; color: #f8fafc; selection-background-color: #38bdf8; }"
        )
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        tools_layout.addWidget(self.combo_presets)

        self.txt_custom_hex = QLineEdit()
        self.txt_custom_hex.setPlaceholderText("e.g. 1ACFFC1D or AA55")
        self.txt_custom_hex.setEnabled(False)
        self.txt_custom_hex.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #38bdf8; font-family: monospace; padding: 4px 8px; border: 1px solid #475569; border-radius: 4px; }"
        )
        tools_layout.addWidget(self.txt_custom_hex)

        lbl_tol = QLabel("Hamming Tol:")
        lbl_tol.setStyleSheet("color: #94a3b8; font-size: 12px;")
        tools_layout.addWidget(lbl_tol)

        self.spin_tol = QSpinBox()
        self.spin_tol.setRange(0, 4)
        self.spin_tol.setValue(1)
        self.spin_tol.setToolTip("Allowed bit flips / errors in sync word match")
        self.spin_tol.setStyleSheet(
            "QSpinBox { background-color: #1e293b; color: #f8fafc; padding: 3px 6px; border: 1px solid #475569; border-radius: 4px; }"
        )
        tools_layout.addWidget(self.spin_tol)

        self.btn_search_sync = QPushButton("🔍 Find Sync Words")
        self.btn_search_sync.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_search_sync.clicked.connect(self._on_find_sync_clicked)
        tools_layout.addWidget(self.btn_search_sync)

        self.btn_frame = QPushButton("📐 Frame Protocol")
        self.btn_frame.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #047857; }"
        )
        self.btn_frame.clicked.connect(self._on_frame_protocol_clicked)
        tools_layout.addWidget(self.btn_frame)

        # FEC quick action
        self.btn_viterbi = QPushButton("⚡ Try Viterbi K=7")
        self.btn_viterbi.setStyleSheet(
            "QPushButton { background-color: #7c3aed; color: white; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #6d28d9; }"
        )
        self.btn_viterbi.clicked.connect(self._on_viterbi_clicked)
        tools_layout.addWidget(self.btn_viterbi)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setStyleSheet(
            "QPushButton { background-color: #475569; color: white; font-weight: bold; padding: 5px 10px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #334155; }"
        )
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        tools_layout.addWidget(self.btn_reset)

        main_layout.addWidget(tools_group)

        # 3. Main Data Splitter: Tables (top) & Dumps (bottom)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top Tabs: Frames & Sync Matches
        self.table_tabs = QTabWidget()
        self.table_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #334155; background-color: #0f172a; }"
            "QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 5px 12px; font-weight: bold; }"
            "QTabBar::tab:selected { background-color: #0f172a; color: #38bdf8; border: 1px solid #334155; border-bottom: none; }"
        )

        # Frames Table
        self.table_frames = QTableWidget()
        self.table_frames.setColumnCount(7)
        self.table_frames.setHorizontalHeaderLabels([
            "Frame #", "Start Bit", "Length (bits)", "Sync Marker", "CRC Scheme", "CRC Status", "Payload Hex"
        ])
        self.table_frames.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_frames.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.table_frames.setStyleSheet(
            "QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; font-size: 12px; border: none; }"
            "QHeaderView::section { background-color: #1e293b; color: #94a3b8; padding: 4px; font-weight: bold; border: 1px solid #334155; }"
        )
        self.table_tabs.addTab(self.table_frames, "📦 Partitioned Protocol Frames (0)")

        # Sync Matches Table
        self.table_sync = QTableWidget()
        self.table_sync.setColumnCount(6)
        self.table_sync.setHorizontalHeaderLabels([
            "Sync Pattern", "Bit Offset", "Hex Pattern", "Hamming Dist", "Correlation", "Polarity"
        ])
        self.table_sync.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_sync.setStyleSheet(
            "QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; font-size: 12px; border: none; }"
            "QHeaderView::section { background-color: #1e293b; color: #94a3b8; padding: 4px; font-weight: bold; border: 1px solid #334155; }"
        )
        self.table_tabs.addTab(self.table_sync, "🎯 Detected Preambles & Sync Words (0)")

        splitter.addWidget(self.table_tabs)

        # Bottom Tabs: Hex Dump & Binary Dump
        self.dump_tabs = QTabWidget()
        self.dump_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #334155; background-color: #0f172a; }"
            "QTabBar::tab { background-color: #1e293b; color: #94a3b8; padding: 5px 12px; font-weight: bold; }"
            "QTabBar::tab:selected { background-color: #0f172a; color: #38bdf8; border: 1px solid #334155; border-bottom: none; }"
        )

        mono_font = QFont("Courier New", 11)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)

        self.txt_hexdump = QPlainTextEdit()
        self.txt_hexdump.setReadOnly(True)
        self.txt_hexdump.setFont(mono_font)
        self.txt_hexdump.setStyleSheet("background-color: #0f172a; color: #38bdf8; border: none; padding: 8px;")
        self.dump_tabs.addTab(self.txt_hexdump, "🔢 Hexadecimal & ASCII Dump")

        self.txt_bindump = QPlainTextEdit()
        self.txt_bindump.setReadOnly(True)
        self.txt_bindump.setFont(mono_font)
        self.txt_bindump.setStyleSheet("background-color: #0f172a; color: #a7f3d0; border: none; padding: 8px;")
        self.dump_tabs.addTab(self.txt_bindump, "0101 Grouped Binary Dump")

        splitter.addWidget(self.dump_tabs)
        splitter.setSizes([260, 340])

        main_layout.addWidget(splitter)

    def _create_metric_card(self, title: str, initial_val: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 6px 10px;"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        lbl_t = QLabel(title.upper())
        lbl_t.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: bold;")
        lay.addWidget(lbl_t)

        lbl_v = QLabel(initial_val)
        lbl_v.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        lay.addWidget(lbl_v)
        card.value_label = lbl_v  # type: ignore[attr-defined]
        return card

    @Slot(int)
    def _on_preset_changed(self, idx: int) -> None:
        is_custom = "Custom" in self.combo_presets.currentText()
        self.txt_custom_hex.setEnabled(is_custom)

    def populate(
        self,
        demod: DemodulationResult | None,
        bitstream_analysis: BitstreamAnalysis | None = None,
        symbol_rate: float | None = None,
    ) -> None:
        """Populate the inspector with demodulated bits and analysis metrics."""
        self.symbol_rate = symbol_rate
        if demod is None or len(demod.bits) == 0:
            self.clear()
            return

        self.raw_bits = np.asarray(demod.bits, dtype=np.uint8)
        self.current_bits = np.copy(self.raw_bits)

        self._refresh_display(bitstream_analysis)

    def _refresh_display(self, analysis: BitstreamAnalysis | None = None) -> None:
        bits = self.current_bits
        if len(bits) == 0:
            self.clear()
            return

        # Update metrics
        if analysis is None:
            analysis = BitstreamAnalyzer.analyze(bits, symbol_rate_hz=self.symbol_rate)

        self.card_bits.value_label.setText(f"{len(bits):,}")  # type: ignore
        self.card_rate.value_label.setText(analysis.bit_rate_est.formatted_value)  # type: ignore
        self.card_entropy.value_label.setText(f"{analysis.entropy_per_bit:.4f} b/b")  # type: ignore
        self.card_trans.value_label.setText(f"{analysis.transition_density:.3f}")  # type: ignore

        p1 = float(np.mean(bits))
        self.card_bias.value_label.setText(f"{1.0 - p1:.1%} / {p1:.1%}")  # type: ignore

        # Hex and binary dumps
        self.txt_hexdump.setPlainText(BitstreamFormatter.format_hexdump(bits, max_rows=1500))
        self.txt_bindump.setPlainText(BitstreamFormatter.format_binary_dump(bits, max_rows=1000))

        # Auto search standard sync words
        self._on_find_sync_clicked()

    @Slot()
    def _on_find_sync_clicked(self) -> None:
        """Run preamble and sync-word correlation search."""
        if len(self.current_bits) == 0:
            return

        choice = self.combo_presets.currentText()
        tol = self.spin_tol.value()

        if "All Standard" in choice:
            self.detected_matches = SyncWordCorrelator.scan_standard_presets(
                self.current_bits, max_hamming_distance=tol
            )
        elif "Custom" in choice:
            hex_val = self.txt_custom_hex.text().strip()
            if not hex_val:
                return
            try:
                pat = SyncWordCorrelator.pattern_from_hex(hex_val)
                self.detected_matches = SyncWordCorrelator.search_pattern(
                    self.current_bits, pat, name="Custom", max_hamming_distance=tol
                )
            except Exception as e:
                QMessageBox.warning(self, "Invalid Pattern", f"Error parsing hex pattern: {e}")
                return
        else:
            preset_key = choice.split(" (")[0]
            for key in SyncWordCorrelator.PRESETS:
                if key.startswith(preset_key):
                    pat = SyncWordCorrelator.PRESETS[key]
                    self.detected_matches = SyncWordCorrelator.search_pattern(
                        self.current_bits, pat, name=key, max_hamming_distance=tol
                    )
                    break

        # Populate sync table
        self.table_sync.setRowCount(len(self.detected_matches))
        for r, m in enumerate(self.detected_matches):
            self.table_sync.setItem(r, 0, QTableWidgetItem(m.name))
            self.table_sync.setItem(r, 1, QTableWidgetItem(f"{m.bit_index:,}"))
            self.table_sync.setItem(r, 2, QTableWidgetItem(m.pattern_hex))
            self.table_sync.setItem(r, 3, QTableWidgetItem(f"{m.hamming_distance} bits"))
            self.table_sync.setItem(r, 4, QTableWidgetItem(f"{m.correlation_score * 100:.1f}%"))
            pol_item = QTableWidgetItem("Inverted (180°)" if m.inverted else "Normal")
            pol_item.setForeground(Qt.GlobalColor.yellow if m.inverted else Qt.GlobalColor.green)
            self.table_sync.setItem(r, 5, pol_item)

        self.table_tabs.setTabText(1, f"🎯 Detected Preambles & Sync Words ({len(self.detected_matches)})")

        # Auto frame
        self._on_frame_protocol_clicked()

    @Slot()
    def _on_frame_protocol_clicked(self) -> None:
        """Partition bitstream into frames based on detected sync markers."""
        if not self.detected_matches or len(self.current_bits) == 0:
            self.table_frames.setRowCount(0)
            self.table_tabs.setTabText(0, "📦 Partitioned Protocol Frames (0)")
            return

        self.detected_frames = ProtocolFramer.extract_frames(
            self.current_bits,
            self.detected_matches,
        )

        self.table_frames.setRowCount(len(self.detected_frames))
        for r, f in enumerate(self.detected_frames):
            self.table_frames.setItem(r, 0, QTableWidgetItem(f"Frame #{f.frame_index}"))
            self.table_frames.setItem(r, 1, QTableWidgetItem(f"{f.start_bit:,}"))
            self.table_frames.setItem(r, 2, QTableWidgetItem(f"{f.length_bits}"))
            self.table_frames.setItem(r, 3, QTableWidgetItem(f.sync_word))
            self.table_frames.setItem(r, 4, QTableWidgetItem(f.crc_algorithm))

            crc_item = QTableWidgetItem("VALID" if f.crc_valid else "INVALID / NONE")
            crc_item.setForeground(Qt.GlobalColor.green if f.crc_valid else Qt.GlobalColor.gray)
            self.table_frames.setItem(r, 5, crc_item)

            preview = f.payload_hex[:64] + ("..." if len(f.payload_hex) > 64 else "")
            self.table_frames.setItem(r, 6, QTableWidgetItem(preview))

        self.table_tabs.setTabText(0, f"📦 Partitioned Protocol Frames ({len(self.detected_frames)})")

    @Slot()
    def _on_viterbi_clicked(self) -> None:
        """Apply rate 1/2 Viterbi decoder to the current bitstream."""
        if len(self.current_bits) < 16:
            QMessageBox.information(self, "Viterbi Decoder", "Not enough bits to decode.")
            return

        decoded, metric = FecManager.decode_viterbi(self.current_bits)
        metric_ratio = metric / (len(self.current_bits) or 1)

        reply = QMessageBox.question(
            self,
            "Viterbi Decoding Result",
            f"Viterbi decoding completed.\n"
            f"Output length: {len(decoded):,} bits (half rate)\n"
            f"Total path metric: {metric} ({metric_ratio:.2%} error rate)\n\n"
            f"Apply decoded bitstream to current view?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.current_bits = decoded
            self._refresh_display()

    @Slot()
    def _on_reset_clicked(self) -> None:
        """Reset view back to original raw demodulated bits."""
        if len(self.raw_bits) > 0:
            self.current_bits = np.copy(self.raw_bits)
            self._refresh_display()

    def clear(self) -> None:
        """Clear all tables and views."""
        self.raw_bits = np.empty(0, dtype=np.uint8)
        self.current_bits = np.empty(0, dtype=np.uint8)
        self.card_bits.value_label.setText("0")  # type: ignore
        self.card_rate.value_label.setText("--")  # type: ignore
        self.card_entropy.value_label.setText("--")  # type: ignore
        self.card_trans.value_label.setText("--")  # type: ignore
        self.card_bias.value_label.setText("--")  # type: ignore
        self.txt_hexdump.clear()
        self.txt_bindump.clear()
        self.table_sync.setRowCount(0)
        self.table_frames.setRowCount(0)
        self.table_tabs.setTabText(0, "📦 Partitioned Protocol Frames (0)")
        self.table_tabs.setTabText(1, "🎯 Detected Preambles & Sync Words (0)")
