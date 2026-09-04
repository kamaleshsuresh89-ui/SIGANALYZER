# SIGANALYZER — Project Status & Development Tracker

**Project Objective**: Automated model for analysis of `.IQ` and `.wav` files along with extensive, scientifically rigorous signal parameter extraction (SIH Problem Statement).  
**Core Principle**: *"Extract everything that can be technically justified — never invent data."*

---

## 1. Project Overview & Vital Statistics
- **Platform**: Python 3.12 (CPython Apple Silicon / macOS + cross-platform design)
- **GUI Engine**: PySide6 (Qt 6) + pyqtgraph (OpenGL accelerated)
- **DSP / Compute**: NumPy, SciPy, Numba JIT (Costas PLL, Gardner TED), scikit-learn, ONNX Runtime
- **Storage**: Local SQLite (`.siganalyzer/history.db`)
- **Reporting**: Standalone offline HTML & JSON exports (zero network dependencies)
- **Current Test Suite**: 66 passing unit & integration tests (2.52s run time)
- **Code Coverage**: 85% across 5,641 statements
- **Parameter Registry**: 250 verified extractable parameters across all 16 categories
- **Last Updated**: 2026-09-04 12:54 IST

---

## 2. Completed Milestones (Phases 1–10)
- [x] **Subsystem 1: Core Types & Provenance Tracking (`siganalyzer.common`)**
  - Strict confidence taxonomy: `DIRECT`, `MEASURED`, `ESTIMATED`, `INFERRED`, `UNKNOWN`.
  - Quantified containers with metadata provenance (`Measurement[T]`).
  - Error hierarchy (`FileFormatError`, `CorruptedFileError`, `InvalidSampleError`).
  - Thread-safe local file logging with fallback.
- [x] **Subsystem 2: File I/O & Format Identification (`siganalyzer.io`)**
  - RIFF/WAV, RF64, SigMF metadata detection.
  - Statistical probing for raw binary IQ formats (Float32, Int16, RTL-SDR Uint8 with 127.5 bias, HackRF Int8).
  - Memory-mapped zero-copy streaming (`numpy.memmap`).
  - Numerical integrity validator (NaN/Inf, clipping, excessive silence).
- [x] **Subsystem 3: Signal Conditioning & Spectral DSP (`siganalyzer.dsp`)**
  - DC offset subtraction and 1-pole high-pass IIR filter.
  - Blind IQ imbalance estimation ($\alpha$ in dB, $\phi$ in deg) & Gram-Schmidt orthogonalization.
  - Welch PSD with Blackman-Harris windowing & percentile noise floor tracking.
  - STFT Spectrogram generation with adaptive LOD downsampling.
  - Adaptive CFAR burst & region energy detection.
- [x] **Subsystem 4: Parameter Estimation & Modulation Classification (`siganalyzer.estimation`, `siganalyzer.classification`)**
  - Carrier frequency (midpoint of main lobe), 99% OBW, -3dB BW, SNR estimation.
  - Symbol rate estimation via cyclostationary magnitude squaring non-linearity ($|s[n]|^2$).
  - Higher-order cumulants ($C_{20}, C_{21}, C_{40}, C_{41}, C_{42}, C_{60}, C_{63}$).
  - Hybrid classification for BPSK, QPSK, 8PSK, 2FSK, 4FSK, 16-QAM, 64-QAM, ASK/OOK.
- [x] **Subsystem 5: Synchronization & Demodulation (`siganalyzer.demodulation`)**
  - Numba JIT-compiled Costas loop (carrier recovery) & Gardner timing error detector (clock recovery).
  - Constellation clustering and RMS / Peak EVM calculation.
  - PSK, FSK, ASK, and QAM symbol slicers with Gray mapping.
- [x] **Subsystem 6: Coding & Bitstream Extraction (`siganalyzer.coding`, `siganalyzer.bitstream`)**
  - Block/matrix and convolutional de-interleaving with autocorrelation depth discovery.
  - Rate 1/2 $K=7$ NASA/CCSDS Viterbi convolutional decoder (hard & soft decision).
  - Pure Galois Field $GF(2^8)$ Reed-Solomon codec with Berlekamp-Massey and Forney algorithms.
  - Multi-standard CRC engine (CRC-8, CRC-16-CCITT, CRC-16-IBM, CRC-32).
  - Fast bipolar preamble/sync-word cross-correlation search (Barker, 0xAA, CCSDS, custom hex) with Hamming tolerance.
  - Frame length periodicity detection and protocol frame boundary partitioner.
  - Bitstream Shannon entropy, transition density, 0/1 bias, hex & binary formatters.
- [x] **Subsystem 7: Storage, Reporting & Packaging (`siganalyzer.storage`, `siganalyzer.reporting`, `scripts`)**
  - Embedded SQLite session persistence and presets table.
  - Offline self-contained HTML report with dark styling and JSON exporter.
  - PyInstaller `.spec` and standalone build script.
- [x] **Subsystem 8: Desktop GUI (`siganalyzer.ui`)**
  - PySide6 dark-themed window with drag-and-drop loading and async `QThreadPool` worker.
  - 5 Main Tabs: Signal Overview (Time, Spectrum, Spectrogram), Constellation & EVM, Bitstream & Protocol Inspector, Parameter Explorer (A–P), Analysis Report.
  - Findings & Parameters Dock and Detected Bursts Dock.
- [x] **Subsystem 9: Central Parameter Registry & 16-Category Extractors (`siganalyzer.parameters`)**
  - Modular, extensible plugin architecture: `BaseParameterExtractor`, `ParameterRegistry`, `ExtractionEngine`, `SignalProfile`.
  - All 16 formal SIH categories fully implemented (250 total parameters):
    - Category A: File & Capture Information (24 parameters)
    - Category B: Time-Domain Parameters (31 parameters)
    - Category C: Frequency-Domain Parameters (25 parameters)
    - Category D: Power & Signal Quality (14 parameters)
    - Category E: Modulation Analysis (15 parameters)
    - Category F: Constellation & Modulation Quality (17 parameters)
    - Category G: Synchronization Parameters (12 parameters)
    - Category H: Burst / Signal Segmentation (15 parameters)
    - Category I: Digital Communication Parameters (11 parameters)
    - Category J: FEC / Coding Analysis (15 parameters)
    - Category K: Bitstream Analysis (14 parameters)
    - Category L: Header / Payload / Frame Analysis (13 parameters)
    - Category M: Noise & Interference Analysis (10 parameters)
    - Category N: Statistical & Higher-Order Features (13 parameters)
    - Category O: Hardware / ADC / Capture Quality (11 parameters)
    - Category P: Signal Anomaly Detection (10 parameters)
  - Strict classification: Every parameter strictly categorized as `DIRECT`, `MEASURED`, `ESTIMATED`, `INFERRED`, or `UNKNOWN`.
  - Fault isolation: Extractors safely catch mathematical errors, returning `UNKNOWN` with explanatory notes without aborting the master pipeline.
- [x] **Subsystem 10: Interactive Parameter Explorer UI (`siganalyzer.ui.views.parameter_explorer`)**
  - Virtualized `QTableView` with `ParameterTableModel` for fast scrolling across 250+ parameters.
  - Real-time text search filter across names, IDs, units, sources, formulas, and values.
  - Dropdown filtering by Category (All Categories, Cat A to P) and Status (`DIRECT`, `MEASURED`, `ESTIMATED`, `INFERRED`, `UNKNOWN`).
  - Side drawer inspector showing formatted value, status badge, confidence rating, derivation formula, and validity notes.
  - Export utilities for CSV and JSON.
  - Integrated as dedicated Tab 4 in `MainWindow`.

---

## 3. Current Architecture Decisions & Foundation
1. **Offline Isolation**: All signal processing, inference, and report generation execute locally with zero external network calls.
2. **Strict Provenance**: Inferred/estimated values are never presented as direct truths.
3. **Decoupled Architecture**: All core DSP and parameter estimation engines are pure Python/NumPy/SciPy classes with no Qt dependencies, ensuring portability to headless CLI and future cross-platform targets (Windows, Linux, Android/iOS).
4. **Memory Scalability**: Multi-gigabyte recordings are handled via `np.memmap` slices and downsampled overview pyramids.
5. **Centralized Registry**: Easily add new parameter extraction plugins by subclassing `BaseParameterExtractor` and annotating with `@ParameterRegistry.register`.

---

## 4. Current Status: Production Ready & Fully Verified
- **All 66 automated tests passing in 2.52s**.
- **85% Code Coverage across 5,641 lines of Python code**.
- **Verified on synthetic QPSK, BPSK, 2FSK, 4FSK, 16-QAM, ASK, and raw IQ recordings**.
