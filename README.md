# SIGANALYZER

**SIGANALYZER** is a production-quality, standalone, 100% offline desktop application for automated analysis of recorded RF and digital communication signals (WAV, IQ).

---

## Features

- **100% Offline-First**: Zero cloud dependencies, zero external network requests.
- **Automatic Format Detection**: Automatically distinguishes standard WAV (PCM16, Float32, PCM24, PCM32), RF64 large-file containers, and raw binary IQ files (Float32, Int16, RTL-SDR Uint8, HackRF Int8).
- **Large-File Performance**: Memory-mapped (`numpy.memmap`) chunked streaming ensures files of any size can be inspected with near-zero RAM overhead.
- **Interactive Visualizations**:
  - Time-Domain: I and Q waveforms with peak-preserving downsampling.
  - Frequency-Domain: Welch Power Spectral Density with noise floor baseline and spectral peak markers.
  - Time-Frequency: 2D Spectrogram / Waterfall with self-contained Viridis colormap.
  - Constellation Diagram: I/Q symbol scatter plot with EVM estimation.
- **Parameter Estimation**:
  - Carrier Frequency & Center Frequency Offset.
  - 99% Occupied Bandwidth (OBW) and -3dB Bandwidth.
  - In-band Signal-to-Noise Ratio (SNR).
  - Symbol Rate and preliminary Bit Rate.
  - Higher-Order Cumulants ($C_{20}, C_{21}, C_{40}, C_{41}, C_{42}, C_{60}, C_{63}$).
- **Automatic Modulation Classification**: Hybrid rule-based and cumulant classifier supporting BPSK, QPSK, 8PSK, 2FSK, 4FSK, 16-QAM, 64-QAM, and OOK/ASK with confidence scores and mathematical supporting evidence.
- **Signal Impairment & Anomaly Detection**: Detects ADC full-scale clipping, signal dropouts, and sudden inter-segment power steps.
- **Offline Reporting**: One-click generation of self-contained, print-ready HTML reports and machine-readable JSON files.

---

## Quickstart

### Launch Desktop Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Launch SIGANALYZER
python -m siganalyzer
```

Or pass a recorded file directly on launch:

```bash
python -m siganalyzer /path/to/signal.wav
```

### Run Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/
```
