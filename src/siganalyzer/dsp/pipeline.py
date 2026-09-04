"""Master Signal Analysis Pipeline orchestrator."""

from datetime import datetime
from pathlib import Path
import time
from typing import Callable
import numpy as np

from siganalyzer.anomaly.detector import AnomalyDetector
from siganalyzer.classification.classifier import ModulationClassifier
from siganalyzer.common.logger import logger
from siganalyzer.dsp.detection import SignalDetector
from siganalyzer.dsp.preprocessing import SignalPreprocessor
from siganalyzer.dsp.spectrogram import SpectrogramData, SpectrogramGenerator
from siganalyzer.dsp.spectrum import SpectralAnalyzer, SpectrumResult
from siganalyzer.estimation.estimator import ParameterEstimateResult, ParameterEstimator
from siganalyzer.io.metadata_extractor import SignalLoader
from siganalyzer.io.readers.base import BaseSignalReader
from siganalyzer.reporting.generator import CompleteAnalysisReport


class AnalysisPipeline:
    """End-to-end automated signal analysis pipeline coordinator."""

    def __init__(
        self,
        progress_callback: Callable[[int, str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.progress_callback = progress_callback or (lambda pct, msg: None)
        self.is_cancelled = is_cancelled or (lambda: False)

    def analyze_file(
        self,
        file_path: str | Path,
        sample_rate_override: float | None = None,
        center_freq_override: float | None = None,
        max_samples_to_load: int = 1_000_000,
    ) -> tuple[CompleteAnalysisReport, SpectrogramData, np.ndarray]:
        """Execute complete analysis pipeline on a signal recording.

        Returns:
            Tuple of (CompleteAnalysisReport, SpectrogramData, raw_samples_slice).
        """
        start_time = time.perf_counter()
        logger.info(f"Starting analysis on file: {file_path}")

        # 1. File loading & metadata extraction
        self._check_cancelled()
        self.progress_callback(10, "Detecting file format and reading headers...")
        reader, detection, validation = SignalLoader.load(
            file_path,
            sample_rate_override=sample_rate_override,
            center_freq_override=center_freq_override,
        )

        sample_rate = reader.metadata.sample_rate.value or 1_000_000.0

        # Read samples slice for analysis (capped for responsiveness)
        total_available = reader.total_samples
        to_read = min(max_samples_to_load, total_available)
        self._check_cancelled()
        self.progress_callback(25, f"Reading {to_read:,} samples from disk...")
        samples = reader.read_samples(0, to_read)

        # 2. Preprocessing & Conditioning
        self._check_cancelled()
        self.progress_callback(40, "Conditioning signal (DC offset & IQ imbalance)...")
        dc_removed = SignalPreprocessor.remove_dc_offset(samples)
        conditioned, iq_metrics = SignalPreprocessor.correct_iq_imbalance(dc_removed)

        # 3. Spectral Analysis
        self._check_cancelled()
        self.progress_callback(55, "Computing Welch PSD and spectral peaks...")
        spectrum_res = SpectralAnalyzer.analyze_spectrum(conditioned, sample_rate, nfft=2048)

        # 4. Spectrogram generation
        self._check_cancelled()
        self.progress_callback(70, "Generating 2D time-frequency spectrogram...")
        spectrogram_data = SpectrogramGenerator.generate(
            conditioned, sample_rate, nfft=1024, overlap=512, max_time_bins=800
        )

        # 5. Signal Detection & Burst Segmentation
        self._check_cancelled()
        self.progress_callback(80, "Detecting active signal regions and bursts...")
        regions = SignalDetector.detect_regions(conditioned, sample_rate)

        # 6. Parameter Estimation on primary active region
        self._check_cancelled()
        self.progress_callback(88, "Estimating carrier frequency, bandwidth, SNR, and symbol rate...")
        primary_slice = conditioned
        if regions and regions[0].is_signal and (regions[0].end_sample - regions[0].start_sample) > 512:
            r0 = regions[0]
            primary_slice = conditioned[r0.start_sample : r0.end_sample]

        params = ParameterEstimator.estimate_all(
            primary_slice, sample_rate, reader.metadata.center_frequency.value
        )

        # 7. Modulation Classification
        self._check_cancelled()
        self.progress_callback(90, "Classifying modulation scheme and cumulants...")
        snr_val = params.snr_db.value or 20.0
        modulation = ModulationClassifier.classify(
            primary_slice, sample_rate, snr_db=snr_val, cumulants=params.cumulants
        )

        # 8. Carrier & Clock Synchronization / Demodulation
        self._check_cancelled()
        self.progress_callback(95, "Synchronizing carrier and clock, extracting symbols and bits...")
        from siganalyzer.demodulation.manager import DemodulationManager

        carrier_offset = (
            params.carrier_freq_hz.value
            if reader.metadata.center_frequency.value is None
            else (params.carrier_freq_hz.value - reader.metadata.center_frequency.value)
        ) or 0.0

        demod_res, const_metrics = DemodulationManager.demodulate_signal(
            primary_slice,
            sample_rate,
            detected_scheme=modulation.scheme,
            symbol_rate_hz=params.symbol_rate_sps.value,
            carrier_freq_offset_hz=carrier_offset,
        )

        # Update constellation scatter points with synchronized recovered symbols
        if demod_res.symbols is not None and len(demod_res.symbols) > 0:
            step_sym = max(1, len(demod_res.symbols) // 2000)
            modulation.constellation_points = demod_res.symbols[::step_sym]
            if const_metrics.evm_rms_percent.value is not None:
                modulation.evm_percent = const_metrics.evm_rms_percent

        # 9. Bitstream Extraction, Sync-Word Correlation, and FEC Probing
        self._check_cancelled()
        self.progress_callback(97, "Analyzing bitstream entropy, preambles, and FEC coding...")
        from siganalyzer.bitstream.analyzer import BitstreamAnalyzer
        from siganalyzer.bitstream.correlation import SyncWordCorrelator
        from siganalyzer.bitstream.framing import ProtocolFramer
        from siganalyzer.coding.fec_manager import FecManager
        from siganalyzer.storage.db import DatabaseManager

        raw_bits = demod_res.bits
        sync_matches = SyncWordCorrelator.scan_standard_presets(raw_bits, max_hamming_distance=1)
        sync_dicts = [
            {
                "name": m.name,
                "bit_index": m.bit_index,
                "pattern_hex": m.pattern_hex,
                "pattern_length": m.pattern_length,
                "hamming_distance": m.hamming_distance,
                "correlation_score": m.correlation_score,
                "inverted": m.inverted,
            }
            for m in sync_matches
        ]
        inferred_len = ProtocolFramer.infer_frame_length(sync_matches)
        bits_per_sym = 2 if "QPSK" in modulation.scheme or "4FSK" in modulation.scheme else (
            3 if "8PSK" in modulation.scheme else (
                4 if "16" in modulation.scheme else (
                    6 if "64" in modulation.scheme else 1
                )
            )
        )
        bitstream_res = BitstreamAnalyzer.analyze(
            raw_bits,
            symbol_rate_hz=params.symbol_rate_sps.value,
            bits_per_symbol=bits_per_sym,
            detected_sync_words=sync_dicts,
            inferred_frame_length_bits=inferred_len,
        )

        fec_res = FecManager.probe_and_decode(raw_bits)
        frames = ProtocolFramer.extract_frames(raw_bits, sync_matches, nominal_frame_length_bits=inferred_len)

        # 10. Anomaly Detection
        self._check_cancelled()
        self.progress_callback(98, "Scanning for clipping, dropouts, and power steps...")
        anomalies = AnomalyDetector.analyze_anomalies(samples, sample_rate)

        # 11. Complete Parameter Extraction across Categories A to P
        self._check_cancelled()
        self.progress_callback(99, "Extracting comprehensive parameters (Categories A through P)...")
        from siganalyzer.parameters.context import SignalContext
        from siganalyzer.parameters.engine import ExtractionEngine

        dur_sec = reader.metadata.duration_seconds.value or (len(samples) / sample_rate)
        context = SignalContext(
            file_path=Path(file_path),
            reader=reader,
            metadata=reader.metadata,
            validation=validation,
            raw_samples=samples,
            conditioned_samples=conditioned,
            sample_rate=sample_rate,
            duration_s=float(dur_sec),
            center_frequency_rf=reader.metadata.center_frequency.value,
            spectrum=spectrum_res,
            spectrogram=spectrogram_data,
            regions=regions,
            primary_parameters=params,
            modulation=modulation,
            demodulation=demod_res,
            constellation_metrics=const_metrics,
            fec=fec_res,
            bitstream=bitstream_res,
            frames=frames,
            anomalies=anomalies,
        )
        signal_profile = ExtractionEngine.extract_all(context)

        duration = time.perf_counter() - start_time
        logger.info(f"Analysis complete in {duration:.3f} s. Detected: {modulation.scheme} ({len(demod_res.bits)} bits, {signal_profile.total_count} parameters)")
        self.progress_callback(100, "Analysis complete.")

        report = CompleteAnalysisReport(
            app_version="0.1.0",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            processing_duration_s=duration,
            metadata=reader.metadata,
            validation=validation,
            spectrum=spectrum_res,
            regions=regions,
            primary_parameters=params,
            primary_modulation=modulation,
            anomalies=anomalies,
            demodulation=demod_res,
            constellation_metrics=const_metrics,
            fec=fec_res,
            bitstream=bitstream_res,
            signal_profile=signal_profile,
        )

        # 11. Record Session into Local SQLite History
        try:
            db = DatabaseManager()
            db.save_session(
                file_path=str(file_path),
                file_name=reader.metadata.file_name,
                file_type=reader.metadata.file_type.value,
                file_size_bytes=reader.metadata.file_size_bytes,
                sample_rate=reader.metadata.sample_rate.value,
                center_freq=reader.metadata.center_frequency.value,
                duration_s=reader.metadata.duration_seconds.value,
                modulation_scheme=modulation.scheme,
                modulation_confidence=modulation.confidence_score,
                snr_db=params.snr_db.value,
                obw_hz=params.occupied_bandwidth_hz.value,
                symbol_rate=params.symbol_rate_sps.value,
                recovered_bits_count=len(raw_bits),
                bitstream_entropy=bitstream_res.entropy_per_bit,
                fec_scheme=fec_res.scheme if fec_res else "None",
                processing_time_s=duration,
            )
        except Exception as e:
            logger.debug(f"Could not persist session to SQLite (non-fatal): {e}")

        return report, spectrogram_data, samples

    def _check_cancelled(self) -> None:
        if self.is_cancelled():
            raise InterruptedError("Analysis cancelled by user.")
