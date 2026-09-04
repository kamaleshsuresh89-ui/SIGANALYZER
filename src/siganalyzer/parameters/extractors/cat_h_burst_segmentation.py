"""Category H: Burst / Signal Segmentation parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class BurstSegmentationExtractor(BaseParameterExtractor):
    """Extracts burst counts, durations, duty cycle, Pulse Repetition Interval (PRI), and burst power."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.BURST_SEGMENTATION

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("burst.transmission_mode", "Transmission Mode", cat, "", str, "Signal regime: Continuous vs Burst / Packetized", ConfidenceLevel.INFERRED, export_priority=1),
            ParameterDefinition("burst.detected_burst_count", "Detected Bursts Count", cat, "bursts", int, "Total number of isolated RF energy bursts", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("burst.total_active_time_s", "Total Active Duration", cat, "s", float, "Cumulative time signal power exceeded threshold", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("burst.total_idle_time_s", "Total Idle Duration", cat, "s", float, "Cumulative silence or noise duration", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("burst.duty_cycle_percent", "Duty Cycle", cat, "%", float, "Percentage of observation time occupied by transmission", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("burst.duration_mean_ms", "Mean Burst Duration", cat, "ms", float, "Average length of active burst emissions", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("burst.duration_min_ms", "Min Burst Duration", cat, "ms", float, "Shortest detected burst duration", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("burst.duration_max_ms", "Max Burst Duration", cat, "ms", float, "Longest detected burst duration", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("burst.duration_std_ms", "Burst Duration Std Dev", cat, "ms", float, "Variance in burst length", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("burst.pri_mean_ms", "Mean PRI", cat, "ms", float, "Average Pulse Repetition Interval (burst start to start)", ConfidenceLevel.MEASURED, export_priority=10),
            ParameterDefinition("burst.pri_min_ms", "Min PRI", cat, "ms", float, "Minimum Pulse Repetition Interval", ConfidenceLevel.MEASURED, export_priority=11),
            ParameterDefinition("burst.pri_max_ms", "Max PRI", cat, "ms", float, "Maximum Pulse Repetition Interval", ConfidenceLevel.MEASURED, export_priority=12),
            ParameterDefinition("burst.mean_burst_power_dbfs", "Mean Burst Power", cat, "dBFS", float, "Average power level strictly during active burst intervals", ConfidenceLevel.MEASURED, export_priority=13),
            ParameterDefinition("burst.mean_burst_snr_db", "Mean Burst SNR", cat, "dB", float, "Average signal-to-noise ratio within detected bursts", ConfidenceLevel.MEASURED, export_priority=14),
            ParameterDefinition("burst.burst_frequency_variation_hz", "Burst Freq Variation", cat, "Hz", float, "Standard deviation of center frequencies across bursts", ConfidenceLevel.MEASURED, export_priority=15),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        regions = context.regions
        active_regions = [r for r in regions if r.is_signal]
        burst_count = len(active_regions)
        duration_total = max(context.duration_s, 1e-6)

        results: list[ParameterResult] = []

        if burst_count > 0:
            durations_ms = np.array([r.duration_s * 1000.0 for r in active_regions])
            total_active_s = float(np.sum([r.duration_s for r in active_regions]))
            total_idle_s = max(0.0, float(duration_total - total_active_s))
            duty_cycle = float(min(100.0, (total_active_s / duration_total) * 100.0))

            mean_dur = float(np.mean(durations_ms))
            min_dur = float(np.min(durations_ms))
            max_dur = float(np.max(durations_ms))
            std_dur = float(np.std(durations_ms))

            # Pulse Repetition Intervals (start of burst i+1 minus start of burst i)
            if burst_count > 1:
                start_times_ms = np.array([r.start_time * 1000.0 for r in active_regions])
                pris_ms = np.diff(start_times_ms)
                mean_pri = float(np.mean(pris_ms))
                min_pri = float(np.min(pris_ms))
                max_pri = float(np.max(pris_ms))
            else:
                mean_pri = 0.0
                min_pri = 0.0
                max_pri = 0.0

            # Burst power and SNR
            powers = [r.mean_power_db for r in active_regions]
            mean_pwr = float(np.mean(powers))
            snrs = [r.snr_db.value for r in active_regions if r.snr_db.value is not None]
            mean_snr = float(np.mean(snrs)) if snrs else 0.0

            # Frequency variation across bursts
            freq_offsets = [r.center_freq_offset_hz.value for r in active_regions if r.center_freq_offset_hz.value is not None]
            freq_var = float(np.std(freq_offsets)) if len(freq_offsets) > 1 else 0.0

            mode = "Continuous" if (burst_count == 1 and duty_cycle > 90.0) else "Burst / Packetized"

            results.append(self._make_res("burst.transmission_mode", mode, "CFAR segmentation duty cycle thresholding", f"Classified transmission format: {mode}."))
            results.append(self._make_res("burst.detected_burst_count", burst_count, "Two-pass CFAR energy detector", f"Detected {burst_count} distinct signal burst(s)."))
            results.append(self._make_res("burst.total_active_time_s", total_active_s, "Sum of burst durations", f"Active transmission time: {total_active_s:.4f} s."))
            results.append(self._make_res("burst.total_idle_time_s", total_idle_s, "Total duration - Active duration", f"Idle silence time: {total_idle_s:.4f} s."))
            results.append(self._make_res("burst.duty_cycle_percent", duty_cycle, "(Active / Total) * 100%", f"Signal duty cycle: {duty_cycle:.2f}%."))
            results.append(self._make_res("burst.duration_mean_ms", mean_dur, "mean(tau_burst)", f"Mean burst duration: {mean_dur:.2f} ms."))
            results.append(self._make_res("burst.duration_min_ms", min_dur, "min(tau_burst)", f"Shortest burst: {min_dur:.2f} ms."))
            results.append(self._make_res("burst.duration_max_ms", max_dur, "max(tau_burst)", f"Longest burst: {max_dur:.2f} ms."))
            results.append(self._make_res("burst.duration_std_ms", std_dur, "std(tau_burst)", f"Burst duration dispersion: {std_dur:.2f} ms."))
            results.append(self._make_res("burst.pri_mean_ms", mean_pri, "mean(t_{start, k+1} - t_{start, k})", f"Mean Pulse Repetition Interval: {mean_pri:.2f} ms."))
            results.append(self._make_res("burst.pri_min_ms", min_pri, "min(PRI)", f"Minimum PRI: {min_pri:.2f} ms."))
            results.append(self._make_res("burst.pri_max_ms", max_pri, "max(PRI)", f"Maximum PRI: {max_pri:.2f} ms."))
            results.append(self._make_res("burst.mean_burst_power_dbfs", mean_pwr, "mean(P_{burst})", f"Mean burst active power: {mean_pwr:.2f} dBFS."))
            results.append(self._make_res("burst.mean_burst_snr_db", mean_snr, "mean(SNR_{burst})", f"Mean in-burst SNR: {mean_snr:.2f} dB."))
            results.append(self._make_res("burst.burst_frequency_variation_hz", freq_var, "std(f_{center, burst})", f"Burst center frequency jitter: {freq_var:.2f} Hz."))

        else:
            # Continuous full duration or no bursts segmented
            results.append(self._make_res("burst.transmission_mode", "Continuous", "Full duration envelope continuous threshold", "Signal is continuous across capture duration."))
            results.append(self._make_res("burst.detected_burst_count", 1, "Single continuous block", "No discrete burst transitions detected."))
            results.append(self._make_res("burst.total_active_time_s", float(duration_total), "Total duration", f"Total active time: {duration_total:.4f} s."))
            results.append(self._make_res("burst.total_idle_time_s", 0.0, "Zero silence detected", "0.0 s."))
            results.append(self._make_res("burst.duty_cycle_percent", 100.0, "Continuous transmission", "100.0%."))
            results.append(self._make_res("burst.duration_mean_ms", float(duration_total * 1000.0), "Total capture duration", f"{duration_total * 1000.0:.2f} ms."))
            results.append(self._make_res("burst.duration_min_ms", float(duration_total * 1000.0), "Total capture duration", f"{duration_total * 1000.0:.2f} ms."))
            results.append(self._make_res("burst.duration_max_ms", float(duration_total * 1000.0), "Total capture duration", f"{duration_total * 1000.0:.2f} ms."))
            results.append(self._make_res("burst.duration_std_ms", 0.0, "Single duration", "0.0 ms."))
            results.append(self._make_res("burst.pri_mean_ms", 0.0, "N/A for continuous", "0.0 ms."))
            results.append(self._make_res("burst.pri_min_ms", 0.0, "N/A for continuous", "0.0 ms."))
            results.append(self._make_res("burst.pri_max_ms", 0.0, "N/A for continuous", "0.0 ms."))
            
            pwr = float(10.0 * np.log10(np.mean(np.abs(context.conditioned_samples) ** 2) + 1e-12))
            snr_val = float(context.primary_parameters.snr_db.value) if (context.primary_parameters and context.primary_parameters.snr_db.value is not None) else 20.0
            results.append(self._make_res("burst.mean_burst_power_dbfs", pwr, "Continuous average power", f"{pwr:.2f} dBFS."))
            results.append(self._make_res("burst.mean_burst_snr_db", snr_val, "Signal SNR", f"{snr_val:.2f} dB."))
            results.append(self._make_res("burst.burst_frequency_variation_hz", 0.0, "Single transmission", "0.0 Hz."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.95,
            source="Burst Detection Subsystem",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
