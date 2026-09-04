"""Category O: Hardware / ADC / Capture Quality parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class HardwareAdcQualityExtractor(BaseParameterExtractor):
    """Extracts ENOB, full-scale utilization, headroom, ADC clipping, and hardware impairments."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.HARDWARE_ADC

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("hw.estimated_enob_bits", "Effective Number of Bits (ENOB)", cat, "bits", float, "Effective ADC resolution derived from SINAD", ConfidenceLevel.ESTIMATED, export_priority=1),
            ParameterDefinition("hw.full_scale_utilization_percent", "Full-Scale Utilization", cat, "%", float, "Peak sample amplitude relative to ADC full scale (1.0)", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("hw.headroom_dbfs", "ADC Dynamic Headroom", cat, "dBFS", float, "Unused margin below full-scale clipping threshold", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("hw.clipping_samples_count", "Clipping Samples Count", cat, "samples", int, "Number of samples saturated at rails (|s| >= 0.999)", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("hw.clipping_duration_ms", "Clipping Total Duration", cat, "ms", float, "Cumulative time signal spent in rail saturation", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("hw.saturation_flag", "ADC Saturation Detected", cat, "", bool, "Indicates active digital/analog receiver overload", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("hw.dc_bias_magnitude_dbfs", "Hardware DC Bias", cat, "dBFS", float, "Magnitude of hardware DC offset relative to full scale", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("hw.iq_gain_imbalance_severity", "I/Q Gain Imbalance Severity", cat, "", str, "Qualitative rating of frontend analog mismatch", ConfidenceLevel.INFERRED, export_priority=8),
            ParameterDefinition("hw.quadrature_orthogonality_error_deg", "Orthogonality Error", cat, "deg", float, "Receiver mixer phase imbalance from 90 degrees", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("hw.sampling_clock_drift_indicator", "Clock Stability Rating", cat, "", str, "Stability evaluation of ADC sampling crystal", ConfidenceLevel.INFERRED, export_priority=10),
            ParameterDefinition("hw.quantization_noise_floor_dbfs", "Theoretical Quantization Floor", cat, "dBFS", float, "Ideal 6.02*N + 1.76 dB noise floor for container bit depth", ConfidenceLevel.MEASURED, export_priority=11),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        raw = context.raw_samples
        bit_depth = context.metadata.bit_depth if context.metadata else 16
        if bit_depth <= 0:
            bit_depth = 16

        results: list[ParameterResult] = []

        # 1. Full-scale utilization and headroom
        abs_raw = np.abs(raw)
        peak_val = float(np.max(abs_raw)) if len(abs_raw) > 0 else 0.0
        fs_util_pct = float(min(100.0, peak_val * 100.0))
        headroom_db = float(-20.0 * np.log10(max(peak_val, 1e-6)))

        # 2. Clipping detection (|s| >= 0.999)
        clip_mask = abs_raw >= 0.999
        clip_count = int(np.sum(clip_mask))
        clip_duration_ms = float((clip_count / max(context.sample_rate, 1.0)) * 1000.0)
        saturation_flag = clip_count > 10

        # 3. DC bias relative to full scale
        dc_mag = float(np.abs(np.mean(raw))) if len(raw) > 0 else 0.0
        dc_dbfs = float(20.0 * np.log10(max(dc_mag, 1e-9)))

        # 4. ENOB derivation from SINAD / SNR
        snr_val = float(context.primary_parameters.snr_db.value) if (context.primary_parameters and context.primary_parameters.snr_db.value is not None) else 35.0
        enob = float(max(1.0, min(float(bit_depth), (snr_val - 1.76) / 6.02)))

        # 5. Theoretical quantization floor: - (6.02 * bit_depth + 1.76)
        quant_floor_dbfs = float(-(6.02 * bit_depth + 1.76))

        # 6. Gain imbalance & orthogonality error from raw I/Q
        i_raw = np.real(raw)
        q_raw = np.imag(raw) if np.iscomplexobj(raw) else np.zeros_like(i_raw)
        i_std = float(np.std(i_raw)) if len(i_raw) > 0 else 1.0
        q_std = float(np.std(q_raw)) if len(q_raw) > 0 else 1.0
        gain_imb_db = float(abs(20.0 * np.log10(max(i_std, 1e-12) / max(q_std, 1e-12)))) if np.iscomplexobj(raw) else 0.0

        if gain_imb_db < 0.2:
            gain_sev = "Negligible (< 0.2 dB)"
        elif gain_imb_db < 1.0:
            gain_sev = f"Moderate ({gain_imb_db:.2f} dB)"
        else:
            gain_sev = f"Severe ({gain_imb_db:.2f} dB)"

        if np.iscomplexobj(raw) and len(i_raw) > 1 and i_std > 1e-9 and q_std > 1e-9:
            corr = float(np.dot(i_raw - np.mean(i_raw), q_raw - np.mean(q_raw)) / (len(i_raw) * i_std * q_std))
            corr = float(np.clip(corr, -1.0, 1.0))
            ortho_err_deg = float(np.degrees(np.arcsin(corr)))
        else:
            ortho_err_deg = 0.0

        # Clock stability rating from frequency drift
        drift = float(context.scratch.get("drift_rate", 0.0))
        drift_ppm = abs(drift) / max(context.sample_rate, 1.0) * 1e6
        if drift_ppm < 5.0:
            clk_rating = "Stable (< 5 ppm TCXO/OCXO Grade)"
        elif drift_ppm < 50.0:
            clk_rating = f"Standard ({drift_ppm:.1f} ppm Crystal)"
        else:
            clk_rating = f"Unstable ({drift_ppm:.1f} ppm High Drift)"

        results.append(self._make_res("hw.estimated_enob_bits", enob, "(SINAD - 1.76) / 6.02", f"Effective ADC precision: {enob:.2f} bits (Container: {bit_depth}-bit).", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("hw.full_scale_utilization_percent", fs_util_pct, "max(|s|) / 1.0 * 100%", f"Full scale utilization: {fs_util_pct:.1f}%."))
        results.append(self._make_res("hw.headroom_dbfs", headroom_db, "-20 * log10(max(|s|))", f"Available dynamic headroom: {headroom_db:.1f} dBFS."))
        results.append(self._make_res("hw.clipping_samples_count", clip_count, "count(|s| >= 0.999)", f"{clip_count} saturated samples."))
        results.append(self._make_res("hw.clipping_duration_ms", clip_duration_ms, "clipping_samples / f_s", f"{clip_duration_ms:.2f} ms."))
        results.append(self._make_res("hw.saturation_flag", saturation_flag, "clipping_samples > 10", "WARNING: ADC Saturation detected!" if saturation_flag else "Normal signal scaling (no saturation)."))
        results.append(self._make_res("hw.dc_bias_magnitude_dbfs", dc_dbfs, "20 * log10(|mean(s)|)", f"DC offset: {dc_dbfs:.1f} dBFS."))
        results.append(self._make_res("hw.iq_gain_imbalance_severity", gain_sev, "20 * log10(std(I)/std(Q)) thresholding", gain_sev, ConfidenceLevel.INFERRED))
        results.append(self._make_res("hw.quadrature_orthogonality_error_deg", ortho_err_deg, "arcsin(cov(I,Q) / (std_I * std_Q))", f"{ortho_err_deg:+.2f} degrees."))
        results.append(self._make_res("hw.sampling_clock_drift_indicator", clk_rating, "Linear instantaneous frequency drift rate", clk_rating, ConfidenceLevel.INFERRED))
        results.append(self._make_res("hw.quantization_noise_floor_dbfs", quant_floor_dbfs, "-(6.02 * N + 1.76)", f"Ideal quantization floor: {quant_floor_dbfs:.1f} dBFS."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.95,
            source="Hardware / ADC Diagnostic Engine",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
