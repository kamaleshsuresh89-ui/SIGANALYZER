"""Category B: Time-Domain Parameters extractor."""

from typing import Any
import numpy as np
from scipy import stats

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class TimeDomainExtractor(BaseParameterExtractor):
    """Calculates comprehensive time-domain statistical and instantaneous signal properties."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.TIME_DOMAIN

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("time.mean_composite", "Composite Signal Mean", cat, "", float, "Mean complex sample value", ConfidenceLevel.MEASURED, export_priority=1),
            ParameterDefinition("time.mean_i", "In-Phase (I) Mean (DC Bias)", cat, "", float, "DC offset on In-Phase channel", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("time.mean_q", "Quadrature (Q) Mean (DC Bias)", cat, "", float, "DC offset on Quadrature channel", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("time.median_mag", "Median Magnitude", cat, "", float, "Median sample magnitude", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("time.variance_composite", "Signal Variance", cat, "", float, "Total power variance of samples", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("time.std_dev_composite", "Standard Deviation", cat, "", float, "Sample spread around mean", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("time.rms_amplitude_composite", "RMS Amplitude (Composite)", cat, "", float, "Root mean square amplitude", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("time.rms_amplitude_i", "RMS Amplitude (I)", cat, "", float, "In-phase RMS level", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("time.rms_amplitude_q", "RMS Amplitude (Q)", cat, "", float, "Quadrature RMS level", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("time.peak_amplitude_composite", "Peak Amplitude", cat, "", float, "Maximum absolute instantaneous sample value", ConfidenceLevel.MEASURED, export_priority=10),
            ParameterDefinition("time.peak_i", "Peak In-Phase (I)", cat, "", float, "Peak absolute level on I-channel", ConfidenceLevel.MEASURED, export_priority=11),
            ParameterDefinition("time.peak_q", "Peak Quadrature (Q)", cat, "", float, "Peak absolute level on Q-channel", ConfidenceLevel.MEASURED, export_priority=12),
            ParameterDefinition("time.peak_to_peak", "Peak-to-Peak Amplitude", cat, "", float, "Span between maximum and minimum value", ConfidenceLevel.MEASURED, export_priority=13),
            ParameterDefinition("time.crest_factor_db", "Crest Factor", cat, "dB", float, "Ratio of peak amplitude to RMS amplitude", ConfidenceLevel.MEASURED, export_priority=14),
            ParameterDefinition("time.papr_db", "PAPR", cat, "dB", float, "Peak-to-Average Power Ratio", ConfidenceLevel.MEASURED, export_priority=15),
            ParameterDefinition("time.dynamic_range_db", "Dynamic Range (Time)", cat, "dB", float, "Time-domain peak to noise baseline ratio", ConfidenceLevel.MEASURED, export_priority=16),
            ParameterDefinition("time.zero_crossing_rate_i", "Zero Crossing Rate (I)", cat, "crossings/s", float, "Rate of sign changes on I channel", ConfidenceLevel.MEASURED, export_priority=17),
            ParameterDefinition("time.zero_crossing_rate_q", "Zero Crossing Rate (Q)", cat, "crossings/s", float, "Rate of sign changes on Q channel", ConfidenceLevel.MEASURED, export_priority=18),
            ParameterDefinition("time.envelope_mean", "Envelope Mean", cat, "", float, "Average magnitude of analytic envelope", ConfidenceLevel.MEASURED, export_priority=19),
            ParameterDefinition("time.envelope_rms", "Envelope RMS", cat, "", float, "RMS level of magnitude envelope", ConfidenceLevel.MEASURED, export_priority=20),
            ParameterDefinition("time.envelope_variance", "Envelope Variance", cat, "", float, "Magnitude variance (constant envelope indicator)", ConfidenceLevel.MEASURED, export_priority=21),
            ParameterDefinition("time.envelope_peak", "Envelope Peak", cat, "", float, "Maximum envelope amplitude", ConfidenceLevel.MEASURED, export_priority=22),
            ParameterDefinition("time.envelope_dynamic_range_db", "Envelope Dynamic Range", cat, "dB", float, "Peak to minimum envelope ratio", ConfidenceLevel.MEASURED, export_priority=23),
            ParameterDefinition("time.instantaneous_phase_std_deg", "Instantaneous Phase Std", cat, "deg", float, "Standard deviation of phase", ConfidenceLevel.MEASURED, export_priority=24),
            ParameterDefinition("time.unwrapped_phase_slope", "Unwrapped Phase Slope", cat, "rad/s", float, "Linear phase trajectory slope (carrier offset indicator)", ConfidenceLevel.MEASURED, export_priority=25),
            ParameterDefinition("time.instantaneous_freq_mean_hz", "Instantaneous Freq Mean", cat, "Hz", float, "Mean instantaneous frequency derivative", ConfidenceLevel.MEASURED, export_priority=26),
            ParameterDefinition("time.instantaneous_freq_std_hz", "Instantaneous Freq Std", cat, "Hz", float, "Frequency fluctuation standard deviation", ConfidenceLevel.MEASURED, export_priority=27),
            ParameterDefinition("time.signal_activity_ratio", "Signal Activity Ratio", cat, "%", float, "Fraction of time signal is above noise threshold", ConfidenceLevel.ESTIMATED, export_priority=28),
            ParameterDefinition("time.clipping_percentage", "Clipping Percentage", cat, "%", float, "Percentage of samples reaching full scale", ConfidenceLevel.MEASURED, export_priority=29),
            ParameterDefinition("time.iq_covariance", "I/Q Covariance", cat, "", float, "Covariance between I and Q channels", ConfidenceLevel.MEASURED, export_priority=30),
            ParameterDefinition("time.iq_correlation", "I/Q Correlation Coefficient", cat, "", float, "Pearson correlation between I and Q (orthogonality metric)", ConfidenceLevel.MEASURED, export_priority=31),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        samples = context.conditioned_samples
        raw = context.raw_samples
        sr = context.sample_rate
        results: list[ParameterResult] = []

        N = len(samples)
        if N == 0:
            return [
                ParameterResult(d, None, ConfidenceLevel.UNKNOWN, 0.0, "Empty", "N/A", "No samples available")
                for d in self.supported_parameters()
            ]

        # Extract I and Q components
        if np.iscomplexobj(samples):
            i_ch = np.real(samples)
            q_ch = np.imag(samples)
        else:
            i_ch = samples.astype(np.float32)
            q_ch = np.zeros_like(i_ch)

        # Raw I and Q for DC offset check
        if np.iscomplexobj(raw):
            raw_i = np.real(raw)
            raw_q = np.imag(raw)
        else:
            raw_i = raw.astype(np.float32)
            raw_q = np.zeros_like(raw_i)

        env = context.envelope

        # 1. Means
        mean_c = complex(np.mean(samples)) if np.iscomplexobj(samples) else float(np.mean(samples))
        mean_i_val = float(np.mean(raw_i))
        mean_q_val = float(np.mean(raw_q))

        results.append(self._make_res("time.mean_composite", abs(mean_c), "mean(s[n])", "Average sample amplitude across complex plane."))
        results.append(self._make_res("time.mean_i", mean_i_val, "mean(I[n])", "Direct DC component on In-Phase channel before filter."))
        results.append(self._make_res("time.mean_q", mean_q_val, "mean(Q[n])", "Direct DC component on Quadrature channel before filter."))

        # 2. Medians & Variances
        med_mag = float(np.median(env))
        var_c = float(np.var(samples))
        std_c = float(np.std(samples))

        results.append(self._make_res("time.median_mag", med_mag, "median(|s[n]|)", "Robust central tendency of sample envelope."))
        results.append(self._make_res("time.variance_composite", var_c, "E[|s - mu|^2]", "Total variance of signal energy."))
        results.append(self._make_res("time.std_dev_composite", std_c, "sqrt(variance)", "Standard deviation of sample distribution."))

        # 3. RMS & Peak Amplitudes
        rms_c = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
        rms_i = float(np.sqrt(np.mean(i_ch ** 2)))
        rms_q = float(np.sqrt(np.mean(q_ch ** 2)))

        peak_c = float(np.max(env))
        peak_i_val = float(np.max(np.abs(i_ch)))
        peak_q_val = float(np.max(np.abs(q_ch)))
        ptp_val = float(np.ptp(i_ch))

        results.append(self._make_res("time.rms_amplitude_composite", rms_c, "sqrt(mean(|s|^2))", "Root mean square amplitude of composite signal."))
        results.append(self._make_res("time.rms_amplitude_i", rms_i, "sqrt(mean(I^2))", "RMS amplitude of In-Phase component."))
        results.append(self._make_res("time.rms_amplitude_q", rms_q, "sqrt(mean(Q^2))", "RMS amplitude of Quadrature component."))
        results.append(self._make_res("time.peak_amplitude_composite", peak_c, "max(|s[n]|)", "Peak amplitude observed in recording."))
        results.append(self._make_res("time.peak_i", peak_i_val, "max(|I[n]|)", "Peak amplitude on I-channel."))
        results.append(self._make_res("time.peak_q", peak_q_val, "max(|Q[n]|)", "Peak amplitude on Q-channel."))
        results.append(self._make_res("time.peak_to_peak", ptp_val, "max(I) - min(I)", "Total peak-to-peak span on I channel."))

        # 4. Crest factor & PAPR
        eps = 1e-12
        crest_factor = 20.0 * np.log10(max(peak_c, eps) / max(rms_c, eps))
        papr_val = 10.0 * np.log10((max(peak_c, eps) ** 2) / (max(rms_c, eps) ** 2))
        dr_time = 20.0 * np.log10(max(peak_c, eps) / max(np.min(env) + eps, eps))

        results.append(self._make_res("time.crest_factor_db", float(crest_factor), "20*log10(Peak / RMS)", "Crest factor indicates envelope peaking above average."))
        results.append(self._make_res("time.papr_db", float(papr_val), "10*log10(PeakPower / AvgPower)", "Peak-to-Average Power Ratio (critical for PA linearity)."))
        results.append(self._make_res("time.dynamic_range_db", float(dr_time), "20*log10(Peak / Min)", "Instantaneous dynamic range across envelope."))

        # 5. Zero-crossing rate
        zcr_i = float(np.sum(np.diff(np.signbit(i_ch)) != 0)) * (sr / (2.0 * max(N - 1, 1)))
        zcr_q = float(np.sum(np.diff(np.signbit(q_ch)) != 0)) * (sr / (2.0 * max(N - 1, 1)))
        results.append(self._make_res("time.zero_crossing_rate_i", zcr_i, "count(sign_flips) * fs / N", "Zero-crossing rate on I-channel."))
        results.append(self._make_res("time.zero_crossing_rate_q", zcr_q, "count(sign_flips) * fs / N", "Zero-crossing rate on Q-channel."))

        # 6. Envelope properties
        env_mean = float(np.mean(env))
        env_rms = float(np.sqrt(np.mean(env ** 2)))
        env_var = float(np.var(env))
        env_peak = float(np.max(env))
        env_dr = float(20.0 * np.log10(max(env_peak, eps) / max(np.min(env) + eps, eps)))

        results.append(self._make_res("time.envelope_mean", env_mean, "mean(|s|)", "Average magnitude of analytic envelope."))
        results.append(self._make_res("time.envelope_rms", env_rms, "sqrt(mean(|s|^2))", "RMS level of envelope."))
        results.append(self._make_res("time.envelope_variance", env_var, "var(|s|)", "Variance of magnitude envelope (low for PSK/FSK, high for QAM/ASK)."))
        results.append(self._make_res("time.envelope_peak", env_peak, "max(|s|)", "Peak envelope level."))
        results.append(self._make_res("time.envelope_dynamic_range_db", env_dr, "20*log10(env_max / env_min)", "Envelope peak-to-minimum ratio."))

        # 7. Instantaneous Phase & Frequency
        phase = context.phase
        u_phase = context.unwrapped_phase
        f_inst = context.instantaneous_frequency

        phase_std_deg = float(np.rad2deg(np.std(phase)))
        slope_rad_per_s = float(np.polyfit(np.arange(len(u_phase)), u_phase, 1)[0] * sr) if len(u_phase) > 2 else 0.0

        f_inst_mean = float(np.mean(f_inst)) if len(f_inst) > 0 else 0.0
        f_inst_std = float(np.std(f_inst)) if len(f_inst) > 0 else 0.0

        results.append(self._make_res("time.instantaneous_phase_std_deg", phase_std_deg, "std(angle(s)) * 180 / pi", "Phase spread in degrees across recording."))
        results.append(self._make_res("time.unwrapped_phase_slope", slope_rad_per_s, "d(unwrap(theta))/dt", "Linear phase drift indicating carrier frequency offset."))
        results.append(self._make_res("time.instantaneous_freq_mean_hz", f_inst_mean, "mean(f_inst)", "Average instantaneous frequency of the signal."))
        results.append(self._make_res("time.instantaneous_freq_std_hz", f_inst_std, "std(f_inst)", "Frequency fluctuation std dev (high for FSK/FM)."))

        # 8. Activity & Clipping
        noise_floor = float(np.percentile(env, 15))
        threshold = noise_floor * 2.5
        active_ratio = float(np.mean(env > threshold) * 100.0)
        clipping_pct = float(np.mean(np.abs(raw_i) >= 0.999) * 100.0)

        results.append(self._make_res("time.signal_activity_ratio", active_ratio, "count(env > threshold) / N * 100", "Percentage of recording with active transmission.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("time.clipping_percentage", clipping_pct, "count(|sample| >= 0.999) / N * 100", "ADC clipping sample percentage."))

        # 9. I/Q Covariance & Correlation
        cov_matrix = np.cov(i_ch, q_ch)
        iq_cov = float(cov_matrix[0, 1])
        std_prod = float(np.std(i_ch) * np.std(q_ch) + eps)
        iq_corr = float(iq_cov / std_prod)

        results.append(self._make_res("time.iq_covariance", iq_cov, "cov(I, Q)", "Cross-covariance between In-Phase and Quadrature."))
        results.append(self._make_res("time.iq_correlation", iq_corr, "cov(I,Q)/(std(I)*std(Q))", "Pearson correlation (ideally 0.0 for orthogonal balanced IQ)."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = None
        for d in self.supported_parameters():
            if d.id == pid:
                defn = d
                break
        if defn is None:
            raise KeyError(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.98 if status == ConfidenceLevel.MEASURED else 0.85,
            source="Time Domain DSP",
            method=method,
            explanation=expl,
        )
