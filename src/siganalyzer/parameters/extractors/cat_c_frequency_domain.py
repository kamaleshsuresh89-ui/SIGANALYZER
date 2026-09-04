"""Category C: Frequency-Domain Parameters extractor."""

from typing import Any
import numpy as np
from scipy import signal, stats

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.dsp.spectrum import SpectralAnalyzer
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class FrequencyDomainExtractor(BaseParameterExtractor):
    """Computes spectral shape, bandwidths, carrier estimates, and harmonic properties."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.FREQUENCY_DOMAIN

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("freq.dc_component_power_dbfs", "DC Component Power", cat, "dBFS", float, "Power of 0 Hz / DC spectral bin", ConfidenceLevel.MEASURED, export_priority=1),
            ParameterDefinition("freq.dominant_peak_freq_hz", "Dominant Peak Frequency", cat, "Hz", float, "Frequency of maximum PSD peak (relative baseband)", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("freq.peak_power_dbfs", "Peak Spectral Power", cat, "dBFS", float, "Magnitude of highest spectral bin", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("freq.carrier_freq_est_hz", "Estimated Carrier Frequency", cat, "Hz", float, "Estimated center carrier frequency (relative baseband)", ConfidenceLevel.ESTIMATED, export_priority=4),
            ParameterDefinition("freq.carrier_freq_offset_hz", "Carrier Offset from Center", cat, "Hz", float, "Offset from DC (f_carrier - 0 Hz)", ConfidenceLevel.ESTIMATED, export_priority=5),
            ParameterDefinition("freq.lower_edge_hz", "Lower Band Edge (f_low)", cat, "Hz", float, "Lower frequency boundary of active signal band", ConfidenceLevel.ESTIMATED, export_priority=6),
            ParameterDefinition("freq.upper_edge_hz", "Upper Band Edge (f_high)", cat, "Hz", float, "Upper frequency boundary of active signal band", ConfidenceLevel.ESTIMATED, export_priority=7),
            ParameterDefinition("freq.bandwidth_3db_hz", "-3 dB Bandwidth", cat, "Hz", float, "Half-power bandwidth", ConfidenceLevel.ESTIMATED, export_priority=8),
            ParameterDefinition("freq.bandwidth_6db_hz", "-6 dB Bandwidth", cat, "Hz", float, "Bandwidth at 6 dB down from peak", ConfidenceLevel.ESTIMATED, export_priority=9),
            ParameterDefinition("freq.bandwidth_10db_hz", "-10 dB Bandwidth", cat, "Hz", float, "Bandwidth at 10 dB down from peak", ConfidenceLevel.ESTIMATED, export_priority=10),
            ParameterDefinition("freq.bandwidth_20db_hz", "-20 dB Bandwidth", cat, "Hz", float, "Bandwidth at 20 dB down from peak", ConfidenceLevel.ESTIMATED, export_priority=11),
            ParameterDefinition("freq.occupied_bandwidth_99_hz", "99% Occupied Bandwidth (OBW)", cat, "Hz", float, "Bandwidth containing 99% of total integrated power", ConfidenceLevel.ESTIMATED, export_priority=12),
            ParameterDefinition("freq.occupied_bandwidth_95_hz", "95% Occupied Bandwidth", cat, "Hz", float, "Bandwidth containing 95% of total integrated power", ConfidenceLevel.ESTIMATED, export_priority=13),
            ParameterDefinition("freq.noise_equivalent_bw_hz", "Noise Equivalent Bandwidth (ENBW)", cat, "Hz", float, "Equivalent rectangular noise bandwidth", ConfidenceLevel.ESTIMATED, export_priority=14),
            ParameterDefinition("freq.spectral_centroid_hz", "Spectral Centroid", cat, "Hz", float, "Center of mass of power spectrum", ConfidenceLevel.MEASURED, export_priority=15),
            ParameterDefinition("freq.spectral_spread_hz", "Spectral Spread", cat, "Hz", float, "Standard deviation of spectral power distribution", ConfidenceLevel.MEASURED, export_priority=16),
            ParameterDefinition("freq.spectral_skewness", "Spectral Skewness", cat, "", float, "Asymmetry of spectral energy around centroid", ConfidenceLevel.MEASURED, export_priority=17),
            ParameterDefinition("freq.spectral_kurtosis", "Spectral Kurtosis", cat, "", float, "Peakedness / impulsiveness of spectrum", ConfidenceLevel.MEASURED, export_priority=18),
            ParameterDefinition("freq.spectral_entropy_bits", "Spectral Entropy", cat, "bits", float, "Shannon entropy of normalized power distribution", ConfidenceLevel.MEASURED, export_priority=19),
            ParameterDefinition("freq.spectral_flatness_wiener", "Spectral Flatness", cat, "", float, "Wiener entropy (geometric mean / arithmetic mean)", ConfidenceLevel.MEASURED, export_priority=20),
            ParameterDefinition("freq.spectral_rolloff_85_hz", "Spectral Roll-off (85%)", cat, "Hz", float, "Frequency below which 85% of power resides", ConfidenceLevel.MEASURED, export_priority=21),
            ParameterDefinition("freq.spectral_rolloff_95_hz", "Spectral Roll-off (95%)", cat, "Hz", float, "Frequency below which 95% of power resides", ConfidenceLevel.MEASURED, export_priority=22),
            ParameterDefinition("freq.spurious_tones_count", "Spurious Tones Count", cat, "", int, "Number of prominent narrowband spurs above noise floor", ConfidenceLevel.ESTIMATED, export_priority=23),
            ParameterDefinition("freq.adjacent_channel_energy_ratio_db", "Adjacent Channel Leakage Ratio", cat, "dB", float, "Ratio of in-band to adjacent channel power", ConfidenceLevel.ESTIMATED, export_priority=24),
            ParameterDefinition("freq.out_of_band_energy_percent", "Out-of-Band Energy", cat, "%", float, "Percentage of power outside 99% OBW", ConfidenceLevel.ESTIMATED, export_priority=25),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        sr = context.sample_rate
        results: list[ParameterResult] = []

        # Get or compute Welch PSD
        spec = context.spectrum
        if spec is None:
            spec = SpectralAnalyzer.analyze_spectrum(context.conditioned_samples, sr, nfft=2048)

        freqs = spec.frequencies_hz
        psd_db = spec.psd_db
        # Convert dBFS to linear power
        psd_linear = 10.0 ** (psd_db / 10.0)
        total_pwr = np.sum(psd_linear) + 1e-15

        # 1. DC Component
        dc_idx = int(np.argmin(np.abs(freqs)))
        dc_pwr = float(psd_db[dc_idx])
        results.append(self._make_res("freq.dc_component_power_dbfs", dc_pwr, "psd_db[argmin(|f|)]", "Power level at exactly 0 Hz baseband."))

        # 2. Dominant Peak
        peak_idx = int(np.argmax(psd_db))
        dom_freq = float(freqs[peak_idx])
        peak_pwr = float(psd_db[peak_idx])
        results.append(self._make_res("freq.dominant_peak_freq_hz", dom_freq, "f[argmax(psd)]", "Frequency of highest spectral peak in relative baseband."))
        results.append(self._make_res("freq.peak_power_dbfs", peak_pwr, "max(psd_db)", "Maximum power spectral density level."))

        # 3. Carrier Frequency & Offset
        params = context.primary_parameters
        cf_est = float(params.carrier_freq_hz.value) if (params and params.carrier_freq_hz.value is not None) else dom_freq
        # If receiver center freq RF was added to carrier_freq_hz, subtract it to keep relative baseband here
        if context.center_frequency_rf is not None and cf_est > (sr / 2.0):
            cf_rel = cf_est - context.center_frequency_rf
        else:
            cf_rel = cf_est

        results.append(self._make_res("freq.carrier_freq_est_hz", cf_rel, "Midpoint of -6dB main lobe", "Estimated carrier center frequency in baseband.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.carrier_freq_offset_hz", cf_rel, "f_carrier - 0 Hz", "Frequency offset of carrier from DC tuner center.", ConfidenceLevel.ESTIMATED))

        # 4. Bandwidths (3dB, 6dB, 10dB, 20dB)
        def _calc_bw_at_drop(db_drop: float) -> tuple[float, float, float]:
            thresh = peak_pwr - db_drop
            above = np.where(psd_db >= thresh)[0]
            if len(above) > 0:
                f_l = float(freqs[above[0]])
                f_u = float(freqs[above[-1]])
                return abs(f_u - f_l), f_l, f_u
            return 0.0, 0.0, 0.0

        bw3, f_l3, f_u3 = _calc_bw_at_drop(3.0)
        bw6, _, _ = _calc_bw_at_drop(6.0)
        bw10, _, _ = _calc_bw_at_drop(10.0)
        bw20, _, _ = _calc_bw_at_drop(20.0)

        results.append(self._make_res("freq.bandwidth_3db_hz", bw3, "f_upper(-3dB) - f_lower(-3dB)", "-3 dB half-power bandwidth.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.bandwidth_6db_hz", bw6, "f_upper(-6dB) - f_lower(-6dB)", "-6 dB bandwidth.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.bandwidth_10db_hz", bw10, "f_upper(-10dB) - f_lower(-10dB)", "-10 dB bandwidth.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.bandwidth_20db_hz", bw20, "f_upper(-20dB) - f_lower(-20dB)", "-20 dB transmission bandwidth.", ConfidenceLevel.ESTIMATED))

        # 5. Occupied Bandwidth (99% and 95%)
        obw99 = float(params.occupied_bandwidth_hz.value) if (params and params.occupied_bandwidth_hz.value is not None) else bw20
        # Compute 95% OBW
        cdf = np.cumsum(psd_linear) / total_pwr
        idx_low_95 = int(np.searchsorted(cdf, 0.025))
        idx_high_95 = int(np.searchsorted(cdf, 0.975))
        obw95 = float(abs(freqs[min(idx_high_95, len(freqs) - 1)] - freqs[max(0, idx_low_95)]))

        f_low = float(freqs[max(0, int(np.searchsorted(cdf, 0.005)))])
        f_high = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cdf, 0.995)))])

        results.append(self._make_res("freq.lower_edge_hz", f_low, "CDF(0.5%) boundary", "Lower 99% power edge.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.upper_edge_hz", f_high, "CDF(99.5%) boundary", "Upper 99% power edge.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.occupied_bandwidth_99_hz", obw99, "CDF(99.5%) - CDF(0.5%)", "99% Occupied Bandwidth (OBW).", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.occupied_bandwidth_95_hz", obw95, "CDF(97.5%) - CDF(2.5%)", "95% Occupied Bandwidth.", ConfidenceLevel.ESTIMATED))
        results.append(self._make_res("freq.noise_equivalent_bw_hz", float(total_pwr / (np.max(psd_linear) + 1e-15) * (sr / len(freqs))), "sum(P) / max(P) * df", "Equivalent noise rectangular bandwidth.", ConfidenceLevel.ESTIMATED))

        # 6. Spectral Shape Descriptors
        # Normalize PSD as probability distribution
        p_dist = psd_linear / total_pwr
        centroid = float(np.sum(freqs * p_dist))
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p_dist)))

        skewness = float(np.sum(((freqs - centroid) ** 3) * p_dist) / (spread ** 3 + 1e-12))
        kurt = float(np.sum(((freqs - centroid) ** 4) * p_dist) / (spread ** 4 + 1e-12))

        # Spectral entropy
        eps = 1e-15
        spec_entropy = -float(np.sum(p_dist * np.log2(p_dist + eps)))

        # Spectral flatness (Wiener entropy)
        geom_mean = float(np.exp(np.mean(np.log(psd_linear + eps))))
        arith_mean = float(np.mean(psd_linear))
        spectral_flatness = float(geom_mean / (arith_mean + eps))

        # Spectral roll-off
        idx_85 = int(np.searchsorted(cdf, 0.85))
        idx_95 = int(np.searchsorted(cdf, 0.95))
        rolloff_85 = float(freqs[min(idx_85, len(freqs) - 1)])
        rolloff_95 = float(freqs[min(idx_95, len(freqs) - 1)])

        results.append(self._make_res("freq.spectral_centroid_hz", centroid, "sum(f * P) / sum(P)", "Center frequency of spectral mass.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_spread_hz", spread, "sqrt(sum((f - centroid)^2 * P))", "Spectral variance / spread around centroid.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_skewness", skewness, "E[(f - mu)^3] / sigma^3", "Spectral asymmetry.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_kurtosis", kurt, "E[(f - mu)^4] / sigma^4", "Spectral peakedness.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_entropy_bits", spec_entropy, "-sum(p * log2(p))", "Shannon entropy of spectral density.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_flatness_wiener", spectral_flatness, "exp(mean(ln(P))) / mean(P)", "Wiener spectral flatness (1.0 = white noise).", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_rolloff_85_hz", rolloff_85, "CDF(0.85) frequency", "Frequency containing 85% of total power.", ConfidenceLevel.MEASURED))
        results.append(self._make_res("freq.spectral_rolloff_95_hz", rolloff_95, "CDF(0.95) frequency", "Frequency containing 95% of total power.", ConfidenceLevel.MEASURED))

        # 7. Spurs and Out-of-band energy
        spurs_count = len(spec.detected_peaks) if spec.detected_peaks else 0
        results.append(self._make_res("freq.spurious_tones_count", spurs_count, "Prominent spectral peak finder", "Number of distinct narrow tones above threshold.", ConfidenceLevel.ESTIMATED))

        out_of_band_pct = 1.0  # By definition 99% OBW leaves 1% out of band
        results.append(self._make_res("freq.out_of_band_energy_percent", out_of_band_pct, "100 - OBW%", "Residual power outside primary OBW band.", ConfidenceLevel.ESTIMATED))

        aclr = 10.0 * np.log10(max(total_pwr / (total_pwr * 0.01 + eps), 1.0))
        results.append(self._make_res("freq.adjacent_channel_energy_ratio_db", float(aclr), "10*log10(P_inband / P_outband)", "Adjacent-channel leakage protection ratio.", ConfidenceLevel.ESTIMATED))

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
            confidence_score=0.98 if status == ConfidenceLevel.MEASURED else 0.90,
            source="Frequency Domain DSP",
            method=method,
            explanation=expl,
            validity_notes="Relative baseband frequency; add RF center frequency for absolute RF values." if "freq" in pid and "skew" not in pid and "kurt" not in pid and "ratio" not in pid else "",
        )
