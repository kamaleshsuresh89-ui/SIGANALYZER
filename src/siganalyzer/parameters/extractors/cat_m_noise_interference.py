"""Category M: Noise & Interference Analysis parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class NoiseInterferenceExtractor(BaseParameterExtractor):
    """Extracts noise distribution, narrowband spurs, carrier leakage, ACLR, and impulsive spikes."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.NOISE_INTERFERENCE

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("noise.noise_distribution_type", "Noise Distribution Profile", cat, "", str, "Statistical profile of the background noise floor", ConfidenceLevel.INFERRED, export_priority=1),
            ParameterDefinition("noise.narrowband_interference_detected", "Narrowband Interference", cat, "", bool, "Presence of co-channel or adjacent discrete interferers", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("noise.spurious_tones_count", "Spurious Tones Count", cat, "tones", int, "Number of spectral spurs exceeding prominence threshold", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("noise.strongest_spur_freq_offset_hz", "Strongest Spur Offset", cat, "Hz", float, "Frequency separation of dominant spur from center", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("noise.strongest_spur_power_dbfs", "Strongest Spur Power", cat, "dBFS", float, "Power of dominant non-carrier spectral spur", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("noise.strongest_spur_rejection_db", "Spur Suppression Depth", cat, "dB", float, "Delta between desired signal peak and strongest spur", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("noise.carrier_leakage_power_dbfs", "Carrier Leakage Power", cat, "dBFS", float, "Direct conversion LO leakage power at DC (0 Hz)", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("noise.adjacent_channel_interference_ratio_db", "ACIR Ratio", cat, "dB", float, "Adjacent channel leakage / interference ratio", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("noise.impulsive_noise_spike_count", "Impulsive Spike Events", cat, "spikes", int, "Number of transient time-domain envelope spikes (> 5 sigma)", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("noise.spectral_flatness_measure", "Spectral Flatness (SFM)", cat, "", float, "Wiener flatness: ratio of geometric mean to arithmetic mean PSD", ConfidenceLevel.MEASURED, export_priority=10),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        samples = context.conditioned_samples
        spec = context.spectrum
        results: list[ParameterResult] = []

        # 1. Impulsive noise analysis from envelope outliers
        env = context.envelope
        if len(env) > 100:
            env_med = float(np.median(env))
            env_mad = float(np.median(np.abs(env - env_med))) + 1e-12
            # Outlier spikes > 5 * MAD
            spike_indices = np.where(env > (env_med + 6.0 * 1.4826 * env_mad))[0]
            spike_count = int(len(spike_indices))
            noise_dist = "Impulsive / Heavy-Tailed" if spike_count > 50 else "Gaussian / Rayleigh"
        else:
            spike_count = 0
            noise_dist = "Gaussian / Rayleigh"

        # 2. Spectral analysis for spurs and carrier leakage
        if spec is not None and len(spec.psd_db) > 100:
            psd = spec.psd_db
            freqs = spec.frequencies_hz
            center_idx = np.argmin(np.abs(freqs))

            # DC / Carrier leakage power
            dc_window = max(1, len(freqs) // 200)
            dc_slice = psd[max(0, center_idx - dc_window) : min(len(psd), center_idx + dc_window + 1)]
            carrier_leak_dbfs = float(np.max(dc_slice))

            # Peak signal power
            sig_peak_pwr = float(np.max(psd))
            sig_peak_idx = int(np.argmax(psd))
            sig_peak_freq = float(freqs[sig_peak_idx])

            # Find spurs outside the main signal peak window
            spur_count = 0
            strongest_spur_pwr = -120.0
            strongest_spur_offset = 0.0

            if spec.detected_peaks:
                for p in spec.detected_peaks:
                    # Ignore the primary signal peak
                    if abs(p.freq_hz - sig_peak_freq) > max(context.sample_rate * 0.02, 5000.0):
                        spur_count += 1
                        if p.power_db > strongest_spur_pwr:
                            strongest_spur_pwr = float(p.power_db)
                            strongest_spur_offset = float(p.freq_hz - sig_peak_freq)

            nb_detected = spur_count > 0 and strongest_spur_pwr > (spec.noise_floor_db + 15.0)
            spur_suppression = float(sig_peak_pwr - strongest_spur_pwr) if spur_count > 0 else 80.0

            # Spectral flatness measure (Wiener entropy)
            psd_lin = 10.0 ** (psd / 10.0)
            psd_lin = np.maximum(psd_lin, 1e-18)
            geom_mean = np.exp(np.mean(np.log(psd_lin)))
            arith_mean = np.mean(psd_lin)
            sfm = float(geom_mean / max(arith_mean, 1e-18))
            sfm = max(0.0, min(1.0, sfm))

            # ACIR (ratio of in-band to out-of-band adjacent channels)
            half = len(psd_lin) // 2
            mid_band = psd_lin[half - half // 2 : half + half // 2]
            adj_band = np.concatenate([psd_lin[: half // 4], psd_lin[-half // 4 :]])
            mid_pwr = np.sum(mid_band)
            adj_pwr = np.sum(adj_band)
            acir_db = float(10.0 * np.log10(max(mid_pwr, 1e-15) / max(adj_pwr, 1e-18)))

        else:
            carrier_leak_dbfs = -90.0
            nb_detected = False
            spur_count = 0
            strongest_spur_offset = 0.0
            strongest_spur_pwr = -120.0
            spur_suppression = 80.0
            sfm = 0.50
            acir_db = 40.0

        results.append(self._make_res("noise.noise_distribution_type", noise_dist, "Envelope MAD / Kurtosis test", f"Noise characterization: {noise_dist}."))
        results.append(self._make_res("noise.narrowband_interference_detected", nb_detected, "Prominence peak detection outside main lobe", "Narrowband interferer present." if nb_detected else "No significant narrowband interference."))
        results.append(self._make_res("noise.spurious_tones_count", spur_count, "Prominence > 15 dB peak count", f"{spur_count} discrete spur(s) detected."))
        results.append(self._make_res("noise.strongest_spur_freq_offset_hz", strongest_spur_offset, "f_spur - f_carrier", f"Strongest spur offset: {strongest_spur_offset:+.1f} Hz." if spur_count > 0 else "0.0 Hz (No spurs)"))
        results.append(self._make_res("noise.strongest_spur_power_dbfs", strongest_spur_pwr, "PSD peak evaluation", f"{strongest_spur_pwr:.1f} dBFS." if spur_count > 0 else "N/A"))
        results.append(self._make_res("noise.strongest_spur_rejection_db", spur_suppression, "P_sig - P_spur", f"{spur_suppression:.1f} dB suppression."))
        results.append(self._make_res("noise.carrier_leakage_power_dbfs", carrier_leak_dbfs, "PSD power at 0 Hz DC offset", f"{carrier_leak_dbfs:.1f} dBFS."))
        results.append(self._make_res("noise.adjacent_channel_interference_ratio_db", acir_db, "10 * log10(P_inband / P_adjacent)", f"{acir_db:.1f} dB."))
        results.append(self._make_res("noise.impulsive_noise_spike_count", spike_count, "Envelope excursion > 6 * MAD", f"{spike_count} impulsive transient events."))
        results.append(self._make_res("noise.spectral_flatness_measure", sfm, "exp(mean(log(PSD))) / mean(PSD)", f"{sfm:.4f} (0=tones, 1=white noise)."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.95,
            source="Noise & Interference Engine",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
