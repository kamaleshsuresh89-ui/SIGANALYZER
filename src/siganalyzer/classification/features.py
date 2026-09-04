"""Statistical, spectral, and instantaneous feature extraction for modulation classification."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ModulationFeatures:
    """Discriminant features extracted from an I/Q segment."""

    gamma_max: float  # Maximum value of spectral power of normalized amplitude
    sigma_ap: float  # Standard deviation of the absolute value of non-linear phase
    sigma_dp: float  # Standard deviation of direct non-linear phase
    sigma_aa: float  # Standard deviation of normalized amplitude
    sigma_af: float  # Standard deviation of normalized instantaneous frequency
    envelope_variance: float  # Var(|s|) / Mean(|s|)^2 (< 0.1 for constant envelope)
    kurtosis_amp: float  # Kurtosis of amplitude
    freq_peaks_count: int  # Number of distinct frequency modes (for FSK)
    phase_peaks_count: int  # Number of distinct phase histogram clusters (for PSK)


class FeatureExtractor:
    """Extracts instantaneous amplitude, phase, and frequency features."""

    @classmethod
    def extract_features(cls, signal: np.ndarray, sample_rate: float) -> ModulationFeatures:
        """Compute comprehensive statistical and physical features for classification."""
        if len(signal) < 64:
            return ModulationFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Remove DC and normalize
        s = signal - np.mean(signal)
        rms = np.sqrt(np.mean(np.abs(s) ** 2))
        if rms > 0:
            s = s / rms

        # 1. Instantaneous amplitude
        a = np.abs(s)
        mean_a = np.mean(a)
        var_a = np.var(a)
        a_n = a / (mean_a + 1e-12) - 1.0  # Zero-center normalized amplitude

        # gamma_max: max value of normalized centered instantaneous amplitude spectrum
        nfft = min(len(a_n), 2048)
        fft_an = np.abs(np.fft.fft(a_n[:nfft])) ** 2 / nfft
        gamma_max = float(np.max(fft_an))

        # sigma_aa: std of normalized amplitude
        sigma_aa = float(np.std(a_n))
        envelope_var = float(var_a / (mean_a ** 2 + 1e-12))

        # Kurtosis of amplitude
        m4_a = np.mean((a - mean_a) ** 4)
        m2_a = np.mean((a - mean_a) ** 2)
        kurt_a = float(m4_a / (m2_a ** 2 + 1e-12) - 3.0)

        # 2. Instantaneous phase (unwrapped)
        phi = np.unwrap(np.angle(s))
        # Remove linear trend (frequency offset)
        t = np.arange(len(phi))
        if len(t) > 1:
            poly = np.polyfit(t, phi, 1)
            phi_nl = phi - np.polyval(poly, t)  # Non-linear phase
        else:
            phi_nl = phi

        sigma_dp = float(np.std(phi_nl))
        sigma_ap = float(np.std(np.abs(phi_nl)))

        # 3. Instantaneous frequency
        diff_phi = np.diff(phi)
        inst_freq = diff_phi / (2 * np.pi) * sample_rate
        mean_if = np.mean(inst_freq)
        inst_freq_n = (inst_freq - mean_if) / sample_rate
        sigma_af = float(np.std(inst_freq_n))

        # Frequency histogram modes (detecting 2FSK vs 4FSK)
        hist_f, _ = np.histogram(inst_freq, bins=64)
        max_hf = np.max(hist_f)
        freq_peaks = 0
        if max_hf > 0:
            for i in range(1, len(hist_f) - 1):
                if hist_f[i] > hist_f[i - 1] and hist_f[i] > hist_f[i + 1] and hist_f[i] > (max_hf * 0.25):
                    freq_peaks += 1

        # Phase histogram modes (wrapped to [-pi, pi])
        wrapped_phi = np.angle(s)
        hist_p, _ = np.histogram(wrapped_phi, bins=36)
        max_hp = np.max(hist_p)
        phase_peaks = 0
        if max_hp > 0:
            for i in range(1, len(hist_p) - 1):
                if hist_p[i] > hist_p[i - 1] and hist_p[i] > hist_p[i + 1] and hist_p[i] > (max_hp * 0.25):
                    phase_peaks += 1

        return ModulationFeatures(
            gamma_max=gamma_max,
            sigma_ap=sigma_ap,
            sigma_dp=sigma_dp,
            sigma_aa=sigma_aa,
            sigma_af=sigma_af,
            envelope_variance=envelope_var,
            kurtosis_amp=kurt_a,
            freq_peaks_count=int(freq_peaks),
            phase_peaks_count=int(phase_peaks),
        )
