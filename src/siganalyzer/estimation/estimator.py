"""Automated signal parameter estimation: carrier, bandwidth, SNR, symbol rate, and cumulants."""

from dataclasses import dataclass, field
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import Measurement
from siganalyzer.dsp.spectrum import SpectralAnalyzer


@dataclass
class CumulantFeatures:
    """Calculated higher-order statistical cumulants for modulation analysis."""

    c20: complex
    c21: float
    c40: complex
    c41: complex
    c42: float
    c60: complex
    c63: float
    norm_c42: float  # Key discriminator (QPSK ≈ -1.0, BPSK ≈ -2.0, 16QAM ≈ -0.68)
    norm_c40: complex
    norm_c63: float


@dataclass
class ParameterEstimateResult:
    """Aggregated parameter estimates with confidence annotations."""

    carrier_freq_hz: Measurement[float]
    occupied_bandwidth_hz: Measurement[float]
    bandwidth_3db_hz: Measurement[float]
    snr_db: Measurement[float]
    symbol_rate_sps: Measurement[float]
    bit_rate_bps: Measurement[float]
    cumulants: CumulantFeatures


class ParameterEstimator:
    """Estimates fundamental RF and digital communication parameters from I/Q samples."""

    @classmethod
    def estimate_all(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        nominal_center_freq: float | None = None,
    ) -> ParameterEstimateResult:
        """Run complete parameter extraction pipeline."""
        if len(signal) == 0:
            return cls._empty_result()

        # Step 1: Compute spectral profile
        spec = SpectralAnalyzer.analyze_spectrum(signal, sample_rate, nfft=2048)

        # Step 2: Estimate Carrier / Center frequency
        carrier_meas = cls.estimate_carrier_freq(spec.frequencies_hz, spec.psd_db, nominal_center_freq)

        # Step 3: Estimate Bandwidth (99% OBW and -3dB)
        obw_meas, bw3db_meas = cls.estimate_bandwidth(spec.frequencies_hz, spec.psd_db, spec.noise_floor_db)

        # Step 4: Estimate SNR
        snr_meas = cls.estimate_snr(spec.psd_db, spec.noise_floor_db, spec.peak_power_db)

        # Step 5: Estimate Symbol Rate
        sym_meas = cls.estimate_symbol_rate(signal, sample_rate, obw_meas.value)

        # Step 6: Compute Higher-Order Cumulants
        cumulants = cls.compute_cumulants(signal)

        # Preliminary bit rate estimate (assuming 1-2 bits/symbol)
        bit_rate_val = sym_meas.value * 2.0 if sym_meas.value is not None else None
        bit_rate_meas = Measurement(
            name="Bit Rate",
            value=bit_rate_val,
            unit="bps",
            confidence=ConfidenceLevel.INFERRED if bit_rate_val is not None else ConfidenceLevel.UNKNOWN,
            confidence_score=0.7 if bit_rate_val is not None else 0.0,
            source="Estimated from symbol rate (nominal 2 bits/sym)",
        )

        return ParameterEstimateResult(
            carrier_freq_hz=carrier_meas,
            occupied_bandwidth_hz=obw_meas,
            bandwidth_3db_hz=bw3db_meas,
            snr_db=snr_meas,
            symbol_rate_sps=sym_meas,
            bit_rate_bps=bit_rate_meas,
            cumulants=cumulants,
        )

    @staticmethod
    def estimate_carrier_freq(
        freqs_hz: np.ndarray, psd_db: np.ndarray, nominal_center_freq: float | None
    ) -> Measurement[float]:
        """Estimate center frequency offset or absolute RF carrier frequency using band midpoint."""
        peak_idx = int(np.argmax(psd_db))
        peak_pwr = psd_db[peak_idx]
        target_6db = peak_pwr - 6.0

        # Scan left from peak to find -6dB boundary
        left_idx = peak_idx
        while left_idx > 0 and psd_db[left_idx] > target_6db:
            left_idx -= 1

        # Scan right from peak to find -6dB boundary
        right_idx = peak_idx
        while right_idx < len(psd_db) - 1 and psd_db[right_idx] > target_6db:
            right_idx += 1

        # Midpoint of the main lobe
        offset_hz = float((freqs_hz[left_idx] + freqs_hz[right_idx]) / 2.0)

        if nominal_center_freq is not None:
            absolute_carrier = nominal_center_freq + offset_hz
            return Measurement(
                name="Carrier Frequency",
                value=absolute_carrier,
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.95,
                source=f"Nominal ({nominal_center_freq / 1e6:.3f} MHz) + Band midpoint ({offset_hz / 1e3:.1f} kHz)",
            )
        else:
            return Measurement(
                name="Carrier Frequency Offset",
                value=offset_hz,
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.90,
                source="Baseband band midpoint",
            )

    @staticmethod
    def estimate_bandwidth(
        freqs_hz: np.ndarray, psd_db: np.ndarray, noise_floor_db: float
    ) -> tuple[Measurement[float], Measurement[float]]:
        """Compute 99% Occupied Bandwidth (OBW) and -3dB bandwidth."""
        # Convert dB to linear power
        pwr_linear = 10.0 ** (psd_db / 10.0)
        total_pwr = np.sum(pwr_linear)

        if total_pwr <= 0:
            return (
                Measurement("Occupied Bandwidth (99%)", None, "Hz", ConfidenceLevel.UNKNOWN),
                Measurement("Bandwidth (-3dB)", None, "Hz", ConfidenceLevel.UNKNOWN),
            )

        # 99% OBW: cumulative power from 0.5% to 99.5%
        cum_pwr = np.cumsum(pwr_linear) / total_pwr
        idx_low = int(np.searchsorted(cum_pwr, 0.005))
        idx_high = int(np.searchsorted(cum_pwr, 0.995))
        idx_high = min(idx_high, len(freqs_hz) - 1)

        obw = float(abs(freqs_hz[idx_high] - freqs_hz[idx_low]))

        # -3dB Bandwidth
        peak_idx = int(np.argmax(psd_db))
        peak_db = psd_db[peak_idx]
        target_3db = peak_db - 3.0

        # Scan left
        left_idx = peak_idx
        while left_idx > 0 and psd_db[left_idx] > target_3db:
            left_idx -= 1

        # Scan right
        right_idx = peak_idx
        while right_idx < len(psd_db) - 1 and psd_db[right_idx] > target_3db:
            right_idx += 1

        bw_3db = float(abs(freqs_hz[right_idx] - freqs_hz[left_idx]))

        return (
            Measurement(
                name="Occupied Bandwidth (99%)",
                value=obw,
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.92,
                source="99% cumulative power integral",
            ),
            Measurement(
                name="Bandwidth (-3dB)",
                value=bw_3db,
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.88,
                source="-3dB power drop from peak",
            ),
        )

    @staticmethod
    def estimate_snr(psd_db: np.ndarray, noise_floor_db: float, peak_power_db: float) -> Measurement[float]:
        """Estimate Signal-to-Noise Ratio (SNR)."""
        snr_val = float(max(peak_power_db - noise_floor_db, 0.0))
        return Measurement(
            name="SNR",
            value=snr_val,
            unit="dB",
            confidence=ConfidenceLevel.ESTIMATED,
            confidence_score=0.85,
            source="Peak spectral power minus estimated noise floor",
        )

    @classmethod
    def estimate_symbol_rate(
        cls, signal: np.ndarray, sample_rate: float, obw_hz: float | None
    ) -> Measurement[float]:
        """Estimate symbol rate using cyclostationary non-linearities and spectral line analysis."""
        if len(signal) < 1024:
            return Measurement("Symbol Rate", None, "sps", ConfidenceLevel.UNKNOWN)

        # Normalize signal
        norm_sig = signal / np.sqrt(np.mean(np.abs(signal) ** 2) + 1e-12)

        # 1. Squaring magnitude non-linearity |s[n]|^2
        mag_sq = np.abs(norm_sig) ** 2
        mag_sq_zero_mean = mag_sq - np.mean(mag_sq)

        # FFT of non-linearity
        nfft = min(len(mag_sq_zero_mean), 8192)
        fft_nl = np.abs(np.fft.rfft(mag_sq_zero_mean[:nfft]))
        freqs_nl = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

        # Ignore DC and extreme low frequencies
        valid_mask = freqs_nl > (sample_rate * 0.005)
        if obw_hz is not None and obw_hz > 0:
            # Symbol rate is typically bounded by OBW
            valid_mask &= (freqs_nl <= obw_hz * 1.5)

        if not np.any(valid_mask):
            valid_mask = freqs_nl > (sample_rate * 0.01)

        masked_fft = fft_nl.copy()
        masked_fft[~valid_mask] = 0.0

        peak_idx = int(np.argmax(masked_fft))
        cand_sym_rate = float(freqs_nl[peak_idx])
        peak_amp = masked_fft[peak_idx]
        mean_floor = np.mean(masked_fft[valid_mask]) + 1e-9
        snr_line = peak_amp / mean_floor

        # If spectral line is strong (> 3x background floor)
        if snr_line > 3.0 and cand_sym_rate > 0:
            return Measurement(
                name="Symbol Rate",
                value=cand_sym_rate,
                unit="sps",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=min(0.95, 0.5 + 0.1 * snr_line),
                source="Cyclostationary magnitude squaring line",
            )
        elif obw_hz is not None and obw_hz > 0:
            # Heuristic fallback: OBW / (1 + alpha) with alpha ~ 0.35
            est_sym = obw_hz / 1.35
            return Measurement(
                name="Symbol Rate",
                value=est_sym,
                unit="sps",
                confidence=ConfidenceLevel.INFERRED,
                confidence_score=0.60,
                source="Inferred from Occupied Bandwidth (assuming RRC alpha=0.35)",
            )
        else:
            return Measurement("Symbol Rate", None, "sps", ConfidenceLevel.UNKNOWN)

    @staticmethod
    def compute_cumulants(signal: np.ndarray) -> CumulantFeatures:
        """Compute 2nd, 4th, and 6th-order cumulants."""
        if len(signal) < 128:
            return CumulantFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Normalize to unit variance
        s = signal - np.mean(signal)
        std_val = np.sqrt(np.mean(np.abs(s) ** 2))
        if std_val > 0:
            s = s / std_val

        s_sq = s ** 2
        s_conj = np.conj(s)
        abs_sq = np.abs(s) ** 2

        # 2nd Order
        m20 = np.mean(s_sq)
        m21 = float(np.mean(abs_sq))

        c20 = m20
        c21 = m21

        # 4th Order moments
        m40 = np.mean(s ** 4)
        m41 = np.mean((s ** 3) * s_conj)
        m42 = float(np.mean(abs_sq ** 2))

        # 4th Order cumulants
        c40 = m40 - 3 * (c20 ** 2)
        c41 = m41 - 3 * c20 * c21
        c42 = m42 - np.abs(c20) ** 2 - 2 * (c21 ** 2)

        # 6th Order
        m60 = np.mean(s ** 6)
        m63 = float(np.mean(abs_sq ** 3))
        c60 = m60 - 15 * c20 * c40 - 30 * (c20 ** 3)
        c63 = m63 - 6 * c20 * np.conj(c41) - 9 * (c21 * c42) - 6 * (c21 ** 3)

        # Normalized cumulants
        norm_c42 = float(c42 / (c21 ** 2)) if c21 > 0 else 0.0
        norm_c40 = c40 / (c21 ** 2) if c21 > 0 else 0.0
        norm_c63 = float(np.real(c63) / (c21 ** 3)) if c21 > 0 else 0.0

        return CumulantFeatures(
            c20=c20,
            c21=c21,
            c40=c40,
            c41=c41,
            c42=c42,
            c60=c60,
            c63=float(np.real(c63)),
            norm_c42=norm_c42,
            norm_c40=norm_c40,
            norm_c63=norm_c63,
        )

    @classmethod
    def _empty_result(cls) -> ParameterEstimateResult:
        return ParameterEstimateResult(
            carrier_freq_hz=Measurement("Carrier Frequency", None, "Hz", ConfidenceLevel.UNKNOWN),
            occupied_bandwidth_hz=Measurement("Occupied Bandwidth", None, "Hz", ConfidenceLevel.UNKNOWN),
            bandwidth_3db_hz=Measurement("Bandwidth (-3dB)", None, "Hz", ConfidenceLevel.UNKNOWN),
            snr_db=Measurement("SNR", None, "dB", ConfidenceLevel.UNKNOWN),
            symbol_rate_sps=Measurement("Symbol Rate", None, "sps", ConfidenceLevel.UNKNOWN),
            bit_rate_bps=Measurement("Bit Rate", None, "bps", ConfidenceLevel.UNKNOWN),
            cumulants=CumulantFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
