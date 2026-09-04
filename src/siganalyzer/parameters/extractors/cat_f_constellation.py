"""Category F: Constellation & Modulation Quality parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class ConstellationQualityExtractor(BaseParameterExtractor):
    """Extracts constellation metrics, EVM, MER, phase/magnitude errors, and IQ impairments."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.CONSTELLATION

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("const.cluster_count_detected", "Detected Cluster Count", cat, "clusters", int, "Number of distinct symbol decision clusters", ConfidenceLevel.INFERRED, export_priority=1),
            ParameterDefinition("const.cluster_separation_metric", "Cluster Separation Metric", cat, "", float, "Normalized minimum distance between cluster centroids", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("const.constellation_spread_radius", "Constellation Spread Radius", cat, "", float, "RMS variance of symbols about their decision centers", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("const.evm_rms_percent", "EVM RMS", cat, "%", float, "Root-mean-square Error Vector Magnitude", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("const.evm_peak_percent", "EVM Peak", cat, "%", float, "Maximum instantaneous symbol error vector magnitude", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("const.evm_95th_percentile_percent", "EVM 95th Percentile", cat, "%", float, "95th percentile Error Vector Magnitude", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("const.mer_db", "Modulation Error Ratio (MER)", cat, "dB", float, "Ratio of reference signal power to error vector power", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("const.phase_error_rms_deg", "RMS Phase Error", cat, "deg", float, "Root-mean-square angular symbol displacement", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("const.magnitude_error_rms_percent", "RMS Magnitude Error", cat, "%", float, "Normalized root-mean-square radial symbol error", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("const.common_phase_error_deg", "Common Phase Error (CPE)", cat, "deg", float, "Constant angular bias across constellation", ConfidenceLevel.MEASURED, export_priority=10),
            ParameterDefinition("const.quadrature_phase_error_deg", "Quadrature Phase Error", cat, "deg", float, "Orthogonality departure of I/Q axes from 90 degrees", ConfidenceLevel.MEASURED, export_priority=11),
            ParameterDefinition("const.iq_gain_imbalance_db", "I/Q Gain Imbalance", cat, "dB", float, "Amplitude imbalance between in-phase and quadrature branches", ConfidenceLevel.MEASURED, export_priority=12),
            ParameterDefinition("const.carrier_leakage_db", "Carrier (LO) Leakage", cat, "dB", float, "Relative power of residual DC carrier tone", ConfidenceLevel.MEASURED, export_priority=13),
            ParameterDefinition("const.image_rejection_ratio_db", "Image Rejection Ratio (IRR)", cat, "dB", float, "Rejection depth against negative frequency image", ConfidenceLevel.MEASURED, export_priority=14),
            ParameterDefinition("const.dc_offset_i", "DC Offset (I)", cat, "", float, "Mean DC offset on in-phase branch", ConfidenceLevel.MEASURED, export_priority=15),
            ParameterDefinition("const.dc_offset_q", "DC Offset (Q)", cat, "", float, "Mean DC offset on quadrature branch", ConfidenceLevel.MEASURED, export_priority=16),
            ParameterDefinition("const.snr_from_evm_db", "EVM-Inferred SNR", cat, "dB", float, "Signal-to-noise ratio calculated from EVM", ConfidenceLevel.ESTIMATED, export_priority=17),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        demod = context.demodulation
        metrics = context.constellation_metrics
        raw = context.raw_samples
        results: list[ParameterResult] = []

        # 1. Hardware/Raw I/Q impairments (computable directly from raw samples)
        i_raw = np.real(raw)
        q_raw = np.imag(raw) if np.iscomplexobj(raw) else np.zeros_like(i_raw)

        dc_i = float(np.mean(i_raw)) if len(i_raw) > 0 else 0.0
        dc_q = float(np.mean(q_raw)) if len(q_raw) > 0 else 0.0

        i_std = float(np.std(i_raw)) if len(i_raw) > 0 else 1.0
        q_std = float(np.std(q_raw)) if len(q_raw) > 0 else 1.0

        gain_imb_db = float(20.0 * np.log10(max(i_std, 1e-12) / max(q_std, 1e-12))) if np.iscomplexobj(raw) else 0.0

        # Quadrature orthogonality error: arcsin( correlation(I, Q) )
        if np.iscomplexobj(raw) and len(i_raw) > 1 and i_std > 1e-9 and q_std > 1e-9:
            i_centered = i_raw - dc_i
            q_centered = q_raw - dc_q
            corr = float(np.dot(i_centered, q_centered) / (len(i_raw) * i_std * q_std))
            corr = float(np.clip(corr, -1.0, 1.0))
            quad_err_deg = float(np.degrees(np.arcsin(corr)))
        else:
            quad_err_deg = 0.0

        # Carrier leakage (relative DC power)
        total_pwr = float(np.mean(np.abs(raw) ** 2)) if len(raw) > 0 else 1.0
        dc_pwr = dc_i ** 2 + dc_q ** 2
        carrier_leakage_db = float(10.0 * np.log10(max(dc_pwr, 1e-15) / max(total_pwr, 1e-12)))

        # Image Rejection Ratio estimation:
        gain_lin = 10.0 ** (gain_imb_db / 20.0)
        phi_rad = np.radians(quad_err_deg)
        num = 1.0 - 2.0 * gain_lin * np.cos(phi_rad) + gain_lin ** 2
        denom = 1.0 + 2.0 * gain_lin * np.cos(phi_rad) + gain_lin ** 2
        if denom > 1e-12 and num > 1e-12:
            irr_db = float(-10.0 * np.log10(num / denom))
            irr_db = max(0.0, min(100.0, irr_db))
        else:
            irr_db = 60.0

        results.append(self._make_res("const.dc_offset_i", dc_i, "mean(real(s_raw))", "Direct current bias on in-phase branch."))
        results.append(self._make_res("const.dc_offset_q", dc_q, "mean(imag(s_raw))", "Direct current bias on quadrature branch."))
        results.append(self._make_res("const.iq_gain_imbalance_db", gain_imb_db, "20 * log10(std(I) / std(Q))", "In-phase to quadrature amplitude ratio imbalance."))
        results.append(self._make_res("const.quadrature_phase_error_deg", quad_err_deg, "arcsin(cov(I,Q) / (std(I)*std(Q)))", "Departure from ideal 90 degree orthogonality."))
        results.append(self._make_res("const.carrier_leakage_db", carrier_leakage_db, "10 * log10(P_dc / P_total)", "Residual carrier LO leakage relative to total power."))
        results.append(self._make_res("const.image_rejection_ratio_db", irr_db, "Analytic IRR from gain/quadrature imbalance", "Image channel rejection suppression ratio."))

        # 2. Symbol constellation metrics (from demodulation / constellation analyzer)
        symbols = demod.symbols if (demod and len(demod.symbols) > 0) else None

        if symbols is not None and len(symbols) >= 16:
            norm_syms = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2) + 1e-12)

            # Determine reference constellation
            order_m = 4
            if context.modulation:
                if "BPSK" in context.modulation.scheme or "2FSK" in context.modulation.scheme:
                    order_m = 2
                elif "8PSK" in context.modulation.scheme:
                    order_m = 8
                elif "16-QAM" in context.modulation.scheme:
                    order_m = 16

            if order_m == 2:
                ref = np.array([-1.0 + 0j, 1.0 + 0j], dtype=np.complex64)
            elif order_m == 8:
                angles = np.arange(8) * (2.0 * np.pi / 8.0)
                ref = np.exp(1j * angles).astype(np.complex64)
            elif order_m == 16:
                grid = np.array([-3, -1, 1, 3])
                i_grid, q_grid = np.meshgrid(grid, grid)
                ref = (i_grid.ravel() + 1j * q_grid.ravel()).astype(np.complex64)
                ref = ref / np.sqrt(np.mean(np.abs(ref) ** 2))
            else:  # QPSK default
                ref = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex64) / np.sqrt(2)

            # Nearest neighbor association
            diffs = norm_syms[:, np.newaxis] - ref[np.newaxis, :]
            distances = np.abs(diffs)
            best_idx = np.argmin(distances, axis=1)
            ideal_syms = ref[best_idx]
            err_vec = norm_syms - ideal_syms

            err_pwr = np.mean(np.abs(err_vec) ** 2)
            ref_pwr = np.mean(np.abs(ideal_syms) ** 2)
            evm_rms = float(np.sqrt(err_pwr / max(ref_pwr, 1e-12)) * 100.0)
            evm_peak = float(np.max(np.abs(err_vec) / np.sqrt(max(ref_pwr, 1e-12))) * 100.0)
            evm_95 = float(np.percentile(np.abs(err_vec) / np.sqrt(max(ref_pwr, 1e-12)), 95) * 100.0)
            mer = float(-20.0 * np.log10(max(evm_rms / 100.0, 1e-6)))

            # Phase & Magnitude error
            phase_err = np.angle(norm_syms) - np.angle(ideal_syms)
            phase_err = (phase_err + np.pi) % (2 * np.pi) - np.pi
            cpe = float(np.degrees(np.mean(phase_err)))
            phase_err_rms = float(np.degrees(np.std(phase_err)))

            mag_meas = np.abs(norm_syms)
            mag_ideal = np.abs(ideal_syms)
            mag_err_rms = float(np.sqrt(np.mean(((mag_meas - mag_ideal) / np.maximum(mag_ideal, 1e-6)) ** 2)) * 100.0)

            # Cluster separation and spread
            unique_clusters = np.unique(best_idx)
            cluster_count = len(unique_clusters)

            if len(ref) > 1:
                ref_diffs = ref[:, np.newaxis] - ref[np.newaxis, :]
                ref_dists = np.abs(ref_diffs)
                np.fill_diagonal(ref_dists, np.inf)
                min_sep = float(np.min(ref_dists))
            else:
                min_sep = 1.0

            spread_rad = float(np.sqrt(err_pwr))

            results.append(self._make_res("const.cluster_count_detected", cluster_count, "Voronoi cluster assignment", "Number of distinct occupied symbol decision regions."))
            results.append(self._make_res("const.cluster_separation_metric", min_sep, "min(||c_i - c_j||)", "Normalized minimum Euclidean distance between constellation points."))
            results.append(self._make_res("const.constellation_spread_radius", spread_rad, "RMS(||s_i - c_i||)", "Root-mean-square symbol cloud radius around reference centroids."))
            results.append(self._make_res("const.evm_rms_percent", evm_rms, "sqrt(P_err / P_ref) * 100%", "RMS Error Vector Magnitude across all decoded symbols."))
            results.append(self._make_res("const.evm_peak_percent", evm_peak, "max(||e_k|| / ||s_ref||) * 100%", "Peak instantaneous Error Vector Magnitude."))
            results.append(self._make_res("const.evm_95th_percentile_percent", evm_95, "Percentile 95 of EVM", "95th percentile symbol error magnitude threshold."))
            results.append(self._make_res("const.mer_db", mer, "-20 * log10(EVM_RMS)", "Modulation Error Ratio in decibels."))
            results.append(self._make_res("const.phase_error_rms_deg", phase_err_rms, "std(angle(s_meas) - angle(s_ref))", "RMS phase jitter / angular error."))
            results.append(self._make_res("const.magnitude_error_rms_percent", mag_err_rms, "RMS radial distance error", "RMS magnitude deviation from ideal constellation radius."))
            results.append(self._make_res("const.common_phase_error_deg", cpe, "mean(angle(s_meas) - angle(s_ref))", "Residual carrier phase bias across all symbols."))
            results.append(self._make_res("const.snr_from_evm_db", mer, "SNR ≈ MER = -20 * log10(EVM)", "Effective SNR derived from constellation scatter."))

        else:
            for pid, desc in [
                ("const.cluster_count_detected", "Demodulated symbols unavailable"),
                ("const.cluster_separation_metric", "Constellation symbols unavailable"),
                ("const.constellation_spread_radius", "Constellation symbols unavailable"),
                ("const.evm_rms_percent", "Demodulation did not produce recovered symbols"),
                ("const.evm_peak_percent", "Demodulation did not produce recovered symbols"),
                ("const.evm_95th_percentile_percent", "Demodulation did not produce recovered symbols"),
                ("const.mer_db", "Demodulation did not produce recovered symbols"),
                ("const.phase_error_rms_deg", "Phase error cannot be computed without symbol decisions"),
                ("const.magnitude_error_rms_percent", "Magnitude error cannot be computed without symbol decisions"),
                ("const.common_phase_error_deg", "Common phase error cannot be computed without symbol decisions"),
                ("const.snr_from_evm_db", "EVM SNR cannot be derived without symbol decisions"),
            ]:
                results.append(ParameterResult(
                    definition=self._get_def(pid),
                    value=None,
                    status=ConfidenceLevel.UNKNOWN,
                    confidence_score=0.0,
                    source="Constellation Analyzer",
                    method="Euclidean decision slicer",
                    explanation=f"UNKNOWN: {desc}.",
                ))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.MEASURED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.95 if status == ConfidenceLevel.MEASURED else 0.85,
            source="Constellation Analyzer",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
