"""Category N: Statistical & Higher-Order Features parameter extractor."""

from typing import Any
import numpy as np
from scipy import stats

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class StatisticalHocExtractor(BaseParameterExtractor):
    """Extracts Higher-Order Cumulants (C20..C63), skewness, kurtosis, and bimodality coefficients."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.STATISTICAL_HOC

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("stat.cumulant_c20_mag", "Cumulant |C20|", cat, "", float, "Magnitude of 2nd-order non-conjugate cumulant E[s^2]", ConfidenceLevel.MEASURED, export_priority=1),
            ParameterDefinition("stat.cumulant_c21_mag", "Cumulant |C21|", cat, "", float, "Magnitude of 2nd-order conjugate cumulant E[|s|^2]", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("stat.cumulant_c40_mag", "Cumulant |C40|", cat, "", float, "Magnitude of 4th-order non-conjugate cumulant", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("stat.cumulant_c41_mag", "Cumulant |C41|", cat, "", float, "Magnitude of 4th-order 1-conjugate cumulant", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("stat.cumulant_c42_real", "Cumulant Re(C42)", cat, "", float, "Real part of 4th-order 2-conjugate cumulant", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("stat.cumulant_c60_mag", "Cumulant |C60|", cat, "", float, "Magnitude of 6th-order non-conjugate cumulant", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("stat.cumulant_c63_real", "Cumulant Re(C63)", cat, "", float, "Real part of 6th-order 3-conjugate cumulant", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("stat.amplitude_skewness", "Amplitude Skewness", cat, "", float, "3rd standardized moment of envelope magnitude distribution", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("stat.amplitude_kurtosis", "Amplitude Excess Kurtosis", cat, "", float, "4th standardized moment of envelope magnitude distribution", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("stat.bimodality_coefficient", "Bimodality Coefficient", cat, "", float, "Sarle's bimodality metric (> 0.555 indicates bimodal)", ConfidenceLevel.MEASURED, export_priority=10),
            ParameterDefinition("stat.phase_variance", "Phase Variance", cat, "rad^2", float, "Variance of wrapped phase distribution", ConfidenceLevel.MEASURED, export_priority=11),
            ParameterDefinition("stat.envelope_moment_m2", "Envelope Central Moment M2", cat, "", float, "Second central moment of instantaneous envelope", ConfidenceLevel.MEASURED, export_priority=12),
            ParameterDefinition("stat.envelope_moment_m4", "Envelope Central Moment M4", cat, "", float, "Fourth central moment of instantaneous envelope", ConfidenceLevel.MEASURED, export_priority=13),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        samples = context.conditioned_samples
        results: list[ParameterResult] = []

        if len(samples) < 16:
            for d in self.supported_parameters():
                results.append(ParameterResult(d, None, ConfidenceLevel.UNKNOWN, 0.0, "HOC Engine", "N/A", "Insufficient samples."))
            return results

        # Normalize to unit average power for scale-invariant cumulants
        s = samples / np.sqrt(np.mean(np.abs(samples) ** 2) + 1e-12)
        s_conj = np.conj(s)

        # Moments
        m20 = np.mean(s ** 2)
        m21 = np.mean(np.abs(s) ** 2)
        m40 = np.mean(s ** 4)
        m41 = np.mean((s ** 3) * s_conj)
        m42 = np.mean(np.abs(s) ** 4)
        m60 = np.mean(s ** 6)
        m63 = np.mean(np.abs(s) ** 6)

        # Cumulants
        c20 = m20
        c21 = m21
        c40 = m40 - 3.0 * (m20 ** 2)
        c41 = m41 - 3.0 * m20 * m21
        c42 = m42 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)

        # 6th order cumulants
        c60 = m60 - 15.0 * m40 * m20 + 30.0 * (m20 ** 3)
        c63 = m63 - 6.0 * c42 * c21 - 9.0 * (c20 * np.conj(c41) + np.conj(c20) * c41) - 6.0 * (c21 ** 3)

        # Envelope statistics
        env = context.envelope
        env_skew = float(stats.skew(env)) if len(env) > 2 else 0.0
        env_kurt = float(stats.kurtosis(env)) if len(env) > 3 else 0.0

        # Sarle's bimodality coefficient: b = (skew^2 + 1) / (kurt + 3)
        # Standard excess kurtosis is used, so denominator is kurt + 3
        bimod = float((env_skew ** 2 + 1.0) / max(env_kurt + 3.0, 1e-6))
        bimod = max(0.0, min(1.0, bimod))

        # Phase variance
        phase_var = float(np.var(context.phase)) if len(context.phase) > 0 else 0.0

        # Envelope central moments
        env_mean = float(np.mean(env))
        env_cent = env - env_mean
        m2_env = float(np.mean(env_cent ** 2))
        m4_env = float(np.mean(env_cent ** 4))

        results.append(self._make_res("stat.cumulant_c20_mag", float(np.abs(c20)), "|E[s^2]|", f"|C20| = {float(np.abs(c20)):.4f}."))
        results.append(self._make_res("stat.cumulant_c21_mag", float(np.abs(c21)), "|E[|s|^2]|", f"|C21| = {float(np.abs(c21)):.4f}."))
        results.append(self._make_res("stat.cumulant_c40_mag", float(np.abs(c40)), "|E[s^4] - 3 E^2[s^2]|", f"|C40| = {float(np.abs(c40)):.4f}."))
        results.append(self._make_res("stat.cumulant_c41_mag", float(np.abs(c41)), "|E[s^3 s*] - 3 E[s^2] E[|s|^2]|", f"|C41| = {float(np.abs(c41)):.4f}."))
        results.append(self._make_res("stat.cumulant_c42_real", float(np.real(c42)), "Re(E[|s|^4] - |E[s^2]|^2 - 2 E^2[|s|^2])", f"Re(C42) = {float(np.real(c42)):.4f} (Key PSK/QAM discriminator)."))
        results.append(self._make_res("stat.cumulant_c60_mag", float(np.abs(c60)), "|6th order cumulant C60|", f"|C60| = {float(np.abs(c60)):.4f}."))
        results.append(self._make_res("stat.cumulant_c63_real", float(np.real(c63)), "Re(6th order cumulant C63)", f"Re(C63) = {float(np.real(c63)):.4f}."))
        results.append(self._make_res("stat.amplitude_skewness", env_skew, "scipy.stats.skew(envelope)", f"Envelope asymmetry: {env_skew:.4f}."))
        results.append(self._make_res("stat.amplitude_kurtosis", env_kurt, "scipy.stats.kurtosis(envelope)", f"Envelope peakedness: {env_kurt:.4f}."))
        results.append(self._make_res("stat.bimodality_coefficient", bimod, "(skew^2 + 1) / (kurt + 3)", f"Bimodality: {bimod:.4f} (threshold: 0.555)."))
        results.append(self._make_res("stat.phase_variance", phase_var, "var(angle(s))", f"Phase variance: {phase_var:.4f} rad^2."))
        results.append(self._make_res("stat.envelope_moment_m2", m2_env, "mean((env - mean(env))^2)", f"M2: {m2_env:.6e}."))
        results.append(self._make_res("stat.envelope_moment_m4", m4_env, "mean((env - mean(env))^4)", f"M4: {m4_env:.6e}."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=ConfidenceLevel.MEASURED,
            confidence_score=0.98,
            source="HOC & Statistical Engine",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
