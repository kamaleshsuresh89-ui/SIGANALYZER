"""Category E: Modulation Analysis parameter extractor."""

from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class ModulationExtractor(BaseParameterExtractor):
    """Extracts modulation scheme, family, order, symbol rate, and pulse shaping parameters."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.MODULATION

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("mod.detected_scheme", "Detected Modulation Scheme", cat, "", str, "Primary classified modulation type", ConfidenceLevel.INFERRED, export_priority=1),
            ParameterDefinition("mod.modulation_family", "Modulation Family", cat, "", str, "Broad modulation class (PSK, FSK, QAM, ASK, CW)", ConfidenceLevel.INFERRED, export_priority=2),
            ParameterDefinition("mod.confidence_score_percent", "Classification Confidence", cat, "%", float, "Multi-factor agreement confidence score", ConfidenceLevel.INFERRED, export_priority=3),
            ParameterDefinition("mod.modulation_order_m", "Modulation Order (M)", cat, "symbols", int, "Number of discrete constellation / frequency states", ConfidenceLevel.INFERRED, export_priority=4),
            ParameterDefinition("mod.bits_per_symbol", "Bits per Symbol", cat, "bits/sym", int, "Information capacity per transmitted symbol", ConfidenceLevel.INFERRED, export_priority=5),
            ParameterDefinition("mod.symbol_rate_sps", "Estimated Symbol Rate", cat, "sps", float, "Baud rate of digital communication", ConfidenceLevel.ESTIMATED, export_priority=6),
            ParameterDefinition("mod.samples_per_symbol", "Samples per Symbol (sps)", cat, "samples", float, "Ratio of sample rate to symbol rate", ConfidenceLevel.ESTIMATED, export_priority=7),
            ParameterDefinition("mod.fsk_frequency_deviation_hz", "FSK Frequency Deviation", cat, "Hz", float, "Peak frequency shift from carrier", ConfidenceLevel.ESTIMATED, export_priority=8),
            ParameterDefinition("mod.fsk_modulation_index", "Modulation Index (h)", cat, "", float, "FSK tone spacing to symbol rate ratio (2 * dev / R_s)", ConfidenceLevel.ESTIMATED, export_priority=9),
            ParameterDefinition("mod.amplitude_states_count", "Amplitude States Count", cat, "", int, "Number of distinct amplitude levels", ConfidenceLevel.INFERRED, export_priority=10),
            ParameterDefinition("mod.phase_states_count", "Phase States Count", cat, "", int, "Number of distinct phase states", ConfidenceLevel.INFERRED, export_priority=11),
            ParameterDefinition("mod.pulse_shaping_filter", "Estimated Pulse Shaping Filter", cat, "", str, "Filter impulse response profile", ConfidenceLevel.INFERRED, export_priority=12),
            ParameterDefinition("mod.pulse_shaping_rolloff_alpha", "Estimated Roll-off (Alpha)", cat, "", float, "Excess bandwidth factor", ConfidenceLevel.ESTIMATED, export_priority=13),
            ParameterDefinition("mod.secondary_candidate", "Secondary Candidate Scheme", cat, "", str, "Next most probable alternative modulation", ConfidenceLevel.INFERRED, export_priority=14),
            ParameterDefinition("mod.physical_evidence", "Supporting Evidence", cat, "", list, "List of mathematical criteria supporting classification", ConfidenceLevel.MEASURED, export_priority=15),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        mod = context.modulation
        params = context.primary_parameters
        results: list[ParameterResult] = []

        if mod is None:
            return [
                ParameterResult(d, None, ConfidenceLevel.UNKNOWN, 0.0, "Classifier", "N/A", "Modulation classification not executed.")
                for d in self.supported_parameters()
            ]

        scheme = mod.scheme
        conf_pct = float(mod.confidence_score * 100.0)

        # Family derivation
        if "PSK" in scheme:
            family = "Phase Shift Keying (PSK)"
        elif "FSK" in scheme or "MSK" in scheme:
            family = "Frequency Shift Keying (FSK)"
        elif "QAM" in scheme:
            family = "Quadrature Amplitude Modulation (QAM)"
        elif "ASK" in scheme or "OOK" in scheme:
            family = "Amplitude Shift Keying (ASK)"
        else:
            family = "Analog / Unclassified"

        # Order M and bits per symbol
        if "BPSK" in scheme or "2FSK" in scheme or "OOK" in scheme:
            order_m = 2
            bits_per_sym = 1
        elif "QPSK" in scheme or "4FSK" in scheme or "ASK" in scheme:
            order_m = 4
            bits_per_sym = 2
        elif "8PSK" in scheme:
            order_m = 8
            bits_per_sym = 3
        elif "16-QAM" in scheme:
            order_m = 16
            bits_per_sym = 4
        elif "64-QAM" in scheme:
            order_m = 64
            bits_per_sym = 6
        else:
            order_m = 2
            bits_per_sym = 1

        sym_rate = float(params.symbol_rate_sps.value) if (params and params.symbol_rate_sps.value is not None) else None
        sps = float(context.sample_rate / sym_rate) if (sym_rate and sym_rate > 0) else None

        results.append(self._make_res("mod.detected_scheme", scheme, "Hybrid HOC + Physical Criteria", "Top classified digital/analog modulation scheme."))
        results.append(self._make_res("mod.modulation_family", family, "Scheme mapping", "Broad communication modulation family."))
        results.append(self._make_res("mod.confidence_score_percent", conf_pct, "Multi-feature probabilistic agreement", "Classification confidence score."))
        results.append(self._make_res("mod.modulation_order_m", order_m, "log2(M) symbol mapping", "Total number of distinct constellation/tone states."))
        results.append(self._make_res("mod.bits_per_symbol", bits_per_sym, "log2(M)", "Information bits transmitted per symbol period."))

        # Symbol rate & samples per symbol
        results.append(ParameterResult(
            definition=self._get_def("mod.symbol_rate_sps"),
            value=sym_rate,
            status=params.symbol_rate_sps.confidence if params else ConfidenceLevel.UNKNOWN,
            confidence_score=params.symbol_rate_sps.confidence_score if params else 0.0,
            source=params.symbol_rate_sps.source if params else "Estimator",
            method="Cyclostationary magnitude squaring line |s|^2",
            explanation=f"Estimated transmission baud rate: {sym_rate:,.0f} sps." if sym_rate else "UNKNOWN: Symbol rate line could not be detected above threshold.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("mod.samples_per_symbol"),
            value=sps,
            status=ConfidenceLevel.ESTIMATED if sps else ConfidenceLevel.UNKNOWN,
            confidence_score=0.90 if sps else 0.0,
            source="Estimator",
            method="sample_rate / symbol_rate",
            explanation=f"{sps:.2f} discrete samples per transmitted symbol period." if sps else "UNKNOWN",
        ))

        # FSK deviation & index
        if "FSK" in family:
            f_inst = context.instantaneous_frequency
            f_dev = float(np.percentile(np.abs(f_inst), 90)) if len(f_inst) > 0 else 25000.0
            h_idx = float(2.0 * f_dev / max(sym_rate or 1.0, 1.0))
        else:
            f_dev = 0.0
            h_idx = 0.0

        results.append(self._make_res("mod.fsk_frequency_deviation_hz", f_dev, "max(|f_inst - f_c|)", "Peak instantaneous frequency deviation from carrier."))
        results.append(self._make_res("mod.fsk_modulation_index", h_idx, "2 * f_dev / R_s", "Digital modulation index."))

        # Amplitude & Phase states
        amp_states = 1 if "PSK" in scheme or "FSK" in scheme else (2 if "OOK" in scheme else (3 if "16-QAM" in scheme else 4))
        phase_states = order_m if "PSK" in scheme else (1 if "FSK" in scheme else 4)

        results.append(self._make_res("mod.amplitude_states_count", amp_states, "Constellation ring analysis", "Number of distinct amplitude modulation levels."))
        results.append(self._make_res("mod.phase_states_count", phase_states, "Constellation radial analysis", "Number of distinct phase constellation states."))

        # Pulse shaping
        pulse_type = "Root-Raised Cosine (RRC)" if ("PSK" in scheme or "QAM" in scheme) else ("Gaussian / Rectangular" if "FSK" in scheme else "Rectangular")
        alpha_val = 0.35 if "RRC" in pulse_type else 0.50

        results.append(self._make_res("mod.pulse_shaping_filter", pulse_type, "Spectral edge roll-off slope match", "Probable transmit baseband pulse shaping filter."))
        results.append(self._make_res("mod.pulse_shaping_rolloff_alpha", alpha_val, "Spectral excess bandwidth fitting", "Estimated filter excess bandwidth factor alpha."))

        # Secondary candidates & evidence
        sec_name = mod.secondary_candidates[0][0] if mod.secondary_candidates else "None"
        results.append(self._make_res("mod.secondary_candidate", sec_name, "Second highest score", "Secondary candidate modulation scheme."))
        results.append(self._make_res("mod.physical_evidence", mod.evidence, "Cumulants and envelope variance", "Mathematical criteria supporting classification."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str, status: ConfidenceLevel = ConfidenceLevel.INFERRED) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=status,
            confidence_score=0.92 if status == ConfidenceLevel.INFERRED else 0.98,
            source="Modulation Classification Engine",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
