"""Category P: Signal Anomaly Detection parameter extractor."""

from typing import Any

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class AnomalyDetectionExtractor(BaseParameterExtractor):
    """Extracts signal impairments, dropouts, clipping, phase slips, and anomaly tallies."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.ANOMALY_DETECTION

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("anomaly.total_anomalies_detected", "Total Anomalies Count", cat, "events", int, "Cumulative count of detected anomalies and impairment warnings", ConfidenceLevel.MEASURED, export_priority=1),
            ParameterDefinition("anomaly.critical_count", "Critical Severity Count", cat, "events", int, "Count of critical anomalies (severe clipping, deep dropouts)", ConfidenceLevel.MEASURED, export_priority=2),
            ParameterDefinition("anomaly.warning_count", "Warning Severity Count", cat, "events", int, "Count of moderate impairment warnings (IQ imbalance, CRC errors)", ConfidenceLevel.MEASURED, export_priority=3),
            ParameterDefinition("anomaly.info_count", "Info Severity Count", cat, "events", int, "Count of informational notices (mild DC offset, low SNR)", ConfidenceLevel.MEASURED, export_priority=4),
            ParameterDefinition("anomaly.clipping_events_count", "Clipping Events Count", cat, "events", int, "Number of distinct ADC full-scale saturation occurrences", ConfidenceLevel.MEASURED, export_priority=5),
            ParameterDefinition("anomaly.power_dropout_events_count", "Power Dropout Events", cat, "events", int, "Number of sudden unexpected transmission dropouts / deep fades", ConfidenceLevel.MEASURED, export_priority=6),
            ParameterDefinition("anomaly.frequency_jump_events_count", "Frequency Jump Events", cat, "events", int, "Sudden carrier frequency discontinuities / LO hops", ConfidenceLevel.MEASURED, export_priority=7),
            ParameterDefinition("anomaly.phase_discontinuity_count", "Phase Slip Discontinuities", cat, "events", int, "Unexplained abrupt phase tears exceeding pi radians", ConfidenceLevel.MEASURED, export_priority=8),
            ParameterDefinition("anomaly.frame_crc_error_events_count", "CRC Integrity Failures", cat, "events", int, "Count of corrupted frames failing CRC polynomial verification", ConfidenceLevel.MEASURED, export_priority=9),
            ParameterDefinition("anomaly.anomalies_summary_list", "Anomaly Findings Summary", cat, "", list, "List of detected impairment summaries with timestamps and severity", ConfidenceLevel.MEASURED, export_priority=10),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        anomalies = context.anomalies
        results: list[ParameterResult] = []

        total = len(anomalies)
        crit = sum(1 for a in anomalies if a.severity == "CRITICAL")
        warn = sum(1 for a in anomalies if a.severity == "WARNING")
        info = sum(1 for a in anomalies if a.severity == "INFO")

        clipping_cnt = sum(1 for a in anomalies if "clip" in a.anomaly_type.lower())
        dropout_cnt = sum(1 for a in anomalies if "dropout" in a.anomaly_type.lower() or "drop" in a.anomaly_type.lower())
        freq_jump_cnt = sum(1 for a in anomalies if "frequency" in a.anomaly_type.lower() or "jump" in a.anomaly_type.lower())
        phase_disc_cnt = sum(1 for a in anomalies if "phase" in a.anomaly_type.lower() or "slip" in a.anomaly_type.lower())

        # CRC errors from frames
        crc_err_cnt = sum(1 for f in context.frames if f.crc_valid is False)

        summary_list = [
            f"[{a.severity}] t={a.time_seconds:.3f}s: {a.anomaly_type} - {a.description}"
            for a in anomalies
        ]
        if not summary_list:
            summary_list = ["No anomalous events or impairments detected."]

        results.append(self._make_res("anomaly.total_anomalies_detected", total, "Heuristic impairment engine accumulator", f"{total} event(s) recorded."))
        results.append(self._make_res("anomaly.critical_count", crit, "Severity filter (CRITICAL)", f"{crit} critical impairment(s)."))
        results.append(self._make_res("anomaly.warning_count", warn, "Severity filter (WARNING)", f"{warn} warning(s)."))
        results.append(self._make_res("anomaly.info_count", info, "Severity filter (INFO)", f"{info} informational notice(s)."))
        results.append(self._make_res("anomaly.clipping_events_count", clipping_cnt, "Saturation detector", f"{clipping_cnt} clipping incident(s)."))
        results.append(self._make_res("anomaly.power_dropout_events_count", dropout_cnt, "Power threshold detector", f"{dropout_cnt} dropout incident(s)."))
        results.append(self._make_res("anomaly.frequency_jump_events_count", freq_jump_cnt, "d(f_inst)/dt outlier detector", f"{freq_jump_cnt} frequency hop/jump incident(s)."))
        results.append(self._make_res("anomaly.phase_discontinuity_count", phase_disc_cnt, "Wrapped phase derivative detector", f"{phase_disc_cnt} phase slip incident(s)."))
        results.append(self._make_res("anomaly.frame_crc_error_events_count", crc_err_cnt, "CRC failure accumulator", f"{crc_err_cnt} CRC verification error(s)."))
        results.append(self._make_res("anomaly.anomalies_summary_list", summary_list, "Chronological event log", f"{len(summary_list)} summary items."))

        return results

    def _make_res(self, pid: str, val: Any, method: str, expl: str) -> ParameterResult:
        defn = self._get_def(pid)
        return ParameterResult(
            definition=defn,
            value=val,
            status=ConfidenceLevel.MEASURED,
            confidence_score=0.98,
            source="Anomaly Detection Engine",
            method=method,
            explanation=expl,
        )

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(param_id)
