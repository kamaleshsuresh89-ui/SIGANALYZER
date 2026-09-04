"""JSON export for SIGANALYZER reports."""

import json
from pathlib import Path
from typing import Any

from siganalyzer.reporting.generator import CompleteAnalysisReport


def report_to_dict(report: CompleteAnalysisReport) -> dict[str, Any]:
    """Convert CompleteAnalysisReport to serializable dictionary."""
    meta = report.metadata
    params = report.primary_parameters
    mod = report.primary_modulation

    return {
        "siganalyzer_version": report.app_version,
        "generated_at": report.generated_at,
        "processing_duration_seconds": report.processing_duration_s,
        "file_info": {
            "path": meta.file_path,
            "name": meta.file_name,
            "type": meta.file_type.value,
            "size_bytes": meta.file_size_bytes,
            "total_samples": meta.total_samples,
            "channels": meta.channels,
            "bit_depth": meta.bit_depth,
            "representation": meta.representation.value,
            "sample_rate": {
                "value": meta.sample_rate.value,
                "unit": meta.sample_rate.unit,
                "confidence": meta.sample_rate.confidence.value,
                "source": meta.sample_rate.source,
            },
            "center_frequency": {
                "value": meta.center_frequency.value,
                "unit": meta.center_frequency.unit,
                "confidence": meta.center_frequency.confidence.value,
                "source": meta.center_frequency.source,
            },
            "duration": {
                "value": meta.duration_seconds.value,
                "unit": meta.duration_seconds.unit,
                "confidence": meta.duration_seconds.confidence.value,
            },
        },
        "parameters": {
            "carrier_frequency": {
                "value": params.carrier_freq_hz.value,
                "unit": params.carrier_freq_hz.unit,
                "confidence": params.carrier_freq_hz.confidence.value,
                "confidence_score": params.carrier_freq_hz.confidence_score,
                "source": params.carrier_freq_hz.source,
            },
            "occupied_bandwidth_99": {
                "value": params.occupied_bandwidth_hz.value,
                "unit": params.occupied_bandwidth_hz.unit,
                "confidence": params.occupied_bandwidth_hz.confidence.value,
                "confidence_score": params.occupied_bandwidth_hz.confidence_score,
            },
            "bandwidth_3db": {
                "value": params.bandwidth_3db_hz.value,
                "unit": params.bandwidth_3db_hz.unit,
                "confidence": params.bandwidth_3db_hz.confidence.value,
            },
            "snr": {
                "value": params.snr_db.value,
                "unit": params.snr_db.unit,
                "confidence": params.snr_db.confidence.value,
                "confidence_score": params.snr_db.confidence_score,
            },
            "symbol_rate": {
                "value": params.symbol_rate_sps.value,
                "unit": params.symbol_rate_sps.unit,
                "confidence": params.symbol_rate_sps.confidence.value,
                "confidence_score": params.symbol_rate_sps.confidence_score,
                "source": params.symbol_rate_sps.source,
            },
        },
        "modulation": {
            "detected_scheme": mod.scheme,
            "confidence_score": mod.confidence_score,
            "confidence_percent": f"{mod.confidence_score * 100:.1f}%",
            "evidence": mod.evidence,
            "secondary_candidates": [
                {"scheme": c[0], "score": c[1]} for c in mod.secondary_candidates
            ],
            "evm_percent": {
                "value": mod.evm_percent.value,
                "unit": mod.evm_percent.unit,
            },
        },
        "detected_regions_count": len(report.regions),
        "demodulation": {
            "scheme": report.demodulation.scheme if report.demodulation else None,
            "recovered_bits": len(report.demodulation.bits) if report.demodulation else 0,
            "carrier_locked": report.demodulation.carrier_locked if report.demodulation else False,
            "timing_locked": report.demodulation.timing_locked if report.demodulation else False,
        } if report.demodulation else None,
        "bitstream": {
            "total_bits": report.bitstream.total_bits,
            "bit_rate_bps": report.bitstream.bit_rate_est.value,
            "transition_density": report.bitstream.transition_density,
            "entropy_per_bit": report.bitstream.entropy_per_bit,
            "detected_sync_words": report.bitstream.detected_sync_words,
            "hex_preview": report.bitstream.hex_preview,
        } if report.bitstream else None,
        "fec": {
            "scheme": report.fec.scheme,
            "detected": report.fec.detected,
            "confidence": report.fec.confidence.value,
            "corrected_errors": report.fec.corrected_errors,
            "notes": report.fec.notes,
        } if report.fec else None,
        "anomalies": [
            {
                "type": a.anomaly_type,
                "severity": a.severity,
                "sample_index": a.sample_index,
                "time_seconds": a.time_seconds,
                "description": a.description,
                "recommended_action": a.recommended_action,
            }
            for a in report.anomalies
        ],
        "signal_profile": report.signal_profile.to_list_of_dicts() if (report.signal_profile and hasattr(report.signal_profile, "to_list_of_dicts")) else None,
    }


def export_json(report: CompleteAnalysisReport, output_path: str | Path) -> Path:
    """Save report to a JSON file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report_to_dict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
