"""Self-contained, offline HTML report generator for SIGANALYZER."""

from pathlib import Path
from siganalyzer.reporting.generator import CompleteAnalysisReport


def generate_html_report(report: CompleteAnalysisReport) -> str:
    """Generate self-contained HTML string with inline CSS (zero external network assets)."""
    meta = report.metadata
    params = report.primary_parameters
    mod = report.primary_modulation

    # Build regions rows
    regions_html = ""
    for r in report.regions:
        regions_html += f"""
        <tr>
            <td>#{r.region_id}</td>
            <td>{r.start_time:.4f} s</td>
            <td>{r.end_time:.4f} s</td>
            <td>{r.duration_s * 1000:.2f} ms</td>
            <td>{r.snr_db.formatted_value}</td>
            <td>{r.mean_power_db:.1f} dBFS</td>
        </tr>
        """

    # Build anomalies list
    anomalies_html = ""
    if report.anomalies:
        for a in report.anomalies:
            severity_color = "#ef4444" if a.severity == "CRITICAL" else "#f59e0b" if a.severity == "WARNING" else "#3b82f6"
            anomalies_html += f"""
            <div class="anomaly-item" style="border-left: 4px solid {severity_color};">
                <div class="anomaly-header">
                    <span class="anomaly-type" style="color: {severity_color};">{a.anomaly_type} [{a.severity}]</span>
                    <span class="anomaly-time">t = {a.time_seconds:.4f} s</span>
                </div>
                <div class="anomaly-desc">{a.description}</div>
                {f'<div class="anomaly-action"><strong>Recommended Action:</strong> {a.recommended_action}</div>' if a.recommended_action else ''}
            </div>
            """
    else:
        anomalies_html = '<div class="no-anomalies">No signal anomalies or impairments detected.</div>'

    # Build Signal Profile HTML (Categories A to P)
    profile_html = ""
    if report.signal_profile and hasattr(report.signal_profile, "results"):
        counts = report.signal_profile.counts_by_status
        profile_rows = ""
        for res in report.signal_profile.results:
            color = res.status.badge_color if hasattr(res.status, "badge_color") else "#64748b"
            profile_rows += f"""
            <tr>
                <td><small style="color: #94a3b8;">{res.definition.category.value.split('. ', 1)[-1]}</small><br><strong>{res.definition.name}</strong></td>
                <td><code style="color: #38bdf8; font-size: 13px;">{res.formatted_value}</code></td>
                <td><span class="badge" style="background-color: {color};">{res.status.value}</span></td>
                <td><small>{res.method}</small></td>
                <td><small>{res.explanation}</small></td>
            </tr>
            """
        profile_html = f"""
    <div class="section">
        <div class="section-title">Extracted Signal Parameters ({report.signal_profile.total_count} Parameters across 16 Categories)</div>
        <div style="margin-bottom: 12px; font-size: 12px; color: #94a3b8; display: flex; gap: 8px;">
            <span class="badge" style="background-color: #22c55e;">Direct: {counts.get('Direct', 0)}</span>
            <span class="badge" style="background-color: #3b82f6;">Measured: {counts.get('Measured', 0)}</span>
            <span class="badge" style="background-color: #f59e0b;">Estimated: {counts.get('Estimated', 0)}</span>
            <span class="badge" style="background-color: #8b5cf6;">Inferred: {counts.get('Inferred', 0)}</span>
            <span class="badge" style="background-color: #64748b;">Unknown: {counts.get('Unknown', 0)}</span>
        </div>
        <div style="max-height: 500px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px;">
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Status</th>
                        <th>Derivation Method</th>
                        <th>Explanation</th>
                    </tr>
                </thead>
                <tbody>
                    {profile_rows}
                </tbody>
            </table>
        </div>
    </div>
    """

    # Build evidence list
    evidence_html = "".join(f"<li>{ev}</li>" for ev in mod.evidence)

    # Build bitstream HTML
    bitstream_html = ""
    if report.bitstream and report.bitstream.total_bits > 0:
        bs = report.bitstream
        sync_rows = ""
        for s in bs.detected_sync_words:
            sync_rows += f"""
            <tr>
                <td>{s.get('name', 'Unknown')}</td>
                <td>{s.get('bit_index', 0)}</td>
                <td><code>{s.get('pattern_hex', '')}</code></td>
                <td>{s.get('hamming_distance', 0)} bits</td>
                <td>{s.get('correlation_score', 1.0) * 100:.1f}%</td>
            </tr>
            """
        sync_table = f"""
        <table style="margin-top: 10px;">
            <thead>
                <tr>
                    <th>Sync Pattern</th>
                    <th>Bit Offset</th>
                    <th>Hex Value</th>
                    <th>Hamming Dist</th>
                    <th>Correlation</th>
                </tr>
            </thead>
            <tbody>
                {sync_rows or '<tr><td colspan="5">No standard sync words detected</td></tr>'}
            </tbody>
        </table>
        """ if bs.detected_sync_words else '<p style="color: var(--text-muted); font-size: 13px;">No standard preambles or sync-words identified in bitstream.</p>'

        bitstream_html = f"""
        <div class="section">
            <div class="section-title">Demodulation & Bitstream Analysis</div>
            <table>
                <tbody>
                    <tr>
                        <td><strong>Total Recovered Bits</strong></td>
                        <td>{bs.total_bits:,} bits</td>
                        <td><strong>Estimated Bit Rate</strong></td>
                        <td>{bs.bit_rate_est.formatted_value}</td>
                    </tr>
                    <tr>
                        <td><strong>Shannon Entropy</strong></td>
                        <td>{bs.entropy_per_bit:.4f} bits/bit (ideal ~1.0)</td>
                        <td><strong>Transition Density</strong></td>
                        <td>{bs.transition_density:.4f} (flips / bit)</td>
                    </tr>
                </tbody>
            </table>
            <div style="margin-top: 14px;">
                <h4 style="font-size: 13px; color: var(--accent); margin-bottom: 6px;">Detected Preambles & Sync Words</h4>
                {sync_table}
            </div>
            <div style="margin-top: 14px;">
                <h4 style="font-size: 13px; color: var(--accent); margin-bottom: 6px;">Hexadecimal Bitstream Preview (First 64 Bytes)</h4>
                <pre style="background: #0f172a; padding: 10px; border-radius: 4px; font-size: 12px; color: #38bdf8; overflow-x: auto;">{bs.hex_preview}</pre>
            </div>
        </div>
        """

    # Build FEC HTML
    fec_html = ""
    if report.fec:
        fec = report.fec
        badge_color = "#22c55e" if fec.detected else "#64748b"
        status_text = "Detected & Decoded" if fec.detected else "Uncoded / None"
        fec_html = f"""
        <div class="section">
            <div class="section-title">Forward Error Correction (FEC) & Channel Coding</div>
            <table>
                <tbody>
                    <tr>
                        <td><strong>FEC Scheme</strong></td>
                        <td>{fec.scheme}</td>
                        <td><strong>Status</strong></td>
                        <td><span class="badge" style="background-color: {badge_color};">{status_text}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Corrected Path Metric / Errors</strong></td>
                        <td>{fec.corrected_errors}</td>
                        <td><strong>Confidence</strong></td>
                        <td><span class="badge" style="background-color: {fec.confidence.badge_color};">{fec.confidence.value}</span></td>
                    </tr>
                </tbody>
            </table>
            <p style="margin-top: 10px; font-size: 13px; color: #cbd5e1;">{fec.notes}</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIGANALYZER Report — {meta.file_name}</title>
<style>
    :root {{
        --bg: #0f172a;
        --card-bg: #1e293b;
        --border: #334155;
        --text: #f8fafc;
        --text-muted: #94a3b8;
        --accent: #38bdf8;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        line-height: 1.5;
        padding: 30px;
    }}
    .container {{ max-width: 1050px; margin: 0 auto; }}
    header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border);
        padding-bottom: 20px;
        margin-bottom: 24px;
    }}
    .logo-area h1 {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
    .logo-area p {{ font-size: 13px; color: var(--text-muted); }}
    .meta-date {{ text-align: right; font-size: 13px; color: var(--text-muted); }}
    .grid-cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
    }}
    .card-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px; }}
    .card-value {{ font-size: 22px; font-weight: 700; color: #fff; }}
    .card-sub {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; display: flex; align-items: center; gap: 6px; }}
    .badge {{
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        padding: 2px 6px;
        border-radius: 4px;
        color: #fff;
    }}
    .section {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 24px;
    }}
    .section-title {{
        font-size: 16px;
        font-weight: 600;
        border-bottom: 1px solid var(--border);
        padding-bottom: 10px;
        margin-bottom: 16px;
        color: var(--accent);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
    ul.evidence {{ list-style-type: square; margin-left: 20px; font-size: 14px; }}
    ul.evidence li {{ margin-bottom: 6px; color: #cbd5e1; }}
    .anomaly-item {{
        background: #0f172a;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
    }}
    .anomaly-header {{ display: flex; justify-content: space-between; font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
    .anomaly-desc {{ font-size: 13px; color: #cbd5e1; }}
    .anomaly-action {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .no-anomalies {{ color: var(--success); font-size: 14px; }}
    footer {{ text-align: center; font-size: 12px; color: var(--text-muted); margin-top: 40px; border-top: 1px solid var(--border); padding-top: 15px; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <div class="logo-area">
            <h1>SIGANALYZER Automated Report</h1>
            <p>Target File: <strong>{meta.file_name}</strong> ({meta.file_type.value})</p>
        </div>
        <div class="meta-date">
            <div>Generated: {report.generated_at}</div>
            <div>Processing Time: {report.processing_duration_s:.3f} s</div>
            <div>SIGANALYZER v{report.app_version} (Offline)</div>
        </div>
    </header>

    <div class="grid-cards">
        <div class="card">
            <div class="card-title">Detected Modulation</div>
            <div class="card-value" style="color: var(--accent);">{mod.scheme}</div>
            <div class="card-sub">
                <span class="badge" style="background-color: #8b5cf6;">Inferred</span>
                Confidence: {mod.confidence_score * 100:.1f}%
            </div>
        </div>
        <div class="card">
            <div class="card-title">Symbol Rate</div>
            <div class="card-value">{params.symbol_rate_sps.formatted_value}</div>
            <div class="card-sub">
                <span class="badge" style="background-color: {params.symbol_rate_sps.confidence.badge_color};">{params.symbol_rate_sps.confidence.value}</span>
                {params.symbol_rate_sps.source}
            </div>
        </div>
        <div class="card">
            <div class="card-title">Occupied Bandwidth (99%)</div>
            <div class="card-value">{params.occupied_bandwidth_hz.formatted_value}</div>
            <div class="card-sub">
                <span class="badge" style="background-color: {params.occupied_bandwidth_hz.confidence.badge_color};">{params.occupied_bandwidth_hz.confidence.value}</span>
                Power integral
            </div>
        </div>
        <div class="card">
            <div class="card-title">Estimated SNR</div>
            <div class="card-value">{params.snr_db.formatted_value}</div>
            <div class="card-sub">
                <span class="badge" style="background-color: {params.snr_db.confidence.badge_color};">{params.snr_db.confidence.value}</span>
                Floor: {report.spectrum.noise_floor_db:.1f} dBFS
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Modulation Classification Evidence</div>
        <ul class="evidence">
            {evidence_html}
        </ul>
    </div>

    {bitstream_html}

    {fec_html}

    <div class="section">
        <div class="section-title">Recording Metadata & File Properties</div>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Classification</th>
                    <th>Source / Derivation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>File Type</td>
                    <td>{meta.file_type.value}</td>
                    <td><span class="badge" style="background-color: #22c55e;">Direct</span></td>
                    <td>Format identification engine</td>
                </tr>
                <tr>
                    <td>Sample Rate</td>
                    <td>{meta.sample_rate.formatted_value}</td>
                    <td><span class="badge" style="background-color: {meta.sample_rate.confidence.badge_color};">{meta.sample_rate.confidence.value}</span></td>
                    <td>{meta.sample_rate.source}</td>
                </tr>
                <tr>
                    <td>Center Frequency</td>
                    <td>{meta.center_frequency.formatted_value}</td>
                    <td><span class="badge" style="background-color: {meta.center_frequency.confidence.badge_color};">{meta.center_frequency.confidence.value}</span></td>
                    <td>{meta.center_frequency.source}</td>
                </tr>
                <tr>
                    <td>Total Samples / Duration</td>
                    <td>{meta.total_samples:,} samples ({meta.duration_seconds.formatted_value})</td>
                    <td><span class="badge" style="background-color: #3b82f6;">Measured</span></td>
                    <td>File size calculation</td>
                </tr>
                <tr>
                    <td>Peak Amplitude / RMS</td>
                    <td>{meta.peak_amplitude.formatted_value} / {meta.rms_power.formatted_value}</td>
                    <td><span class="badge" style="background-color: #3b82f6;">Measured</span></td>
                    <td>Sample scan</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Detected Signal Regions ({len(report.regions)})</div>
        <table>
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Duration</th>
                    <th>SNR</th>
                    <th>Mean Power</th>
                </tr>
            </thead>
            <tbody>
                {regions_html}
            </tbody>
        </table>
    </div>

    <div class="section">
        <div class="section-title">Signal Impairments & Anomalies</div>
        {anomalies_html}
    </div>

    {profile_html}

    <footer>
        SIGANALYZER Offline Signal Analysis Suite — 100% Local Execution — Zero Network Dependencies
    </footer>
</div>
</body>
</html>
"""
    return html


def export_html(report: CompleteAnalysisReport, output_path: str | Path) -> Path:
    """Save report to a standalone HTML file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html_report(report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return path
