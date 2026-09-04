"""Category A: File & Capture Information parameter extractor."""

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.parameters.context import SignalContext
from siganalyzer.parameters.model import ParameterCategory, ParameterDefinition, ParameterResult
from siganalyzer.parameters.registry import BaseParameterExtractor, ParameterRegistry


@ParameterRegistry.register
class FileCaptureExtractor(BaseParameterExtractor):
    """Extracts container, capture, file-system, and format parameters."""

    @property
    def category(self) -> ParameterCategory:
        return ParameterCategory.FILE_CAPTURE

    def supported_parameters(self) -> list[ParameterDefinition]:
        cat = self.category
        return [
            ParameterDefinition("file.name", "File Name", cat, "", str, "Base filename of recording", ConfidenceLevel.DIRECT, export_priority=1),
            ParameterDefinition("file.extension", "File Extension", cat, "", str, "File extension format", ConfidenceLevel.DIRECT, export_priority=2),
            ParameterDefinition("file.size_bytes", "File Size", cat, "bytes", int, "Exact file size on disk", ConfidenceLevel.DIRECT, export_priority=3),
            ParameterDefinition("file.sha256", "SHA-256 Hash", cat, "", str, "Cryptographic hash of recording file", ConfidenceLevel.DIRECT, export_priority=4),
            ParameterDefinition("file.modified_time", "Modified Timestamp", cat, "", str, "Filesystem last modified date/time", ConfidenceLevel.DIRECT, export_priority=5),
            ParameterDefinition("capture.container_format", "Container Format", cat, "", str, "Detected recording container type", ConfidenceLevel.DIRECT, export_priority=10),
            ParameterDefinition("capture.riff_chunks", "WAV/RIFF Chunks", cat, "", list, "Parsed chunks in WAV container header", ConfidenceLevel.DIRECT, export_priority=11),
            ParameterDefinition("capture.header_size_bytes", "Header Size", cat, "bytes", int, "Container header size before sample payload", ConfidenceLevel.DIRECT, export_priority=12),
            ParameterDefinition("capture.data_offset_bytes", "Data Offset", cat, "bytes", int, "Byte offset where sample data begins", ConfidenceLevel.DIRECT, export_priority=13),
            ParameterDefinition("capture.payload_size_bytes", "Payload Size", cat, "bytes", int, "Size of sample data payload in bytes", ConfidenceLevel.DIRECT, export_priority=14),
            ParameterDefinition("capture.sample_count", "Total Sample Count", cat, "samples", int, "Total complex or real samples in file", ConfidenceLevel.MEASURED, export_priority=15),
            ParameterDefinition("capture.channel_count", "Channel Count", cat, "", int, "Number of interleaved audio/data channels", ConfidenceLevel.DIRECT, export_priority=16),
            ParameterDefinition("capture.sample_rate", "Sampling Rate", cat, "Hz", float, "Baseband sampling rate", ConfidenceLevel.DIRECT, export_priority=17),
            ParameterDefinition("capture.duration_s", "Duration", cat, "s", float, "Recording length in seconds", ConfidenceLevel.MEASURED, export_priority=18),
            ParameterDefinition("capture.bit_depth", "Bit Depth", cat, "bits", int, "Bits per sample component", ConfidenceLevel.DIRECT, export_priority=19),
            ParameterDefinition("capture.data_type", "Sample Data Type", cat, "", str, "Numeric sample representation", ConfidenceLevel.DIRECT, export_priority=20),
            ParameterDefinition("capture.signedness", "Signedness", cat, "", str, "Signed or unsigned sample encoding", ConfidenceLevel.DIRECT, export_priority=21),
            ParameterDefinition("capture.endianness", "Endianness", cat, "", str, "Byte ordering on disk", ConfidenceLevel.DIRECT, export_priority=22),
            ParameterDefinition("capture.domain", "Signal Domain", cat, "", str, "Complex I/Q or Real-valued", ConfidenceLevel.DIRECT, export_priority=23),
            ParameterDefinition("capture.iq_arrangement", "I/Q Arrangement", cat, "", str, "Interleaved, planar, or real", ConfidenceLevel.DIRECT, export_priority=24),
            ParameterDefinition("capture.iq_ordering", "I/Q Ordering", cat, "", str, "Component order in memory (I-Q or Q-I)", ConfidenceLevel.DIRECT, export_priority=25),
            ParameterDefinition("capture.bytes_per_sample", "Bytes per Sample", cat, "bytes", int, "Total bytes per complex sample pair", ConfidenceLevel.DIRECT, export_priority=26),
            ParameterDefinition("capture.center_frequency_rf", "RF Center Frequency", cat, "Hz", float, "Receiver tuner center frequency", ConfidenceLevel.UNKNOWN, export_priority=27),
            ParameterDefinition("capture.receiver_device", "Receiver Model", cat, "", str, "SDR hardware identifier if logged", ConfidenceLevel.UNKNOWN, export_priority=28),
        ]

    def extract(self, context: SignalContext) -> list[ParameterResult]:
        meta = context.metadata
        results: list[ParameterResult] = []
        p = context.file_path

        # Compute SHA-256 hash (fast block read on first 64 KB + last 64 KB if huge, or full file if < 50 MB)
        file_hash = "Unavailable"
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                if context.metadata.file_size_bytes < 50_000_000:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                    file_hash = h.hexdigest()
                else:
                    h.update(f.read(65536))
                    f.seek(max(0, context.metadata.file_size_bytes - 65536))
                    h.update(f.read(65536))
                    file_hash = f"{h.hexdigest()} (sampled)"
        except Exception:
            pass

        # Modified time
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = "Unknown"

        # Parameter 1: file.name
        results.append(ParameterResult(
            definition=self._get_def("file.name"),
            value=meta.file_name,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Filesystem Path",
            method="os.path.basename",
            explanation="Exact filename recorded on filesystem.",
        ))

        # Parameter 2: file.extension
        results.append(ParameterResult(
            definition=self._get_def("file.extension"),
            value=p.suffix.lower(),
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Filesystem Path",
            method="pathlib.Path.suffix",
            explanation="Extension indicates standard container convention.",
        ))

        # Parameter 3: file.size_bytes
        results.append(ParameterResult(
            definition=self._get_def("file.size_bytes"),
            value=meta.file_size_bytes,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Filesystem Stat",
            method="os.path.getsize",
            explanation="Exact byte size of the file on storage media.",
        ))

        # Parameter 4: file.sha256
        results.append(ParameterResult(
            definition=self._get_def("file.sha256"),
            value=file_hash,
            status=ConfidenceLevel.DIRECT if "sampled" not in file_hash else ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Cryptographic Digest",
            method="hashlib.sha256",
            explanation="Cryptographic integrity fingerprint of file content.",
        ))

        # Parameter 5: file.modified_time
        results.append(ParameterResult(
            definition=self._get_def("file.modified_time"),
            value=mtime,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Filesystem Stat",
            method="st_mtime",
            explanation="Last modification timestamp on filesystem.",
        ))

        # Parameter 6: capture.container_format
        results.append(ParameterResult(
            definition=self._get_def("capture.container_format"),
            value=meta.file_type.value,
            status=ConfidenceLevel.DIRECT if "WAV" in meta.file_type.value else ConfidenceLevel.INFERRED,
            confidence_score=1.0 if "WAV" in meta.file_type.value else 0.90,
            source="FileTypeDetector",
            method="Magic Bytes & Statistical Distribution Probe",
            explanation=f"Identified as {meta.file_type.value}.",
        ))

        # Parameter 7: capture.riff_chunks
        chunks_list = list(meta.raw_header_fields.keys()) if meta.raw_header_fields else []
        results.append(ParameterResult(
            definition=self._get_def("capture.riff_chunks"),
            value=chunks_list,
            status=ConfidenceLevel.DIRECT if chunks_list else ConfidenceLevel.UNKNOWN,
            confidence_score=1.0 if chunks_list else 0.0,
            source="RIFF Parser",
            method="Chunk FourCC scan",
            explanation=f"Header contains {len(chunks_list)} RIFF chunks: {chunks_list}" if chunks_list else "No RIFF chunk structure found (raw binary).",
        ))

        # Header, offset, payload
        data_offset = int(meta.raw_header_fields.get("data_offset", 0))
        header_size = data_offset
        payload_size = meta.file_size_bytes - data_offset

        results.append(ParameterResult(
            definition=self._get_def("capture.header_size_bytes"),
            value=header_size,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Container Parser",
            method="data_offset - file_start",
            explanation=f"{header_size} bytes of container metadata before payload.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.data_offset_bytes"),
            value=data_offset,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Container Parser",
            method="RIFF 'data' chunk offset or 0",
            explanation=f"Sample array starts at byte offset {data_offset}.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.payload_size_bytes"),
            value=payload_size,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Container Parser",
            method="file_size - header_size",
            explanation=f"{payload_size:,} bytes of raw audio/IQ sample data.",
        ))

        # Sample count & duration
        results.append(ParameterResult(
            definition=self._get_def("capture.sample_count"),
            value=meta.total_samples,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Reader",
            method="payload_size / bytes_per_sample",
            explanation=f"Calculated from byte payload and {meta.channels} channels.",
        ))

        # Sample rate
        sr = meta.sample_rate.value
        sr_status = meta.sample_rate.confidence
        results.append(ParameterResult(
            definition=self._get_def("capture.sample_rate"),
            value=sr,
            status=sr_status,
            confidence_score=1.0 if sr_status == ConfidenceLevel.DIRECT else 0.5,
            source=meta.sample_rate.source,
            method="WAV 'fmt ' chunk parse or default override",
            explanation=f"Sampling frequency {meta.sample_rate.formatted_value}.",
        ))

        # Duration
        dur = meta.duration_seconds.value
        results.append(ParameterResult(
            definition=self._get_def("capture.duration_s"),
            value=dur,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Reader",
            method="total_samples / sample_rate",
            explanation=f"Signal recording spans {meta.duration_seconds.formatted_value}.",
        ))

        # Format details
        results.append(ParameterResult(
            definition=self._get_def("capture.channel_count"),
            value=meta.channels,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Container Parser",
            method="WAV num_channels or 2 for complex IQ",
            explanation=f"{meta.channels} audio/IQ channel(s).",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.bit_depth"),
            value=meta.bit_depth,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Container Parser",
            method="Bits per component",
            explanation=f"{meta.bit_depth}-bit quantization depth.",
        ))

        dtype_str = "float32" if meta.bit_depth == 32 else ("int16" if meta.bit_depth == 16 else "uint8")
        results.append(ParameterResult(
            definition=self._get_def("capture.data_type"),
            value=dtype_str,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Format Detector",
            method="Data type inference",
            explanation=f"Stored as native {dtype_str} values.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.signedness"),
            value="Unsigned" if "uint" in meta.file_type.value.lower() else "Signed",
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Format Detector",
            method="Type inspect",
            explanation="Signed two's complement or unsigned offset binary.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.endianness"),
            value="Little-Endian",
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Format Detector",
            method="Architecture convention (RIFF x86 LE)",
            explanation="Standard Intel little-endian byte ordering.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.domain"),
            value="Complex (IQ)" if meta.channels >= 2 else "Real-valued",
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Reader",
            method="Channel count test",
            explanation="Two channels representing orthogonal In-Phase and Quadrature components.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.iq_arrangement"),
            value=meta.representation.value,
            status=ConfidenceLevel.DIRECT,
            confidence_score=1.0,
            source="Reader",
            method="Memory mapping layout",
            explanation="Sample arrangement on storage media.",
        ))

        results.append(ParameterResult(
            definition=self._get_def("capture.iq_ordering"),
            value="I then Q (In-Phase first)",
            status=ConfidenceLevel.DIRECT,
            confidence_score=0.95,
            source="Reader",
            method="Standard SDR interleaved convention",
            explanation="Left channel contains I, right channel contains Q.",
        ))

        bytes_per_sample = (meta.bit_depth // 8) * meta.channels
        results.append(ParameterResult(
            definition=self._get_def("capture.bytes_per_sample"),
            value=bytes_per_sample,
            status=ConfidenceLevel.MEASURED,
            confidence_score=1.0,
            source="Reader",
            method="(bit_depth / 8) * channels",
            explanation=f"{bytes_per_sample} bytes consumed per complex sample pair.",
        ))

        # RF Center frequency & device
        cf = meta.center_frequency.value
        cf_status = meta.center_frequency.confidence
        results.append(ParameterResult(
            definition=self._get_def("capture.center_frequency_rf"),
            value=cf,
            status=cf_status,
            confidence_score=1.0 if cf_status == ConfidenceLevel.DIRECT else 0.0,
            source=meta.center_frequency.source,
            method="SDR# 'auxi' chunk or header metadata",
            explanation="Absolute RF center frequency of tuner LO." if cf is not None else "UNKNOWN: File does not contain RF tuner LO metadata.",
            validity_notes="" if cf is not None else "Requires SDR companion metadata or manual RF frequency specification.",
        ))

        device_info = meta.raw_header_fields.get("device_info")
        results.append(ParameterResult(
            definition=self._get_def("capture.receiver_device"),
            value=device_info,
            status=ConfidenceLevel.DIRECT if device_info else ConfidenceLevel.UNKNOWN,
            confidence_score=1.0 if device_info else 0.0,
            source="Container Metadata",
            method="auxi or SigMF hardware field",
            explanation=str(device_info) if device_info else "UNKNOWN: Recording device hardware tag not logged in header.",
        ))

        return results

    def _get_def(self, param_id: str) -> ParameterDefinition:
        for d in self.supported_parameters():
            if d.id == param_id:
                return d
        raise KeyError(f"Unknown parameter ID: {param_id}")
