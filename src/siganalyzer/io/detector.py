"""Automatic signal file type identification engine."""

from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np

from siganalyzer.common.types import SignalFileType


@dataclass
class DetectionResult:
    """Outcome of file format identification."""

    file_type: SignalFileType
    confidence: float  # 0.0 to 1.0
    evidence: list[str]
    suggested_reader: str
    is_ambiguous: bool = False


class FileTypeDetector:
    """Automatically detects WAV, RF64, SigMF, and raw IQ data representations."""

    PROBE_SIZE_BYTES = 131072  # 128 KB

    @classmethod
    def detect(cls, file_path: str | Path) -> DetectionResult:
        """Analyze the file header and statistical distribution to determine file type."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size == 0:
            return DetectionResult(
                file_type=SignalFileType.UNKNOWN,
                confidence=0.0,
                evidence=["File is completely empty (0 bytes)."],
                suggested_reader="",
            )

        # Check for SigMF companion metadata
        sigmf_meta_path = path.with_suffix(".sigmf-meta")
        if sigmf_meta_path.exists():
            return DetectionResult(
                file_type=SignalFileType.SIGMF,
                confidence=1.0,
                evidence=[f"Found companion SigMF metadata: {sigmf_meta_path.name}"],
                suggested_reader="SigMFReader",
            )

        # Read first 16 bytes for magic headers
        with open(path, "rb") as f:
            header = f.read(16)

        if len(header) >= 12:
            magic = header[:4]
            container = header[8:12]

            if magic == b"RIFF" and container == b"WAVE":
                # Standard WAV file, probe bit depth from fmt chunk
                bit_depth, audio_fmt = cls._probe_wav_fmt(path)
                if audio_fmt == 3 and bit_depth == 32:
                    return DetectionResult(
                        file_type=SignalFileType.WAV_FLOAT32,
                        confidence=1.0,
                        evidence=["RIFF WAVE header detected", "IEEE Float 32-bit audio format"],
                        suggested_reader="WavSignalReader",
                    )
                elif bit_depth == 16:
                    return DetectionResult(
                        file_type=SignalFileType.WAV_PCM16,
                        confidence=1.0,
                        evidence=["RIFF WAVE header detected", "PCM 16-bit audio format"],
                        suggested_reader="WavSignalReader",
                    )
                elif bit_depth == 24:
                    return DetectionResult(
                        file_type=SignalFileType.WAV_PCM24,
                        confidence=1.0,
                        evidence=["RIFF WAVE header detected", "PCM 24-bit audio format"],
                        suggested_reader="WavSignalReader",
                    )
                else:
                    return DetectionResult(
                        file_type=SignalFileType.WAV_PCM16,
                        confidence=0.9,
                        evidence=["RIFF WAVE header detected"],
                        suggested_reader="WavSignalReader",
                    )

            elif magic == b"RF64" and container == b"WAVE":
                return DetectionResult(
                    file_type=SignalFileType.RF64_WAV,
                    confidence=1.0,
                    evidence=["RF64 64-bit Large-File WAV container detected"],
                    suggested_reader="WavSignalReader",
                )

        # No RIFF/RF64 header found -> Statistical Raw IQ Probing
        return cls._probe_raw_iq(path, file_size)

    @classmethod
    def _probe_wav_fmt(cls, path: Path) -> tuple[int, int]:
        """Read fmt chunk to find bit depth and audio format."""
        try:
            with open(path, "rb") as f:
                f.seek(12)
                while True:
                    ch_hdr = f.read(8)
                    if len(ch_hdr) < 8:
                        break
                    ch_id, ch_sz = struct.unpack("<4sI", ch_hdr)
                    if ch_id == b"fmt ":
                        fmt_bytes = f.read(min(ch_sz, 16))
                        if len(fmt_bytes) >= 16:
                            audio_fmt, _, _, _, _, bit_depth = struct.unpack("<HHIIHH", fmt_bytes[:16])
                            return bit_depth, audio_fmt
                        break
                    else:
                        f.seek(ch_sz + (ch_sz % 2), 1)
        except Exception:
            pass
        return 16, 1

    @classmethod
    def _probe_raw_iq(cls, path: Path, file_size: int) -> DetectionResult:
        """Statistically probe binary data without headers."""
        probe_len = min(cls.PROBE_SIZE_BYTES, file_size)
        with open(path, "rb") as f:
            raw_bytes = f.read(probe_len)

        scores: dict[SignalFileType, float] = {}
        evidence_dict: dict[SignalFileType, list[str]] = {}

        # 1. Test Float32 Little-Endian
        if file_size % 8 == 0 and len(raw_bytes) >= 8:
            float32_arr = np.frombuffer(raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 8)], dtype="<f4")
            # Check for NaNs or Infs
            if np.any(np.isnan(float32_arr)) or np.any(np.isinf(float32_arr)):
                scores[SignalFileType.RAW_IQ_FLOAT32] = 0.0
            else:
                abs_vals = np.abs(float32_arr)
                max_val = np.max(abs_vals) if len(abs_vals) > 0 else 0
                mean_abs = np.mean(abs_vals) if len(abs_vals) > 0 else 0
                # In RF, float32 IQ is normalized between ~1e-4 and ~10.0
                if 1e-4 < mean_abs < 5.0 and max_val < 50.0:
                    scores[SignalFileType.RAW_IQ_FLOAT32] = 0.95
                    evidence_dict[SignalFileType.RAW_IQ_FLOAT32] = [
                        f"Valid finite float32 range (max={max_val:.2f}, mean={mean_abs:.2f})",
                        "File size is divisible by 8 bytes (complex64)",
                    ]
                elif max_val < 1000.0:
                    scores[SignalFileType.RAW_IQ_FLOAT32] = 0.65
                    evidence_dict[SignalFileType.RAW_IQ_FLOAT32] = [
                        "Finite float32 values, larger dynamic range",
                    ]
                else:
                    scores[SignalFileType.RAW_IQ_FLOAT32] = 0.1

        # 2. Test Int16 Little-Endian
        if file_size % 4 == 0 and len(raw_bytes) >= 4:
            int16_arr = np.frombuffer(raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 4)], dtype="<i2")
            mean_val = float(np.mean(int16_arr))
            std_val = float(np.std(int16_arr))
            min_val = float(np.min(int16_arr))
            max_val = float(np.max(int16_arr))

            # Int16 signals are centered near 0 with standard deviation > 20
            span = max_val - min_val
            if abs(mean_val) < 2000 and std_val > 50 and span > 200:
                scores[SignalFileType.RAW_IQ_INT16] = 0.85
                evidence_dict[SignalFileType.RAW_IQ_INT16] = [
                    f"Int16 centered near 0 (mean={mean_val:.1f}, std={std_val:.1f}, span={span:.0f})",
                    "File size is divisible by 4 bytes (complex int16)",
                ]
            else:
                scores[SignalFileType.RAW_IQ_INT16] = 0.3

        # 3. Test RTL-SDR Unsigned 8-bit (uint8)
        if file_size % 2 == 0 and len(raw_bytes) >= 2:
            uint8_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            u_mean = np.mean(uint8_arr)
            u_std = np.std(uint8_arr)
            # RTL-SDR is centered around 127.5
            if 124.0 < u_mean < 131.0 and 2.0 < u_std < 80.0:
                scores[SignalFileType.RAW_IQ_UINT8_RTL] = 0.90
                evidence_dict[SignalFileType.RAW_IQ_UINT8_RTL] = [
                    f"RTL-SDR characteristic DC bias at ~127.5 (mean={u_mean:.2f})",
                    f"Dynamic variance consistent with ADC noise (std={u_std:.2f})",
                ]
            else:
                scores[SignalFileType.RAW_IQ_UINT8_RTL] = 0.2

        # 4. Test HackRF Signed 8-bit (int8)
        if file_size % 2 == 0 and len(raw_bytes) >= 2:
            int8_arr = np.frombuffer(raw_bytes, dtype=np.int8)
            i_mean = np.mean(int8_arr)
            i_std = np.std(int8_arr)
            if abs(i_mean) < 8.0 and 2.0 < i_std < 60.0:
                scores[SignalFileType.RAW_IQ_INT8_HACKRF] = 0.80
                evidence_dict[SignalFileType.RAW_IQ_INT8_HACKRF] = [
                    f"HackRF signed int8 centered at 0 (mean={i_mean:.2f}, std={i_std:.2f})",
                ]
            else:
                scores[SignalFileType.RAW_IQ_INT8_HACKRF] = 0.2

        if not scores:
            return DetectionResult(
                file_type=SignalFileType.UNKNOWN,
                confidence=0.0,
                evidence=["Could not determine raw binary format from sample distribution."],
                suggested_reader="",
                is_ambiguous=True,
            )

        # Find best candidate
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]
        best_evidence = evidence_dict.get(best_type, [])

        # Check for ambiguity (two types with close scores)
        sorted_scores = sorted(scores.values(), reverse=True)
        is_ambiguous = len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 0.15)

        return DetectionResult(
            file_type=best_type,
            confidence=best_score,
            evidence=best_evidence,
            suggested_reader="RawIQSignalReader",
            is_ambiguous=is_ambiguous,
        )
