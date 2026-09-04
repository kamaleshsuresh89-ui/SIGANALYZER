"""High-performance WAV and RF64 reader supporting PCM16, float32, stereo I/Q, and SDR chunks."""

import os
from pathlib import Path
import struct
from typing import Any
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.errors import CorruptedFileError, FileFormatError
from siganalyzer.common.types import (
    DataRepresentation,
    Measurement,
    SignalFileType,
    SignalMetadata,
)
from siganalyzer.io.readers.base import BaseSignalReader


class WavSignalReader(BaseSignalReader):
    """Memory-mapped WAV and RF64 signal reader supporting standard audio & SDR I/Q formats."""

    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path)
        self._file_handle = open(self.file_path, "rb")
        self._data_offset: int = 0
        self._data_bytes: int = 0
        self._channels: int = 2
        self._sample_rate: int = 48000
        self._bit_depth: int = 16
        self._audio_format: int = 1  # 1 = PCM, 3 = IEEE Float, 0xFFFE = Extensible
        self._is_rf64: bool = False
        self._extra_metadata: dict[str, Any] = {}
        self._mmap_array: np.memmap | None = None
        self._total_samples: int = 0
        self._dtype: np.dtype = np.dtype(np.int16)

        self._parse_headers()
        self._setup_memmap()

    def _parse_headers(self) -> None:
        """Parse RIFF/RF64 headers, chunks, and custom SDR metadata chunks."""
        self._file_handle.seek(0)
        magic = self._file_handle.read(4)

        if magic == b"RIFF":
            self._is_rf64 = False
        elif magic == b"RF64":
            self._is_rf64 = True
        else:
            raise FileFormatError(
                f"Invalid WAV header: expected RIFF or RF64, got {magic!r}",
                f"File {self.file_path.name} does not have a valid RIFF/RF64 signature.",
            )

        riff_size = struct.unpack("<I", self._file_handle.read(4))[0]
        wave_tag = self._file_handle.read(4)
        if wave_tag != b"WAVE":
            raise FileFormatError(f"Expected WAVE container, got {wave_tag!r}")

        file_size = self.file_path.stat().st_size
        fmt_found = False
        data_found = False

        # Read chunks until data chunk or EOF
        while self._file_handle.tell() < file_size - 8:
            chunk_header = self._file_handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)

            if chunk_id == b"ds64" and self._is_rf64:
                # RF64 data sizes (64-bit RIFF, data, and sample count)
                ds64_data = self._file_handle.read(min(chunk_size, 28))
                if len(ds64_data) >= 16:
                    _, data_size_64 = struct.unpack("<QQ", ds64_data[:16])
                    self._data_bytes = data_size_64
                if chunk_size > len(ds64_data):
                    self._file_handle.seek(chunk_size - len(ds64_data), os.SEEK_CUR)

            elif chunk_id == b"fmt ":
                fmt_data = self._file_handle.read(chunk_size)
                fmt_found = True
                if len(fmt_data) < 16:
                    raise CorruptedFileError("Truncated 'fmt ' chunk in WAV file.")
                (
                    self._audio_format,
                    self._channels,
                    self._sample_rate,
                    byte_rate,
                    block_align,
                    self._bit_depth,
                ) = struct.unpack("<HHIIHH", fmt_data[:16])

            elif chunk_id == b"data":
                data_found = True
                self._data_offset = self._file_handle.tell()
                if not self._is_rf64 or self._data_bytes == 0:
                    # Normal 32-bit chunk size
                    self._data_bytes = chunk_size
                    # Sanity check: if reported size extends beyond file, cap to actual file size
                    available = file_size - self._data_offset
                    if self._data_bytes > available or self._data_bytes == 0xFFFFFFFF:
                        self._data_bytes = available
                # Move to next chunk (aligned to 2 bytes)
                self._file_handle.seek(self._data_offset + self._data_bytes + (self._data_bytes % 2), os.SEEK_SET)

            elif chunk_id == b"auxi":
                # SDR# / HDSDR auxi chunk containing center frequency
                auxi_data = self._file_handle.read(chunk_size)
                if len(auxi_data) >= 8:
                    center_freq = struct.unpack("<Q", auxi_data[:8])[0]
                    self._extra_metadata["center_frequency_hz"] = float(center_freq)

            elif chunk_id == b"LIST":
                list_data = self._file_handle.read(chunk_size)
                self._extra_metadata["list_chunk"] = list_data.decode("latin1", errors="ignore")

            else:
                # Skip unknown chunk
                pad = chunk_size % 2
                self._file_handle.seek(chunk_size + pad, os.SEEK_CUR)

        if not fmt_found:
            raise CorruptedFileError("Missing 'fmt ' chunk in WAV file.")
        if not data_found or self._data_bytes <= 0:
            # Check if there is data after fmt
            available = file_size - self._file_handle.tell()
            if available > 0:
                self._data_offset = file_size - available
                self._data_bytes = available
            else:
                raise CorruptedFileError("Missing or empty 'data' chunk in WAV file.")

        bytes_per_sample = self._bit_depth // 8
        bytes_per_frame = bytes_per_sample * self._channels
        if bytes_per_frame > 0:
            self._total_samples = self._data_bytes // bytes_per_frame
        else:
            self._total_samples = 0

    def _setup_memmap(self) -> None:
        """Create a zero-copy np.memmap view over the data chunk."""
        # Determine numpy dtype
        if self._audio_format == 3:  # IEEE Float
            if self._bit_depth == 32:
                self._dtype = np.dtype("<f4")
            elif self._bit_depth == 64:
                self._dtype = np.dtype("<f8")
            else:
                raise FileFormatError(f"Unsupported float bit-depth: {self._bit_depth}")
        elif self._audio_format in (1, 0xFFFE):  # PCM
            if self._bit_depth == 8:
                self._dtype = np.dtype("u1")
            elif self._bit_depth == 16:
                self._dtype = np.dtype("<i2")
            elif self._bit_depth == 32:
                self._dtype = np.dtype("<i4")
            elif self._bit_depth == 24:
                # 24-bit requires special 3-byte unpacking
                self._dtype = np.dtype("u1")
            else:
                raise FileFormatError(f"Unsupported PCM bit-depth: {self._bit_depth}")
        else:
            raise FileFormatError(f"Unsupported audio format code: {self._audio_format}")

        if self._bit_depth != 24:
            total_elements = self._total_samples * self._channels
            self._mmap_array = np.memmap(
                self.file_path,
                dtype=self._dtype,
                mode="r",
                offset=self._data_offset,
                shape=(total_elements,),
            )

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def metadata(self) -> SignalMetadata:
        if self._metadata is not None:
            return self._metadata

        file_size = self.file_path.stat().st_size
        duration = self._total_samples / max(self._sample_rate, 1)

        # Classify file type
        if self._is_rf64:
            file_type = SignalFileType.RF64_WAV
        elif self._audio_format == 3 and self._bit_depth == 32:
            file_type = SignalFileType.WAV_FLOAT32
        elif self._bit_depth == 16:
            file_type = SignalFileType.WAV_PCM16
        elif self._bit_depth == 24:
            file_type = SignalFileType.WAV_PCM24
        elif self._bit_depth == 32:
            file_type = SignalFileType.WAV_PCM32
        else:
            file_type = SignalFileType.UNKNOWN

        representation = (
            DataRepresentation.COMPLEX_INTERLEAVED
            if self._channels >= 2
            else DataRepresentation.REAL_ONLY
        )

        center_freq_meas = (
            Measurement(
                name="Center Frequency",
                value=self._extra_metadata["center_frequency_hz"],
                unit="Hz",
                confidence=ConfidenceLevel.DIRECT,
                confidence_score=1.0,
                source="WAV auxi chunk",
            )
            if "center_frequency_hz" in self._extra_metadata
            else Measurement(
                name="Center Frequency",
                value=None,
                unit="Hz",
                confidence=ConfidenceLevel.UNKNOWN,
                source="Header does not specify RF center frequency",
            )
        )

        # Quick peak and RMS probe on first 65536 samples
        probe_count = min(65536, self._total_samples)
        if probe_count > 0:
            sample_slice = self.read_samples(0, probe_count)
            peak_val = float(np.max(np.abs(sample_slice)))
            rms_val = float(np.sqrt(np.mean(np.abs(sample_slice) ** 2)))
            rms_db = float(20 * np.log10(max(rms_val, 1e-12)))
            clipping = peak_val >= 0.999
        else:
            peak_val = 0.0
            rms_db = -100.0
            clipping = False

        self._metadata = SignalMetadata(
            file_path=str(self.file_path),
            file_name=self.file_path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            total_samples=self._total_samples,
            channels=self._channels,
            bit_depth=self._bit_depth,
            representation=representation,
            sample_rate=Measurement(
                name="Sample Rate",
                value=float(self._sample_rate),
                unit="Hz",
                confidence=ConfidenceLevel.DIRECT,
                confidence_score=1.0,
                source="WAV fmt chunk",
            ),
            center_frequency=center_freq_meas,
            duration_seconds=Measurement(
                name="Duration",
                value=float(duration),
                unit="s",
                confidence=ConfidenceLevel.MEASURED,
                confidence_score=1.0,
                source="Sample count / sample rate",
            ),
            peak_amplitude=Measurement(
                name="Peak Amplitude",
                value=peak_val,
                unit="",
                confidence=ConfidenceLevel.MEASURED,
                confidence_score=1.0,
                source="Signal probe",
            ),
            rms_power=Measurement(
                name="RMS Power",
                value=rms_db,
                unit="dBFS",
                confidence=ConfidenceLevel.MEASURED,
                confidence_score=1.0,
                source="Signal probe",
            ),
            has_clipping=Measurement(
                name="Clipping Detected",
                value=clipping,
                confidence=ConfidenceLevel.MEASURED,
                confidence_score=1.0,
                source="Threshold check (|amp| >= 0.999)",
            ),
            raw_header_fields={
                "audio_format": self._audio_format,
                "channels": self._channels,
                "sample_rate": self._sample_rate,
                "bit_depth": self._bit_depth,
                "data_offset": self._data_offset,
                "data_bytes": self._data_bytes,
                "is_rf64": self._is_rf64,
                **self._extra_metadata,
            },
        )
        return self._metadata

    def read_samples(self, start_sample: int = 0, count: int | None = None) -> np.ndarray:
        """Read a slice of samples normalized to complex64 (or float32 for mono)."""
        if start_sample >= self._total_samples:
            return np.empty(0, dtype=np.complex64 if self._channels >= 2 else np.float32)

        if count is None:
            count = self._total_samples - start_sample
        else:
            count = min(count, self._total_samples - start_sample)

        if count <= 0:
            return np.empty(0, dtype=np.complex64 if self._channels >= 2 else np.float32)

        # Handle 24-bit PCM
        if self._bit_depth == 24:
            byte_start = self._data_offset + start_sample * self._channels * 3
            byte_count = count * self._channels * 3
            self._file_handle.seek(byte_start, os.SEEK_SET)
            raw_bytes = self._file_handle.read(byte_count)
            raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            # Reshape to (N, 3) and convert to int32
            b0 = raw_arr[0::3].astype(np.int32)
            b1 = raw_arr[1::3].astype(np.int32)
            b2 = raw_arr[2::3].astype(np.int32)
            # Sign extend 24-bit to 32-bit
            val = (b0 | (b1 << 8) | (b2 << 16))
            val = np.where(val & 0x800000, val - 0x1000000, val)
            raw_floats = (val / 8388608.0).astype(np.float32)
        else:
            # Memory-mapped read
            elem_start = start_sample * self._channels
            elem_end = elem_start + count * self._channels
            raw_slice = np.array(self._mmap_array[elem_start:elem_end], copy=True)

            if self._bit_depth == 8:
                # Unsigned 8-bit PCM (0-255, 128 is zero)
                raw_floats = (raw_slice.astype(np.float32) - 128.0) / 128.0
            elif self._bit_depth == 16:
                raw_floats = raw_slice.astype(np.float32) / 32768.0
            elif self._bit_depth == 32:
                if self._audio_format == 3:  # Float32
                    raw_floats = raw_slice.astype(np.float32)
                else:  # PCM32
                    raw_floats = raw_slice.astype(np.float32) / 2147483648.0
            else:
                raw_floats = raw_slice.astype(np.float32)

        if self._channels >= 2:
            # Stereo = I/Q interleaved
            i_channel = raw_floats[0::self._channels]
            q_channel = raw_floats[1::self._channels]
            return (i_channel + 1j * q_channel).astype(np.complex64)
        else:
            # Mono real
            return raw_floats.astype(np.float32)

    def close(self) -> None:
        if self._mmap_array is not None:
            del self._mmap_array
            self._mmap_array = None
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
