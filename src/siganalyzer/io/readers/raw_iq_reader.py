"""High-performance raw binary IQ reader with memory-mapping and multi-format support."""

from pathlib import Path
import numpy as np

from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.errors import FileFormatError
from siganalyzer.common.types import (
    DataRepresentation,
    Measurement,
    SignalFileType,
    SignalMetadata,
)
from siganalyzer.io.readers.base import BaseSignalReader


class RawIQSignalReader(BaseSignalReader):
    """Memory-mapped reader for raw binary I/Q recordings."""

    def __init__(
        self,
        file_path: str | Path,
        file_type: SignalFileType = SignalFileType.RAW_IQ_FLOAT32,
        sample_rate: float | None = None,
        center_frequency: float | None = None,
    ) -> None:
        super().__init__(file_path)
        self.file_type = file_type
        self.nominal_sample_rate = sample_rate
        self.nominal_center_freq = center_frequency
        self._mmap_array: np.memmap | None = None
        self._total_samples: int = 0
        self._bytes_per_complex_sample: int = 8
        self._dtype: np.dtype = np.dtype(np.float32)

        self._configure_format()
        self._setup_memmap()

    def _configure_format(self) -> None:
        """Determine bytes per sample and numpy dtype based on identified format."""
        match self.file_type:
            case SignalFileType.RAW_IQ_FLOAT32:
                self._dtype = np.dtype("<f4")
                self._bytes_per_complex_sample = 8  # 2 * 4 bytes
                self.bit_depth = 32
            case SignalFileType.RAW_IQ_INT16:
                self._dtype = np.dtype("<i2")
                self._bytes_per_complex_sample = 4  # 2 * 2 bytes
                self.bit_depth = 16
            case SignalFileType.RAW_IQ_UINT8_RTL:
                self._dtype = np.dtype("u1")
                self._bytes_per_complex_sample = 2  # 2 * 1 bytes
                self.bit_depth = 8
            case SignalFileType.RAW_IQ_INT8_HACKRF:
                self._dtype = np.dtype("i1")
                self._bytes_per_complex_sample = 2  # 2 * 1 bytes
                self.bit_depth = 8
            case _:
                raise FileFormatError(f"Unsupported raw IQ format: {self.file_type}")

        file_size = self.file_path.stat().st_size
        self._total_samples = file_size // self._bytes_per_complex_sample

    def _setup_memmap(self) -> None:
        """Create memory-mapped array for zero-copy random access."""
        total_elements = self._total_samples * 2  # 2 elements (I and Q) per complex sample
        self._mmap_array = np.memmap(
            self.file_path,
            dtype=self._dtype,
            mode="r",
            offset=0,
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

        if self.nominal_sample_rate is not None:
            sr_meas = Measurement(
                name="Sample Rate",
                value=float(self.nominal_sample_rate),
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.9,
                source="User specified / estimated",
            )
            duration_s = self._total_samples / max(self.nominal_sample_rate, 1.0)
            dur_meas = Measurement(
                name="Duration",
                value=float(duration_s),
                unit="s",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.9,
                source="Calculated from nominal sample rate",
            )
        else:
            sr_meas = Measurement(
                name="Sample Rate",
                value=None,
                unit="Hz",
                confidence=ConfidenceLevel.UNKNOWN,
                source="Raw file does not contain sample rate header",
            )
            dur_meas = Measurement(
                name="Duration",
                value=None,
                unit="s",
                confidence=ConfidenceLevel.UNKNOWN,
                source="Unknown without sample rate",
            )

        if self.nominal_center_freq is not None:
            cf_meas = Measurement(
                name="Center Frequency",
                value=float(self.nominal_center_freq),
                unit="Hz",
                confidence=ConfidenceLevel.ESTIMATED,
                confidence_score=0.9,
                source="User specified / estimated",
            )
        else:
            cf_meas = Measurement(
                name="Center Frequency",
                value=None,
                unit="Hz",
                confidence=ConfidenceLevel.UNKNOWN,
                source="Raw file does not contain RF center frequency header",
            )

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
            file_type=self.file_type,
            file_size_bytes=file_size,
            total_samples=self._total_samples,
            channels=2,
            bit_depth=self.bit_depth,
            representation=DataRepresentation.COMPLEX_INTERLEAVED,
            sample_rate=sr_meas,
            center_frequency=cf_meas,
            duration_seconds=dur_meas,
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
                "format": self.file_type.value,
                "bytes_per_sample": self._bytes_per_complex_sample,
            },
        )
        return self._metadata

    def read_samples(self, start_sample: int = 0, count: int | None = None) -> np.ndarray:
        """Read a slice of samples normalized to complex64 NumPy array."""
        if start_sample >= self._total_samples:
            return np.empty(0, dtype=np.complex64)

        if count is None:
            count = self._total_samples - start_sample
        else:
            count = min(count, self._total_samples - start_sample)

        if count <= 0:
            return np.empty(0, dtype=np.complex64)

        elem_start = start_sample * 2
        elem_end = elem_start + count * 2
        raw_slice = np.array(self._mmap_array[elem_start:elem_end], copy=True)

        match self.file_type:
            case SignalFileType.RAW_IQ_FLOAT32:
                floats = raw_slice.astype(np.float32)
            case SignalFileType.RAW_IQ_INT16:
                floats = raw_slice.astype(np.float32) / 32768.0
            case SignalFileType.RAW_IQ_UINT8_RTL:
                # RTL-SDR: unsigned bytes with zero at 127.5
                floats = (raw_slice.astype(np.float32) - 127.5) / 127.5
            case SignalFileType.RAW_IQ_INT8_HACKRF:
                # HackRF: signed 8-bit bytes with zero at 0
                floats = raw_slice.astype(np.float32) / 128.0
            case _:
                floats = raw_slice.astype(np.float32)

        i_ch = floats[0::2]
        q_ch = floats[1::2]
        return (i_ch + 1j * q_ch).astype(np.complex64)

    def close(self) -> None:
        if self._mmap_array is not None:
            del self._mmap_array
            self._mmap_array = None
