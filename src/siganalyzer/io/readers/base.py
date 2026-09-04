"""Abstract base reader for signal recordings with chunking and memory-mapping support."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator
import numpy as np

from siganalyzer.common.types import SignalMetadata


class BaseSignalReader(ABC):
    """Abstract base class for all signal readers.

    Guarantees memory-efficient streaming and random-access slicing
    for multi-gigabyte recordings.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Signal file not found: {self.file_path}")
        self._metadata: SignalMetadata | None = None

    @property
    @abstractmethod
    def metadata(self) -> SignalMetadata:
        """Return the extracted metadata for this signal file."""
        pass

    @property
    @abstractmethod
    def total_samples(self) -> int:
        """Total number of complex or real samples in the recording."""
        pass

    @abstractmethod
    def read_samples(self, start_sample: int = 0, count: int | None = None) -> np.ndarray:
        """Read a slice of samples as a complex64 (or float32 for real) NumPy array.

        Args:
            start_sample: Starting sample index (0-indexed).
            count: Number of samples to read. If None, reads to end of file.

        Returns:
            1D numpy array of complex64 (or float32) normalized samples.
        """
        pass

    def read_chunks(self, chunk_size: int = 65536, overlap: int = 0) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over the recording in chunks for memory-efficient streaming.

        Args:
            chunk_size: Number of samples per chunk.
            overlap: Number of overlapping samples between consecutive chunks.

        Yields:
            Tuple of (chunk_start_sample_index, chunk_samples_array).
        """
        if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly less than chunk_size.")

        total = self.total_samples
        step = chunk_size - overlap
        idx = 0

        while idx < total:
            to_read = min(chunk_size, total - idx)
            samples = self.read_samples(start_sample=idx, count=to_read)
            yield idx, samples
            idx += step

    @abstractmethod
    def close(self) -> None:
        """Release any open file handles or memory maps."""
        pass

    def __enter__(self) -> "BaseSignalReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
