"""Custom exception hierarchy for SIGANALYZER."""


class SigAnalyzerError(Exception):
    """Base exception for all SIGANALYZER errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class FileFormatError(SigAnalyzerError):
    """Raised when an unsupported file format or header is encountered."""
    pass


class CorruptedFileError(SigAnalyzerError):
    """Raised when a file is corrupted, truncated, or has invalid chunk sizes."""
    pass


class InvalidSampleError(SigAnalyzerError):
    """Raised when samples contain NaN, Inf, or impossible values."""
    pass


class SignalProcessingError(SigAnalyzerError):
    """Raised when a signal processing step fails."""
    pass


class SynchronizationError(SignalProcessingError):
    """Raised when carrier or symbol timing recovery fails to acquire lock."""
    pass


class DemodulationError(SignalProcessingError):
    """Raised when demodulation fails."""
    pass


class DecodingError(SigAnalyzerError):
    """Raised when FEC or frame decoding encounters uncorrectable errors."""
    pass


class ConfigurationError(SigAnalyzerError):
    """Raised when application configuration is invalid."""
    pass
