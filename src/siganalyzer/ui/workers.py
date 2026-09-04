"""Asynchronous background worker using QRunnable and QThreadPool."""

from pathlib import Path
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from siganalyzer.dsp.pipeline import AnalysisPipeline


class WorkerSignals(QObject):
    """Signals for background analysis worker."""

    progress = Signal(int, str)  # (percent, message)
    finished = Signal(object, object, object)  # (CompleteAnalysisReport, SpectrogramData, samples)
    error = Signal(str)  # error message


class AnalysisWorker(QRunnable):
    """Executes signal analysis pipeline in a background thread."""

    def __init__(
        self,
        file_path: str | Path,
        sample_rate_override: float | None = None,
        center_freq_override: float | None = None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.sample_rate_override = sample_rate_override
        self.center_freq_override = center_freq_override
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self) -> None:
        """Request worker cancellation."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    @Slot()
    def run(self) -> None:
        """Worker thread entrypoint."""
        try:
            pipeline = AnalysisPipeline(
                progress_callback=self.signals.progress.emit,
                is_cancelled=self.is_cancelled,
            )
            report, spectrogram_data, samples = pipeline.analyze_file(
                self.file_path,
                sample_rate_override=self.sample_rate_override,
                center_freq_override=self.center_freq_override,
            )
            if not self._cancelled:
                self.signals.finished.emit(report, spectrogram_data, samples)
        except InterruptedError:
            self.signals.progress.emit(0, "Analysis cancelled.")
        except Exception as e:
            self.signals.error.emit(str(e))
