"""Top-level modulation classification coordinator."""

import numpy as np

from siganalyzer.classification.features import FeatureExtractor
from siganalyzer.classification.rule_based import RuleBasedClassifier
from siganalyzer.common.confidence import ConfidenceLevel
from siganalyzer.common.types import Measurement, ModulationResult
from siganalyzer.estimation.estimator import CumulantFeatures, ParameterEstimator


class ModulationClassifier:
    """Classifies signal modulation and extracts constellation metrics."""

    @classmethod
    def classify(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        snr_db: float = 20.0,
        cumulants: CumulantFeatures | None = None,
    ) -> ModulationResult:
        """Classify modulation scheme from raw I/Q samples."""
        if len(signal) < 64:
            return ModulationResult(
                scheme="Unknown",
                confidence_score=0.0,
                evidence=["Signal slice too short for classification."],
            )

        # 1. Feature extraction
        features = FeatureExtractor.extract_features(signal, sample_rate)

        # 2. Cumulant computation (if not passed in)
        if cumulants is None:
            cumulants = ParameterEstimator.compute_cumulants(signal)

        # 3. Rule-based evaluation
        result = RuleBasedClassifier.classify(features, cumulants, snr_db)

        # 4. Extract downsampled constellation scatter points (up to 2000 points)
        step = max(1, len(signal) // 2000)
        scatter_points = signal[::step]
        # Normalize constellation for display
        rms = np.sqrt(np.mean(np.abs(scatter_points) ** 2))
        if rms > 0:
            scatter_points = scatter_points / rms

        result.constellation_points = scatter_points

        # 5. Rough EVM estimation based on SNR
        evm_val = float(100.0 / (10.0 ** (snr_db / 20.0))) if snr_db > 0 else 50.0
        result.evm_percent = Measurement(
            name="EVM",
            value=min(evm_val, 100.0),
            unit="%",
            confidence=ConfidenceLevel.ESTIMATED,
            confidence_score=0.80,
            source="Estimated from in-band SNR",
        )

        return result
