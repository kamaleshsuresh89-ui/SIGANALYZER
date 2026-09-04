"""Deterministic rule-based modulation classification using physical & cumulant criteria."""

import numpy as np

from siganalyzer.classification.features import ModulationFeatures
from siganalyzer.common.types import ModulationResult
from siganalyzer.estimation.estimator import CumulantFeatures


class RuleBasedClassifier:
    """Classifies modulation schemes using physics, cumulants, and signal properties."""

    @classmethod
    def classify(
        cls,
        features: ModulationFeatures,
        cumulants: CumulantFeatures,
        snr_db: float,
    ) -> ModulationResult:
        """Evaluate deterministic rules and cumulants to determine modulation."""
        evidence: list[str] = []
        candidates: list[tuple[str, float]] = []

        abs_c20 = float(np.abs(cumulants.c20))
        norm_c42 = cumulants.norm_c42
        abs_norm_c40 = float(np.abs(cumulants.norm_c40))
        env_var = features.envelope_variance

        # 1. BPSK Check: Characteristic |C20| > 0.40 and C42 < -1.4
        dist_bpsk = abs(norm_c42 - (-2.0)) + abs(abs_norm_c40 - 1.0)
        dist_qpsk = abs(norm_c42 - (-1.0)) + abs(abs_norm_c40 - 0.0)

        if abs_c20 > 0.45 or (dist_bpsk < dist_qpsk and env_var < 0.30):
            scheme = "BPSK"
            conf = min(0.98, max(0.70, 1.0 - 0.25 * dist_bpsk))
            evidence.append(f"C42 cumulant ({norm_c42:.2f}) and |C40| ({abs_norm_c40:.2f}) match BPSK theoretical values (-2.0, 1.0)")
            if abs_c20 > 0.3:
                evidence.append(f"Strong 2nd-order conjugate symmetry |C20| = {abs_c20:.2f}")
            candidates = [("BPSK", conf), ("QPSK", 0.30), ("2FSK", 0.20)]

        # 2. FSK Check: Multiple frequency modes with very low envelope variance
        elif features.freq_peaks_count >= 2 and (env_var < 0.08 or features.sigma_af > 0.02):
            if features.freq_peaks_count <= 2:
                scheme = "2FSK"
                conf = 0.90
                evidence.append("Two distinct frequency modes in instantaneous frequency distribution")
                candidates = [("2FSK", 0.90), ("MSK", 0.60), ("4FSK", 0.30)]
            else:
                scheme = "4FSK"
                conf = 0.85
                evidence.append(f"{features.freq_peaks_count} frequency modes in instantaneous frequency distribution")
                candidates = [("4FSK", 0.85), ("2FSK", 0.40)]

        # 3. Constant vs Non-Constant Envelope for QPSK vs QAM/OOK
        elif env_var < 0.25:
            # QPSK / 8PSK / MSK
            scheme = "QPSK"
            conf = min(0.95, max(0.75, 1.0 - 0.20 * dist_qpsk))
            evidence.append(f"Low envelope variance ({env_var:.3f} < 0.25): Constant-envelope modulation family")
            evidence.append(f"C42 cumulant ({norm_c42:.2f}) matches QPSK theoretical value (-1.0)")
            evidence.append(f"|C40| cumulant ({abs_norm_c40:.2f}) indicates 4-fold rotational symmetry")
            candidates = [("QPSK", conf), ("8PSK", 0.40), ("MSK", 0.35), ("16-QAM", 0.25)]

        else:
            # Non-Constant Envelope: ASK, OOK, QAM
            evidence.append(f"High envelope variance ({env_var:.3f} >= 0.25): Non-constant envelope family")

            # Check for OOK / ASK vs QAM
            # OOK has high kurtosis and distinct off-cycles
            if features.kurtosis_amp > 1.5 and features.sigma_aa > 0.6:
                scheme = "OOK"
                conf = 0.85
                evidence.append("High amplitude kurtosis and on-off envelope transitions consistent with OOK/ASK")
                candidates = [("OOK", 0.85), ("ASK", 0.60), ("16-QAM", 0.25)]
            else:
                # QAM family
                # 16-QAM theoretical c42 ≈ -0.68
                dist_16qam = abs(norm_c42 - (-0.68))
                dist_64qam = abs(norm_c42 - (-0.619))

                if dist_16qam <= dist_64qam:
                    scheme = "16-QAM"
                    conf = min(0.92, max(0.60, 1.0 - 0.4 * dist_16qam))
                    evidence.append(f"C42 cumulant ({norm_c42:.2f}) aligns with 16-QAM theoretical value (-0.68)")
                    candidates = [("16-QAM", conf), ("64-QAM", 0.50), ("QPSK", 0.30)]
                else:
                    scheme = "64-QAM"
                    conf = min(0.88, max(0.55, 1.0 - 0.4 * dist_64qam))
                    evidence.append(f"C42 cumulant ({norm_c42:.2f}) aligns with 64-QAM theoretical value (-0.62)")
                    candidates = [("64-QAM", conf), ("16-QAM", 0.60)]

        # Adjust confidence by SNR
        if snr_db < 6.0:
            conf = max(conf - 0.20, 0.40)
            evidence.append(f"Low SNR ({snr_db:.1f} dB) decreases classification certainty")

        return ModulationResult(
            scheme=scheme,
            confidence_score=float(conf),
            evidence=evidence,
            secondary_candidates=candidates,
        )
