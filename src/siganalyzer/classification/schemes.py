"""Definitions of supported modulation schemes and their characteristics."""

from dataclasses import dataclass
from enum import Enum


class ModulationFamily(str, Enum):
    """Broad families of digital modulation."""

    AMPLITUDE = "Amplitude Shift Keying (ASK)"
    FREQUENCY = "Frequency Shift Keying (FSK)"
    PHASE = "Phase Shift Keying (PSK)"
    QUADRATURE_AMPLITUDE = "Quadrature Amplitude Modulation (QAM)"
    CONTINUOUS_PHASE = "Continuous Phase Modulation (CPM/MSK)"
    UNKNOWN = "Unknown Modulation"


@dataclass(frozen=True)
class ModulationSchemeInfo:
    """Metadata and expected theoretical characteristics of a modulation scheme."""

    name: str
    family: ModulationFamily
    bits_per_symbol: int
    constant_envelope: bool
    theoretical_norm_c42: float  # C42 / C21^2
    theoretical_norm_c40: float  # |C40| / C21^2
    description: str


SCHEMES: dict[str, ModulationSchemeInfo] = {
    "OOK": ModulationSchemeInfo("OOK", ModulationFamily.AMPLITUDE, 1, False, -1.0, 1.0, "On-Off Keying (Binary ASK)"),
    "ASK": ModulationSchemeInfo("ASK", ModulationFamily.AMPLITUDE, 2, False, -0.68, 0.68, "4-Level Amplitude Shift Keying"),
    "2FSK": ModulationSchemeInfo("2FSK", ModulationFamily.FREQUENCY, 1, True, -1.0, 0.0, "Binary Frequency Shift Keying"),
    "4FSK": ModulationSchemeInfo("4FSK", ModulationFamily.FREQUENCY, 2, True, -1.0, 0.0, "4-Level Frequency Shift Keying"),
    "BPSK": ModulationSchemeInfo("BPSK", ModulationFamily.PHASE, 1, True, -2.0, 1.0, "Binary Phase Shift Keying"),
    "QPSK": ModulationSchemeInfo("QPSK", ModulationFamily.PHASE, 2, True, -1.0, 0.0, "Quadrature Phase Shift Keying"),
    "8PSK": ModulationSchemeInfo("8PSK", ModulationFamily.PHASE, 3, True, -1.0, 0.0, "8-Ary Phase Shift Keying"),
    "16-QAM": ModulationSchemeInfo("16-QAM", ModulationFamily.QUADRATURE_AMPLITUDE, 4, False, -0.68, 0.0, "16-State Quadrature Amplitude Modulation"),
    "64-QAM": ModulationSchemeInfo("64-QAM", ModulationFamily.QUADRATURE_AMPLITUDE, 6, False, -0.619, 0.0, "64-State Quadrature Amplitude Modulation"),
    "MSK": ModulationSchemeInfo("MSK", ModulationFamily.CONTINUOUS_PHASE, 1, True, -1.0, 0.0, "Minimum Shift Keying"),
    "GMSK": ModulationSchemeInfo("GMSK", ModulationFamily.CONTINUOUS_PHASE, 1, True, -1.0, 0.0, "Gaussian Minimum Shift Keying"),
}
