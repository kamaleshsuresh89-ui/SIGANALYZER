"""Deterministic synthetic RF signal generator fixture for tests and validation."""

from pathlib import Path
import struct
import numpy as np

from siganalyzer.common.types import SignalFileType


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Additive White Gaussian Noise to achieve specified SNR in dB."""
    if snr_db is None or np.isneginf(snr_db):
        return signal
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = (
        rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
        + 1j * rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
    )
    return signal + noise


def add_iq_imbalance(
    signal: np.ndarray, gain_imbalance_db: float = 0.0, phase_imbalance_deg: float = 0.0
) -> np.ndarray:
    """Simulate hardware IQ gain and phase imbalance."""
    i = np.real(signal)
    q = np.imag(signal)
    gain_linear = 10 ** (gain_imbalance_db / 20.0)
    phase_rad = np.radians(phase_imbalance_deg)

    i_imb = i * gain_linear
    q_imb = i * np.sin(phase_rad) + q * np.cos(phase_rad)
    return i_imb + 1j * q_imb


def rrc_filter(num_taps: int, alpha: float, sps: int) -> np.ndarray:
    """Generate Root-Raised Cosine (RRC) filter taps."""
    t = np.arange(-num_taps // 2, num_taps // 2 + 1) / sps
    h = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif abs(abs(4 * alpha * ti) - 1.0) < 1e-6:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            numerator = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            denominator = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = numerator / denominator
    return h / np.sqrt(np.sum(h ** 2))


def generate_bpsk(
    num_symbols: int = 2000,
    sps: int = 8,
    sample_rate: float = 1_000_000.0,
    carrier_freq: float = 100_000.0,
    snr_db: float = 20.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a clean BPSK signal with ground truth metadata."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols)
    symbols = 2 * bits - 1  # [-1, +1]

    # Pulse shaping
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    taps = rrc_filter(num_taps=6 * sps, alpha=0.35, sps=sps)
    filtered = np.convolve(upsampled, taps, mode="same")

    # Frequency shift to carrier_freq
    t = np.arange(len(filtered)) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    signal = (filtered * carrier).astype(np.complex64)

    noisy = add_awgn(signal, snr_db, rng)

    ground_truth = {
        "scheme": "BPSK",
        "num_symbols": num_symbols,
        "sps": sps,
        "symbol_rate": sample_rate / sps,
        "sample_rate": sample_rate,
        "carrier_freq": carrier_freq,
        "snr_db": snr_db,
        "bits": bits,
    }
    return noisy, ground_truth


def generate_qpsk(
    num_symbols: int = 2000,
    sps: int = 8,
    sample_rate: float = 1_000_000.0,
    carrier_freq: float = 150_000.0,
    snr_db: float = 25.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a clean QPSK signal with ground truth metadata."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols * 2)
    i_bits = 2 * bits[0::2] - 1
    q_bits = 2 * bits[1::2] - 1
    symbols = (i_bits + 1j * q_bits) / np.sqrt(2)

    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    taps = rrc_filter(num_taps=6 * sps, alpha=0.35, sps=sps)
    filtered = np.convolve(upsampled, taps, mode="same")

    t = np.arange(len(filtered)) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    signal = (filtered * carrier).astype(np.complex64)

    noisy = add_awgn(signal, snr_db, rng)

    ground_truth = {
        "scheme": "QPSK",
        "num_symbols": num_symbols,
        "sps": sps,
        "symbol_rate": sample_rate / sps,
        "sample_rate": sample_rate,
        "carrier_freq": carrier_freq,
        "snr_db": snr_db,
        "bits": bits,
    }
    return noisy, ground_truth


def generate_2fsk(
    num_symbols: int = 2000,
    sps: int = 16,
    sample_rate: float = 1_000_000.0,
    carrier_freq: float = 200_000.0,
    freq_deviation: float = 50_000.0,
    snr_db: float = 25.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a continuous-phase 2FSK signal with ground truth."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_symbols)
    freq_offsets = (2 * bits - 1) * freq_deviation

    # Repeat per symbol (rectangular frequency pulse)
    inst_freq = np.repeat(carrier_freq + freq_offsets, sps)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
    signal = np.exp(1j * phase).astype(np.complex64)

    noisy = add_awgn(signal, snr_db, rng)

    ground_truth = {
        "scheme": "2FSK",
        "num_symbols": num_symbols,
        "sps": sps,
        "symbol_rate": sample_rate / sps,
        "sample_rate": sample_rate,
        "carrier_freq": carrier_freq,
        "freq_deviation": freq_deviation,
        "snr_db": snr_db,
        "bits": bits,
    }
    return noisy, ground_truth


def save_as_wav(
    file_path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
    bit_depth: int = 16,
    center_freq: float | None = None,
) -> Path:
    """Write complex samples to a standard stereo WAV file with optional auxi SDR chunk."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize samples to avoid clipping
    max_amp = np.max(np.abs(samples))
    if max_amp > 0:
        norm_samples = samples / max_amp * 0.95
    else:
        norm_samples = samples

    num_samples = len(norm_samples)
    channels = 2

    # Interleave I and Q
    i_channel = np.real(norm_samples)
    q_channel = np.imag(norm_samples)

    if bit_depth == 16:
        i_data = (i_channel * 32767).astype("<i2")
        q_data = (q_channel * 32767).astype("<i2")
        interleaved = np.empty((num_samples * 2,), dtype="<i2")
        interleaved[0::2] = i_data
        interleaved[1::2] = q_data
        raw_bytes = interleaved.tobytes()
        audio_format = 1
        bytes_per_sample = 2
    elif bit_depth == 32:
        i_data = i_channel.astype("<f4")
        q_data = q_channel.astype("<f4")
        interleaved = np.empty((num_samples * 2,), dtype="<f4")
        interleaved[0::2] = i_data
        interleaved[1::2] = q_data
        raw_bytes = interleaved.tobytes()
        audio_format = 3  # IEEE Float
        bytes_per_sample = 4
    else:
        raise ValueError(f"Unsupported bit depth for test WAV: {bit_depth}")

    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample

    with open(path, "wb") as f:
        # RIFF header placeholder
        f.write(b"RIFF\x00\x00\x00\x00WAVE")

        # fmt chunk
        fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, audio_format, channels, sample_rate, byte_rate, block_align, bit_depth)
        f.write(fmt_chunk)

        # Optional SDR# auxi chunk
        if center_freq is not None:
            auxi_data = struct.pack("<Q", int(center_freq))
            auxi_chunk = struct.pack("<4sI", b"auxi", len(auxi_data)) + auxi_data
            f.write(auxi_chunk)

        # data chunk
        data_header = struct.pack("<4sI", b"data", len(raw_bytes))
        f.write(data_header)
        f.write(raw_bytes)

        # Update RIFF size
        total_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", total_size - 8))

    return path


def save_as_raw_iq(
    file_path: str | Path,
    samples: np.ndarray,
    file_type: SignalFileType = SignalFileType.RAW_IQ_FLOAT32,
) -> Path:
    """Save complex samples as raw binary IQ."""
    path = Path(file_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    max_amp = np.max(np.abs(samples))
    norm_samples = samples / max_amp * 0.95 if max_amp > 0 else samples
    num_samples = len(norm_samples)

    i_ch = np.real(norm_samples)
    q_ch = np.imag(norm_samples)

    match file_type:
        case SignalFileType.RAW_IQ_FLOAT32:
            arr = np.empty(num_samples * 2, dtype="<f4")
            arr[0::2] = i_ch.astype("<f4")
            arr[1::2] = q_ch.astype("<f4")
        case SignalFileType.RAW_IQ_INT16:
            arr = np.empty(num_samples * 2, dtype="<i2")
            arr[0::2] = (i_ch * 32767).astype("<i2")
            arr[1::2] = (q_ch * 32767).astype("<i2")
        case SignalFileType.RAW_IQ_UINT8_RTL:
            arr = np.empty(num_samples * 2, dtype="u1")
            arr[0::2] = (i_ch * 127.5 + 127.5).astype("u1")
            arr[1::2] = (q_ch * 127.5 + 127.5).astype("u1")
        case SignalFileType.RAW_IQ_INT8_HACKRF:
            arr = np.empty(num_samples * 2, dtype="i1")
            arr[0::2] = (i_ch * 127.0).astype("i1")
            arr[1::2] = (q_ch * 127.0).astype("i1")
        case _:
            raise ValueError(f"Unsupported raw type for saving: {file_type}")

    with open(path, "wb") as f:
        f.write(arr.tobytes())

    return path
