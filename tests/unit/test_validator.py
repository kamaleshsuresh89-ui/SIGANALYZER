"""Unit tests for FileValidator."""

from pathlib import Path
import numpy as np

from siganalyzer.io.validator import FileValidator


def test_validator_nonexistent_file(tmp_path: Path):
    res = FileValidator.validate_file_structure(tmp_path / "missing.bin")
    assert not res.is_valid
    assert any("does not exist" in err for err in res.errors)


def test_validator_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.bin"
    empty.touch()
    res = FileValidator.validate_file_structure(empty)
    assert not res.is_valid
    assert any("empty" in err for err in res.errors)


def test_validator_samples_nan_inf():
    # Test NaNs
    arr_nan = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.complex64)
    res_nan = FileValidator.validate_samples(arr_nan)
    assert not res_nan.is_valid
    assert res_nan.has_nans

    # Test Infs
    arr_inf = np.array([1.0, np.inf, 3.0], dtype=np.complex64)
    res_inf = FileValidator.validate_samples(arr_inf)
    assert not res_inf.is_valid
    assert res_inf.has_infs


def test_validator_samples_clipping():
    arr_clip = np.array([0.5, 0.9999, 1.0, 0.4], dtype=np.complex64)
    res = FileValidator.validate_samples(arr_clip)
    assert res.is_valid  # Warnings do not invalidate
    assert res.has_clipping
    assert len(res.warnings) > 0


def test_validator_samples_silence():
    arr_silent = np.zeros(1000, dtype=np.complex64)
    res = FileValidator.validate_samples(arr_silent)
    assert res.is_valid
    assert res.has_excessive_silence
