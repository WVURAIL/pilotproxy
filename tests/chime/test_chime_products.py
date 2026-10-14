# coding=utf-8
from __future__ import annotations

import os
import stat

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import products as products_module
from pilot_proxy.chime.products import atomic_savez_compressed, spectrum_before_after


def test_before_after_spectrum_excludes_masked_frames_from_denominator() -> None:
    baseband = np.asarray(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
    )
    mask = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )

    before_db, after_db = spectrum_before_after(baseband, mask)

    np.testing.assert_allclose(before_db, 10.0 * np.log10([20.0, 200.0]))
    np.testing.assert_allclose(after_db, 10.0 * np.log10([20.0, 150.0]))


def test_after_spectrum_all_masked_returns_nan_not_zero() -> None:
    before_db, after_db = spectrum_before_after(
        np.asarray([[10.0], [20.0]]),
        np.asarray([[1], [1]], dtype=np.uint8),
    )

    assert np.isfinite(before_db[0])
    assert np.isnan(after_db[0])


def test_before_after_spectrum_uses_valid_frame_denominators() -> None:
    baseband = np.asarray([[10.0], [1000.0], [30.0], [40.0]])
    mask = np.asarray([[0], [1], [0], [0]], dtype=np.uint8)
    valid = np.asarray([[1], [0], [1], [0]], dtype=np.uint8)

    before_db, after_db = spectrum_before_after(baseband, mask, valid)

    np.testing.assert_allclose(before_db, 10.0 * np.log10([20.0]))
    np.testing.assert_allclose(after_db, 10.0 * np.log10([20.0]))


def test_before_after_spectrum_all_invalid_returns_nan() -> None:
    before_db, after_db = spectrum_before_after(
        np.asarray([[10.0], [20.0]]),
        np.asarray([[0], [0]], dtype=np.uint8),
        np.asarray([[0], [0]], dtype=np.uint8),
    )

    assert np.isnan(before_db[0])
    assert np.isnan(after_db[0])


def test_atomic_npz_failure_preserves_previous_product(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "canonical.npz"
    destination.write_bytes(b"previous-good-product")

    def fail_after_partial_write(path, **arrays):
        del arrays
        path.write_bytes(b"partial")
        raise RuntimeError("simulated interrupted writer")

    monkeypatch.setattr(products_module.np, "savez_compressed", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_savez_compressed(destination, value=np.asarray([1]))

    assert destination.read_bytes() == b"previous-good-product"
    assert not list(tmp_path.glob(".canonical.npz.*.tmp.npz"))


def test_atomic_npz_preserves_existing_destination_mode(tmp_path) -> None:
    destination = tmp_path / "canonical.npz"
    np.savez_compressed(destination, value=np.asarray([0]))
    destination.chmod(0o604)
    if stat.S_IMODE(destination.stat().st_mode) != 0o604:
        pytest.skip("filesystem does not support POSIX permission bits")

    atomic_savez_compressed(destination, value=np.asarray([1]))

    assert stat.S_IMODE(destination.stat().st_mode) == 0o604


def test_atomic_npz_new_file_honours_process_umask(tmp_path) -> None:
    probe = tmp_path / "mode-probe"
    previous_umask = os.umask(0o027)
    try:
        probe.write_bytes(b"")
    finally:
        os.umask(previous_umask)
    if stat.S_IMODE(probe.stat().st_mode) != 0o640:
        pytest.skip("filesystem does not expose normal POSIX umask semantics")
    probe.unlink()

    previous_umask = os.umask(0o027)
    try:
        atomic_savez_compressed(tmp_path / "new.npz", value=np.asarray([1]))
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((tmp_path / "new.npz").stat().st_mode) == 0o640
