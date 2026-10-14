# coding=utf-8
"""Tests for the pilot-tone injection harness (real-baseband file format)."""
from __future__ import annotations

import filecmp
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt  # noqa: E402

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz  # noqa: E402
from pilot_proxy.chime.injection import (  # noqa: E402
    INJECTION_MANIFEST_FILENAME,
    inject_directory,
    inject_tone_into_baseband_file,
    resolve_baseband_frequency_hz,
)

N_TIME = 4096
N_FEEDS = 3
CENTER_MHZ = 470.312500  # coarse channel containing the DTV 14 pilot
# Land the tone on an exact FFT bin so the delta-spectrum check is leakage-free.
TONE_BIN = 37
F_BB = TONE_BIN * fmt.FS / N_TIME


def _make_base_file(path, *, include_minus_eight: bool = True) -> None:
    fmt.make_synth_file(
        str(path), N_TIME, N_FEEDS, CENTER_MHZ,
        f_tone_bb=0.0, tone_amp=0.0, noise_std=1.0, seed=7,
    )
    with h5py.File(str(path), "r+") as handle:
        if include_minus_eight:
            # Byte 0x00 decodes to (-8, -8): the component a symmetric +/-7
            # quantizer cannot round-trip. Identity must still hold.
            data = handle["baseband"][...]
            data[0, 0] = 0x00
            handle["baseband"][...] = data
        handle.create_dataset("sibling", data=np.arange(5, dtype=np.int64))
        handle.attrs["note"] = "preserve me"


def test_zero_amplitude_is_byte_identical(tmp_path) -> None:
    source = tmp_path / "event_506.h5"
    _make_base_file(source, include_minus_eight=True)
    output = tmp_path / "out" / "event_506.h5"

    entry = inject_tone_into_baseband_file(
        source, output,
        baseband_frequency_hz=F_BB, amplitude_lsb=0.0, phase_seed=1,
    )

    assert entry["byte_identical_to_source"] is True
    assert entry["clip_count"] == 0
    assert filecmp.cmp(str(source), str(output), shallow=False)


def test_injection_delta_is_the_requested_tone(tmp_path) -> None:
    amplitude = 2.0
    source = tmp_path / "event_506.h5"
    _make_base_file(source, include_minus_eight=False)
    output = tmp_path / "out" / "event_506.h5"

    entry = inject_tone_into_baseband_file(
        source, output,
        baseband_frequency_hz=F_BB, amplitude_lsb=amplitude, phase_seed=3,
    )

    assert entry["clip_count"] == 0
    assert entry["byte_identical_to_source"] is False
    # make_synth_file draws complex noise with per-component std
    # noise_std/sqrt(2); integer rounding adds ~1/12 variance.
    expected_rms = float(np.sqrt(0.5 + 1.0 / 12.0))
    assert entry["rms_lsb_per_component"] == pytest.approx(expected_rms, rel=0.1)
    with h5py.File(str(source), "r") as src, h5py.File(str(output), "r") as dst:
        delta = fmt.unpack_4bit(dst["baseband"][...]) - fmt.unpack_4bit(
            src["baseband"][...]
        )
        # Siblings preserved verbatim; only baseband changed.
        assert np.array_equal(src["sibling"][...], dst["sibling"][...])
        assert dst.attrs["note"] == "preserve me"
    spectrum = np.fft.fft(delta, axis=0) / N_TIME
    recovered = np.abs(spectrum[TONE_BIN, :])
    # Rounding noise per bin is ~sqrt(1/6/N_TIME) ~ 0.006 LSB against a = 2.
    assert recovered == pytest.approx(amplitude, rel=0.05)
    off_bin = np.abs(spectrum[TONE_BIN + 5, :])
    assert np.all(off_bin < 0.1 * amplitude)


def test_saturation_is_counted(tmp_path) -> None:
    source = tmp_path / "event_506.h5"
    _make_base_file(source, include_minus_eight=False)
    output = tmp_path / "out" / "event_506.h5"

    entry = inject_tone_into_baseband_file(
        source, output,
        baseband_frequency_hz=F_BB, amplitude_lsb=30.0, phase_seed=3,
    )

    assert entry["clip_count"] > 0
    assert 0.0 < entry["clip_fraction"] <= 1.0


def test_inject_directory_writes_manifest_with_per_file_phases(tmp_path) -> None:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for freq_id in (506, 521):
        _make_base_file(inputs / f"event_{freq_id}.h5", include_minus_eight=False)
    out = tmp_path / "out"

    entries = inject_directory(
        sorted(inputs.glob("*.h5")), out,
        amplitude_lsb=1.5, phase_seed=11,
        baseband_frequency_hz=F_BB,
    )

    assert [entry["phase_seed"] for entry in entries] == [11, 12]
    manifest = json.loads((out / INJECTION_MANIFEST_FILENAME).read_text())
    assert manifest["schema_version"] == "pilotproxy_injection_manifest_v1"
    assert {e["output"] for e in manifest["files"]} == {
        str(out / "event_506.h5"), str(out / "event_521.h5")
    }


def test_resolve_baseband_frequency_paths() -> None:
    center = CENTER_MHZ * 1e6
    pilot = physical_channel_to_pilot_hz(14)
    from_channel = resolve_baseband_frequency_hz(
        center, physical_channel=14, sample_rate_hz=fmt.FS
    )
    from_pilot = resolve_baseband_frequency_hz(
        center, pilot_frequency_hz=pilot, sample_rate_hz=fmt.FS
    )
    assert from_channel == pytest.approx(pilot - center)
    assert from_channel == from_pilot

    with pytest.raises(ValueError, match="outside"):
        resolve_baseband_frequency_hz(
            center, baseband_frequency_hz=0.6 * fmt.FS, sample_rate_hz=fmt.FS
        )
    with pytest.raises(ValueError, match="exactly one"):
        resolve_baseband_frequency_hz(
            center, physical_channel=14, pilot_frequency_hz=pilot,
            sample_rate_hz=fmt.FS,
        )
