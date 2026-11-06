"""Calibrate a bundle from synthetic captures, then test it."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy.bundle_calibration import (
    binomial_upper_tail,
    calibrate_bundle,
    nominal_fine_bin,
    simulated_null_scores,
    test_bundle as run_bundle_test,
)

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 16384
INPUTS = 16
FS = 390625.0
CELL = 3051.7578125
# channel: (freq_id, centre, pilot)
CHANNELS = {35: (521, 596484375.0, 596309441.0), 29: (614, 560156250.0, 560309441.0), 37: (491, 608203125.0, 608309441.0)}


def write_capture(path, channel, frames, seed, tone=0.0):
    freq_id, centre, pilot = CHANNELS[channel]
    rng = np.random.default_rng(seed)
    n = frames * BLOCK
    z = rng.standard_normal((n, INPUTS)) + 1j * rng.standard_normal((n, INPUTS))
    if tone:
        # the CHIME axis is descending, so a positive offset is a negative baseband rate
        t = np.arange(n) / FS
        z = z + tone * np.exp(-2j * np.pi * (pilot - centre) * t)[:, None]
    z *= 2.0
    real = np.clip(np.rint(z.real), -8, 7).astype(np.int16) + 8
    imag = np.clip(np.rint(z.imag), -8, 7).astype(np.int16) + 8
    with h5py.File(path, "w") as handle:
        handle.attrs["freq_id"] = freq_id
        handle.attrs["time0_fpga_count"] = 0
        handle["baseband"] = ((real << 4) | imag).astype(np.uint8)
    return str(path)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    configs = ROOT / "configs"
    common = dict(
        receiver_profile=str(configs / "receiver_profiles/chime_dtv_fengine.json"),
        detector_core_profile=str(configs / "detector_core/pilotproxy_cuda_local_reference_power_ratio.json"),
        capture_format="chime_baseband",
    )
    calibrate = dict(common, epoch="synthetic", null_quantile=0.9, channels={
        "35": [write_capture(tmp / "c35.h5", 35, 6, 1, tone=0.8)],
        "29": [write_capture(tmp / "c29.h5", 29, 6, 2)],
    }, control=dict(channel=37, captures=[write_capture(tmp / "c37.h5", 37, 6, 3)]))
    test = dict(capture_format="chime_baseband", max_quiet_masked=0.5,
                strong={"35": [write_capture(tmp / "t35.h5", 35, 4, 4, tone=0.8)]},
                quiet={"29": [write_capture(tmp / "t29.h5", 29, 4, 5)]},
                control=[write_capture(tmp / "t37.h5", 37, 6, 6)])
    calibrate_path, test_path = tmp / "calibrate.json", tmp / "test.json"
    calibrate_path.write_text(json.dumps(calibrate))
    test_path.write_text(json.dumps(test))
    bundle = tmp / "bundle"
    report = calibrate_bundle(plan_path=calibrate_path, output_dir=bundle)
    return dict(tmp=tmp, bundle=bundle, report=report, calibrate=calibrate, test=test, test_path=test_path,
                calibrate_path=calibrate_path)


def test_strong_pilot_is_measured_and_quiet_channel_is_nominal(plans):
    channels = plans["report"]["channels"]
    assert channels["35"]["anchor_source"] == "measured"
    assert abs(channels["35"]["anchor_bin"] - channels["35"]["nominal_bin"]) <= 2
    assert channels["29"]["anchor_source"] == "nominal"


def test_bundle_passes_on_other_captures(plans):
    result = run_bundle_test(bundle_dir=plans["bundle"], plan_path=plans["test_path"])
    assert result["passed"], result


def test_tampered_bundle_fails(plans, tmp_path):
    import shutil

    bundle = tmp_path / "bundle"
    shutil.copytree(plans["bundle"], bundle)
    data = bytearray((bundle / "weights.bin").read_bytes())
    data[0] ^= 1
    (bundle / "weights.bin").write_bytes(bytes(data))
    assert not run_bundle_test(bundle_dir=bundle, plan_path=plans["test_path"])["passed"]


def test_reused_capture_fails(plans):
    plan = dict(plans["test"], strong=plans["calibrate"]["channels"] | {})
    plan["strong"] = {"35": plans["calibrate"]["channels"]["35"]}
    path = plans["tmp"] / "reused.json"
    path.write_text(json.dumps(plan))
    result = run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)
    assert not next(c for c in result["checks"] if c["check"].startswith("test data"))["passed"]


def test_quiet_channel_with_a_pilot_fails(plans):
    plan = dict(plans["test"], quiet={"35": plans["test"]["strong"]["35"]})
    path = plans["tmp"] / "wrong.json"
    path.write_text(json.dumps(plan))
    assert not run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)["passed"]


def test_plan_values_override_the_measurements(plans, tmp_path):
    plan = dict(plans["calibrate"], word_q16=80000, anchors={"29": 200})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    assert report["word_q16"] == 80000
    assert report["channels"]["29"]["anchor_source"] == "plan"
    assert report["channels"]["29"]["anchor_bin"] == 200


def test_existing_output_is_refused(plans):
    with pytest.raises(ValueError, match="never changed"):
        calibrate_bundle(plan_path=plans["calibrate_path"], output_dir=plans["bundle"])


def test_simulated_word_falls_with_input_count():
    few = np.quantile(simulated_null_scores(32, frames=200), 0.99) / 65536
    many = np.quantile(simulated_null_scores(256, frames=200), 0.99) / 65536
    assert few == pytest.approx(1.84, abs=0.1)
    assert many < few


def test_nominal_bin_matches_pathfinder_control():
    assert nominal_fine_bin(608309441.0, 3115 * 195312.5, CELL, False) == 214


def test_binomial_tail():
    assert binomial_upper_tail(0, 10, 0.5) == 1.0
    assert binomial_upper_tail(10, 10, 0.5) == pytest.approx(2.0 ** -10)
