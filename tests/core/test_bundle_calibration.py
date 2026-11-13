"""Calibrate a bundle from synthetic captures, then test it."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy.bundle_calibration import (
    _expand,
    _load_fines,
    binomial_lower_tail,
    binomial_upper_tail,
    calibrate_bundle,
    capture_weights,
    iter_run_frames,
    knee_word,
    nominal_fine_bin,
    simulated_null_scores,
    test_bundle as run_bundle_test,
)
from pilot_proxy.run_plans import write_run_plans
from pilot_proxy.run_spectrogram import FRAME_SECONDS, run_spectrogram, write_masked_summary, write_run_spectrogram

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
    # strong and clean captures in turn, so the kept fraction has a knee
    mixed = [write_capture(tmp / f"c35_{i}.h5", 35, 2, 10 + i, tone=0.8 * (i % 2)) for i in range(6)]
    calibrate = dict(common, epoch="synthetic", null_quantile=0.8, min_knee_captures=4, channels={
        "35": mixed,
        "29": [write_capture(tmp / "c29.h5", 29, 6, 2)],
    }, control=dict(channel=37, captures=[write_capture(tmp / "c37.h5", 37, 26, 3)]))
    test = dict(capture_format="chime_baseband",
                channels={"35": [write_capture(tmp / f"t35_{i}.h5", 35, 2, 20 + i, tone=0.8 * (i % 2)) for i in range(2)],
                          "29": [write_capture(tmp / "t29.h5", 29, 4, 5)]},
                control=[write_capture(tmp / "t37.h5", 37, 12, 6)])
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


def test_manifest_holds_no_local_paths(plans):
    manifest = json.loads((plans["bundle"] / "weights.manifest.json").read_text())
    assert manifest["receiver_profile_path"] == "chime_dtv_fengine.json"
    assert "/" not in manifest["detector_core_profile_path"]


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
    plan = dict(plans["test"], channels={"35": plans["calibrate"]["channels"]["35"]})
    path = plans["tmp"] / "reused.json"
    path.write_text(json.dumps(plan))
    result = run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)
    assert not next(c for c in result["checks"] if c["check"].startswith("test data"))["passed"]


def test_channel_masked_unlike_its_calibration_fails(plans):
    plan = dict(plans["test"], channels={"29": [write_capture(
        plans["tmp"] / "t29_tone.h5", 29, 24, 7, tone=0.8)]})
    path = plans["tmp"] / "wrong.json"
    path.write_text(json.dumps(plan))
    assert not run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)["passed"]


def test_word_is_the_knee_above_the_floor(plans):
    report = plans["report"]
    channel = report["channels"]["35"]
    assert channel["word_source"] == "knee"
    assert channel["word_q16"] == channel["knee_word_q16"] > report["floor_word_q16"]
    profiles = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())["profiles"]
    fine = {p["physical_channel"]: p["fine_calibration"] for p in profiles}
    assert fine[35]["cfar_multiplier_q16"] == channel["word_q16"]
    assert fine[35]["designated_half_width"] == 32


def test_channel_with_no_pilot_takes_the_floor(plans):
    channel = plans["report"]["channels"]["29"]
    assert channel["word_source"] == "floor, no pilot"
    assert channel["word_q16"] == plans["report"]["floor_word_q16"]


def test_too_few_captures_take_the_floor(plans, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], min_knee_captures=30)))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    assert report["channels"]["35"]["word_source"] == "floor, 6 captures of 30"


def test_knee_sits_at_the_end_of_the_clean_population():
    clean = np.exp(np.linspace(np.log(70000), np.log(80000), 900))
    pilots = np.exp(np.linspace(np.log(2e5), np.log(1e8), 100))
    words = np.r_[clean, pilots].astype(np.int64)
    assert knee_word(words, np.ones(words.size)) == words[899]
    assert knee_word(np.full(5, 70000), np.ones(5)) == 70000


def test_capture_weights_are_equal_per_capture():
    weights = capture_weights(np.array([0, 0, 0, 1]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(weights[3])


def write_run_file(path, channels, frames):
    """Write one detect-only dtv_fine_powers file, with an idle row for the other channel."""
    data = np.zeros((len(frames), len(channels), 3, 256), dtype=np.uint64)
    data[:, 0] = frames
    with h5py.File(path, "w") as handle:
        handle["dtv_fine_powers_n0"] = data
        handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = channels
    return str(path)


def test_run_frames_skip_untested_and_missing_blocks(tmp_path):
    frames = np.ones((3, 3, 256), dtype=np.uint64)
    frames[1] = np.iinfo(np.uint64).max
    (tmp_path / "acq_1").mkdir()
    write_run_file(tmp_path / "acq_1" / "run.h5", [521, 614], frames)
    assert len(list(iter_run_frames(tmp_path / "acq_1" / "run.h5", 521))) == 2
    assert not list(iter_run_frames(tmp_path / "acq_1" / "run.h5", 614))
    assert _expand([str(tmp_path / "**" / "run.h5")]) == [str(tmp_path / "acq_1" / "run.h5")]
    with h5py.File(tmp_path / "idle.h5", "w") as handle:
        handle["dtv_fine_powers"] = np.zeros((3, 2, 3, 256), dtype=np.uint64)
    assert not list(iter_run_frames(tmp_path / "idle.h5", 521))


def test_detector_run_gives_the_bundle_of_its_captures(plans, tmp_path):
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    bank = (plans["bundle"] / "weights.bin").read_bytes()
    reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    files = []
    for index, capture in enumerate(plans["calibrate"]["channels"]["35"]):
        fines, _ = _load_fines([capture], "chime_baseband", row, bank, 128, reverse, None)
        files.append(write_run_file(tmp_path / f"run_{index}.h5", [521, 614], fines))
    plan = dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"]})
    run = dict(plan, capture_format="dtv_fine_powers", channels={"35": [str(tmp_path / "run_*.h5")]},
               control=None, num_inputs=INPUTS, word_q16=plans["report"]["floor_word_q16"])
    plan.update(word_q16=plans["report"]["floor_word_q16"], control=None, num_inputs=INPUTS)
    paths = tmp_path / "base.json", tmp_path / "run.json"
    paths[0].write_text(json.dumps(plan))
    paths[1].write_text(json.dumps(run))
    base = calibrate_bundle(plan_path=paths[0], output_dir=tmp_path / "base")
    other = calibrate_bundle(plan_path=paths[1], output_dir=tmp_path / "other")
    for key in ("anchor_bin", "knee_word_q16", "frames", "captures", "masked_fraction"):
        assert other["channels"]["35"][key] == base["channels"]["35"][key]
    assert other["calibration_captures"] == sorted(files)


def test_spectrogram_removes_masked_frames(plans, tmp_path):
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    anchor = row["fine_calibration"]["anchor_bin"]
    rng = np.random.default_rng(7)
    quiet = rng.integers(900, 1100, size=(48, 3, 256)).astype(np.uint64)
    loud = quiet.copy()
    loud[:16, 0, anchor] *= np.uint64(1000)
    run = write_run_file(tmp_path / "run.h5", [521, 614], loud)
    result = run_spectrogram(bundle_dir=plans["bundle"], runs=[run], channel=35, chunk_seconds=16 * FRAME_SECONDS)
    assert result["frames"].tolist() == [16, 16, 16]
    assert result["kept"].tolist() == [0, 16, 16]
    assert result["all_frames"][0, anchor] > 100 * result["all_frames"][1, anchor]
    assert np.isnan(result["kept_frames"][0]).all()
    assert np.array_equal(result["kept_frames"][1], result["all_frames"][1])
    data, figure = write_run_spectrogram(result, tmp_path / "out")
    assert data.exists() and figure.exists()
    # 16 of 48 frames, in one stretch that crosses these 5-frame chunks
    short = run_spectrogram(bundle_dir=plans["bundle"], runs=[run], channel=35, chunk_seconds=5 * FRAME_SECONDS)
    assert short["longest_masked_frames"] == 16
    summary = list(csv.DictReader(write_masked_summary([short], tmp_path / "out").open()))
    assert summary[0]["channel"] == "35" and summary[0]["masked_frames"] == "16"
    assert summary[0]["masked_percent"] == "33.33"
    assert float(summary[0]["longest_masked_seconds"]) == pytest.approx(16 * FRAME_SECONDS, abs=0.005)


def test_run_plans_split_each_channel_in_time(plans, tmp_path):
    captures = plans["calibrate"]["control"]["captures"]
    plan = dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    frames = np.ones((2, 3, 256), dtype=np.uint64)
    for index in range(6):
        name = write_run_file(tmp_path / f"run_{index}.h5", [521, 491], frames)
        with h5py.File(name, "a") as handle:
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = 1000 - index
    # another half, with no bound channel
    write_run_file(tmp_path / "run_other.h5", [614, 700], frames)
    result = write_run_plans(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], output_dir=tmp_path / "out",
                             epoch="synthetic", receiver_profile=Path(plans["calibrate"]["receiver_profile"]))
    first, last = result["calibrate"]["channels"]["37"], result["test"]["channels"]["37"]
    assert [Path(p).name for p in first] == ["run_5.h5", "run_4.h5", "run_3.h5", "run_2.h5"]
    assert [Path(p).name for p in last] == ["run_1.h5", "run_0.h5"]
    assert result["calibrate"]["control"] == dict(channel=37, captures=first)
    assert result["test"]["control"] == last
    assert json.loads((tmp_path / "out" / "test.json").read_text()) == result["test"]


def test_first_bundle_needs_no_captures(plans, tmp_path):
    plan = dict(plans["calibrate"], channels={"29": [], "37": []}, control=None, num_inputs=32, null_quantile=0.99)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    assert report["floor_eta"] > 1.84
    assert report["channels"]["37"]["word_source"] == "floor, no captures"


def test_control_channel_can_be_bound(plans, tmp_path):
    captures = plans["calibrate"]["control"]["captures"]
    plan = dict(plans["calibrate"], channels={**plans["calibrate"]["channels"], "37": captures})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    rows = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    status = {row["physical_channel"]: row["fine_calibration"]["status"] for row in rows}
    assert status == {29: "calibrated", 35: "calibrated", 37: "control"}
    assert report["channels"]["37"]["word_q16"] == report["floor_word_q16"]


def test_plan_values_override_the_measurements(plans, tmp_path):
    plan = dict(plans["calibrate"], word_q16=80000, anchors={"29": 200}, words={"35": 90000})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    assert report["floor_word_q16"] == 80000
    assert report["channels"]["29"]["word_q16"] == 80000
    assert report["channels"]["35"]["word_q16"] == 90000
    assert report["channels"]["29"]["anchor_source"] == "plan"
    assert report["channels"]["29"]["anchor_bin"] == 200


def test_existing_output_is_refused(plans):
    with pytest.raises(ValueError, match="never changed"):
        calibrate_bundle(plan_path=plans["calibrate_path"], output_dir=plans["bundle"])


def test_simulated_word_falls_with_input_count():
    few = np.quantile(simulated_null_scores(32, frames=200), 0.99) / 65536
    many = np.quantile(simulated_null_scores(256, frames=200), 0.99) / 65536
    assert few > 1.84
    assert many < few


def test_nominal_bin_matches_pathfinder_control():
    assert nominal_fine_bin(608309441.0, 3115 * 195312.5, CELL, False) == 214


def test_binomial_tail():
    assert binomial_upper_tail(0, 10, 0.5) == 1.0
    assert binomial_upper_tail(10, 10, 0.5) == pytest.approx(2.0 ** -10)
    assert binomial_lower_tail(0, 10, 0.5) == pytest.approx(2.0 ** -10)
    assert binomial_upper_tail(1000, 5000, 0.2) == pytest.approx(0.5, abs=0.02)
