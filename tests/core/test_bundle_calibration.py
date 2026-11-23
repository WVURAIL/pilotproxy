"""Calibrate a bundle from synthetic captures, then test it."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy.power_limit_word import control_corrections, limit_test, ratio_excess
from pilot_proxy.bundle_calibration import (
    BUNDLE_FILES,
    NO_PILOT_REASON,
    REJECT_WORD_Q16,
    _bulk,
    _exact_word,
    _expand,
    _load_fines,
    binomial_lower_tail,
    binomial_upper_tail,
    bundle_digest,
    calibrate_bundle,
    capture_weights,
    control_words_at,
    default_rank,
    iter_blocks,
    iter_run_frames,
    knee_word,
    nominal_fine_bin,
    qualify_bundle,
    required_words,
    scaled_word,
    simulated_null_scores,
    test_bundle as run_bundle_test,
    weight_norms,
)
from pilot_proxy.characterization import threshold_rule
from pilot_proxy.fine_decision import fine_stat_v3, masks_from_stat, v3_word
from pilot_proxy.runtime_bundle import _write_sha256sums, validate_runtime_weight_bundle
from pilot_proxy.process_run import process_run
from pilot_proxy.run_plans import ordered_run_files, write_run_plans
from pilot_proxy.run_files import combine
from kotekan_files import write_frames_file
from pilot_proxy.run_scores import build_run_scores
from pilot_proxy.run_spectrogram import FRAME_SECONDS, run_spectrogram, write_masked_summary, write_run_spectrogram

ROOT = Path(__file__).resolve().parents[2]
BLOCK = 16384
INPUTS = 16
FS = 390625.0
CELL = 3051.7578125
# channel: (freq_id, centre, pilot)
CHANNELS = {35: (521, 596484375.0, 596309441.0), 29: (614, 560156250.0, 560309441.0), 37: (491, 608203125.0, 608309441.0)}


def write_capture(path, channel, frames, seed, tone=0.0, ctime=None):
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
        if ctime is not None:
            handle.attrs["time0_ctime"] = ctime
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
    control = [write_capture(tmp / "c37.h5", 37, 26, 3)]
    calibrate = dict(common, epoch="synthetic", null_quantile=0.8, min_knee_captures=4, channels={
        "35": mixed,
        "29": [write_capture(tmp / "c29.h5", 29, 6, 2)],
        "37": control,
    }, control=dict(channel=37, captures=control))
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


def test_bundle_test_needs_the_control(plans, tmp_path):
    plan = dict(plans["test"], control=[])
    path = tmp_path / "no_control.json"
    path.write_text(json.dumps(plan))
    result = run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)
    check = next(c for c in result["checks"] if c["check"] == "false alarms on the control channel")
    assert not result["passed"] and not check["passed"] and "no control" in check["detail"]


def _passing_report(plans, tmp_path):
    report = tmp_path / "test_report.json"
    assert run_bundle_test(bundle_dir=plans["bundle"], plan_path=plans["test_path"], output_json=report)["passed"]
    return report


def test_qualified_bundle_names_its_test_and_reviewer(plans, tmp_path):
    report = _passing_report(plans, tmp_path)
    summary = qualify_bundle(bundle_dir=plans["bundle"], test_report=report, reviewer="A. Reviewer",
                             output_dir=tmp_path / "qualified")
    assert summary["qualified_channels"] == [29, 35]
    assert summary["source_bundle_sha256sums"] == bundle_digest(plans["bundle"]) != summary["bundle_sha256sums"]
    profiles = json.loads((tmp_path / "qualified" / "pilot_profiles.json").read_text())["profiles"]
    for row in profiles:
        record = row["fine_calibration"]["provenance"].get("mask_qualification")
        if row["physical_channel"] == 37:
            assert record is None
        else:
            assert record == dict(test_report_sha256=summary["test_report_sha256"], reviewer="A. Reviewer")
    assert not validate_runtime_weight_bundle(bundle_dir=tmp_path / "qualified").get("errors")


def test_qualify_refuses_a_failed_or_foreign_report(plans, tmp_path):
    report = _passing_report(plans, tmp_path)
    failed = json.loads(report.read_text())
    failed["passed"] = False
    failed["checks"][0]["passed"] = False
    (tmp_path / "failed.json").write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="did not pass"):
        qualify_bundle(bundle_dir=plans["bundle"], test_report=tmp_path / "failed.json", reviewer="R",
                       output_dir=tmp_path / "q1")
    foreign = dict(json.loads(report.read_text()), bundle_sha256sums="0" * 64)
    (tmp_path / "foreign.json").write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="is not a test of"):
        qualify_bundle(bundle_dir=plans["bundle"], test_report=tmp_path / "foreign.json", reviewer="R",
                       output_dir=tmp_path / "q2")
    assert not (tmp_path / "q1").exists() and not (tmp_path / "q2").exists()


def test_qualify_refuses_a_channel_the_test_left_out(plans, tmp_path):
    # the test plan leaves out channel 29, so its report passes and never tests it
    plan = dict(plans["test"], channels={"35": plans["test"]["channels"]["35"]})
    path = tmp_path / "test35.json"
    path.write_text(json.dumps(plan))
    report = tmp_path / "report35.json"
    assert run_bundle_test(bundle_dir=plans["bundle"], plan_path=path, output_json=report)["passed"]
    with pytest.raises(ValueError, match="no passing masked-fraction check of channel 29"):
        qualify_bundle(bundle_dir=plans["bundle"], test_report=report, reviewer="R", output_dir=tmp_path / "q")
    assert not (tmp_path / "q").exists()


def test_qualification_is_refused_on_the_control(plans, tmp_path):
    plan = dict(plans["calibrate"], channels={**plans["calibrate"]["channels"],
                                              "37": plans["calibrate"]["control"]["captures"]})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    bundle = tmp_path / "bundle"
    calibrate_bundle(plan_path=path, output_dir=bundle)
    data = json.loads((bundle / "pilot_profiles.json").read_text())
    control = next(p for p in data["profiles"] if p["physical_channel"] == 37)
    control["fine_calibration"]["provenance"]["mask_qualification"] = dict(test_report_sha256="a" * 64, reviewer="R")
    (bundle / "pilot_profiles.json").write_text(json.dumps(data, indent=2) + "\n")
    _write_sha256sums(bundle, BUNDLE_FILES)
    errors = validate_runtime_weight_bundle(bundle_dir=bundle)["errors"]
    assert any("only a calibrated channel" in e["message"] for e in errors), errors


def test_plan_input_count_is_recorded(plans, tmp_path):
    plan = dict(plans["calibrate"], num_inputs=INPUTS)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    profiles = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    assert {row["fine_calibration"]["provenance"]["summed_inputs"] for row in profiles} == {INPUTS}


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


def test_a_copy_of_a_calibration_capture_is_not_test_data(plans, tmp_path):
    import os
    import shutil

    source = plans["calibrate"]["channels"]["35"][0]
    copy, link = tmp_path / "renamed.h5", tmp_path / "linked.h5"
    shutil.copy(source, copy)
    os.link(source, link)
    for capture in (copy, link):
        plan = dict(plans["test"], channels={"35": [str(capture)] + plans["test"]["channels"]["35"]})
        path = tmp_path / "test.json"
        path.write_text(json.dumps(plan))
        result = run_bundle_test(bundle_dir=plans["bundle"], plan_path=path)
        shared = next(c for c in result["checks"] if c["check"] == "test data differ from calibration data")
        assert not shared["passed"] and shared["detail"] == "1 shared captures"
    # a report from before the identities were recorded still compares paths
    report_path = plans["bundle"].with_name(plans["bundle"].name + ".report.json")
    report = json.loads(report_path.read_text())
    assert len(report["calibration_capture_ids"]) == len(report["calibration_captures"])
    assert len(report["control"]["capture_ids"]) == len(report["control"]["captures"])


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
    assert not channel["pilot_in_nearly_every_frame"]
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


def test_pilot_in_every_capture_takes_the_floor(plans, tmp_path):
    # no clean frames, so the knee would sit in the pilot's own spread and keep most of it
    always = [write_capture(tmp_path / f"a35_{i}.h5", 35, 2, 40 + i, tone=0.8) for i in range(6)]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels={"35": always, "37": plans["calibrate"]["channels"]["37"]})))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    assert channel["pilot_in_nearly_every_frame"]
    assert channel["word_source"] == "floor, pilot in nearly every frame"
    assert channel["word_q16"] == report["floor_word_q16"]
    assert channel["masked_fraction"] > 0.9


@pytest.mark.parametrize("decision", ["fine_decision_v1", "fine_decision_v2", "fine_decision_v3"])
@pytest.mark.parametrize("kind", ["noise", "ties", "dead bins", "near the top", "always masked"])
@pytest.mark.parametrize("anchor", [100, 3])
def test_fast_words_equal_the_exact_reference(kind, anchor, decision):
    rng = np.random.default_rng(len(kind) + anchor)
    n = 300
    if kind == "noise":
        fines = rng.gamma(44, 1e6 / 44, size=(n, 3, 256))
        fines[:, 0, 100] *= rng.choice([1, 3, 30, 1e6], n)
    elif kind == "ties":
        fines = rng.integers(0, 4, size=(n, 3, 256))
    elif kind == "dead bins":
        fines = rng.integers(0, 50, size=(n, 3, 256))
        fines[:, 1:, rng.random(256) < 0.5] = 0
        fines[:, 0, rng.random(256) < 0.3] = 0
    elif kind == "near the top":
        fines = rng.integers(2**62, 2**63, size=(n, 3, 256), dtype=np.uint64)
    else:
        fines = rng.integers(1, 3, size=(n, 3, 256)).astype(np.uint64)
        fines[:, 0, 90:110] = np.uint64(2**63)
    fines = np.asarray(fines).astype(np.uint64)
    bulk = _bulk(anchor)
    rank = default_rank(bulk)
    exact = [_exact_word(f, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision) for f in fines]
    assert required_words(fines, anchor_bin=anchor, bulk=bulk, rank=rank, decision_version=decision).tolist() == exact


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


def write_chord_capture(path, voltages, coarse_freq, fpga_seq_num, dish_label=None):
    """Write a Pathfinder baseband file: uint8 [time, coarse channel, polarization, dish] in baseband_subset."""
    with h5py.File(path, "w") as handle:
        handle["baseband_subset"] = voltages
        handle["baseband_subset"].attrs["coarse_freq"] = coarse_freq
        handle["baseband_subset"].attrs["fpga_seq_num"] = fpga_seq_num
        if dish_label is not None:
            handle["baseband_subset"].attrs["dish_label"] = dish_label
    return str(path)


def test_chord_blocks_start_on_the_block_grid_and_leave_out_rfi_dishes(tmp_path):
    block = 8
    voltages = np.random.default_rng(4).integers(0, 256, size=(5 * block + 3, 2, 2, 3), dtype=np.uint8)
    # 3 samples before a block edge
    path = write_chord_capture(tmp_path / "pf.h5", voltages, [2531, 3115], 1000 * block - 3, ["D1", "RFI1", "D2"])
    blocks = list(iter_blocks(path, "chord_baseband", 3115, block))
    # stream = polarization * dishes + dish, over the dishes that are not RFI antennas
    expected = voltages[3:, 1][:, :, [0, 2]].reshape(5, block, 4)
    assert len(blocks) == 5 and all(np.array_equal(b, e) for b, e in zip(blocks, expected))
    assert len(list(iter_blocks(path, "chord_baseband", 3115, block, max_frames=2))) == 2
    # fixed-length byte labels leave out the same dish
    path = write_chord_capture(tmp_path / "bytes.h5", voltages, [2531, 3115], 1000 * block - 3,
                               np.array([b"D1", b"RFI1", b"D2"], dtype="S4"))
    blocks = list(iter_blocks(path, "chord_baseband", 3115, block))
    assert len(blocks) == 5 and all(np.array_equal(b, e) for b, e in zip(blocks, expected))
    # a file with no labels sums every dish
    path = write_chord_capture(tmp_path / "unlabelled.h5", voltages, [2531, 3115], 0)
    blocks = list(iter_blocks(path, "chord_baseband", 2531, block))
    assert len(blocks) == 5 and np.array_equal(blocks[0], voltages[:block, 0].reshape(block, 6))
    with pytest.raises(ValueError, match="does not hold coarse_freq 2869"):
        list(iter_blocks(path, "chord_baseband", 2869, block))


def test_chime_capture_of_another_channel_or_format_is_refused(tmp_path):
    path = write_capture(tmp_path / "c35.h5", 35, 1, 0)
    assert len(list(iter_blocks(path, "chime_baseband", 521, BLOCK))) == 1
    with pytest.raises(ValueError, match="holds freq_id 521, not 614"):
        list(iter_blocks(path, "chime_baseband", 614, BLOCK))
    with pytest.raises(ValueError, match="unknown capture_format 'chime'"):
        list(iter_blocks(path, "chime", 521, BLOCK))


def test_chord_capture_gives_the_fine_powers_of_the_same_voltages_from_chime(plans, tmp_path):
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    bank = (plans["bundle"] / "weights.bin").read_bytes()
    reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    # the capture with the tone, its 16 inputs as 2 polarizations of 8 dishes
    chime = plans["calibrate"]["channels"]["35"][1]
    with h5py.File(chime, "r") as handle:
        inputs = handle["baseband"][()].reshape(-1, 2, 8)
    # an RFI antenna at full scale (+7 + 7j) between the dishes, and another channel beside this one
    rfi = np.full(inputs.shape[:2] + (1,), 0xFF, dtype=np.uint8)
    dishes = np.concatenate([inputs[:, :, :4], rfi, inputs[:, :, 4:]], axis=2)
    voltages = np.stack([np.full_like(dishes, 0x88), dishes], axis=1)
    labels = ["D0", "D1", "D2", "D3", "RFI0", "D4", "D5", "D6", "D7"]
    labelled = write_chord_capture(tmp_path / "pf.h5", voltages, [600, row["receiver_channel_id"]], 0, labels)
    unlabelled = write_chord_capture(tmp_path / "pf_all.h5", voltages, [600, row["receiver_channel_id"]], 0)
    expected, _ = _load_fines([chime], "chime_baseband", row, bank, 128, reverse, None)
    fines, _ = _load_fines([labelled], "chord_baseband", row, bank, 128, reverse, None)
    assert len(fines) == 2 and np.array_equal(fines, expected)
    # summed in, the RFI antenna changes the powers
    fines, _ = _load_fines([unlabelled], "chord_baseband", row, bank, 128, reverse, None)
    assert not np.array_equal(fines, expected)


def test_run_files_keep_recording_order_across_restarts(tmp_path):
    """An F-engine restart resets fpga_seq_num; the files stay in order by their first frame's time."""
    frames = np.ones((2, 3, 256), dtype=np.uint64)
    # (acquisition directory, fpga_seq_num, time in s): the counter starts again in the second acquisition
    layout = [("acq_20261121_000000_000000000", 900, 10), ("acq_20261121_000000_000000000", 1000, 20),
              ("acq_20261121_020000_000000000", 100, 7230), ("acq_20261121_020000_000000000", 200, 7240)]
    for timed in (True, False):
        paths = []
        for index, (acq, seq, seconds) in enumerate(layout):
            (tmp_path / str(timed) / acq).mkdir(parents=True, exist_ok=True)
            path = write_run_file(tmp_path / str(timed) / acq / f"dtv_fine_powers.{index:08d}.h5", [521, 614], frames)
            with h5py.File(path, "a") as handle:
                handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = seq
                if timed:
                    handle["dtv_fine_powers_n0"].attrs["fpga_seq_time_nsec"] = seconds * 10**9
            paths.append(path)
        # an idle file records neither attribute and does not change the order
        with h5py.File(tmp_path / str(timed) / layout[0][0] / "dtv_fine_powers.idle.h5", "w") as handle:
            handle["dtv_fine_powers_n0"] = np.zeros((2, 2, 3, 256), dtype=np.uint64)
        ordered = [path for _, coarse, path in ordered_run_files([str(tmp_path / str(timed) / "acq_*" / "*.h5")]) if coarse]
        assert ordered == paths
        times = [time_ns for time_ns, coarse, _ in ordered_run_files(paths)]
        assert times == ([s * 10**9 for _, _, s in layout] if timed else [None] * 4)


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


def test_combined_run_gives_the_bundle_of_its_windows(plans, tmp_path):
    """A run combined into one file per half, or written as one several-frame file, calibrates as its
    per-window files do."""
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    bank = (plans["bundle"] / "weights.bin").read_bytes()
    reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    acq = tmp_path / "acq_1"
    acq.mkdir()
    for index, capture in enumerate(plans["calibrate"]["channels"]["35"]):
        fines, _ = _load_fines([capture], "chime_baseband", row, bank, 128, reverse, None)
        path = write_run_file(acq / f"dtv_fine_powers.{index:08d}.h5", [521, 614], fines)
        with h5py.File(path, "a") as handle:
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 1024 * 8192
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_time_nsec"] = 10**18 + index * 43 * 10**9
    summary = combine(acq, tmp_path / "combined", "fine_powers")
    assert [r["windows"] for r in summary] == [len(plans["calibrate"]["channels"]["35"])]
    (tmp_path / "frames").mkdir()
    write_frames_file(sorted(acq.glob("*.h5")), tmp_path / "frames" / "dtv_fine_powers.00000000.h5", per_frame_time=True)
    reports = {}
    for name, pattern in (("windows", str(acq / "dtv_fine_powers.*.h5")),
                          ("combined", str(tmp_path / "combined" / "*.combined.h5")),
                          ("frames", str(tmp_path / "frames" / "*.h5"))):
        run = dict(plans["calibrate"], capture_format="dtv_fine_powers", channels={"35": [pattern]},
                   control=None, num_inputs=INPUTS, word_q16=plans["report"]["floor_word_q16"])
        (tmp_path / f"{name}.json").write_text(json.dumps(run))
        reports[name] = calibrate_bundle(plan_path=tmp_path / f"{name}.json", output_dir=tmp_path / f"bundle-{name}")
    for key in ("anchor_bin", "knee_word_q16", "frames", "captures", "masked_fraction"):
        assert reports["combined"]["channels"]["35"][key] == reports["windows"]["channels"]["35"][key]
        assert reports["frames"]["channels"]["35"][key] == reports["windows"]["channels"]["35"][key]
    assert all("#" in c for c in reports["combined"]["calibration_captures"])
    assert all("#" in c for c in reports["frames"]["calibration_captures"])
    # the three layouts hold the same windows, so a bundle calibrated on one is not tested on another
    ids = {name: report["calibration_capture_ids"] for name, report in reports.items()}
    assert ids["combined"] == ids["windows"] == ids["frames"] and len(ids["windows"]) == len(sorted(acq.glob("*.h5")))
    test = dict(plans["test"], capture_format="dtv_fine_powers",
                channels={"35": [str(tmp_path / "combined" / "*.combined.h5")]}, control=[])
    (tmp_path / "test.json").write_text(json.dumps(test))
    checks = run_bundle_test(bundle_dir=tmp_path / "bundle-windows", plan_path=tmp_path / "test.json")["checks"]
    shared = next(c for c in checks if c["check"] == "test data differ from calibration data")
    assert not shared["passed"] and shared["detail"] == f"{len(ids['windows'])} shared captures"


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


def test_run_scores_feed_the_threshold_rule(plans, tmp_path):
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    anchor, word = row["fine_calibration"]["anchor_bin"], row["fine_calibration"]["cfar_multiplier_q16"]
    rng = np.random.default_rng(7)
    runs = []
    for index in range(2):
        frames = rng.integers(900, 1100, size=(48, 3, 256)).astype(np.uint64)
        frames[:16, 0, anchor] *= np.uint64(1000)
        frames[40:] = np.iinfo(np.uint64).max
        runs.append(write_run_file(tmp_path / f"run_{index}.h5", [521, 614], frames))
        with h5py.File(runs[-1], "a") as handle:
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = (1 - index) * 48 * 8192
    result = build_run_scores(bundle_dir=plans["bundle"], runs=runs, channel=35, output_dir=tmp_path / "scores",
                              event_seconds=8 * FRAME_SECONDS, jobs=2)
    assert result["frames"] == 80 and result["events"] == 10
    frames = threshold_rule.load_frames(35, tmp_path / "scores")
    assert frames.current_word == word
    assert np.all(np.diff(frames.time) > 0)
    kept = threshold_rule.keep(frames, word)
    assert kept.sum() == 48 and not kept[:16].any() and not kept[40:56].any()
    assert frames.excess[:16].min() > 10 * abs(frames.excess[16:40]).max()
    sweep = threshold_rule.sweep(frames)
    assert sweep["kappa"][-1] == pytest.approx(1.0)
    assert sweep["x"][-1] > sweep["x"][0]
    with pytest.raises(ValueError, match="already exists"):
        build_run_scores(bundle_dir=plans["bundle"], runs=runs, channel=35, output_dir=tmp_path / "scores")


def test_run_scores_read_the_frame_timing_from_the_file(plans, tmp_path):
    frames = np.random.default_rng(3).integers(900, 1100, size=(4, 3, 256)).astype(np.uint64)
    path = write_run_file(tmp_path / "run_chime.h5", [521, 614], frames)
    with h5py.File(path, "a") as handle:
        handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = 0
        handle["dtv_fine_powers_n0"].attrs["seq_length_nsec"] = 2560
        handle["dtv_fine_powers_n0"].attrs["dim_scalings"] = [16384, 1, 1, 1]
    build_run_scores(bundle_dir=plans["bundle"], runs=[path], channel=35, output_dir=tmp_path / "scores")
    time = threshold_rule.load_frames(35, tmp_path / "scores").time
    assert np.diff(time) == pytest.approx(16384 * 2560e-9)


def _v3_run(plans, tmp_path, **words):
    """Write a bundle with channel 35 as a v3 profile at ``words``, and a run whose v3 keep words are 1, 2 and 3.

    Each keep word, in units of 65536, holds 16 frames in recorded order. The run tools read only the profiles.
    """
    bundle = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    row["fine_calibration"] = dict(status="calibrated", decision_version="fine_decision_v3", **words)
    directory = tmp_path / "bundle_v3"
    directory.mkdir()
    (directory / "pilot_profiles.json").write_text(json.dumps(bundle))
    frames = np.ones((48, 3, 256), dtype=np.uint64)
    frames[16:32, 0, 140] = 2
    frames[32:, 0, 140] = 3
    run = write_run_file(tmp_path / "run_v3.h5", [521, 614], frames)
    with h5py.File(run, "a") as handle:
        handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = 0
    return directory, run


@pytest.mark.parametrize("words, applied, kept", [
    # the stage applies the power-limit word where calibration wrote one
    (dict(cfar_multiplier_q16=65536, power_limit_multiplier_q16=2 * 65536), 2 * 65536, [16, 16, 0]),
    # and the default word otherwise
    (dict(cfar_multiplier_q16=65536), 65536, [16, 0, 0]),
])
def test_v3_spectrogram_keeps_the_frames_the_stage_keeps(plans, tmp_path, words, applied, kept):
    bundle, run = _v3_run(plans, tmp_path, **words)
    result = run_spectrogram(bundle_dir=bundle, runs=[run], channel=35, chunk_seconds=16 * FRAME_SECONDS)
    assert result["word_q16"] == applied and result["kept"].tolist() == kept
    summary = list(csv.DictReader(write_masked_summary([result], tmp_path / "out").open()))
    assert summary[0]["word_q16"] == str(applied) and summary[0]["masked_frames"] == str(48 - sum(kept))


@pytest.mark.parametrize("words, applied, kept", [
    (dict(cfar_multiplier_q16=65536, power_limit_multiplier_q16=2 * 65536), 2 * 65536, 32),
    (dict(cfar_multiplier_q16=65536), 65536, 16),
])
def test_v3_run_scores_keep_the_frames_the_stage_keeps(plans, tmp_path, words, applied, kept):
    bundle, run = _v3_run(plans, tmp_path, **words)
    result = build_run_scores(bundle_dir=bundle, runs=[run], channel=35, output_dir=tmp_path / "scores")
    assert result["current_threshold"] == {"kind": "finite", "eta_q16": str(applied)}
    assert result["masked_percent"] == round(100.0 * (48 - kept) / 48, 3)
    frames = threshold_rule.load_frames(35, tmp_path / "scores")
    assert frames.current_word == applied and int(frames.current_keep.sum()) == kept


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
    # the calibration default unless a decision version is named, as for a Kotekan stage without v2
    assert "decision_version" not in result["calibrate"]
    old = write_run_plans(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], output_dir=tmp_path / "v1",
                          epoch="synthetic", receiver_profile=Path(plans["calibrate"]["receiver_profile"]),
                          decision_version="fine_decision_v1")
    assert json.loads((tmp_path / "v1" / "calibrate.json").read_text())["decision_version"] == "fine_decision_v1"
    assert old["calibrate"]["channels"] == result["calibrate"]["channels"]
    # a half with no bound channel drops coarse_freq, and half the files split a run by day
    with h5py.File(tmp_path / "run_idle.h5", "w") as handle:
        handle["dtv_fine_powers"] = frames[:, None].repeat(2, axis=1)
        handle["dtv_fine_powers"].attrs["fpga_seq_num"] = 0
    half = write_run_plans(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], output_dir=tmp_path / "half",
                           epoch="synthetic", receiver_profile=Path(plans["calibrate"]["receiver_profile"]),
                           calibration_share=0.5)
    assert len(half["calibrate"]["channels"]["37"]) == len(half["test"]["channels"]["37"]) == 3
    # the night option goes into the default_word block of the calibration plan only
    assert "default_word" not in result["calibrate"]
    night = write_run_plans(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], output_dir=tmp_path / "night",
                            epoch="synthetic", receiver_profile=Path(plans["calibrate"]["receiver_profile"]),
                            night_site=(49.3208, -119.6236))
    assert night["calibrate"]["default_word"] == dict(night=DRAO) and "default_word" not in night["test"]
    with pytest.raises(ValueError, match="between 0 and 1"):
        write_run_plans(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], output_dir=tmp_path / "bad",
                        epoch="synthetic", calibration_share=1.0)


def test_process_run_summarizes_a_run(plans, tmp_path):
    captures = plans["calibrate"]["control"]["captures"]
    plan = dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    profiles = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    anchor = next(p for p in profiles if p["physical_channel"] == 35)["fine_calibration"]["anchor_bin"]
    rng = np.random.default_rng(3)
    for index in range(8):
        data = rng.gamma(44, 1e6 / 44, size=(32, 2, 3, 256))
        data[:, 0, 0, anchor] *= 30  # channel 35's pilot in every frame
        with h5py.File(tmp_path / f"run_{index}.h5", "w") as handle:
            handle["dtv_fine_powers_n0"] = data.astype(np.uint64)
            handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = [521, 491]
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 32 * 8192
    out = tmp_path / "processed"
    summary = process_run(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], epoch="synthetic",
                          output_dir=out, calibration_share=0.5,
                          receiver_profile=Path(plans["calibrate"]["receiver_profile"]), log=lambda line: None)
    rows = {row["channel"]: row for row in summary["channels"]}
    assert rows[35]["word_source"].startswith("floor") and rows[35]["pilot_share"] > 0.9
    assert rows[35]["calibrate_masked_percent"] > 90 and rows[35]["test_masked_percent"] > 90
    assert rows[37]["test_masked_percent"] < 10
    assert json.loads((out / "summary.json").read_text()) == summary
    assert "| 35 |" in (out / "summary.md").read_text()
    assert (out / "spectrograms" / "test" / "masked_summary.csv").exists()
    with pytest.raises(ValueError, match="not empty"):
        process_run(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], epoch="synthetic", output_dir=out)


@pytest.mark.parametrize("decision, floor", [(None, 1.7), ("fine_decision_v1", 1.84)])
def test_first_bundle_needs_no_captures(plans, tmp_path, decision, floor):
    plan = dict(plans["calibrate"], channels={"29": [], "37": []}, control=None, num_inputs=32, null_quantile=0.99)
    if decision:
        plan["decision_version"] = decision
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    # a new bundle takes version 2 unless the plan names another
    version = decision or "fine_decision_v2"
    assert report["decision_version"] == version
    assert report["floor_eta"] > floor
    assert report["channels"]["37"]["word_source"] == "floor, no captures"
    rows = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    assert {row["fine_calibration"]["decision_version"] for row in rows} == {version}


def test_v3_first_bundle_scales_the_simulated_control_word(plans, tmp_path):
    plan = dict(plans["calibrate"], channels={"29": [], "35": [], "37": []}, control=None, num_inputs=32,
                null_quantile=0.99, decision_version="fine_decision_v3")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    # the control word is the quantile of white noise through ideal weights, whose null level is 1
    q = float(np.quantile(simulated_null_scores(32, decision_version="fine_decision_v3"), 0.99))
    assert report["control_quantile_q16"] == q and report["reference_level"] == 1.0
    assert report["floor_word_q16"] == report["measured_floor_q16"] == int(np.ceil(q))
    assert 1.8 < report["floor_eta"] < 1.95
    assert report["floor_source"] == "white noise, 32 inputs" and report["control"] is None
    # one independent null score in each simulated frame
    assert report["null_scores"] == 2000
    rows = {r["physical_channel"]: r["fine_calibration"]
            for r in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    for channel in (29, 35, 37):
        r = report["channels"][str(channel)]
        assert r["word_source"] == "control word x scale" and r["frames"] == 0
        assert r["scale"] == r["mu0"] != 1.0 and r["passband_factor_source"] == "not given"
        assert r["word_q16"] == scaled_word(q, r["mu0"]) == rows[channel]["cfar_multiplier_q16"]
        assert rows[channel]["decision_version"] == "fine_decision_v3"
        assert rows[channel]["provenance"]["summed_inputs"] == 32


def test_unknown_decision_version_is_refused(plans, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], decision_version="fine_decision_v9")))
    with pytest.raises(ValueError, match="decision_version"):
        calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")


def test_control_channel_must_be_listed_under_channels(plans, tmp_path):
    channels = {c: p for c, p in plans["calibrate"]["channels"].items() if c != "37"}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels=channels)))
    with pytest.raises(ValueError, match="must also be listed under channels"):
        calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_control_channel_is_bound(plans):
    report = plans["report"]
    rows = json.loads((plans["bundle"] / "pilot_profiles.json").read_text())["profiles"]
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
    few = np.quantile(simulated_null_scores(32, frames=200, decision_version="fine_decision_v1"), 0.99) / 65536
    many = np.quantile(simulated_null_scores(256, frames=200, decision_version="fine_decision_v1"), 0.99) / 65536
    assert few > 1.84
    assert many < few


def test_median_reference_floor_lies_below_the_bin_ratio_floor():
    # the reference pooled over the background is less noisy than the reference of each bin
    floors = {v: [np.quantile(simulated_null_scores(n, frames=200, decision_version=v), 0.99) / 65536 for n in (32, 256)]
              for v in ("fine_decision_v1", "fine_decision_v2")}
    assert floors["fine_decision_v2"][1] < floors["fine_decision_v2"][0]
    assert all(a < b for a, b in zip(floors["fine_decision_v2"], floors["fine_decision_v1"]))
    assert all(f > 1 for f in floors["fine_decision_v2"])


def test_nominal_bin_matches_pathfinder_control():
    assert nominal_fine_bin(608309441.0, 3115 * 195312.5, CELL, False) == 214


def test_binomial_tail():
    assert binomial_upper_tail(0, 10, 0.5) == 1.0
    assert binomial_upper_tail(10, 10, 0.5) == pytest.approx(2.0 ** -10)
    assert binomial_lower_tail(0, 10, 0.5) == pytest.approx(2.0 ** -10)
    assert binomial_upper_tail(1000, 5000, 0.2) == pytest.approx(0.5, abs=0.02)


def limit_plan(plans, tmp_path, limit, **given):
    """Write a power-limit plan on the captures of the knee plan and return its path."""
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"35": limit, "29": limit}))
    table = ROOT / "src/pilot_proxy/characterization/data/tolerance_table.json"
    path = tmp_path / "limit_plan.json"
    # the curvature and the margin are required, and zero here
    given = dict(dict(curvature={c: [0.0, 0.0] for c in plans["calibrate"]["channels"]}, margin=0.0), **given)
    path.write_text(json.dumps(dict(plans["calibrate"], threshold="power_limit",
                                    power_limit=dict(table=str(table), limits=str(limits), min_captures=2, **given))))
    return path


def test_knee_is_the_default_threshold(plans):
    assert plans["report"]["threshold"] == "knee"
    assert "power_limit" not in plans["report"] and "power_limit" not in plans["report"]["channels"]["35"]
    assert not plans["bundle"].with_name("bundle.decisions.json").exists()


def test_power_limit_threshold_keeps_the_most_frames_under_the_limit(plans, tmp_path):
    report = calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-3), output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    tested = channel["power_limit"]
    assert report["threshold"] == "power_limit" and report["power_limit"]["no_passing_word"] == [29]
    assert tested["outcome"] == "useful" and channel["word_source"] == "power limit"
    assert channel["word_q16"] == tested["word"]
    # half of the captures hold the tone, and no word that keeps one of them meets the limit
    assert tested["kappa"] == pytest.approx(0.5) and tested["ratio"] <= 1.0
    assert tested["coupling"] == 1.0 and tested["coupling_source"] == "worst case"
    assert tested["corrections"]["control_frames"] == 26
    profiles = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    fine = {p["physical_channel"]: p["fine_calibration"] for p in profiles}
    assert fine[35]["cfar_multiplier_q16"] == tested["word"]
    # a channel whose pilot the run never showed takes the floor and no pass
    quiet = report["channels"]["29"]
    assert quiet["power_limit"]["outcome"] == "not useful" and quiet["power_limit"]["reason"] == NO_PILOT_REASON
    assert "word" not in quiet["power_limit"]
    assert quiet["word_q16"] == report["floor_word_q16"]


def test_power_limit_decisions_are_read_by_the_operating_policy(plans, tmp_path):
    from pilot_proxy.operating_policy import _decisions, _threshold_kind

    calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-3), output_dir=tmp_path / "bundle")
    rows = _decisions(tmp_path / "bundle.decisions.json")
    assert sorted(rows) == [29, 35]
    # a run that never showed the pilot is masked whole: it is not the threshold rule's "DTV without a matching pilot"
    assert _threshold_kind(rows[35]) == "finite" and _threshold_kind(rows[29]) == "reject_all"
    report = json.loads((tmp_path / "bundle.report.json").read_text())
    assert rows[35]["deployed_word"] == report["channels"]["35"]["word_q16"]


def test_power_limit_channel_with_no_passing_word_takes_the_floor(plans, tmp_path):
    from pilot_proxy.operating_policy import _decisions, _threshold_kind

    report = calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-12), output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    assert channel["power_limit"]["outcome"] == "not useful" and channel["power_limit"]["least_ratio"] > 1.0
    assert channel["word_source"] == "floor, power limit: no supported threshold meets tolerance"
    assert channel["word_q16"] == report["floor_word_q16"]
    assert report["power_limit"]["no_passing_word"] == [29, 35]
    assert _threshold_kind(_decisions(tmp_path / "bundle.decisions.json")[35]) == "reject_all"


def test_power_limit_coupling_scales_the_power(plans, tmp_path):
    worst = calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-12), output_dir=tmp_path / "worst")
    (tmp_path / "coupled").mkdir()
    coupled = calibrate_bundle(plan_path=limit_plan(plans, tmp_path / "coupled", 1e-12, coupling={"35": 1e-6}),
                               output_dir=tmp_path / "bundle")
    a, b = worst["channels"]["35"]["power_limit"], coupled["channels"]["35"]["power_limit"]
    assert b["coupling_source"] == "plan"
    assert b["least_ratio"] == pytest.approx(1e-6 * a["least_ratio"])


def test_power_limit_plan_errors(plans, tmp_path):
    base = json.loads(limit_plan(plans, tmp_path, 1e-3).read_text())
    path = tmp_path / "bad.json"
    for change, message in ((dict(threshold="best"), "threshold must be"), (dict(power_limit={}), "needs"),
                            (dict(words={"35": 70000}), "takes no words"),
                            (dict(control=None, num_inputs=16), "needs a control"),
                            (dict(power_limit=dict(base["power_limit"], coupling={"35": 2.0})), "at most 1"),
                            (dict(power_limit={k: v for k, v in base["power_limit"].items() if k != "margin"}),
                             r"needs \['margin'\]"),
                            (dict(power_limit={k: v for k, v in base["power_limit"].items() if k != "curvature"}),
                             r"needs \['curvature'\]"),
                            (dict(power_limit=dict(base["power_limit"], curvature={"35": [0.0, 0.0]})),
                             r"no entry for channels \[29\]")):
        path.write_text(json.dumps(dict(base, **change)))
        with pytest.raises(ValueError, match=message):
            calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
        assert not (tmp_path / "bundle").exists()


def test_ratio_excess_reads_the_pilot_over_the_null_level():
    fines = np.full((3, 3, 256), 50, dtype=np.uint64)
    fines[:, 0] = 100                # F = 2 P_0 / (P_1 + P_2) = 2, the null level
    fines[1, 0, 7] += 25600          # a pilot that doubles the frame power
    fines[2, 1:] = 0                 # no reference
    excess = ratio_excess(fines, (1, 1))     # mu_0 = 2 |w_t|^2 / (|w_lo|^2 + |w_up|^2) = 2
    assert excess[0] == pytest.approx(0.0) and excess[1] == pytest.approx(1.0) and np.isnan(excess[2])


def test_ratio_excess_is_the_archive_pilot_excess(monkeypatch):
    import pilot_proxy.power_limit_word as power_limit_word
    from pilot_proxy.product_contract import CurrentProductContractError, pilot_excess_from_fine_powers

    fines = np.random.default_rng(5).integers(0, 2**40, (50, 3, 256), dtype=np.uint64)
    fines[9, 1:] = 0
    product = dict(fine_power_u64=fines, target_norm_sq=np.asarray([3]), reference_norm_sum_sq=np.asarray([5]))
    monkeypatch.setattr(power_limit_word, "EXCESS_CHUNK", 7)
    np.testing.assert_array_equal(ratio_excess(fines, (3, 5)), pilot_excess_from_fine_powers(product))
    # float powers would be truncated by the exact sums, so they are refused
    with pytest.raises(CurrentProductContractError, match="uint64"):
        ratio_excess(fines.astype(np.float64), (3, 5))


def test_control_corrections_equal_the_direct_sums():
    rng = np.random.default_rng(11)
    n = 600
    words = rng.integers(100, 160, n)
    excess = rng.normal(1e-3, 0.03, n) - 1e-4 * (words - 130)
    unit = rng.integers(0, 9, n) * 5
    at = np.r_[50, np.unique(words), 500]
    got = control_corrections(words, excess, unit, at)
    mean = excess.mean()
    sums = np.bincount(unit, excess - mean)
    assert got["null"] == pytest.approx(mean) and got["null_se"] == pytest.approx(np.sqrt(sums @ sums) / n)
    for i, word in enumerate(at):
        keep = words <= word
        if not keep.any():
            # no control frame kept, so no measured shift
            assert got["shift"][i] == 0.0 and got["shift_se"][i] == np.inf
            continue
        if keep.all():
            assert got["shift"][i] == 0.0 and got["shift_se"][i] == 0.0
            continue
        kept = excess[keep].mean()
        influence = np.where(keep, (excess - kept) / keep.sum(), 0.0) - (excess - mean) / n
        cluster = np.bincount(unit, influence)
        assert got["shift"][i] == pytest.approx(kept - mean, rel=1e-9, abs=1e-15)
        assert got["shift_se"][i] == pytest.approx(np.sqrt(cluster @ cluster), rel=1e-7)


def test_limit_test_takes_the_largest_passing_word():
    sweep = dict(word=np.array([10, 20, 30, 40], dtype=np.uint64), kappa=np.array([0.2, 0.5, 0.8, 1.0]),
                 mean_excess=np.array([0.0, 1.0, 2.0, 30.0]), se=np.zeros(4), kept_frames=np.array([2, 5, 8, 10]),
                 kept_events=np.array([1, 3, 3, 4]))
    common = dict(shelf=0.01, power_scale=4.0, power_limit=4.0 * 0.021**2)
    result = limit_test(sweep, min_captures=3, **common)
    assert (result["word"], result["kappa"], result["passing_words"]) == (30, 0.8, 2)
    assert result["ratio"] == pytest.approx((0.02 / 0.021) ** 2)
    # the offset and the shift come off the estimate, and the margin goes onto it
    shifted = limit_test(sweep, min_captures=3, offset=0.005, shift=np.full(4, -0.005), margin=0.001, **common)
    assert shifted["upper_shelf"] == pytest.approx(0.021) and shifted["word"] == 30
    assert limit_test(sweep, min_captures=5, **common)["reason"] == "no word keeps 5 captures"
    assert limit_test(sweep, min_captures=3, coupling=1e-3, **common)["word"] == 40
    # a word with no measured shift cannot pass
    unmeasured = limit_test(sweep, min_captures=3, margin=np.array([0.0, 0.0, np.inf, 0.0]), **common)
    assert (unmeasured["word"], unmeasured["passing_words"]) == (20, 1)
    with pytest.raises(ValueError, match="at most 1"):
        limit_test(sweep, coupling=1.5, **common)


def _power_limit_run(plans, tmp_path, decision, min_captures=2):
    """Process a run with channel 35's pilot in every other file at the power-limit threshold.

    Return the summary, the bundle report and the arguments common to process_run.
    """
    captures = plans["calibrate"]["control"]["captures"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    profiles = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]
    anchor = next(p for p in profiles if p["physical_channel"] == 35)["fine_calibration"]["anchor_bin"]
    rng = np.random.default_rng(5)
    for index in range(8):
        data = rng.gamma(44, 1e6 / 44, size=(32, 2, 3, 256))
        data[:, 0, 0, anchor] *= 1 + 299 * (index % 2)  # channel 35's pilot in every other file
        with h5py.File(tmp_path / f"run_{index}.h5", "w") as handle:
            handle["dtv_fine_powers_n0"] = data.astype(np.uint64)
            handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = [521, 491]
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 32 * 8192
    (tmp_path / "limits.json").write_text(json.dumps({"35": 1e-5, "37": 1e-5}))
    table = ROOT / "src/pilot_proxy/characterization/data/tolerance_table.json"
    (tmp_path / "inputs.json").write_text(json.dumps(dict(
        table=str(table), limits="limits.json", min_captures=min_captures,
        curvature={"35": [0.0, 0.0], "37": [0.0, 0.0]}, margin=0.0)))
    common = dict(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], epoch="synthetic",
                  calibration_share=0.5, receiver_profile=Path(plans["calibrate"]["receiver_profile"]), log=lambda line: None)
    summary = process_run(output_dir=tmp_path / "out", threshold="power_limit", power_limit=tmp_path / "inputs.json",
                          decision_version=decision, **common)
    return summary, json.loads((tmp_path / "out" / "bundle.report.json").read_text()), common


def _summary_md_eta(tmp_path, channel):
    """Return the Eta cell of a channel's row in summary.md."""
    lines = (tmp_path / "out" / "summary.md").read_text().splitlines()
    return next(line for line in lines if line.startswith(f"| {channel} |")).split("|")[3].strip()


@pytest.mark.parametrize("decision", [None, "fine_decision_v3"])
def test_process_run_takes_the_power_limit_threshold(plans, tmp_path, decision):
    summary, report, common = _power_limit_run(plans, tmp_path, decision)
    rows = {row["channel"]: row for row in summary["channels"]}
    assert summary["threshold"] == "power_limit" and rows[35]["word_source"] == "power limit"
    # the word keeps the files with no pilot and masks the others
    assert 40 < rows[35]["calibrate_masked_percent"] < 60
    assert report["channels"]["37"]["power_limit"]["corrections"] == "none, the control itself"
    # the summary and the spectrograms name the word the stage applies; v3 keeps it beside its default word
    channel = report["channels"]["35"]
    applied = channel["power_limit_word_q16"] if decision else channel["word_q16"]
    assert decision is None or applied != channel["word_q16"]
    masked = next(csv.DictReader((tmp_path / "out" / "spectrograms" / "calibrate" / "masked_summary.csv").open()))
    assert rows[35]["word_q16"] == applied and masked["channel"] == "35" and masked["word_q16"] == str(applied)
    assert rows[35]["eta"] == round(applied / 65536, 4) and _summary_md_eta(tmp_path, 35) == str(rows[35]["eta"])
    assert "Threshold: power limit" in (tmp_path / "out" / "summary.md").read_text()
    assert (tmp_path / "out" / "bundle.decisions.json").exists()
    with pytest.raises(ValueError, match="needs a JSON file"):
        process_run(output_dir=tmp_path / "none", threshold="power_limit", **common)


def test_v3_process_run_names_the_reject_word_where_no_power_limit_word_passes(plans, tmp_path):
    # no word keeps 100 captures, so the channel carries the reject word, which the stage applies to mask it whole
    summary, report, _ = _power_limit_run(plans, tmp_path, "fine_decision_v3", min_captures=100)
    channel = report["channels"]["35"]
    assert channel["power_limit"]["outcome"] == "not useful" and channel["power_limit_word_q16"] == REJECT_WORD_Q16
    row = next(r for r in summary["channels"] if r["channel"] == 35)
    assert row["word_q16"] == REJECT_WORD_Q16 and row["word_source"] == channel["power_limit_word_source"]
    assert row["eta"] == round(REJECT_WORD_Q16 / 65536, 4) and _summary_md_eta(tmp_path, 35) == str(row["eta"])
    assert row["calibrate_masked_percent"] == 100.0 and row["test_masked_percent"] == 100.0
    masked = next(csv.DictReader((tmp_path / "out" / "spectrograms" / "calibrate" / "masked_summary.csv").open()))
    assert masked["channel"] == "35" and masked["word_q16"] == str(REJECT_WORD_Q16)
    assert "the bundle holds the reject word on them" in (tmp_path / "out" / "summary.md").read_text()


@pytest.fixture(scope="module")
def v3_plans(plans):
    path = plans["tmp"] / "calibrate_v3.json"
    path.write_text(json.dumps(dict(plans["calibrate"], decision_version="fine_decision_v3")))
    bundle = plans["tmp"] / "bundle_v3"
    return dict(bundle=bundle, report=calibrate_bundle(plan_path=path, output_dir=bundle))


def test_v3_bundle_names_no_reference_bins(v3_plans):
    assert v3_plans["report"]["decision_version"] == "fine_decision_v3"
    rows = json.loads((v3_plans["bundle"] / "pilot_profiles.json").read_text())["profiles"]
    for row in rows:
        fine = row["fine_calibration"]
        assert fine["decision_version"] == "fine_decision_v3"
        assert not {"bulk_mask_words_hex", "cfar_rank", "anchor_bin", "designated_half_width"} & set(fine)
        assert fine["cfar_multiplier_q16"] >= 1
    # the report keeps the geometry the words were scored on
    assert v3_plans["report"]["control"]["cfar_rank"] == 63
    assert v3_plans["report"]["control"]["bulk_bins"] == list(range(0, 256, 2))


def test_v3_bundle_passes_on_other_captures(plans, v3_plans):
    result = run_bundle_test(bundle_dir=v3_plans["bundle"], plan_path=plans["test_path"])
    assert result["passed"], result


def _qualify_as_tested(bundle, test_path, tmp_path):
    """Test a bundle, qualify it on the passing report, check the copy holds the rows that were tested, and
    return the summary and the qualified rows."""
    report = tmp_path / "test_report.json"
    result = run_bundle_test(bundle_dir=bundle, plan_path=test_path, output_json=report)
    assert result["passed"], result
    qualified = tmp_path / "qualified"
    summary = qualify_bundle(bundle_dir=bundle, test_report=report, reviewer="A. Reviewer", output_dir=qualified)
    assert summary["qualified_channels"] == [29, 35]
    assert not validate_runtime_weight_bundle(bundle_dir=qualified).get("errors")
    for name in ("detector_contract.json", "weights.bin", "weights.manifest.json"):
        assert (qualified / name).read_bytes() == (bundle / name).read_bytes()
    before = {p["physical_channel"]: p for p in json.loads((bundle / "pilot_profiles.json").read_text())["profiles"]}
    after = {p["physical_channel"]: p for p in json.loads((qualified / "pilot_profiles.json").read_text())["profiles"]}
    assert sorted(after) == sorted(before)
    for channel, row in after.items():
        unqualified = json.loads(json.dumps(row))
        record = unqualified["fine_calibration"]["provenance"].pop("mask_qualification", None)
        expected = dict(test_report_sha256=summary["test_report_sha256"], reviewer="A. Reviewer")
        assert record == (None if channel == 37 else expected)
        # the words and the rest of the row are those that were tested
        assert unqualified == before[channel]
    return summary, after


def test_v3_bundle_is_qualified_at_its_tested_words(plans, tmp_path):
    # the control is bound, as in the bundle of a run, and qualification leaves its row as it was
    channels = dict(plans["calibrate"]["channels"], **{"37": plans["calibrate"]["control"]["captures"]})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels=channels, decision_version="fine_decision_v3")))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    _, rows = _qualify_as_tested(tmp_path / "bundle", plans["test_path"], tmp_path)
    assert sorted(rows) == [29, 35, 37] and rows[37]["fine_calibration"]["status"] == "control"
    assert {row["fine_calibration"]["decision_version"] for row in rows.values()} == {"fine_decision_v3"}
    assert not any("power_limit_multiplier_q16" in row["fine_calibration"] for row in rows.values())


def test_v3_power_limit_bundle_is_qualified_with_its_power_limit_word(plans, tmp_path):
    path = limit_plan(plans, tmp_path, 1e-3)
    # 36 frames of channel 35, so its pilot is seen, as in test_v3_power_limit_word_is_the_power_limit_word
    mixed = [write_capture(tmp_path / f"c35_{i}.h5", 35, 6, 10 + i, tone=0.8 * (i % 2)) for i in range(6)]
    plan = json.loads(path.read_text())
    plan["channels"] = dict(plan["channels"], **{"35": mixed})
    path.write_text(json.dumps(dict(plan, decision_version="fine_decision_v3")))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    word = report["channels"]["35"]["power_limit_word_q16"]
    assert word not in (None, REJECT_WORD_Q16) and report["channels"]["29"]["power_limit_word_q16"] == REJECT_WORD_Q16
    _, rows = _qualify_as_tested(tmp_path / "bundle", plans["test_path"], tmp_path)
    # the stage applies the power-limit word of channel 35 and the reject word of channel 29, the words tested
    fine = {channel: row["fine_calibration"] for channel, row in rows.items()}
    assert fine[35]["power_limit_multiplier_q16"] == v3_word(fine[35]) == word != fine[35]["cfar_multiplier_q16"]
    assert fine[29]["power_limit_multiplier_q16"] == v3_word(fine[29]) == REJECT_WORD_Q16


def test_v3_power_limit_word_is_the_power_limit_word(plans, tmp_path):
    path = limit_plan(plans, tmp_path, 1e-3)
    # v3 searches all 256 bins, whose noise peak on these few-input captures lies above that of the 65-bin
    # search, so 12 frames are too few for the pilot to pass the seen test; 36 are enough
    mixed = [write_capture(tmp_path / f"c35_{i}.h5", 35, 6, 10 + i, tone=0.8 * (i % 2)) for i in range(6)]
    plan = json.loads(path.read_text())
    plan["channels"] = dict(plan["channels"], **{"35": mixed})
    path.write_text(json.dumps(dict(plan, decision_version="fine_decision_v3")))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    tested = channel["power_limit"]
    assert tested["outcome"] == "useful"
    assert channel["power_limit_word_q16"] == tested["word"] and channel["power_limit_word_source"] == "power limit"
    # the default word is the scaled control word, as without a power limit
    assert channel["word_source"] == "control word x scale"
    fine = {p["physical_channel"]: p["fine_calibration"]
            for p in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    assert fine[35]["power_limit_multiplier_q16"] == tested["word"]
    assert fine[35]["cfar_multiplier_q16"] == channel["word_q16"]
    # test-bundle tests the word the stage applies, against the interval calibration gave at that word
    assert tested["word"] != channel["word_q16"] and "applied_masked_interval" in channel
    result = run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=plans["test_path"])
    check = next(c for c in result["checks"] if c["check"] == "masked fraction on channel 35")
    assert "at the power-limit word" in check["detail"]
    low, high = channel["applied_masked_interval"]
    assert f"calibration gave {low:.3f} to {high:.3f}" in check["detail"]
    # a report from before this interval existed cannot test the power-limit word
    report_path = tmp_path / "bundle.report.json"
    stale = json.loads(report_path.read_text())
    del stale["channels"]["35"]["applied_masked_interval"]
    report_path.write_text(json.dumps(stale))
    check = next(c for c in run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=plans["test_path"])["checks"]
                 if c["check"] == "masked fraction on channel 35")
    assert not check["passed"] and "recalibrate" in check["detail"]
    # a channel with no passing word carries the reject word, so every consumer masks it whole
    assert fine[29]["power_limit_multiplier_q16"] == REJECT_WORD_Q16


def test_bundle_test_of_a_channel_calibrated_without_captures_fails_cleanly(plans, tmp_path):
    # channel 29 calibrated with no captures has a word and no masked_interval; its test captures fail the check
    plan = dict(plans["calibrate"], channels=dict(plans["calibrate"]["channels"], **{"29": []}))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    result = run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=plans["test_path"])
    check = next(c for c in result["checks"] if c["check"] == "masked fraction on channel 29")
    assert not check["passed"] and "no captures" in check["detail"] and not result["passed"]


def test_v3_power_limit_estimate_is_the_ratio_at_each_null_level(plans, tmp_path):
    from pilot_proxy.bundle_calibration import _shelf
    from pilot_proxy.detector_contract import null_power_ratio_from_weight_norms

    path = limit_plan(plans, tmp_path, 1e-3)
    plan = dict(json.loads(path.read_text()), decision_version="fine_decision_v3")
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    weights = bytes.fromhex(report["control"]["weights_hex"])
    control = null_power_ratio_from_weight_norms(*weight_norms(weights))
    # the control's null is its mean ratio at its own null level, which tells the two levels apart
    bundle = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())
    reverse = bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"])
    row = dict(physical_channel=report["control"]["channel"], receiver_channel_id=report["control"]["receiver_channel_id"],
               weight_bank_offset_bytes=0, weight_bank_nbytes=len(weights))
    fines, _ = _load_fines(plan["control"]["captures"], "chime_baseband", row, weights, 128, reverse, None)
    null = _shelf(plan, path.parent) * ratio_excess(fines, weight_norms(weights)).mean()
    for channel in ("35", "29"):
        tested = report["channels"][channel]["power_limit"]
        assert tested["estimate"] == "ratio, F / mu_0 - 1"
        assert tested["mu0"] == report["channels"][channel]["mu0"] != control
        assert tested["corrections"]["control_mu0"] == control
        assert tested["corrections"]["control_frames"] == len(fines)
        assert tested["corrections"]["null"] == pytest.approx(null, rel=1e-12)


def test_power_limit_corrections_take_only_decided_control_frames(plans, tmp_path, monkeypatch):
    import pilot_proxy.bundle_calibration as calibration

    words = calibration.required_words

    def first_masked_always(*args, **kwargs):
        out = words(*args, **kwargs)
        out[0] = calibration.MAX_SCORE
        return out

    monkeypatch.setattr(calibration, "required_words", first_masked_always)
    report = calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-3), output_dir=tmp_path / "bundle")
    corrections = report["channels"]["35"]["power_limit"]["corrections"]
    assert corrections["control_frames"] == report["control"]["frames"] - 1


def test_v3_channel_with_no_passing_word_carries_the_reject_word(plans, tmp_path):
    from pilot_proxy.operating_policy import _decisions, _threshold_kind

    path = limit_plan(plans, tmp_path, 1e-12)
    # 36 frames, so the pilot is seen (as in the test above) and the channel fails on its power limit
    mixed = [write_capture(tmp_path / f"c35_{i}.h5", 35, 6, 10 + i, tone=0.8 * (i % 2)) for i in range(6)]
    plan = json.loads(path.read_text())
    plan["channels"] = dict(plan["channels"], **{"35": mixed})
    path.write_text(json.dumps(dict(plan, decision_version="fine_decision_v3")))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    assert channel["power_limit"]["outcome"] == "not useful"
    assert channel["power_limit"]["reason"] == "no supported threshold meets tolerance"
    assert channel["power_limit_word_q16"] == REJECT_WORD_Q16
    assert channel["power_limit_word_source"] == "reject, power limit: no supported threshold meets tolerance"
    # DtvRfiMask and the relay apply the power-limit word, so they mask the channel whole, as the policy does
    fine = {p["physical_channel"]: p["fine_calibration"]
            for p in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    assert v3_word(fine[35]) == REJECT_WORD_Q16
    assert fine[35]["cfar_multiplier_q16"] == channel["word_q16"]
    assert _threshold_kind(_decisions(tmp_path / "bundle.decisions.json")[35]) == "reject_all"
    # every frame with target power is rejected at that word, so the test of the applied word passes on new data
    bundle = json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())
    row = next(p for p in bundle["profiles"] if p["physical_channel"] == 35)
    fines, _ = _load_fines(plans["test"]["channels"]["35"], "chime_baseband", row,
                           (tmp_path / "bundle" / "weights.bin").read_bytes(), 128,
                           bool(bundle["input_preprocessing"]["time_reverse_detector_windows_before_kernel"]), None)
    stats = np.array([[s.n_peak, s.d_rho] for s in map(fine_stat_v3, fines)], dtype=np.uint64)
    assert masks_from_stat(stats, v3_word(fine[35])).all()
    assert channel["applied_masked_interval"] == [1.0, 1.0]
    result = run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=plans["test_path"])
    check = next(c for c in result["checks"] if c["check"] == "masked fraction on channel 35")
    assert check["passed"] and "masked 1.000 at the power-limit word" in check["detail"]


def _v3_plan(plans, tmp_path, **extra):
    plan = dict(plans["calibrate"], decision_version="fine_decision_v3", **extra)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def test_v3_pilot_seen_gate_is_the_default_word(plans, tmp_path):
    # channel 29 holds noise only; at half its null level every noise frame passes its default word, which the
    # gate reads, while the unscaled control word would leave it unseen
    report = calibrate_bundle(plan_path=_v3_plan(plans, tmp_path, default_word=dict(passband_factor={"29": 0.5})),
                              output_dir=tmp_path / "bundle")
    assert plans["report"]["channels"]["29"]["anchor_source"] == "nominal"
    assert report["channels"]["29"]["anchor_source"] == "measured"


def test_v3_no_capture_channel_in_a_power_limit_plan_is_masked_whole(plans, tmp_path):
    from pilot_proxy.operating_policy import _decisions, _threshold_kind

    path = limit_plan(plans, tmp_path, 1e-3)
    plan = json.loads(path.read_text())
    plan["channels"] = dict(plan["channels"], **{"29": []})
    path.write_text(json.dumps(dict(plan, decision_version="fine_decision_v3")))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    channel = report["channels"]["29"]
    assert channel["power_limit"]["reason"] == "no captures" and channel["power_limit_word_q16"] == REJECT_WORD_Q16
    assert _threshold_kind(_decisions(tmp_path / "bundle.decisions.json")[29]) == "reject_all"
    fine = {p["physical_channel"]: p["fine_calibration"]
            for p in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    assert fine[29]["power_limit_multiplier_q16"] == REJECT_WORD_Q16


def test_v3_default_word_is_the_scaled_control_word(plans, tmp_path):
    factors = {"35": 1.01, "29": 0.99, "37": 1.002}
    report = calibrate_bundle(plan_path=_v3_plan(plans, tmp_path, default_word=dict(passband_factor=factors)),
                              output_dir=tmp_path / "bundle")
    q = report["control_quantile_q16"]
    assert report["reference_level"] > 0 and report["measured_floor_q16"] == int(np.ceil(q))
    rows = {r["physical_channel"]: r["fine_calibration"]
            for r in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    for channel in ("35", "29"):
        r = report["channels"][channel]
        assert r["word_source"] == "control word x scale" and "knee_word_q16" not in r
        assert r["passband_factor"] == factors[channel] and r["passband_factor_source"] == "plan"
        assert r["scale"] == pytest.approx(r["mu0"] * factors[channel] / report["reference_level"], rel=1e-15)
        assert r["word_q16"] == scaled_word(q, r["scale"]) == rows[int(channel)]["cfar_multiplier_q16"]


def test_v3_control_channel_takes_the_control_word_itself(plans, tmp_path):
    captures = plans["calibrate"]["control"]["captures"]
    path = _v3_plan(plans, tmp_path, channels={**plans["calibrate"]["channels"], "37": captures})
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    r = report["channels"]["37"]
    assert r["scale"] == 1.0 and r["passband_factor_source"] == "not given"
    assert r["word_q16"] == int(np.ceil(report["control_quantile_q16"]))


def test_v3_default_word_needs_no_captures_of_the_channel(plans, tmp_path):
    path = _v3_plan(plans, tmp_path, channels={**plans["calibrate"]["channels"], "29": []})
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    r = report["channels"]["29"]
    assert r["frames"] == 0 and r["word_source"] == "control word x scale"
    assert r["word_q16"] == scaled_word(report["control_quantile_q16"], r["scale"])


@pytest.mark.parametrize("block, match", [(dict(passband_factor={"29": 0.0}), "positive"),
                                          (dict(offsets={"29": 1.0}), "unknown default_word"),
                                          (dict(night=dict(latitude_deg=49.3)), "night takes"),
                                          (dict(night=dict(latitude_deg=95.0, longitude_deg_east=0.0)), "latitude"),
                                          (dict(night=dict(latitude_deg=49.3, longitude_deg_east=400.0)), "longitude")])
def test_v3_default_word_plan_errors(plans, tmp_path, block, match):
    with pytest.raises(ValueError, match=match):
        calibrate_bundle(plan_path=_v3_plan(plans, tmp_path, default_word=block), output_dir=tmp_path / "bundle")


DRAO = dict(latitude_deg=49.3208, longitude_deg_east=-119.6236)
NIGHT_CTIME = 1782028800.0      # 2026-06-21 08:00 UTC, near local midnight at DRAO
DAY_CTIME = 1782072000.0        # 2026-06-21 20:00 UTC, near local noon


def test_sun_elevation_at_drao():
    from pilot_proxy.sun import elevation_deg, is_night

    site = (DRAO["latitude_deg"], DRAO["longitude_deg_east"])
    # 90 - 49.32 +/- 23.44 at the June solstice: about 64.1 at noon, -17.2 at midnight
    assert elevation_deg(DAY_CTIME - 6 * 60, *site) == pytest.approx(64.1, abs=0.2)
    assert elevation_deg(NIGHT_CTIME - 6 * 60, *site) == pytest.approx(-17.2, abs=0.2)
    assert list(is_night([NIGHT_CTIME, DAY_CTIME], *site)) == [True, False]


def _night_calibration(plans, tmp_path, name, captures, **extra):
    plan = dict(plans["calibrate"], decision_version="fine_decision_v3", control=dict(channel=37, captures=captures),
                **extra)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(plan))
    return calibrate_bundle(plan_path=path, output_dir=tmp_path / name)


def test_v3_night_default_word_takes_the_night_control(plans, tmp_path):
    night = write_capture(tmp_path / "n37.h5", 37, 26, 3, ctime=NIGHT_CTIME)
    # in the day a tone stands for the Sun, which raises the scores of the control
    day = write_capture(tmp_path / "d37.h5", 37, 10, 7, tone=0.8, ctime=DAY_CTIME)
    only_night = _night_calibration(plans, tmp_path, "only_night", [night])
    with_night = _night_calibration(plans, tmp_path, "with_night", [night, day], default_word=dict(night=DRAO))
    all_frames = _night_calibration(plans, tmp_path, "all_frames", [night, day])
    assert with_night["control_quantile_q16"] == only_night["control_quantile_q16"]
    assert all_frames["control_quantile_q16"] > with_night["control_quantile_q16"]
    assert with_night["control"]["night"] == dict(DRAO, frames=only_night["control"]["frames"])
    # the corrections of the power-limit test still read every control frame
    assert with_night["control"]["frames"] == all_frames["control"]["frames"]
    assert with_night["floor_source"] == "control channel, night frames"
    assert "night" not in all_frames["control"]


def test_v3_night_needs_frame_times_and_a_night_frame(plans, tmp_path):
    # the control of the fixture records no time0_ctime
    with pytest.raises(ValueError, match="time0_ctime"):
        calibrate_bundle(plan_path=_v3_plan(plans, tmp_path, default_word=dict(night=DRAO)), output_dir=tmp_path / "b")
    day = write_capture(tmp_path / "d37.h5", 37, 26, 3, ctime=DAY_CTIME)
    with pytest.raises(ValueError, match="below the horizon"):
        _night_calibration(plans, tmp_path, "day_only", [day], default_word=dict(night=DRAO))


def test_frame_times_of_run_windows_and_refused_formats(tmp_path):
    from pilot_proxy.bundle_calibration import _frame_times

    frames = np.ones((3, 3, 256), dtype=np.uint64)
    path = write_run_file(tmp_path / "run.h5", [491], frames)
    with h5py.File(path, "a") as handle:
        handle["dtv_fine_powers_n0"].attrs["fpga_seq_time_nsec"] = int(NIGHT_CTIME * 1e9)
    assert list(_frame_times([path], "dtv_fine_powers", BLOCK, np.array([3]))) == [NIGHT_CTIME] * 3
    with pytest.raises(ValueError, match="no frame time"):
        _frame_times([path], "chord_baseband", BLOCK, np.array([3]))


def test_bundle_test_counts_the_night_control(plans, tmp_path):
    night = write_capture(tmp_path / "n37.h5", 37, 26, 3, ctime=NIGHT_CTIME)
    day = write_capture(tmp_path / "d37.h5", 37, 10, 7, tone=0.8, ctime=DAY_CTIME)
    _night_calibration(plans, tmp_path, "bundle", [night, day], default_word=dict(night=DRAO))
    test = dict(plans["test"], control=[write_capture(tmp_path / "tn37.h5", 37, 12, 6, ctime=NIGHT_CTIME),
                                        write_capture(tmp_path / "td37.h5", 37, 6, 8, tone=0.8, ctime=DAY_CTIME)])
    path = tmp_path / "test.json"
    path.write_text(json.dumps(test))
    result = run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=path)
    check = next(c for c in result["checks"] if c["check"] == "false alarms on the control channel")
    assert "of 12 night frames" in check["detail"]
    # a test control recorded only in the day cannot test a night calibration
    path.write_text(json.dumps(dict(test, control=[test["control"][1]])))
    check = next(c for c in run_bundle_test(bundle_dir=tmp_path / "bundle", plan_path=path)["checks"]
                 if c["check"] == "false alarms on the control channel")
    assert not check["passed"] and "no frame recorded with the Sun below the horizon" in check["detail"]


def test_weight_norms_are_the_squared_norms_of_the_terms():
    # target (1, 0) and references (1, 1) and (0, 1) in every tap: |w_t|^2 = k, |w_lo|^2 + |w_up|^2 = 2 k + k
    k = 4
    taps = np.array([[0x10] * k, [0x11] * k, [0x01] * k], dtype=np.uint8)
    assert weight_norms(taps.tobytes()) == (k, 3 * k)


def test_control_words_at_maps_a_channel_word_onto_the_control():
    words = np.array([10, 100, 131072], dtype=np.int64)
    assert control_words_at(words, 1.0).tolist() == [10, 100, 131072]
    assert control_words_at(words, 1.25).tolist() == [8, 80, 104857]
    assert control_words_at(words, 0.98).tolist() == [10, 102, 133746]


def test_v3_power_limit_reads_the_control_at_the_scaled_word(plans, tmp_path, monkeypatch):
    import pilot_proxy.bundle_calibration as calibration
    from pilot_proxy.characterization import threshold_rule

    swept, read = [], []
    sweep, corrections = threshold_rule.sweep, calibration.control_corrections
    monkeypatch.setattr(threshold_rule, "sweep", lambda *a, **k: swept.append(sweep(*a, **k)) or swept[-1])
    monkeypatch.setattr(calibration, "control_corrections", lambda w, e, c, at: read.append(at) or corrections(w, e, c, at))
    path = limit_plan(plans, tmp_path, 1e-3)
    plan = dict(json.loads(path.read_text()), decision_version="fine_decision_v3",
                default_word=dict(passband_factor={"35": 1.03, "29": 0.97, "37": 1.0}))
    path.write_text(json.dumps(plan))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    scale = report["channels"]["35"]["scale"]
    tested = report["channels"]["35"]["power_limit"]
    assert tested["corrections"]["control_word_scale"] == scale != 1.0
    words = next(s["word"] for s, at in zip(swept, read) if len(s["word"]) == len(at)
                 and np.array_equal(at, control_words_at(s["word"].astype(np.int64), scale)))
    assert len(words) > 0


def test_v3_process_run_summary_reports_the_control_word(plans, tmp_path):
    captures = plans["calibrate"]["control"]["captures"]
    plan = dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    rng = np.random.default_rng(3)
    for index in range(8):
        data = rng.gamma(44, 1e6 / 44, size=(32, 2, 3, 256))
        data[:, 0, 0, 140] *= 30  # channel 35's pilot in every frame
        with h5py.File(tmp_path / f"run_{index}.h5", "w") as handle:
            handle["dtv_fine_powers_n0"] = data.astype(np.uint64)
            handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = [521, 491]
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 32 * 8192
    out = tmp_path / "processed"
    summary = process_run(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], epoch="synthetic",
                          output_dir=out, calibration_share=0.5, decision_version="fine_decision_v3",
                          receiver_profile=Path(plans["calibrate"]["receiver_profile"]), log=lambda line: None)
    rows = {row["channel"]: row for row in summary["channels"]}
    assert rows[35]["word_source"] == "control word x scale" and rows[35]["scale"] > 0
    text = (out / "summary.md").read_text()
    assert "Control word eta" in text and "| Scale |" in text and "knee or the floor" not in text


def _calibration_length_tool():
    spec = importlib.util.spec_from_file_location("calibration_length", ROOT / "tools" / "calibration_length.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_length_gives_the_v3_words_and_no_floor(plans, tmp_path, monkeypatch, capsys):
    tool = _calibration_length_tool()
    captures = plans["calibrate"]["control"]["captures"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    # 64 captures that record no time, so 43 s apart: 34 start in the first 0.4 h, and 17 in the first 0.2 h.
    # 8 frames each, so 17 captures hold the 100 control frames of a 1% word
    rng = np.random.default_rng(3)
    (tmp_path / "run").mkdir()
    for index in range(64):
        data = rng.gamma(44, 1e6 / 44, size=(8, 2, 3, 256))
        data[:, 0, 0, 140] *= 1 + 299 * (index % 2)  # channel 35's pilot in every other capture
        with h5py.File(tmp_path / "run" / f"run_{index:02d}.h5", "w") as handle:
            handle["dtv_fine_powers_n0"] = data.astype(np.uint64)
            handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = [521, 491]
            handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 8 * 8192
    (tmp_path / "limits.json").write_text(json.dumps({"35": 1e-3}))
    table = ROOT / "src/pilot_proxy/characterization/data/tolerance_table.json"
    limit = dict(table=str(table), limits="limits.json", curvature={"35": [0.0, 0.0]}, margin=0.0)
    (tmp_path / "limit.json").write_text(json.dumps(limit))
    (tmp_path / "limit10.json").write_text(json.dumps(dict(limit, min_captures=10)))

    def study(name, *extra, mode=("--hours", "0.2", "0.4"), output="length.json"):
        monkeypatch.setattr(sys, "argv", ["calibration_length.py", "--bundle-dir", str(tmp_path / "bundle"), "--run",
                                          str(tmp_path / "run" / "run_*.h5"), *mode,
                                          "--output-dir", str(tmp_path / name), *extra])
        tool.main()
        return json.loads((tmp_path / name / output).read_text()), capsys.readouterr().out

    # v1 keeps its floor, anchors and knee gate
    rows, out = study("v1", "--decision-version", "fine_decision_v1")
    assert [(r["hours"], r["captures"], r["test_captures"]) for r in rows] == [(0.4, 34, 30)]
    assert "0.2 h: 17 captures to calibrate and 47 to test, each needs 30; skipped" in out
    assert f"floor {rows[0]['floor_eta']};" in out and all("anchor" in c for c in rows[0]["channels"].values())

    # v3 gives the default word and the power-limit word, with no floor or anchor
    rows, out = study("v3", "--decision-version", "fine_decision_v3", "--power-limit", str(tmp_path / "limit.json"))
    report = json.loads((tmp_path / "v3" / "0.4h" / "bundle.report.json").read_text())["channels"]
    assert [r["hours"] for r in rows] == [0.4] and "floor_eta" not in rows[0] and "floor" not in out
    assert "each needs 30 (a power-limit word keeps at least 30); skipped" in out
    channels = rows[0]["channels"]
    assert not any("anchor" in c for c in channels.values())
    assert channels["35"]["eta"] == round(report["35"]["word_q16"] / 65536, 4)
    assert channels["35"]["source"] == "control word x scale"
    assert channels["35"]["power_limit_eta"] == round(report["35"]["power_limit_word_q16"] / 65536, 4)
    assert f"power-limit word {channels['35']['power_limit_eta']}" in out
    assert "power_limit_eta" not in channels["37"]

    # the gate is the power-limit plan's min_captures, under v1 as well
    rows, out = study("v3-10", "--decision-version", "fine_decision_v3", "--power-limit", str(tmp_path / "limit10.json"))
    assert [r["hours"] for r in rows] == [0.2, 0.4] and "skipped" not in out
    assert tool.fewest_captures(dict(decision_version="fine_decision_v1", power_limit=dict(min_captures=10))) == (
        10, "a power-limit word keeps at least 10")

    # blocks under v3 have no floor, and with no power limit the tool sets 30 captures for the masked fractions
    blocks = ("--block-hours", "0.2")
    rows, out = study("v3-blocks", "--decision-version", "fine_decision_v3", "--power-limit",
                      str(tmp_path / "limit10.json"), mode=blocks, output="blocks.json")
    assert [r["captures"] for r in rows] == [17, 17, 17] and "floor" not in out
    assert not any("floor_eta" in r for r in rows) and all("power_limit_eta" in r["channels"]["35"] for r in rows)
    with pytest.raises(SystemExit, match="fewer than 30 captures, and the tool sets 30 for the masked fractions"):
        study("v3-blocks-30", "--decision-version", "fine_decision_v3", mode=blocks, output="blocks.json")

    # a channel with no passing word carries the reject word; a report from before the reject word has none
    reject = tool.summary(dict(decision_version="fine_decision_v3", channels={"29": dict(
        eta=1.5, word_source="control word x scale", masked_fraction=0.01, power_limit_word_q16=1,
        power_limit_word_source="reject, power limit: pilot not seen in the calibration captures")}))
    assert tool.describe(reject) == "ch29 1.5 (control word x scale, 1.0%, reject word, masked whole)"
    none = tool.summary(dict(decision_version="fine_decision_v3", channels={"29": dict(
        eta=1.5, word_source="control word x scale", masked_fraction=0.01, power_limit_word_q16=None,
        power_limit_word_source="none, power limit: pilot not seen")}))
    assert tool.describe(none) == "ch29 1.5 (control word x scale, 1.0%, no power-limit word)"


def test_calibration_length_takes_the_night_control_word(plans, tmp_path, monkeypatch, capsys):
    from pilot_proxy.sun import is_night

    tool = _calibration_length_tool()
    captures = plans["calibrate"]["control"]["captures"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], channels={"35": plans["calibrate"]["channels"]["35"], "37": captures})))
    calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    minutes = DAY_CTIME + 60 * np.arange(12 * 60)
    sunset = minutes[np.argmax(is_night(minutes, DRAO["latitude_deg"], DRAO["longitude_deg_east"]))]
    site = (str(DRAO["latitude_deg"]), str(DRAO["longitude_deg_east"]))

    def write_run(run, before_sunset_s):
        """Write 64 captures starting before_sunset_s before sunset at DRAO, and return which are in the day."""
        times = sunset - before_sunset_s + tool.CAPTURE_SECONDS * np.arange(64)
        day = ~is_night(times, DRAO["latitude_deg"], DRAO["longitude_deg_east"])
        rng = np.random.default_rng(3)
        (tmp_path / run).mkdir()
        for index in range(64):
            data = rng.gamma(44, 1e6 / 44, size=(8, 2, 3, 256))
            data[:, 1, 0] *= 1 + 0.5 * day[index]        # the Sun raises the control's target in the day
            with h5py.File(tmp_path / run / f"run_{index:02d}.h5", "w") as handle:
                handle["dtv_fine_powers_n0"] = data.astype(np.uint64)
                handle["dtv_fine_powers_n0"].attrs["coarse_freq"] = [521, 491]
                handle["dtv_fine_powers_n0"].attrs["fpga_seq_num"] = index * 8 * 8192
                handle["dtv_fine_powers_n0"].attrs["fpga_seq_time_nsec"] = int(times[index] * 1e9)
        return day

    def study(name, run, *extra):
        monkeypatch.setattr(sys, "argv", ["calibration_length.py", "--bundle-dir", str(tmp_path / "bundle"), "--run",
                                          str(tmp_path / run / "run_*.h5"), "--decision-version",
                                          "fine_decision_v3", "--output-dir", str(tmp_path / name), *extra])
        tool.main()
        return capsys.readouterr().out

    # from 0.2 h before sunset, the 0.4 h calibration holds about 17 day and 17 night captures
    day = write_run("run", 720)
    study("night", "run", "--hours", "0.4", "--night", *site)
    study("all", "run", "--hours", "0.4")
    night, all_frames = (json.loads((tmp_path / name / "0.4h" / "bundle.report.json").read_text())
                         for name in ("night", "all"))
    assert night["control"]["night"] == dict(DRAO, frames=8 * int((~day[:34]).sum()))
    assert night["control_quantile_q16"] < all_frames["control_quantile_q16"]

    # from 1400 s before sunset, the first 0.38 h is all day: it is skipped, and the night after it calibrates
    day = write_run("dusk", 1400)
    assert day[:32].all()
    out = study("dusk-hours", "dusk", "--hours", "0.38", "--night", *site)
    assert "0.38 h: no night control capture; skipped" in out
    assert json.loads((tmp_path / "dusk-hours" / "length.json").read_text()) == []
    study("dusk-blocks", "dusk", "--block-hours", "0.38", "--night", *site)
    rows = json.loads((tmp_path / "dusk-blocks" / "blocks.json").read_text())
    assert rows[0]["skipped"] == "no night control capture" and "channels" in rows[1]
    report = json.loads((tmp_path / "dusk-blocks" / "block1" / "bundle.report.json").read_text())
    assert report["control"]["night"] == dict(DRAO, frames=8 * int((~day[32:]).sum()))


def test_v3_rows_need_a_receiver_namespace():
    from pilot_proxy.runtime_bundle import _validate_receiver_channel_ids

    def errors_for(version):
        rows = [dict(physical_channel=29, receiver_channel_id_namespace=None,
                     fine_calibration=dict(decision_version=version))]
        errors = []
        _validate_receiver_channel_ids(profiles=rows, receiver_channel_id_namespace=None, errors=errors)
        return [e["message"] for e in errors]

    # Kotekan maps v3 words by receiver_channel_id and refuses such a bundle, so validation refuses it first
    assert any("needs a receiver_channel_id_namespace" in m for m in errors_for("fine_decision_v3"))
    assert errors_for("fine_decision_v2") == []
