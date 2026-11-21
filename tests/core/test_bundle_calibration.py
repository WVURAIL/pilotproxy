"""Calibrate a bundle from synthetic captures, then test it."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy.power_limit_word import control_corrections, limit_test, median_excess
from pilot_proxy.bundle_calibration import (
    BUNDLE_FILES,
    _bulk,
    _exact_word,
    _expand,
    _load_fines,
    binomial_lower_tail,
    binomial_upper_tail,
    bundle_digest,
    calibrate_bundle,
    capture_weights,
    default_rank,
    iter_run_frames,
    knee_word,
    nominal_fine_bin,
    qualify_bundle,
    required_words,
    simulated_null_scores,
    test_bundle as run_bundle_test,
)
from pilot_proxy.characterization import threshold_rule
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
    path.write_text(json.dumps(dict(plans["calibrate"], channels={"35": always})))
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


def test_unknown_decision_version_is_refused(plans, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(plans["calibrate"], decision_version="fine_decision_v9")))
    with pytest.raises(ValueError, match="decision_version"):
        calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")


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
    assert quiet["power_limit"]["outcome"] == "not useful" and quiet["power_limit"]["reason"] == "pilot not seen"
    assert "word" not in quiet["power_limit"]
    assert quiet["word_q16"] == report["floor_word_q16"]


def test_power_limit_decisions_are_read_by_the_operating_policy(plans, tmp_path):
    from pilot_proxy.operating_policy import _decisions, _threshold_kind

    calibrate_bundle(plan_path=limit_plan(plans, tmp_path, 1e-3), output_dir=tmp_path / "bundle")
    rows = _decisions(tmp_path / "bundle.decisions.json")
    assert sorted(rows) == [29, 35]
    assert _threshold_kind(rows[35]) == "finite" and _threshold_kind(rows[29]) == "pass_all"
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
                            (dict(power_limit=dict(base["power_limit"], coupling={"35": 2.0})), "at most 1")):
        path.write_text(json.dumps(dict(base, **change)))
        with pytest.raises(ValueError, match=message):
            calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
        assert not (tmp_path / "bundle").exists()


def test_median_excess_reads_the_pilot_over_the_background():
    fines = np.full((3, 3, 256), 50, dtype=np.uint64)
    fines[:, 0] = 100
    fines[1, 0, 7] += 25600          # a pilot that doubles the frame power
    fines[2, 1:] = 0                 # no reference
    bulk = _bulk(7)
    excess = median_excess(fines, bulk, default_rank(bulk))
    assert excess[0] == pytest.approx(0.0) and excess[1] == pytest.approx(1.0) and np.isnan(excess[2])


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
        if not keep.any() or keep.all():
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
    with pytest.raises(ValueError, match="at most 1"):
        limit_test(sweep, coupling=1.5, **common)


def test_process_run_takes_the_power_limit_threshold(plans, tmp_path):
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
    (tmp_path / "inputs.json").write_text(json.dumps(dict(table=str(table), limits="limits.json", min_captures=2)))
    common = dict(bundle_dir=tmp_path / "bundle", runs=[str(tmp_path / "run_*.h5")], epoch="synthetic",
                  calibration_share=0.5, receiver_profile=Path(plans["calibrate"]["receiver_profile"]), log=lambda line: None)
    summary = process_run(output_dir=tmp_path / "out", threshold="power_limit", power_limit=tmp_path / "inputs.json", **common)
    rows = {row["channel"]: row for row in summary["channels"]}
    assert summary["threshold"] == "power_limit" and rows[35]["word_source"] == "power limit"
    # the word keeps the files with no pilot and masks the others
    assert 40 < rows[35]["calibrate_masked_percent"] < 60
    report = json.loads((tmp_path / "out" / "bundle.report.json").read_text())
    assert report["channels"]["37"]["power_limit"]["corrections"] == "none, the control itself"
    assert "Threshold: power limit" in (tmp_path / "out" / "summary.md").read_text()
    assert (tmp_path / "out" / "bundle.decisions.json").exists()
    with pytest.raises(ValueError, match="needs a JSON file"):
        process_run(output_dir=tmp_path / "none", threshold="power_limit", **common)


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
        assert "bulk_mask_words_hex" not in fine and "cfar_rank" not in fine
        assert fine["cfar_multiplier_q16"] >= 1
    # the report keeps the geometry the words were scored on
    assert v3_plans["report"]["control"]["cfar_rank"] == 63
    assert v3_plans["report"]["control"]["bulk_bins"] == list(range(0, 256, 2))


def test_v3_bundle_passes_on_other_captures(plans, v3_plans):
    result = run_bundle_test(bundle_dir=v3_plans["bundle"], plan_path=plans["test_path"])
    assert result["passed"], result


def test_v3_power_limit_word_is_the_power_limit_word(plans, tmp_path):
    path = limit_plan(plans, tmp_path, 1e-3)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), decision_version="fine_decision_v3")))
    report = calibrate_bundle(plan_path=path, output_dir=tmp_path / "bundle")
    channel = report["channels"]["35"]
    tested = channel["power_limit"]
    assert tested["outcome"] == "useful"
    assert channel["power_limit_word_q16"] == tested["word"] and channel["power_limit_word_source"] == "power limit"
    # the default word follows the knee rule, as without a power limit
    assert channel["word_source"] != "power limit"
    fine = {p["physical_channel"]: p["fine_calibration"]
            for p in json.loads((tmp_path / "bundle" / "pilot_profiles.json").read_text())["profiles"]}
    assert fine[35]["power_limit_multiplier_q16"] == tested["word"]
    assert fine[35]["cfar_multiplier_q16"] == channel["word_q16"]
    # a channel with no passing word holds the floor as its power-limit word
    assert fine[29]["power_limit_multiplier_q16"] == report["floor_word_q16"]
