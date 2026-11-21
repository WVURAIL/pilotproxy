"""Recorded fine decision v3 statistics: their decisions, and the Pathfinder tools that read them."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from pilot_proxy.fine_decision import MISSING_ROW, MULTIPLIER_ONE, fine_keep_v3, masks_from_stat
from pilot_proxy.run_files import v3_words

ROOT = Path(__file__).resolve().parents[2]
V3, V1 = 2531, 2869
BLOCKS = 16


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_masks_equal_the_exact_comparison_near_the_boundary():
    rng = np.random.default_rng(5)
    word = 98304
    d = rng.integers(1, 2**40, size=4000, dtype=np.uint64)
    # n on either side of the keep boundary n 2^16 = word d, where floats cannot tell
    boundary = (d.astype(object) * word) // MULTIPLIER_ONE
    n = np.array([int(b) + int(o) for b, o in zip(boundary, rng.integers(-2, 3, size=d.size))], dtype=np.uint64)
    got = masks_from_stat(np.stack([n, d], axis=1), word)
    want = [1 - fine_keep_v3(int(a), int(b), word) for a, b in zip(n, d)]
    assert got.tolist() == want


def test_unevaluated_missing_and_undecided_rows():
    rows = np.array([[0, MISSING_ROW], [MISSING_ROW, MISSING_ROW], [7, 0], [1, 1], [2, 1]], dtype=np.uint64)
    # kept, missing, no decision (rejected), at the boundary (kept), above it (rejected)
    assert masks_from_stat(rows, MULTIPLIER_ONE).tolist() == [0, -1, 1, 0, 1]
    with pytest.raises(ValueError):
        masks_from_stat(rows[:, :1], MULTIPLIER_ONE)


def _bundle(tmp_path):
    def row(freq, channel, status, version, **words):
        return dict(receiver_channel_id=freq, physical_channel=channel,
                    fine_calibration=dict(status=status, decision_version=version, anchor_bin=141, **words))
    bundle = dict(profiles=[
        row(V3, 18, "calibrated", "fine_decision_v3", cfar_multiplier_q16=65536, power_limit_multiplier_q16=2 * 65536),
        row(2715, 24, "calibrated", "fine_decision_v3", cfar_multiplier_q16=3 * 65536),
        row(V1, 29, "calibrated", "fine_decision_v1", cfar_multiplier_q16=65536),
        row(3115, 37, "control", "fine_decision_v3", cfar_multiplier_q16=65536),
    ])
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "pilot_profiles.json").write_text(json.dumps(bundle))
    return tmp_path / "bundle"


def test_v3_words_are_the_words_n2_applies(tmp_path):
    bundle = _bundle(tmp_path)
    # the power-limit word where there is one, the default word otherwise, the control only with mask_control
    assert v3_words(bundle) == {V3: 2 * 65536, 2715: 3 * 65536}
    assert v3_words(bundle, mask_control=True) == {V3: 2 * 65536, 2715: 3 * 65536, 3115: 65536}
    assert v3_words(None) == {}


def _write(path, product, rows, freqs, seq=0):
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset(f"dtv_{product}", data=rows)
        data.attrs.update(fpga_seq_num=seq, fpga_seq_time_nsec=1_800_000_000 * 10**9 + seq * 5120,
                          coarse_freq=np.array(freqs))


@pytest.fixture
def run(tmp_path):
    dtv = tmp_path / "dtv"
    dtv.mkdir()
    freqs = [V3, V1]
    mask = np.zeros((BLOCKS, 2), np.int8)
    mask[:4, 1] = 1                                       # the v1 channel masks 4 blocks itself
    stat = np.zeros((BLOCKS, 2, 2), np.uint64)
    stat[:, :, 1] = 100
    stat[:, 0, 0] = 100 * 1                               # n/d = 1: kept at the power-limit word 2
    stat[:6, 0, 0] = 100 * 3                              # n/d = 3: rejected
    stat[6, 0] = MISSING_ROW                              # a missing block
    stat[:, 1] = [0, MISSING_ROW]                         # the v1 channel has no statistic
    _write(dtv / "dtv_mask.0.h5", "mask", mask, freqs)
    _write(dtv / "dtv_stat.0.h5", "stat", stat, freqs)
    return dict(dtv=dtv, bundle=_bundle(tmp_path), tmp=tmp_path)


def test_compare_tool_decides_v3_channels_from_the_statistic(run):
    tool = _tool("compare_pathfinder_dtv_mask")
    masks = tool.read_masks(str(run["dtv"]), v3_words(run["bundle"]))
    assert masks[V3][1].tolist() == [1] * 6 + [-1] + [0] * (BLOCKS - 7)
    assert masks[V1][1].tolist() == [1] * 4 + [0] * (BLOCKS - 4)
    # without a bundle the v3 channel reads its dtv_mask rows, which hold 0
    assert not tool.read_masks(str(run["dtv"]))[V3][1].any()


def test_check_tool_reports_v3_decisions(run):
    out = run["tmp"] / "check"
    subprocess.run([sys.executable, str(ROOT / "tools" / "check_pathfinder_dtv_run.py"), "--dtv-dir", str(run["dtv"]),
                    "--bundle-dir", str(run["bundle"]), "--channels", str(V3), str(V1), "--output-dir", str(out)],
                   check=True, capture_output=True)
    reports = {r["channel"]: r for r in json.loads((out / "check.json").read_text())}
    assert reports[V3]["missing_blocks"] == 1
    assert reports[V3]["masked_percent"] == round(100 * 6 / (BLOCKS - 1), 3)
    assert reports[V3]["valid_test_percent"] == 100.0
    assert reports[V1]["masked_percent"] == round(100 * 4 / BLOCKS, 3)
