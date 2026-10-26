"""Era boundaries at on/off transitions: Bayesian Blocks, the computed split, the excursion merge and the era list of record."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pilot_proxy.characterization import era_boundaries as e
from pilot_proxy.config.project import default_project

RECORD = default_project().directory / "eras" / "on_off_2026-10-14"
needs_record = pytest.mark.skipif(not (RECORD / "monthly_levels.json").is_file(), reason="withheld from the code-only cut (CHIME_DERIVED.md)")
STARTS = {14: "2018-12", 15: "2018-12", 16: "2018-12", 17: "2018-12", 18: "2018-12", 19: "2024-12", 20: "2018-12",
          21: "2018-12", 22: "2018-12", 23: "2018-12", 24: "2018-12", 25: "2018-12", 26: "2023-04", 27: "2022-10",
          28: "2020-06", 29: "2018-12", 30: "2018-12", 31: "2019-06", 32: "2023-02", 33: "2022-10", 34: "2018-12",
          35: "2021-11", 36: "2018-12", 37: "2018-12"}


def same(a, b):
    """Equal up to the last few bits of a float, since numpy's sums differ between AVX-512 and AVX2 runners."""
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-12, abs=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def levels():
    if not (RECORD / "monthly_levels.json").is_file():
        pytest.skip("withheld from the code-only cut (CHIME_DERIVED.md)")
    return {int(ch): rows for ch, rows in json.loads((RECORD / "monthly_levels.json").read_text())["channels"].items()}


@pytest.fixture(scope="module")
def result(levels):
    return e.era_boundaries(levels)


def test_blocks_find_a_step_and_keep_noise_whole():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 0.3, 40), rng.normal(5, 0.3, 30)]
    assert e.bayesian_blocks(x, np.full(70, 0.3)) == [0, 40]
    assert e.bayesian_blocks(rng.normal(0, 0.3, 70), np.full(70, 0.3)) == [0]


def test_blocks_match_astropy():
    astropy = pytest.importorskip("astropy.stats")
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(5, 100))
        t = np.sort(rng.choice(300, n, replace=False)).astype(float)
        x = np.cumsum(rng.normal(0, 10, n) * (rng.random(n) < 0.1)) + rng.normal(0, 1, n)
        s = rng.uniform(0.2, 2, n)
        edges = astropy.bayesian_blocks(t, x, s, fitness="measures", p0=0.05)
        assert e.bayesian_blocks(x, s) == [int(np.searchsorted(t, a)) for a in edges[:-1]]


def test_the_split_is_the_largest_gap_between_sorted_steps():
    assert e.largest_gap([0.1, -0.5, 3.0, 9.0, -30.0, 31.0]) == (9.0, 30.0)


def test_the_split_of_record_is_computed_from_the_steps(levels, result):
    every = [s["step_db"] for r in result["channels"] for s in r["steps"]]
    lower, upper = result["split"]["largest_below_db"], result["split"]["smallest_above_db"]
    assert (lower, upper) == e.largest_gap(every)
    assert 19 < lower < upper < 30
    above = sorted(abs(s) for s in every if abs(s) >= upper)
    assert all(b - a < upper - lower for a, b in zip(above[:-1], above[1:]))


def test_an_excursion_that_returns_is_not_a_boundary():
    blocks = [e.Block(0, 9, 40.0), e.Block(10, 10, 5.0), e.Block(11, 40, 40.5)]
    cut, merged = e.boundaries(blocks, upper=20.0)
    assert cut == [] and merged[0]["first_month"] == e.month_name(10)


def test_a_short_era_that_does_not_return_stays():
    cut, merged = e.boundaries([e.Block(0, 9, 40.0), e.Block(10, 10, 5.0), e.Block(11, 40, 70.0)], upper=20.0)
    assert cut == [1, 2] and merged == []
    cut, _ = e.boundaries([e.Block(0, 9, 40.0), e.Block(10, 30, 5.0), e.Block(31, 40, 40.0)], upper=20.0)
    assert cut == [1, 2]   # longer than a neighbor, so the off era is kept and not merged as an excursion


def test_the_control_has_no_boundary(result):
    control = next(r for r in result["channels"] if r["channel"] == 37)
    assert control["boundaries"] == [] and not any(s["boundary"] for s in control["steps"])


def test_the_outages_on_30_and_35_are_excursions(result):
    rows = {r["channel"]: r for r in result["channels"]}
    assert [x["first_month"] for x in rows[30]["excursions"]] == ["2019-04"]
    assert [x["first_month"] for x in rows[35]["excursions"]] == ["2022-09"]


def test_the_starts_of_record(result):
    assert {r["channel"]: r["current_first_month"] for r in result["channels"]} == STARTS


def test_the_record_reproduces(levels, result):
    blocks = json.loads((RECORD / "blocks.json").read_text())
    eras = json.loads((RECORD / "eras.json").read_text())
    assert same(json.loads(json.dumps(result["channels"])), blocks["channels"])
    assert blocks["split"] == result["split"]
    again = e.era_list(result, eras["provenance"], json.loads((RECORD.parent / "power_eras_2026-10-04.json").read_text()))
    assert same(again, eras)
    assert {r["channel"]: r["current_months"][0] for r in eras["channels"]} == STARTS


def test_monthly_levels_skip_empty_months_and_pool_small_ones():
    month = np.r_[[24000] * 6, [24002] * 2]
    level = np.r_[1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 9.0]
    rows = e.monthly_levels(month, level, [24000, 24001, 24002], [30, 0, 9])
    assert [r["month"] for r in rows] == ["2000-01", "2000-03"]
    assert rows[0]["level_db"] == 3.0 and rows[0]["events"] == 5 and rows[0]["spread_db"] == pytest.approx(1.4826)
    assert rows[1]["spread_db"] is None and rows[1]["frames"] == 9


@needs_record
def test_the_tool_writes_the_era_list(tmp_path, capsys):
    assert e.main(["--levels", str(RECORD / "monthly_levels.json"), "--output", str(tmp_path / "out"),
                   "--previous", str(RECORD.parent / "power_eras_2026-10-04.json")]) == 0
    out = {name: json.loads((tmp_path / "out" / name).read_text()) for name in ("monthly_levels.json", "blocks.json", "eras.json")}
    rec = {name: json.loads((RECORD / name).read_text()) for name in out}
    # eras.json cites the blocks.json written beside it, whose last float bits follow the CPU
    assert out["eras.json"]["provenance"].pop("sha256") == hashlib.sha256((tmp_path / "out" / "blocks.json").read_bytes()).hexdigest()
    rec["eras.json"]["provenance"].pop("sha256")
    for name in out:
        assert same(out[name], rec[name]), name
    assert "ch37: 2018-12 .. 2026-08  boundaries -" in capsys.readouterr().out


# Line-identity option

def _lines(offsets_by_month: dict[int, float], mad: float = 0.0) -> dict[int, dict]:
    return {m: dict(median_bins=v, mad_bins=mad, events=40) for m, v in offsets_by_month.items()}


def _blocks(spec):
    """Block list from [(first, last, level), ...] tuples."""
    return [e.Block(a, b, lvl) for a, b, lvl in spec]


def test_monthly_lines_round_to_the_half_bin_grid_and_pool_small_months():
    month = np.r_[np.full(40, 10), np.full(3, 11)]
    offset = np.r_[np.full(40, -99.0), np.full(3, 5.0)]
    rows = e.monthly_lines(month, offset, 23.84)
    assert set(rows) == {10}
    assert rows[10]["median_bins"] == -4.0 and rows[10]["mad_bins"] == 0.0 and rows[10]["events"] == 40


def test_a_steady_line_that_jumps_more_than_the_half_width_cuts_when_the_level_backs_it():
    blocks = _blocks([(0, 23, 20.0), (24, 47, 12.0)])
    lines = _lines({m: -4.0 for m in range(24)} | {m: 0.0 for m in range(24, 48)})
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert cut == [1] and evidence["edges"][0]["change"] == "jump" and evidence["calibration_db"] == 0.0


def test_drift_one_bin_per_month_never_cuts():
    blocks = _blocks([(0, 23, 20.0), (24, 47, 12.0)])
    lines = _lines({m: -4.0 + 0.5 * max(0, m - 20) for m in range(48)})   # creeps half a bin a month from month 20
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert cut == [] and evidence["edges"][0]["change"] in ("drift", "same")


def test_a_jump_needs_a_level_step_above_the_same_line_steps():
    blocks = _blocks([(0, 11, 20.0), (12, 23, 25.0), (24, 35, 20.0), (36, 47, 21.0)])   # the same line steps 5 dB twice
    lines = _lines({m: -4.0 for m in range(36)} | {m: 0.0 for m in range(36, 48)})
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert evidence["calibration_db"] == 5.0 and cut == []


def test_edges_of_a_reversing_block_never_cut_and_short_ones_do_not_calibrate():
    blocks = _blocks([(0, 23, 30.0), (24, 25, 12.0), (26, 47, 31.0), (48, 71, 36.0)])   # a two-month dip, then a new line
    lines = _lines({m: -1.5 for m in range(48)} | {m: 1.5 for m in range(48, 72)})
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert evidence["reversing_blocks"] == [dict(first_month="0002-01", last_month="0002-02", resolved=False)]
    assert evidence["calibration_db"] == 0.0 and cut == [3]


def test_a_steady_line_becoming_unsteady_cuts():
    blocks = _blocks([(0, 23, 20.0), (24, 47, 24.0)])
    lines = _lines({m: -1.0 for m in range(24)}, mad=3.0) | _lines({m: -1.0 for m in range(24, 48)}, mad=0.0)
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert cut == [1] and evidence["edges"][0]["change"] == "steadiness"


def test_a_two_state_line_hands_over_without_a_boundary():
    lines = _lines({m: (0.0 if (m // 6) % 2 == 0 else 3.0) for m in range(48)})
    assert e.two_state_line(lines, 2)
    blocks = _blocks([(0, 5, 20.0), (6, 11, 21.2), (12, 17, 20.0), (18, 47, 21.1)])
    cut, evidence = e.line_boundaries(blocks, lines, upper=29.0, h=2)
    assert cut == [] and all(x["change"] == "jump" for x in evidence["edges"])
    assert not e.two_state_line(_lines({m: float(m // 6) for m in range(48)}), 2)   # a staircase is not two-state


def test_era_list_names_the_line_basis():
    result = dict(channels=[dict(channel=20, current_first_month="2022-09", current_last_month="2026-06",
                                 boundaries=["2022-09"], line=dict(boundaries=["2022-09"]))])
    rows = e.era_list(result, dict(path="x", sha256="y"))["channels"]
    assert rows[0]["era_basis"] == "line_identity_transition" and rows[0]["current_months"][0] == "2022-09"


def test_the_marker_half_width_is_the_detector_s():
    from pilot_proxy.detectors.narrowband_marker.anchors import DESIGNATED_HALF_WIDTH
    from pilot_proxy.products.reader import FINE_BIN_HZ
    assert e.MARKER_HALF_WIDTH_FINE_BINS == DESIGNATED_HALF_WIDTH
    assert e.FINE_BIN_HZ == FINE_BIN_HZ
    assert e.MARKER_HALF_WIDTH_HZ == DESIGNATED_HALF_WIDTH * FINE_BIN_HZ
    assert abs(e.MARKER_HALF_WIDTH_HZ - 23.84) < 0.01
