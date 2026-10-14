"""Era boundaries at on/off transitions: Bayesian Blocks, the computed split, the excursion merge and the era list of record."""
from __future__ import annotations

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
    assert cut == [1, 2]   # longer than a neighbour: a real off era, not an excursion


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
    assert blocks["channels"] == json.loads(json.dumps(result["channels"]))
    assert blocks["split"] == result["split"]
    again = e.era_list(result, eras["provenance"], json.loads((RECORD.parent / "power_eras_2026-10-04.json").read_text()))
    assert again == eras
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
    for name in ("monthly_levels.json", "blocks.json", "eras.json"):
        assert (tmp_path / "out" / name).read_bytes() == (RECORD / name).read_bytes()
    assert "ch37: 2018-12 .. 2026-08  boundaries -" in capsys.readouterr().out
