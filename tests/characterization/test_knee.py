import json
import os
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import knee as wt

# The 24 knees of record: exact Q16 word and current era span (the knee_threshold of the
# threshold-calibration-bridge score cache, the input of the threshold rule).
KNEES = {
    14: (188812, "2019-10", "2026-08"),
    15: (229615, "2021-07", "2026-08"),
    16: (189501, "2018-12", "2026-08"),
    17: (9359224, "2022-10", "2026-08"),
    18: (746143, "2024-08", "2026-08"),
    19: (115160, "2024-12", "2026-04"),
    20: (2044265, "2024-07", "2026-06"),
    21: (96190, "2018-12", "2026-08"),
    22: (18173076, "2018-12", "2026-08"),
    23: (89630, "2018-12", "2026-08"),
    24: (60777923, "2018-12", "2026-04"),
    25: (105644, "2019-10", "2026-08"),
    26: (127624, "2023-04", "2026-08"),
    27: (455516, "2022-11", "2026-08"),
    28: (1486899, "2020-06", "2026-08"),
    29: (92429, "2018-12", "2026-08"),
    30: (34520786, "2018-12", "2023-09"),
    31: (114188364, "2022-10", "2026-04"),
    32: (153316, "2023-02", "2026-08"),
    33: (157656, "2022-10", "2026-08"),
    34: (589103, "2022-09", "2026-07"),
    35: (26516480, "2021-11", "2026-08"),
    36: (2949136, "2018-12", "2026-08"),
    37: (72850, "2018-12", "2026-08"),
}
# Channels 20 and 34 were rescored on their revised era geometry.
RESCORED = {20, 34}
INPUTS = os.environ.get("PILOT_PROXY_WORKING_THRESHOLD_INPUTS")
SCORES = os.environ.get("PILOT_PROXY_SCORE_CACHE")


def _months(first, last):
    start, stop = wt._month_index(first), wt._month_index(last)
    return [f"{index // 12}-{index % 12 + 1:02d}" for index in range(start, stop + 1)]


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "projects" / "chime_atsc" / "eras"
                         / "power_eras_2026-10-04.json").is_file(),
                    reason="withheld from the code-only cut (CHIME_DERIVED.md)")
def test_the_era_list_of_record_gives_the_knee_cohorts():
    era_list = Path(__file__).resolve().parents[2] / "projects" / "chime_atsc" / "eras" / "power_eras_2026-10-04.json"
    cohorts = {row["channel"]: row["current_months"] for row in json.loads(era_list.read_text())["channels"]}
    assert sorted(cohorts) == sorted(KNEES)
    for channel, (_, first, last) in KNEES.items():
        assert cohorts[channel] == _months(first, last), channel


def test_the_word_is_the_knee():
    rng = np.random.default_rng(7)
    words = (np.exp(rng.normal(12.0, 0.4, 400)) // 1).astype(np.uint64)
    events = [f"e{i // 4}" for i in range(len(words))]
    result = wt.propose(words, events)
    assert result["status"] == "knee"
    assert result["threshold_q16"] == result["knee_word_q16"]
    assert "outer_fence_word_q16" not in result and "q25_word_q16" not in result


def test_every_event_carries_equal_weight():
    # One event of 99 low frames and three single-frame high events: the knee keeps the
    # low event's frames, which are most of the frames but one event in four.
    words = [1000] * 99 + [5000, 6000, 7000]
    events = ["burst"] * 99 + ["a", "b", "c"]
    result = wt.propose(words, events)
    assert result["current_events"] == 4
    assert result["knee_word_q16"] == "1000"
    assert result["frame_retention"] == pytest.approx(99 / 102)
    assert result["equal_event_retention"] == pytest.approx(0.25)


def test_sentinel_and_invalid_frames_are_not_scores():
    words = [10, 20, wt.ALWAYS_MASKED, 30]
    events = ["a", "b", "c", "d"]
    result = wt.propose(words, events, valid=[True, True, True, False])
    assert result["always_masked_frames"] == 1
    assert result["invalid_frames"] == 1
    assert result["finite_supported_frames"] == 2
    assert result["kept_frames"] <= 2


def test_zero_score_maps_to_word_one_and_constant_keeps_all():
    result = wt.propose([0, 1, 1], ["a", "b", "c"])
    assert result["zero_score_frames"] == 1
    assert result["threshold_q16"] == "1"
    assert result["kept_frames"] == 3


def test_no_supported_scores_gives_no_word():
    result = wt.propose([wt.ALWAYS_MASKED], ["a"])
    assert result["status"] == "unavailable_no_finite_supported_scores"
    assert result["threshold_q16"] is None


def test_cohort_refuses_to_split_an_event():
    data = {
        "selected": np.array([True, True, True]),
        "finite_time": np.array([True, True, True]),
        "frame_month": np.array([wt._month_index("2024-01")] * 2 + [wt._month_index("2024-02")]),
        "frame_unit_index": np.array([0, 1, 1]),
        "source_event_keys": np.array(["a", "b"]),
    }
    assert wt.cohort_rows(data, ["2024-01", "2024-02"]).tolist() == [0, 1, 2]
    with pytest.raises(wt.CohortSupportError):
        wt.cohort_rows(data, ["2024-01"])


def _channel_result(root, channel, first, last):
    if channel in RESCORED:
        path = root / f"calibration-physical-review-2026-10-04/geometry/exploratory_rescore/ch{channel}_exact_comparison.npz"
        with np.load(path, allow_pickle=False) as z:
            events = z["event_keys"][z["frame_event_inverse"]]
            return wt.propose(z["new_required_q16"], list(map(str, events)),
                              valid=list(z["new_rank_supported"]),
                              always_masked=list(z["new_always_masked"]))
    directory = ("fine-dsp-characterization-2026-10-01/threshold-link/control-anchor45-export"
                 if channel == 37 else "fine-histogram-export-2026-10-01/export/archive")
    with np.load(root / directory / f"ch{channel}.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    return wt.propose_from_export(data, _months(first, last))


@pytest.mark.skipif(not INPUTS, reason="set PILOT_PROXY_WORKING_THRESHOLD_INPUTS to the recorded score exports")
@pytest.mark.parametrize("channel", sorted(KNEES))
def test_recorded_knees_reproduce_from_the_exports(channel):
    word, first, last = KNEES[channel]
    assert int(_channel_result(Path(INPUTS), channel, first, last)["knee_word_q16"]) == word


@pytest.mark.skipif(not SCORES, reason="set PILOT_PROXY_SCORE_CACHE to the score cache of record")
@pytest.mark.parametrize("channel", sorted(KNEES))
def test_recorded_knees_reproduce_from_the_score_cache(channel):
    meta = json.loads((Path(SCORES) / f"ch{channel}.json").read_text())
    with np.load(Path(SCORES) / f"ch{channel}.npz", allow_pickle=False) as z:
        result = wt.propose(z["required_q16"], [str(int(e)) for e in z["frame_event_index"]],
                            valid=list(z["rank_supported"] & z["score_computed"]),
                            always_masked=list(z["always_masked"]))
    assert int(result["knee_word_q16"]) == int(meta["knee_threshold"]["eta_q16"]) == KNEES[channel][0]
    assert meta["current_months"][0] == KNEES[channel][1] and meta["current_months"][-1] == KNEES[channel][2]
