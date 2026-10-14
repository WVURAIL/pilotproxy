"""Adversarial checks for exploratory power eras and their input contract."""
import pytest

from pilot_proxy.characterization.power_eras import MonthlyPower


def month(name="2020-01", level=0.0, **overrides):
    data = dict(month=name, level_db=level, event_count=8, frame_count=40, day_count=4)
    data.update(overrides)
    return MonthlyPower(**data)


@pytest.mark.parametrize("changes", [
    {"month": "2020-1"}, {"month": "2020-00"}, {"month": "2020-13"},
    {"month": "0000-01"}, {"month": 202001},
    {"level_db": float("nan")}, {"level_db": float("inf")},
    {"level_db": True}, {"level_db": "1"},
    {"peak_offset_hz": float("-inf")},
    {"event_count": True}, {"event_count": 5.0}, {"event_count": -1},
    {"frame_count": 7}, {"day_count": -1}, {"tracking_usable": 1},
    {"metadata": {"bad": float("nan")}},
])
def test_invalid_month_values_are_refused(changes):
    with pytest.raises(ValueError):
        month(**changes)


def test_missing_level_is_explicit_and_negative_relative_power_is_valid():
    assert month(level=None).level_db is None
    assert month(level=-3.5).level_db == -3.5
    assert month(day_count=None).day_count is None


def test_unknown_fields_and_missing_counts_are_refused():
    row = dict(month="2020-01", level_db=0, event_count=8, frame_count=40)
    with pytest.raises(ValueError, match="unknown"):
        MonthlyPower.from_dict(dict(row, event_counts=9))
    del row["event_count"]
    with pytest.raises(ValueError, match="missing"):
        MonthlyPower.from_dict(row)


def test_metadata_is_detached_from_callers_dictionary():
    metadata = {"instrument": {"input_map": "map-a"}}
    feature = month(metadata=metadata)
    metadata["instrument"]["input_map"] = "map-b"
    assert feature.metadata["instrument"]["input_map"] == "map-a"

from dataclasses import asdict, replace
import hashlib
import json

from pilot_proxy.characterization.power_eras import (
    INPUT_VERSION,
    PowerEraConfig,
    input_from_dict,
    main,
    propose_power_eras,
)


def series(values, *, start=2020 * 12, **overrides):
    return [month(f"{(start+i)//12:04d}-{(start+i)%12+1:02d}", value, **overrides)
            for i, value in enumerate(values)]


def propose(values, **kwargs):
    return propose_power_eras(series(values), channel=19,
                              provenance={"source": "synthetic", "reduction": "equal-event monthly median"},
                              **kwargs)


@pytest.mark.parametrize("values", [
    [0.0] * 20, [-8.0] * 20, [0.1, -0.1, 0.2, -0.2] * 6,
    [0.0] * 8 + [1000.0] + [0.0] * 8,
    [0.0] * 8 + [-1000.0] * 2 + [0.0] * 8,
    [0.0] * 8 + [1000.0] * 3 + [0.0] * 8,
    [10.0] * 8 + [12.0] * 8,
])
def test_stationary_short_excursions_and_small_step_do_not_define_regimes(values):
    result = propose(values)
    assert result["boundaries"] == []
    assert result["accepted_eras_changed"] is False
    assert result["science_selection_eligible"] is False


@pytest.mark.parametrize("old,new", [(0.0, 6.0), (20.0, 26.0), (-20.0, -26.0), (0.0, 3.0)])
def test_sustained_relative_step_has_one_cluster_and_correct_observed_bounds(old, new):
    result = propose([old] * 8 + [new] * 8)
    assert len(result["boundaries"]) == 1
    boundary = result["boundaries"][0]
    assert boundary["observation_bounds"]["old_last_month"] == "2020-08"
    assert boundary["observation_bounds"]["new_first_month"] == "2020-09"
    assert boundary["proposed_cohort_first_month"] == "2020-09"
    assert len(boundary["passing_edges"]) >= 1
    assert boundary["representative"]["delete_one_min_separation_db"] >= 3
    assert boundary["candidate_edge_interval"]["old_last_month"] <= "2020-08"
    assert boundary["candidate_edge_interval"]["new_first_month"] >= "2020-09"
    assert result["partition_status"] == "not_constructed_candidate_review_required"


def test_just_below_inclusive_effect_boundary_is_not_accepted():
    assert propose([0.0] * 8 + [2.999999999] * 8)["boundaries"] == []


@pytest.mark.parametrize("left_outlier,right_outlier", [(1000.0, -1000.0), (6.0, 0.0)])
def test_opposite_regime_interior_outliers_do_not_reverse_observation_bounds(left_outlier, right_outlier):
    values = [0.0] * 6 + [6.0] * 6
    values[3], values[8] = left_outlier, right_outlier
    boundary = propose(values)["boundaries"][0]
    assert boundary["observation_bounds"]["old_last_month"] == "2020-06"
    assert boundary["observation_bounds"]["new_first_month"] == "2020-07"
    assert boundary["observation_bounds"]["ordered"]


@pytest.mark.parametrize("post_months", [1, 2, 3, 4])
def test_archive_end_with_fewer_than_five_new_months_remains_pending(post_months):
    result = propose([0.0] * 10 + [6.0] * post_months)
    assert result["boundaries"] == []
    assert result["pending_changes"]
    assert any("confirmation" in item["reason"] for item in result["pending_changes"])


def test_five_new_months_at_archive_end_confirm_actual_onset():
    boundary = propose([0.0] * 10 + [6.0] * 5)["boundaries"][0]
    assert boundary["proposed_cohort_first_month"] == "2020-11"


@pytest.mark.parametrize("gap_months", [1, 36])
@pytest.mark.parametrize("new_level", [0.0, 6.0])
def test_gap_is_explicit_and_never_an_ordinary_adjacency(gap_months, new_level):
    rows = series([0.0] * 6) + series([new_level] * 6, start=2020*12 + 6 + gap_months)
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    assert result["gaps"][0]["unsupported_month_count"] == gap_months
    assert result["gaps"][0]["continuity_established"] is False
    if new_level == 0:
        assert result["boundaries"] == []
        assert result["gaps"][0]["status"] == "compatible_observed_levels"
    else:
        assert len(result["boundaries"]) == 1
        assert result["boundaries"][0]["kind"] == "gap_censored_power_change"
    for edge in result["edge_checks"]:
        if edge["kind"] == "observed_power_change":
            left, right = edge["old_last_month"], edge["new_first_month"]
            assert left != rows[5].month or right != rows[6].month


def test_missing_days_are_not_silently_waived_and_gate_can_be_explicitly_disabled():
    rows = series([0.0] * 5 + [6.0] * 5, day_count=None)
    blocked = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    assert blocked["status"] == "no_supported_months"
    assert all("day_support_unverified" in row["support_reasons"] for row in blocked["months"])
    config = PowerEraConfig(min_days=0)
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True}, config=config)
    assert len(result["boundaries"]) == 1
    assert result["config"]["min_days"] == 0


def test_tracking_unusable_and_weak_months_remain_missing_without_compression():
    rows = series([0.0] * 6 + [6.0] * 6)
    rows[5] = replace(rows[5], tracking_usable=False)
    rows[6] = replace(rows[6], event_count=2)
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    assert result["gaps"][0]["unsupported_month_count"] == 2
    assert "tracking_unusable" in result["months"][5]["support_reasons"]
    assert "insufficient_events" in result["months"][6]["support_reasons"]
    assert all(edge["kind"] == "gap_censored_power_change"
               for edge in result["edge_checks"] if edge["status"] == "passing")


def test_frequency_drift_and_metadata_changes_never_trigger_power_boundaries():
    rows = [replace(row, peak_offset_hz=100.0*i, metadata={"instrument": f"map-{i}"})
            for i, row in enumerate(series([10.0] * 20))]
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    assert result["boundaries"] == []


def test_window_exit_can_only_be_relative_feature_candidate_with_identity_caveat():
    rows = [replace(row, peak_offset_hz=200.0*i, metadata={"window_edge_warning": i >= 7})
            for i, row in enumerate(series([10.0] * 8 + [0.0] * 8))]
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    boundary = result["boundaries"][0]
    assert "dominant_feature_identity_unverified" in boundary["warnings"]
    assert any(item["metadata"]["window_edge_warning"] for item in boundary["feature_diagnostics"])
    assert result["feature_interpretation"] == "dominant_relative_spectral_feature"


def test_short_excursion_is_preserved_for_display():
    result = propose([0.0] * 8 + [50.0] * 2 + [0.0] * 8)
    assert result["excursions"]
    assert all(not row["creates_boundary"] for row in result["excursions"])


def test_overlapping_confirmation_windows_are_disclosed():
    result = propose([0.0]*8 + [6.0]*6 + [12.0]*8)
    assert len(result["boundaries"]) == 2
    assert all(row["overlapping_confirmation_windows"] for row in result["boundaries"])


def test_short_transition_preserves_timing_interval():
    result = propose([0.0]*8 + [2.0, 4.0] + [6.0]*8)
    boundary = result["boundaries"][0]
    assert boundary["candidate_edge_interval"]["old_last_month"] <= "2020-09"
    assert boundary["candidate_edge_interval"]["new_first_month"] >= "2020-10"


@pytest.mark.parametrize("values", [[], [None] * 8])
def test_empty_and_all_missing_inputs_do_not_invent_zero(values):
    result = propose(values)
    assert result["status"] == "no_supported_months"
    assert result["boundaries"] == []
    json.dumps(result, allow_nan=False)


def test_canonical_config_and_input_digest_are_deterministic_and_sensitive():
    result = propose([0.0] * 8 + [6.0] * 8)
    repeated = propose([0.0] * 8 + [6.0] * 8)
    assert result == repeated
    config = PowerEraConfig()
    assert config.digest == hashlib.sha256(config.canonical_json().encode()).hexdigest()
    changed = propose([0.0] * 8 + [6.1] * 8)
    assert changed["input_digest"] != result["input_digest"]
    stricter = propose([0.0] * 8 + [6.0] * 8,
                       config=PowerEraConfig(minimum_practical_shift_db=7))
    assert stricter["config_digest"] != result["config_digest"]
    assert stricter["boundaries"] == []


@pytest.mark.parametrize("config", [
    {"block_months": 3}, {"required_consistent_months": 3}, {"min_days": -1},
    {"minimum_practical_shift_db": 0}, {"scale_floor_db": float("nan")},
    {"robust_scale_multiplier": True}, {"version": "invented"},
    {"within_regime_band_max_fraction_of_separation": 0.5},
])
def test_invalid_config_is_refused(config):
    with pytest.raises(ValueError):
        PowerEraConfig(**config)


def test_duplicate_and_unordered_months_are_refused():
    rows = series([0, 0])
    for invalid in ([rows[0], rows[0]], rows[::-1]):
        with pytest.raises(ValueError, match="unique and strictly increasing"):
            propose_power_eras(invalid, channel=19, provenance={"synthetic": True})


def test_nested_nonstring_metadata_keys_are_refused():
    with pytest.raises(ValueError, match="string keys"):
        month(metadata={"nested": {1: "value"}})


def document():
    return {"schema_version": INPUT_VERSION, "channel": 19,
            "provenance": {"source": "synthetic", "reduction": "equal-event"},
            "months": [asdict(row) for row in series([0.0]*6 + [6.0]*6)]}


def test_cli_writes_bound_proposal_and_refuses_overwrite(tmp_path):
    source, output = tmp_path/"input.json", tmp_path/"proposal.json"
    source.write_text(json.dumps(document()))
    assert main(["--input", str(source), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["input_file"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize("raw", [
    '{"schema_version":"monthly-power-features-v1","schema_version":"other"}',
    '{"bad":NaN}', '{"bad":Infinity}', '[]',
])
def test_malformed_json_never_creates_output(tmp_path, raw):
    source, output = tmp_path/"input.json", tmp_path/"proposal.json"
    source.write_text(raw)
    with pytest.raises(SystemExit):
        main(["--input", str(source), "--output", str(output)])
    assert not output.exists()


def test_document_contract_refuses_unknown_version_and_fields():
    for changes in ({"schema_version": "future"}, {"typo": True}, {"months": {}}):
        with pytest.raises(ValueError):
            input_from_dict(dict(document(), **changes))

def test_four_new_months_before_gap_cannot_borrow_later_confirmation():
    rows = series([0.0] * 18 + [6.0] * 4)
    rows += series([6.0] * 8, start=2020 * 12 + 22 + 18)
    result = propose_power_eras(rows, channel=19, provenance={"synthetic": True})
    assert result["boundaries"] == []
    assert result["pending_changes"]
    assert any("fewer_than_five_adjacent_supported_months_from_observed_new" in row["reasons"]
               for row in result["pending_changes"])


def test_finite_input_arithmetic_overflow_is_refused_as_nonfinite_evidence():
    with pytest.raises(ValueError):
        propose([-1e308] * 6 + [1e308] * 6)
