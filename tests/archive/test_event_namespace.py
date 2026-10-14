# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.archive import combine as combine_module
from pilot_proxy.archive.chime_coarse import source_event_key
from pilot_proxy.archive.combine import _align_frames, combine_detector_products


def _product(channel: int, freq_id: int, event: str, *, time0: float) -> dict:
    return {
        "physical_channel": np.asarray([channel], dtype=np.int32),
        "freq_id": np.asarray([freq_id], dtype=np.int64),
        "frame_index": np.asarray([0], dtype=np.int64),
        "source_event_keys": np.asarray([event]),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
        "unit_event_id": np.asarray([123], dtype=np.int64),
        "unit_time0_ctime": np.asarray([time0], dtype=np.float64),
        "unit_time0_fpga": np.asarray([456], dtype=np.uint64),
        "unit_delta_time": np.asarray([1.0 / 390_625.0], dtype=np.float64),
        "coarse_power_ratio": np.asarray([[1.0]], dtype=np.float64),
    }


def test_source_event_key_preserves_campaign_namespace() -> None:
    assert source_event_key("/campaign-a/baseband_100_844.h5", 844) == (
        "/campaign-a/baseband_100.h5"
    )
    assert source_event_key("/campaign-b/baseband_100_829.h5", 829) == (
        "/campaign-b/baseband_100.h5"
    )
    assert source_event_key("cadc:ARCHIVE/baseband_100_844.h5", 844) == (
        "cadc:ARCHIVE/baseband_100.h5"
    )


def test_same_basename_in_different_namespaces_does_not_align() -> None:
    first = _product(14, 844, "/campaign-a/baseband_100.h5", time0=10.0)
    second = _product(15, 829, "/campaign-b/baseband_100.h5", time0=10.0)

    with pytest.raises(ValueError, match="share no common"):
        _align_frames([first, second])


def test_path_matched_event_with_different_acquisition_id_is_rejected() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    second["unit_event_id"] = np.asarray([124], dtype=np.int64)

    with pytest.raises(ValueError, match="disagrees on unit_event_id"):
        _align_frames([first, second])


@pytest.mark.parametrize(
    "field, key",
    [
        ("unit_git_version_tag", "n_events_mixed_git_version_tag"),
        ("unit_input_map_sha256", "n_events_mixed_input_map"),
    ],
)
def test_mixed_per_node_provenance_is_recorded_not_refused(field, key) -> None:
    # Each frequency's file comes from its own baseband node; a rolling deploy
    # leaves one burst with two writer versions or input maps across channels.
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    first[field] = np.asarray(["first"], dtype=str)
    second[field] = np.asarray(["second"], dtype=str)

    aligned, _, info = _align_frames([first, second])

    assert len(aligned) == 2
    assert info[key] == 1
    if field == "unit_input_map_sha256":
        assert info["events_mixed_input_map"] == [event]


def test_dispersed_start_times_are_accepted_and_recorded() -> None:
    # One burst reaches lower frequencies later, so each channel's file starts
    # later; 12.458 s is the widest spread seen in the 2026-09 archive run.
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0 + 12.458)
    second["unit_time0_fpga"] = np.asarray(
        [456 + round(12.458 * 390_625)], dtype=np.uint64
    )

    aligned, _, info = _align_frames([first, second])

    assert len(aligned) == 2
    assert info["time0_spread_max_s"] == pytest.approx(12.458, abs=1e-5)
    assert info["n_events_partial_time0"] == 0
    assert info["n_events_partial_sample_period"] == 0


def test_start_time_spread_beyond_dispersion_bound_is_rejected() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0 + 121.0)

    with pytest.raises(ValueError, match="beyond the 120 s dispersion sweep"):
        _align_frames([first, second])


def test_fpga_start_spread_is_scaled_by_the_sample_period() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    second["unit_time0_fpga"] = np.asarray(
        [456 + 121 * 390_625], dtype=np.uint64
    )

    with pytest.raises(ValueError, match="on unit_time0_fpga, beyond"):
        _align_frames([first, second])


def test_missing_sample_period_on_some_pilots_is_counted_not_refused() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    second["unit_delta_time"] = np.asarray([np.nan], dtype=np.float64)

    aligned, _, info = _align_frames([first, second])

    assert len(aligned) == 2
    assert info["n_events_partial_sample_period"] == 1


def test_single_product_without_sample_period_is_accepted() -> None:
    # A full-depth single-channel combine has nothing to align across pilots.
    only = _product(14, 844, "/campaign-a/baseband_100.h5", time0=10.0)
    only["unit_delta_time"] = np.asarray([np.nan], dtype=np.float64)

    aligned, _, info = _align_frames([only])

    assert len(aligned) == 1
    assert info["n_events_partial_sample_period"] == 0


@pytest.mark.parametrize("field", ["unit_scope", "archive_version"])
def test_path_matched_event_with_different_acquisition_identity_is_rejected(field) -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=10.0)
    second = _product(15, 829, event, time0=10.0)
    first[field] = np.asarray(["first"], dtype=str)
    second[field] = np.asarray(["second"], dtype=str)

    with pytest.raises(ValueError, match=rf"disagrees on {field}"):
        _align_frames([first, second])


def test_start_time_shift_below_half_sample_is_accepted() -> None:
    event = "/campaign-a/baseband_100.h5"
    sample_period = 1.0 / 390_625.0
    first = _product(14, 844, event, time0=10.0)
    second = _product(
        15,
        829,
        event,
        time0=10.0 + 0.49 * sample_period,
    )

    aligned, _, _ = _align_frames([first, second])

    assert len(aligned) == 2


def test_identical_start_times_are_accepted_below_timestamp_resolution() -> None:
    event = "/campaign-a/baseband_100.h5"
    first = _product(14, 844, event, time0=1_700_000_000.0)
    second = _product(15, 829, event, time0=1_700_000_000.0)
    first["unit_delta_time"] = np.asarray([1.0e-9], dtype=np.float64)
    second["unit_delta_time"] = np.asarray([1.0e-9], dtype=np.float64)

    aligned, _, _ = _align_frames([first, second])

    assert len(aligned) == 2


def test_failed_combined_build_does_not_publish_partial_set(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    canonical = destination / "chime_detector_outputs.npz"
    canonical.write_bytes(b"previous-complete-set")

    def fail_build(product_paths, run_dir, **kwargs):
        del product_paths, kwargs
        (run_dir / "chime_detector_outputs.npz").write_bytes(b"partial-new-set")
        raise RuntimeError("simulated terminal combine failure")

    monkeypatch.setattr(combine_module, "_combine_detector_products", fail_build)

    with pytest.raises(RuntimeError, match="terminal combine failure"):
        combine_detector_products([], destination)

    assert canonical.read_bytes() == b"previous-complete-set"
    assert not list(tmp_path.glob(".run.combine.*"))
