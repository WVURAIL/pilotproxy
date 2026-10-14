# coding=utf-8
"""Opt-in invariants for the large attached 23-product dissertation archive."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.archive_health import (
    REASON_BASEBAND_CEILING,
    REASON_DETECTOR_INVALID,
    REASON_DETECTOR_POWERS_ALL_ZERO,
    evaluate_frame_health,
    frame_utc_seconds,
    health_correct_integrated_spectra,
    recompute_corrected_fine_diagnostics,
    verify_supporting_evidence,
    _git_package_source_identity,
    _time_exposure_summary,
)


def test_attached_all23_health_invariants() -> None:
    value = os.environ.get("PILOT_PROXY_ATTACHED_PRODUCTS")
    if not value:
        pytest.skip("set PILOT_PROXY_ATTACHED_PRODUCTS to run large-archive invariants")
    root = Path(value)
    paths = sorted(root.glob("*.npz"), key=lambda path: int(path.stem))
    assert len(paths) == 23
    totals: Counter[str] = Counter()
    stored = included = 0
    versions: Counter[str] = Counter()
    ceiling_before = ceiling_after = zero_frame_units = 0
    product_events: set[int] = set()
    stored_frame_events: set[int] = set()
    healthy_frame_events: set[int] = set()
    channels_without_health_included_kept_frames: set[int] = set()
    channels_with_measured_epoch_anchor: set[int] = set()
    channel_23_q3_edge_refusal = False
    parseval_errors: list[float] = []
    temporal_stored = temporal_included = 0
    product_summaries: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as product:
            health = evaluate_frame_health(product)
            spectra = health_correct_integrated_spectra(product, health)
            fine_diagnostic = recompute_corrected_fine_diagnostics(
                product, health, chunk_rows=8192
            )
            assert np.count_nonzero(fine_diagnostic.null_bulk_mask) >= 8
            assert not np.any(
                fine_diagnostic.null_bulk_mask[
                    fine_diagnostic.predicted_acquisition_bins
                ]
            )
            assert sum(
                int(row["health_included_frames"])
                for row in fine_diagnostic.epoch_anchor_records
            ) == np.count_nonzero(health.include)
            for record in fine_diagnostic.epoch_anchor_records:
                candidate = record.get("candidate_anchor_bin")
                if candidate is not None:
                    assert int(candidate) in set(
                        fine_diagnostic.predicted_acquisition_bins.tolist()
                    )
                    assert abs(
                        int(record["candidate_minus_predicted_circular_bins"])
                    ) <= 30
                if record["status"] == "measured_narrow_line_anchor":
                    channels_with_measured_epoch_anchor.add(
                        int(np.asarray(product["physical_channel"]).item())
                    )
                    assert not np.any(
                        fine_diagnostic.null_bulk_mask[
                            np.asarray(record["selected_window_bins"], dtype=np.int64)
                        ]
                    )
                if (
                    int(np.asarray(product["physical_channel"]).item()) == 23
                    and record["epoch_key"] == "2026Q3"
                ):
                    assert record["status"] == "fallback_predicted_acquisition"
                    assert (
                        "candidate_at_acquisition_edge_with_stronger_external_sentinel"
                        in record["refusal_reasons"]
                    )
                    channel_23_q3_edge_refusal = True
            exposure = _time_exposure_summary(
                product, frame_utc_seconds(product), health.include
            )
            assert exposure["status"] in {"available", "partial"}
            resolved_stored = int(
                exposure["coverage"]["frames_with_finite_timestamp"]
            )
            resolved_included = int(
                exposure["coverage"][
                    "health_included_frames_with_finite_timestamp"
                ]
            )
            for key in (
                "utc_calendar_month",
                "utc_hour_of_day",
                "local_civil_calendar_month",
                "local_civil_hour_of_day",
                "local_meteorological_season",
                "local_mean_sidereal_hour",
            ):
                assert sum(
                    int(row["stored_frames"]) for row in exposure[key]["bins"]
                ) == resolved_stored
                assert sum(
                    int(row["health_included_frames"])
                    for row in exposure[key]["bins"]
                ) == resolved_included
            temporal_stored += resolved_stored
            temporal_included += resolved_included
            stored += int(health.include.size)
            included += int(np.count_nonzero(health.include))
            totals.update(health.reason_counts)
            versions[str(np.asarray(product["detector_version"]).item())] += 1
            assert spectra.exact is True
            assert spectra.parseval_pass is True
            if spectra.healthy_after_count == 0:
                channels_without_health_included_kept_frames.add(
                    int(np.asarray(product["physical_channel"]).item())
                )
            assert np.all(spectra.before >= 0.0)
            assert np.all(spectra.after >= 0.0)
            ceiling_before += spectra.ceiling_count_before
            ceiling_after += spectra.ceiling_count_after
            parseval_errors.extend(
                [
                    abs(spectra.before_parseval_relative_error),
                    abs(spectra.after_parseval_relative_error),
                ]
            )
            unit_events = np.asarray(product["unit_event_id"], dtype=np.int64)
            frame_units = np.asarray(product["frame_unit_index"], dtype=np.int64)
            units_with_frames = np.unique(frame_units)
            units_with_healthy_frames = np.unique(frame_units[health.include])
            zero_frame_units += int(unit_events.size - units_with_frames.size)
            product_events.update(int(value) for value in unit_events)
            stored_frame_events.update(
                int(value) for value in unit_events[units_with_frames]
            )
            healthy_frame_events.update(
                int(value) for value in unit_events[units_with_healthy_frames]
            )
            unit_keys = np.asarray(product["unit_order"]).reshape(-1)
            product_summaries.append(
                {
                    "product_name": path.name,
                    "product_sha256": "not-needed-without-product-archive",
                    "freq_id": int(np.asarray(product["freq_id"]).item()),
                    "unit_count": int(unit_keys.size),
                    "detector_version_tokens": {},
                }
            )
            frame_units = np.asarray(product["frame_unit_index"], dtype=np.int64)
            for frame in np.flatnonzero(~health.include):
                ledger.append(
                    {
                        "product_name": path.name,
                        "frame_index": int(frame),
                        "unit_key": str(unit_keys[frame_units[frame]]),
                    }
                )
    assert stored == 750_461
    assert included == 750_279
    assert temporal_stored == 747_972
    assert temporal_included == 747_790
    assert stored - included == 182
    assert totals[REASON_DETECTOR_INVALID] == 4
    assert totals[REASON_DETECTOR_POWERS_ALL_ZERO] == 4
    assert totals[REASON_BASEBAND_CEILING] == 178
    assert ceiling_before == 178
    assert ceiling_after == 118
    assert ceiling_before - ceiling_after == 60
    assert zero_frame_units == 4_692
    assert len(product_events) == 9_214
    assert len(stored_frame_events) == 8_983
    assert len(healthy_frame_events) == 8_980
    assert channels_without_health_included_kept_frames == {22, 24}
    assert channels_with_measured_epoch_anchor
    assert channel_23_q3_edge_refusal
    assert max(parseval_errors) < 5.0e-7
    assert len(versions) == 2
    assert all("kernel_sha256=c85f50ddf898517bc0101d1882c854c3df70b09f0ab0b58803dc32f59e3c6d12" in version for version in versions)

    inventory_value = os.environ.get("PILOT_PROXY_ATTACHED_INVENTORY")
    if inventory_value:
        evidence = verify_supporting_evidence(
            product_summaries,
            ledger,
            product_paths=paths,
            inventory_archive=Path(inventory_value),
        )
        observation = evidence["inventory_archive"][
            "health_filtered_exposure_by_observation_class"
        ]
        assert observation["status"] == "partial"
        assert "unit_delta_time" in observation["duration_unavailable_reason"]
        assert set(observation["classes"]) == {"triggered_event", "scheduled"}
        triggered = observation["classes"]["triggered_event"]
        scheduled = observation["classes"]["scheduled"]
        assert triggered["processed_units"] == 166_581
        assert triggered["stored_frames"] == 699_723
        assert triggered["health_included_frames"] == 699_579
        assert triggered["health_included_exposure_seconds"] == pytest.approx(
            29_238.0737536
        )
        assert scheduled["processed_units"] == 3_793
        assert scheduled["stored_frames"] == 50_738
        assert scheduled["health_included_frames"] == 50_700
        assert scheduled["health_included_exposure_seconds"] == pytest.approx(
            2_126.512128
        )
        assert sum(
            int(row["processed_units"])
            for row in observation["classes"].values()
        ) == 170_374
        assert sum(
            int(row["stored_frames"])
            for row in observation["classes"].values()
        ) == stored
        assert sum(
            int(row["health_included_frames"])
            for row in observation["classes"].values()
        ) == included


def test_attached_product_source_hash_maps_to_clean_historical_commit() -> None:
    if not os.environ.get("PILOT_PROXY_ATTACHED_PRODUCTS"):
        pytest.skip("set PILOT_PROXY_ATTACHED_PRODUCTS to run archive provenance")
    repository = Path(__file__).resolve().parents[2]
    resolved, digest = _git_package_source_identity(
        repository,
        "94b1de0e07bdbabb7e544aff22b62ef866e9cf0c",
    )
    assert resolved == "94b1de0e07bdbabb7e544aff22b62ef866e9cf0c"
    assert digest == "9317d5be23a309e1226dbb5b1b2c5cc950f3aef71ec54cb89e034b1ae921d0ce"
