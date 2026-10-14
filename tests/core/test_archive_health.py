# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.archived_product_keys import (
    ARCHIVED_REFERENCE_NORM_SUM_SQ,
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_DATA_SHELF_SNR_DB,
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB,
    ARCHIVED_PILOT_EXCESS_DB,
)
from pilot_proxy.archive_health import (
    ARCHIVE_HEALTH_SUMMARY_SCHEMA_VERSION,
    RESIDUAL_HEALTH_VIEW_SCHEMA_VERSION,
    CORRECTED_SPECTRA_SCHEMA_VERSION,
    ENCODING_INTERPRETATION,
    EXCLUSION_LEDGER_SCHEMA_VERSION,
    FRAME_HEALTH_GATE_SCHEMA_VERSION,
    LMST_FORMULA_IMPLEMENTATION_SHA256,
    REASON_BASEBAND_CEILING,
    REASON_BASEBAND_NONFINITE,
    REASON_BASEBAND_OUT_OF_BOUNDS,
    REASON_DETECTOR_INVALID,
    REASON_DETECTOR_POWERS_ALL_ZERO,
    REASON_FINE_NEGATIVE,
    ArchiveHealthError,
    audit_product,
    corrected_fine_geometry,
    evaluate_frame_health,
    health_correct_integrated_spectra,
    main,
    proportion_summary,
    recompute_corrected_fine_diagnostics,
    temporary_residual_health_views,
    unix_utc_to_lmst_hours,
    write_residual_health_view,
)
from pilot_proxy import archive_health
from pilot_proxy.fine_reduction import calibrate_cfar


NFFT = 16_384
K = 128
FINE_BINS = 256
SAMPLE_RATE_HZ = 390_625.0


def _product(
    *,
    valid: list[int],
    reject: list[int],
    baseband_power: list[float],
    p_target: list[int] | None = None,
    p_reference: list[int] | None = None,
    fine: np.ndarray | None = None,
    num_streams: int = 2,
) -> dict[str, np.ndarray]:
    n = len(valid)
    if not (len(reject) == len(baseband_power) == n):
        raise ValueError("fixture frame columns must align")
    target = np.asarray(p_target or [10] * n, dtype=np.uint64)
    reference = np.asarray(p_reference or [20] * n, dtype=np.uint64)
    if fine is None:
        rng = np.random.default_rng(8)
        fine = rng.normal(1.0, 0.04, size=(n, FINE_BINS)).astype(np.float32)
    frame_unit = np.arange(n, dtype=np.int32)
    fstat = 2.0 * target.astype(float) / np.where(reference > 0, reference, 1)
    fstat = np.where(reference > 0, fstat, 0.0)
    return {
        "schema_version": np.asarray("pilotproxy_detector_datatrawl_v3"),
        "physical_channel": np.asarray([17], dtype=np.int32),
        "freq_id": np.asarray([798], dtype=np.int64),
        "pilot_frequency_hz": np.asarray([488_309_440.559], dtype=np.float64),
        "chime_frequency_hz": np.asarray([488_281_250.0], dtype=np.float64),
        "frame_index": np.arange(n, dtype=np.int64),
        "p_target_u64": target[:, None],
        "p_ref_sum_u64": reference[:, None],
        ARCHIVED_COARSE_POWER_RATIO: fstat[:, None],
        ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB: (
            10.0 * np.log10(np.maximum(fstat, 1.0e-12))
        )[:, None],
        ARCHIVED_PILOT_EXCESS_DB: np.full((n, 1), -10.0),
        ARCHIVED_DATA_SHELF_SNR_DB: np.full((n, 1), -31.0),
        "reject_mask": np.asarray(reject, dtype=np.uint8)[:, None],
        "valid": np.asarray(valid, dtype=np.uint8)[:, None],
        "baseband_power_linear": np.asarray(baseband_power, dtype=np.float64)[:, None],
        ARCHIVED_FINE_POWER_RATIO: np.asarray(fine, dtype=np.float32),
        "fine_pad_factor": np.asarray(2, dtype=np.int64),
        "fine_num_bins": np.asarray(FINE_BINS, dtype=np.int64),
        "fine_p_fa": np.asarray(1.0e-3),
        "fine_guard_fine_bins": np.asarray(1, dtype=np.int64),
        "fine_designated_bins": np.asarray([0], dtype=np.int64),
        "fine_census_excluded_bins": np.asarray([], dtype=np.int64),
        "frame_unit_index": frame_unit,
        "frame_in_unit": np.zeros(n, dtype=np.int32),
        "source_event_keys": np.asarray([f"baseband_{100+i}.h5" for i in range(n)]),
        "unit_order": np.asarray([f"cadc:test/baseband_{100+i}_798.h5" for i in range(n)]),
        "unit_event_id": np.arange(100, 100 + n, dtype=np.int64),
        "unit_time0_ctime": 1_700_000_000.0 + np.arange(n) * 100.0,
        "unit_delta_time": np.full(n, 1.0 / SAMPLE_RATE_HZ),
        "nfft": np.asarray(NFFT, dtype=np.int64),
        "sample_rate_hz": np.asarray(SAMPLE_RATE_HZ),
        "detector_window_samples": np.asarray(K, dtype=np.int64),
        "num_input_streams": np.asarray(num_streams, dtype=np.int64),
        "sense": np.asarray(-1, dtype=np.int64),
        "mu0": np.asarray([1.0]),
        # The correction is derived from these; mu0 above is the
        # archived float the migration still accepts.
        "target_norm_sq": np.asarray([1], dtype=np.int64),
        ARCHIVED_REFERENCE_NORM_SUM_SQ: np.asarray([2], dtype=np.int64),
        "weight_bank_sha256": np.asarray("a" * 64),
        "weight_manifest_sha256": np.asarray("b" * 64),
        "weights_hash": np.asarray("c" * 64),
        "detector_version": np.asarray(
            "pilot-proxy/1.0.0 source=" + "d" * 64 + " kernel=2.1.0 "
            "kernel_sha256=" + "e" * 64 + " pilotproxy_detector_datatrawl_v3 K=128"
        ),
        "detector_contract_json": np.asarray(
            json.dumps({"equivalent_mask_rule": "F > mu0"})
        ),
        "mask_rule": np.asarray("F > mu0"),
        "pilot_below_data_db": np.asarray(-16.0),
        "dtv_bandwidth_hz": np.asarray(5.38e6),
        "bin_enbw_hz": np.asarray(390_625.0 / K),
        "integrated_spectrum_before_mask": np.zeros(NFFT, dtype=np.float64),
        "integrated_spectrum_after_mask": np.zeros(NFFT, dtype=np.float64),
    }


def _save_product(path: Path, product: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **product)
    return path


def test_health_gate_is_fail_closed_and_uses_correct_encoding_language() -> None:
    fine = np.ones((6, FINE_BINS), dtype=np.float32)
    fine[5, 7] = -0.1
    product = _product(
        valid=[1, 0, 1, 1, 1, 1],
        reject=[0, 0, 0, 0, 0, 0],
        baseband_power=[4.0, 0.0, 128.0, np.nan, 129.0, 4.0],
        p_target=[10, 0, 10, 10, 10, 10],
        p_reference=[20, 0, 20, 20, 20, 20],
        fine=fine,
    )
    health = evaluate_frame_health(product)
    assert health.include.tolist() == [True, False, False, False, False, False]
    assert health.reason_counts == {
        REASON_DETECTOR_INVALID: 1,
        REASON_DETECTOR_POWERS_ALL_ZERO: 1,
        REASON_BASEBAND_NONFINITE: 1,
        REASON_BASEBAND_OUT_OF_BOUNDS: 1,
        REASON_BASEBAND_CEILING: 1,
        REASON_FINE_NEGATIVE: 1,
    }
    assert "native raw byte is 0x00" in ENCODING_INTERPRETATION
    assert "two's-complement 0x88" in ENCODING_INTERPRETATION


def test_archive_pdf_metadata_suppresses_wall_clock_dates() -> None:
    metadata = archive_health._pdf_metadata("Deterministic test")
    assert metadata["CreationDate"] is None
    assert metadata["ModDate"] is None
    assert metadata["Title"] == "Deterministic test"


def test_lmst_matches_hard_coded_j2000_and_usno_reference_vectors() -> None:
    # J2000 is also an algebraic anchor for the declared polynomial: T=0 and
    # GMST=280.46061837 degrees.  The modern reference values are copied from
    # the USNO Sidereal Time API v4.0.1 (queried as UT1, longitude -119.6175).
    unix = np.asarray([946_728_000.0, 1_700_000_000.0, 1_717_200_000.0])
    got = unix_utc_to_lmst_hours(unix)
    assert got[0] == pytest.approx(10.722874558, abs=1.0e-10)
    usno_hours = np.asarray(
        [
            10.0 + 43.0 / 60.0 + 22.3494 / 3600.0,
            17.0 + 49.0 / 60.0 + 52.8028 / 3600.0,
            8.0 + 41.0 / 60.0 + 24.8438 / 3600.0,
        ]
    )
    np.testing.assert_allclose(got, usno_hours, atol=0.01 / 3600.0, rtol=0.0)
    assert len(LMST_FORMULA_IMPLEMENTATION_SHA256) == 64


def test_lmst_modern_hour_boundary_is_stable() -> None:
    got = unix_utc_to_lmst_hours([1_704_096_990.0, 1_704_096_991.0])
    np.testing.assert_allclose(
        got,
        [6.999998193637778, 7.000276730426898],
        atol=1.0e-10,
        rtol=0.0,
    )
    assert np.floor(got).astype(int).tolist() == [6, 7]


def test_health_gate_refuses_missing_or_misaligned_identity() -> None:
    product = _product(valid=[1], reject=[0], baseband_power=[4.0])
    del product["unit_order"]
    with pytest.raises(ArchiveHealthError, match="unit_order"):
        evaluate_frame_health(product)


def test_corrected_fine_diagnostic_matches_scalar_reference() -> None:
    rng = np.random.default_rng(12)
    fine = rng.normal(1.0, 0.06, size=(4, FINE_BINS)).astype(np.float32)
    product = _product(
        valid=[1, 1, 1, 1],
        reject=[0, 0, 0, 0],
        baseband_power=[4.0, 4.1, 3.9, 4.2],
        fine=fine,
    )
    health = evaluate_frame_health(product)
    got = recompute_corrected_fine_diagnostics(product, health, chunk_rows=2)
    geometry = corrected_fine_geometry(product)
    assert geometry["stored_differs_from_predicted_acquisition"] is True
    assert len(got.predicted_acquisition_bins) == 61
    assert not np.any(got.measured_anchor_used_by_frame)
    for row in range(4):
        expected = calibrate_cfar(
            fine[row],
            designated_bins=got.predicted_acquisition_bins,
            census_excluded_bins=(),
            guard_fine_bins=1,
            pad_factor=2,
            p_fa=1.0e-3,
        )
        assert got.location[row] == pytest.approx(expected.location)
        assert got.scale[row] == pytest.approx(expected.scale)
        assert got.threshold[row] == pytest.approx(expected.threshold)
        assert got.null_bulk_exceedance_fraction[row] == pytest.approx(
            expected.null_bulk_exceedance_fraction
        )


def test_fine_anchor_audit_accepts_persistent_epoch_line_and_excludes_it_from_null() -> None:
    n = 40
    fine = np.ones((n, FINE_BINS), dtype=np.float32)
    product = _product(
        valid=[1] * n,
        reject=[0] * n,
        baseband_power=[4.0] * n,
        fine=fine,
    )
    predicted = corrected_fine_geometry(product)["predicted_anchor_bin"]
    measured_anchor = (predicted + 10) % FINE_BINS
    product[ARCHIVED_FINE_POWER_RATIO][:, measured_anchor] = 3.0
    got = recompute_corrected_fine_diagnostics(product, chunk_rows=11)
    assert np.all(got.measured_anchor_used_by_frame)
    assert np.all(got.selected_anchor_bin_by_frame == measured_anchor)
    assert len(got.epoch_anchor_records) == 1
    record = got.epoch_anchor_records[0]
    assert record["status"] == "measured_narrow_line_anchor"
    assert record["candidate_anchor_bin"] == measured_anchor
    assert record["candidate_persistence_fraction"] == pytest.approx(1.0)
    assert record["refusal_reasons"] == []
    assert not got.null_bulk_mask[measured_anchor]
    assert got.detected_count_selected_epoch_window.shape == (n,)
    np.testing.assert_allclose(got.selected_epoch_window_peak, 3.0)


def test_outside_line_cannot_steal_circular_predicted_acquisition_anchor() -> None:
    n = 40
    fine = np.ones((n, FINE_BINS), dtype=np.float32)
    product = _product(
        valid=[1] * n,
        reject=[0] * n,
        baseband_power=[4.0] * n,
        fine=fine,
    )
    fine_bin_hz = (SAMPLE_RATE_HZ / K) / FINE_BINS
    product["pilot_frequency_hz"] = np.asarray(
        float(product["chime_frequency_hz"].item()) - fine_bin_hz
    )
    predicted = corrected_fine_geometry(product)["predicted_anchor_bin"]
    assert predicted == 1
    target = (predicted - 5) % FINE_BINS
    external = (predicted + 80) % FINE_BINS
    product[ARCHIVED_FINE_POWER_RATIO][:, target] = 3.0
    product[ARCHIVED_FINE_POWER_RATIO][:, external] = 8.0
    got = recompute_corrected_fine_diagnostics(product, chunk_rows=13)
    assert np.all(got.measured_anchor_used_by_frame)
    assert np.all(got.selected_anchor_bin_by_frame == target)
    record = got.epoch_anchor_records[0]
    assert record["epoch_key"] == "2023Q4"
    assert "provisional" in record["epoch_definition"]
    assert "not an authoritative station epoch" in record["epoch_definition"]
    assert record["candidate_anchor_bin"] == target
    sentinel = record["strongest_outside_acquisition_sentinel"]
    assert sentinel["bin"] == external
    assert "cannot redefine" in sentinel["interpretation"]


def test_edge_candidate_with_stronger_external_sentinel_is_refused() -> None:
    n = 40
    fine = np.ones((n, FINE_BINS), dtype=np.float32)
    product = _product(
        valid=[1] * n,
        reject=[0] * n,
        baseband_power=[4.0] * n,
        fine=fine,
    )
    predicted = corrected_fine_geometry(product)["predicted_anchor_bin"]
    boundary = (predicted + 30) % FINE_BINS
    external = (predicted + 35) % FINE_BINS
    product[ARCHIVED_FINE_POWER_RATIO][:20, boundary] = 3.0
    product[ARCHIVED_FINE_POWER_RATIO][:, external] = 5.0
    got = recompute_corrected_fine_diagnostics(product, chunk_rows=13)
    assert not np.any(got.measured_anchor_used_by_frame)
    record = got.epoch_anchor_records[0]
    assert record["candidate_anchor_bin"] == boundary
    assert record["strongest_outside_acquisition_sentinel"]["bin"] == external
    assert (
        "candidate_at_acquisition_edge_with_stronger_external_sentinel"
        in record["refusal_reasons"]
    )


def test_fine_anchor_audit_refuses_unseparated_broad_elevation() -> None:
    n = 40
    fine = np.ones((n, FINE_BINS), dtype=np.float32)
    fine[:, ::2] = 3.0
    product = _product(
        valid=[1] * n,
        reject=[0] * n,
        baseband_power=[4.0] * n,
        fine=fine,
    )
    got = recompute_corrected_fine_diagnostics(product, chunk_rows=13)
    assert not np.any(got.measured_anchor_used_by_frame)
    record = got.epoch_anchor_records[0]
    assert record["status"] == "fallback_predicted_acquisition"
    assert "peak_not_separated_from_off_window_competitor" in record[
        "refusal_reasons"
    ]


def test_ceiling_spectrum_is_exactly_reconstructible() -> None:
    product = _product(
        valid=[1, 0, 1],
        reject=[0, 0, 0],
        baseband_power=[4.0, 0.0, 128.0],
        p_target=[10, 0, 10],
        p_reference=[20, 0, 20],
        num_streams=2,
    )
    health = evaluate_frame_health(product)
    # The production CuPy expression abs(complex64 FFT)**2 rounds before the
    # float64 stream sum.  The helper must reproduce that exact contribution.
    dc_per_stream = 34_359_736_320.0
    ceiling_dc = 2 * dc_per_stream
    healthy_parseval_sum = float(NFFT**2 * 2 * 4.0)
    background_before = np.zeros(NFFT, dtype=np.float64)
    background_after = np.zeros(NFFT, dtype=np.float64)
    background_before[9] = healthy_parseval_sum
    background_after[9] = healthy_parseval_sum
    product["integrated_spectrum_before_mask"] = background_before.copy()
    product["integrated_spectrum_before_mask"][0] += ceiling_dc
    product["integrated_spectrum_after_mask"] = background_after.copy()
    product["integrated_spectrum_after_mask"][0] += ceiling_dc

    corrected = health_correct_integrated_spectra(product, health)
    assert corrected.exact is True
    assert corrected.ceiling_dc_power_per_frame == ceiling_dc
    np.testing.assert_array_equal(corrected.before, background_before)
    np.testing.assert_array_equal(corrected.after, background_after)
    assert corrected.healthy_before_count == 1
    assert corrected.healthy_after_count == 1


def test_unknown_valid_exclusion_refuses_spectral_repair() -> None:
    product = _product(valid=[1], reject=[0], baseband_power=[np.nan])
    result = health_correct_integrated_spectra(product)
    assert result.exact is False
    assert "do not have a reconstructible spectrum" in str(result.unavailable_reason)


def test_proportion_summary_reports_wilson_uncertainty() -> None:
    summary = proportion_summary(20, 100)
    assert summary["fraction"] == pytest.approx(0.2)
    assert summary["standard_error"] == pytest.approx(0.04)
    assert summary["wilson_95"]["low"] < 0.2 < summary["wilson_95"]["high"]


def test_residual_view_filters_frames_and_records_source(tmp_path) -> None:
    product = _product(
        valid=[1, 0, 1, 1],
        reject=[0, 0, 0, 1],
        baseband_power=[4.0, 0.0, 128.0, 5.0],
        p_target=[10, 0, 10, 20],
        p_reference=[20, 0, 20, 20],
    )
    source = _save_product(tmp_path / "798.npz", product)
    destination = tmp_path / "view" / source.name
    assert write_residual_health_view(source, destination) == destination
    with np.load(destination, allow_pickle=False) as view:
        assert str(view["schema_version"].item()) == RESIDUAL_HEALTH_VIEW_SCHEMA_VERSION
        assert str(view["archive_health_gate_schema_version"].item()) == (
            FRAME_HEALTH_GATE_SCHEMA_VERSION
        )
        assert int(view["source_frame_count"]) == 4
        assert int(view["health_included_frame_count"]) == 2
        assert int(view["health_excluded_frame_count"]) == 2
        assert view["frame_index"].tolist() == [0, 1]
        # The view is a published interface, so it keeps
        # the archived column names until that migration is coordinated.
        assert view[ARCHIVED_COARSE_POWER_RATIO][:, 0].tolist() == [1.0, 2.0]
        assert "coarse_power_ratio" not in view.files
        # RFIsher's residual reader refuses a view without the per-frame
        # shelf estimate beside the statistic.
        assert view[ARCHIVED_DATA_SHELF_SNR_DB].shape == (2, 1)
        assert np.all(view["valid"] == 1)
        assert ARCHIVED_FINE_POWER_RATIO not in view.files

    with temporary_residual_health_views([source]) as paths:
        transient = paths[0]
        assert transient.is_file()
    assert not transient.exists()


def test_cli_writes_versioned_summary_ledger_and_corrected_spectra(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    product = _product(
        valid=[1, 0, 1],
        reject=[0, 0, 0],
        baseband_power=[4.0, 0.0, 128.0],
        p_target=[10, 0, 10],
        p_reference=[20, 0, 20],
    )
    dc = 2 * 34_359_736_320.0
    product["integrated_spectrum_before_mask"][0] = dc
    product["integrated_spectrum_after_mask"][0] = dc
    healthy_parseval_sum = float(NFFT**2 * 2 * 4.0)
    product["integrated_spectrum_before_mask"][9] = healthy_parseval_sum
    product["integrated_spectrum_after_mask"][9] = healthy_parseval_sum
    product_path = _save_product(tmp_path / "798.npz", product)
    output = tmp_path / "audit"

    assert main(
        [
            "--product",
            str(product_path),
            "--output-dir",
            str(output),
            "--no-plots",
            "--expect-products",
            "1",
            "--expect-excluded-frames",
            "2",
            "--expect-invalid-frames",
            "1",
            "--expect-ceiling-frames",
            "1",
        ]
    ) == 0
    summary = json.loads((output / "archive_health_summary.json").read_text())
    assert summary["schema_version"] == ARCHIVE_HEALTH_SUMMARY_SCHEMA_VERSION
    assert summary["health_gate"]["schema_version"] == FRAME_HEALTH_GATE_SCHEMA_VERSION
    assert summary["created_utc"] == "2023-11-14T22:13:20+00:00"
    assert summary["created_utc_source"] == "SOURCE_DATE_EPOCH"
    coordinates = summary["exposure_coordinates"]
    assert coordinates["status"] == "available"
    assert coordinates["local_mean_sidereal_hour"]["status"] == "available"
    assert (
        coordinates["triggered_versus_scheduled_observation_class"]["status"]
        == "unavailable"
    )
    implementation = summary["provenance"]["audit_implementation"]
    assert len(implementation["package_source_sha256"]) == 64
    assert len(implementation["archive_health_module_sha256"]) == 64
    manifest = json.loads((output / "diagnostic_manifest.json").read_text())
    assert manifest["path_semantics"] == "release_root_relative_posix"
    assert manifest["summary"] == "archive_health_summary.json"
    assert manifest["exclusion_ledger"] == "archive_exclusion_ledger.jsonl"
    assert manifest["corrected_spectra"] == "health_corrected_integrated_spectra.npz"
    assert manifest["generator"] == implementation
    ledger = [
        json.loads(line)
        for line in (output / "archive_exclusion_ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == 2
    assert all(row["schema_version"] == EXCLUSION_LEDGER_SCHEMA_VERSION for row in ledger)
    assert all(row["reason_codes"] for row in ledger)
    with np.load(output / "health_corrected_integrated_spectra.npz") as spectra:
        assert str(spectra["schema_version"].item()) == CORRECTED_SPECTRA_SCHEMA_VERSION
        assert int(spectra["ceiling_frames_subtracted_before"][0]) == 1
        assert float(spectra["integrated_spectrum_before_health_gate"][0, 0]) == 0.0


def test_cli_snapshot_expectation_fails_before_publishing(tmp_path) -> None:
    product = _product(
        valid=[0],
        reject=[0],
        baseband_power=[0.0],
        p_target=[0],
        p_reference=[0],
    )
    source = _save_product(tmp_path / "798.npz", product)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ArchiveHealthError, match="expected products=2, observed 1"):
        main(
            [
                "--product",
                str(source),
                "--output-dir",
                str(output),
                "--no-plots",
                "--expect-products",
                "2",
            ]
        )
    assert not output.exists()


def test_audit_product_ledger_uses_reason_coded_stable_keys(tmp_path) -> None:
    product = _product(
        valid=[0],
        reject=[0],
        baseband_power=[0.0],
        p_target=[0],
        p_reference=[0],
    )
    path = _save_product(tmp_path / "798.npz", product)
    summary, ledger, spectra = audit_product(path)
    assert summary["frame_counts"]["excluded_unique"] == 1
    assert spectra.exact is True
    assert len(ledger) == 1
    assert ledger[0]["ledger_key"].startswith(
        FRAME_HEALTH_GATE_SCHEMA_VERSION + "/fid-798/"
    )
    assert REASON_DETECTOR_INVALID in ledger[0]["reason_codes"]
    assert REASON_DETECTOR_POWERS_ALL_ZERO in ledger[0]["reason_codes"]
    exposure = summary["health_filtered_exposure"]
    assert exposure["status"] == "available"
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
        ) == 1
        assert sum(
            int(row["health_included_frames"]) for row in exposure[key]["bins"]
        ) == 0
    assert (
        exposure["local_mean_sidereal_hour"][
            "formula_implementation_sha256"
        ]
        == LMST_FORMULA_IMPLEMENTATION_SHA256
    )


def test_inventory_scope_join_exports_health_filtered_observation_classes(
    tmp_path,
) -> None:
    product = _product(
        valid=[1, 0, 1],
        reject=[0, 0, 0],
        baseband_power=[4.0, 0.0, 4.0],
        p_target=[10, 0, 10],
        p_reference=[20, 0, 20],
    )
    healthy_parseval_sum = float(NFFT**2 * 2 * 8.0)
    product["integrated_spectrum_before_mask"][9] = healthy_parseval_sum
    product["integrated_spectrum_after_mask"][9] = healthy_parseval_sum
    path = _save_product(tmp_path / "798.npz", product)
    _, ledger, _ = audit_product(path)
    inventory = []
    for index, key in enumerate(product["unit_order"].tolist()):
        common_path, name = str(key).rsplit("/", 1)
        inventory.append(
            {
                "common_path": common_path,
                "name": name,
                "scope": (
                    "chime.event.baseband.raw"
                    if index == 0
                    else "chime.scheduled.baseband.raw"
                ),
                "event": str(100 + index),
                "freq_id": 798,
                "size_bytes": 1000 + index,
            }
        )
    inventory_by_key = {
        f"{row['common_path']}/{row['name']}": row for row in inventory
    }
    result = archive_health._inventory_observation_class_exposure(
        inventory, inventory_by_key, [path], ledger, []
    )
    assert result["status"] == "available"
    triggered = result["classes"]["triggered_event"]
    scheduled = result["classes"]["scheduled"]
    assert triggered["processed_units"] == 1
    assert triggered["stored_frames"] == 1
    assert triggered["health_included_frames"] == 1
    assert scheduled["processed_units"] == 2
    assert scheduled["stored_frames"] == 2
    assert scheduled["health_included_frames"] == 1
    frame_seconds = NFFT / SAMPLE_RATE_HZ
    assert triggered["health_included_exposure_seconds"] == pytest.approx(
        frame_seconds
    )
    assert scheduled["health_included_exposure_seconds"] == pytest.approx(
        frame_seconds
    )
