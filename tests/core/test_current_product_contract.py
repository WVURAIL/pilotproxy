# coding=utf-8
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_contract import NORMALIZED_POSITIVE_EXCESS_MASK_RULE
from pilot_proxy.product_contract import (
    CurrentProductContractError,
    FINE_DERIVED_FRAME_COLUMNS,
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    SOURCE_EVENT_KEY_SCHEMA_VERSION,
    coarse_columns_from_fine,
    current_decision_contract_json,
    frame_powers_from_fine,
    pilot_excess_from_fine_powers,
    validate_current_product_identity,
)

# One frame, four fine bins: the target, lower and upper spectra each sum to
# one, so P = (1, 1, 1), p_ref_sum = 2, R = 1 and, with norms 1 : 2, Q = 1.
FIXTURE_FINE = np.asarray(
    [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]], dtype=np.uint64
)


def current_product() -> dict[str, np.ndarray]:
    return {
        "schema_name": np.asarray(PER_PILOT_PRODUCT_SCHEMA_NAME),
        "schema_revision": np.asarray(PER_PILOT_PRODUCT_SCHEMA_REVISION),
        "schema_version": np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        "source_event_key_schema_version": np.asarray(
            SOURCE_EVENT_KEY_SCHEMA_VERSION
        ),
        "decision_contract_json": np.asarray(current_decision_contract_json()),
        "detector_contract_json": np.asarray("{\"geometry\":\"current\"}"),
        "physical_channel": np.asarray([14], dtype=np.int32),
        "freq_id": np.asarray([844], dtype=np.int64),
        "pilot_in_band": np.asarray([1], dtype=np.uint8),
        "pilot_frequency_hz": np.asarray([470_309_441.0]),
        "chime_frequency_hz": np.asarray([470_312_500.0]),
        "nfft": np.asarray(1, dtype=np.int64),
        "sample_rate_hz": np.asarray(390_625.0, dtype=np.float64),
        "detector_window_samples": np.asarray(1, dtype=np.int64),
        "num_input_streams": np.asarray(1, dtype=np.int64),
        "sense": np.asarray(-1, dtype=np.int64),
        "frame_index": np.asarray([0], dtype=np.int64),
        "p_target_u64": np.asarray([[1]], dtype=np.uint64),
        "p_ref_sum_u64": np.asarray([[2]], dtype=np.uint64),
        "p_ref_lower_u64": np.asarray([[1]], dtype=np.uint64),
        "p_ref_upper_u64": np.asarray([[1]], dtype=np.uint64),
        "coarse_power_ratio": np.asarray([[1.0]]),
        "normalized_coarse_power_ratio_db": np.asarray([[0.0]]),
        "pilot_excess_db": np.asarray([[np.nan]]),
        "estimated_data_shelf_snr_db": np.asarray([[np.nan]]),
        "reject_mask": np.asarray([[0]], dtype=np.uint8),
        "valid": np.asarray([[1]], dtype=np.uint8),
        "target_norm_sq": np.asarray([1], dtype=np.int64),
        "reference_norm_sum_sq": np.asarray([2], dtype=np.int64),
        "null_power_ratio": np.asarray([1.0]),
        "normalized_pilot_excess": np.asarray([[0.0]]),
        "baseband_power_linear": np.asarray([[1.0]]),
        "railed_sample_count": np.asarray([[0]], dtype=np.uint64),
        "fill_sample_count": np.asarray([[0]], dtype=np.uint64),
        # 2 * nfft * num_input_streams for the fixture's 1x1 geometry.
        "railed_sample_total": np.asarray([[2]], dtype=np.uint64),
        "integrated_spectrum_before_mask": np.asarray([1.0]),
        "integrated_spectrum_after_mask": np.asarray([1.0]),
        "fine_power_u64": FIXTURE_FINE.copy(),
        "psd_frame_db_i16": np.zeros((1, 0), dtype=np.int16),
        "psd_db_reference": np.asarray([[1.0]]),
        "fine_cfar_location": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_scale": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_threshold": np.asarray([[np.nan]], dtype=np.float64),
        "fine_cfar_mode": np.asarray([[0]], dtype=np.uint8),
        "fine_threshold_exceedance_count": np.asarray([[0]], dtype=np.int32),
        "fine_threshold_exceedance_frame": np.asarray([], dtype=np.int64),
        "fine_threshold_exceedance_bin": np.asarray([], dtype=np.int64),
        "fine_pad_factor": np.asarray(4, dtype=np.int64),
        "fine_num_bins": np.asarray(4, dtype=np.int64),
        "fine_p_fa": np.asarray(0.001, dtype=np.float64),
        "fine_guard_fine_bins": np.asarray(1, dtype=np.int64),
        "fine_designated_bins": np.asarray([0], dtype=np.int64),
        "fine_census_excluded_bins": np.asarray([], dtype=np.int64),
        "fine_status": np.asarray("enabled"),
        "fine_null_bulk_exceedance_fraction": np.asarray([[np.nan]]),
        "source_event_keys": np.asarray(["event"], dtype=str),
        "unit_keys": np.asarray(["event"], dtype=str),
        "unit_order": np.asarray(["event"], dtype=str),
        "unit_time0_ctime": np.asarray([np.nan], dtype=np.float64),
        "unit_time0_fpga": np.asarray([0], dtype=np.uint64),
        "unit_event_id": np.asarray([-1], dtype=np.int64),
        "unit_delta_time": np.asarray([1.0 / 390_625.0], dtype=np.float64),
        "archive_version": np.asarray([""], dtype=str),
        "unit_git_version_tag": np.asarray(["receiver-build"], dtype=str),
        "unit_input_map_sha256": np.asarray(["a" * 64], dtype=str),
        "unit_collection_server": np.asarray(["host-a"], dtype=str),
        "unit_scope": np.asarray(["triggered"], dtype=str),
        "frame_unit_index": np.asarray([0], dtype=np.int32),
        "frame_in_unit": np.asarray([0], dtype=np.int32),
        "max_chunks_per_file": np.asarray(-1, dtype=np.int64),
        "rational_overflow_count": np.asarray(0, dtype=np.uint64),
        "weights_hash": np.asarray("weights"),
        "weight_bank_sha256": np.asarray("bank"),
        "weight_manifest_sha256": np.asarray("manifest"),
        "detector_version": np.asarray("detector"),
        "mask_rule": np.asarray(NORMALIZED_POSITIVE_EXCESS_MASK_RULE),
        "reference_placement_json": np.asarray("{}"),
        "pilot_below_data_db": np.asarray(11.3, dtype=np.float64),
        "bin_enbw_hz": np.asarray(1.0, dtype=np.float64),
        "dtv_bandwidth_hz": np.asarray(6.0e6, dtype=np.float64),
        "pilot_capture_efficiency": np.asarray(1.0, dtype=np.float64),
    }


def test_current_product_is_accepted():
    validate_current_product_identity(current_product())


@pytest.mark.parametrize(
    "field",
    [
        "schema_name",
        "source_event_key_schema_version",
        "decision_contract_json",
        "source_event_keys",
        "unit_order",
        "unit_input_map_sha256",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "reject_mask",
    ],
)
def test_missing_current_field_is_refused(field: str):
    product = current_product()
    product.pop(field)
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_alternate_field_name_is_refused():
    product = current_product()
    # A required measurement under any other name is not that measurement.
    product["fine_terms_u64"] = product.pop("fine_power_u64")
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_legacy_basename_event_identity_is_refused():
    product = current_product()
    product["source_event_key_schema_version"] = np.asarray(
        "pilotproxy_basename_source_event_key_v1"
    )
    with pytest.raises(CurrentProductContractError, match="source-event identity"):
        validate_current_product_identity(product)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("schema_revision", np.asarray(1.9, dtype=np.float64)),
        ("freq_id", np.asarray([844.9], dtype=np.float64)),
        ("physical_channel", np.asarray([14.9], dtype=np.float64)),
        ("frame_unit_index", np.asarray([0.1], dtype=np.float64)),
        ("frame_in_unit", np.asarray([False], dtype=np.bool_)),
        ("unit_event_id", np.asarray([123.1], dtype=np.float64)),
        ("unit_time0_fpga", np.asarray([123.1], dtype=np.float64)),
        ("unit_git_version_tag", np.asarray([""], dtype=str)),
        ("unit_input_map_sha256", np.asarray([""], dtype=str)),
        ("unit_input_map_sha256", np.asarray(["A" * 64], dtype=str)),
        ("unit_scope", np.asarray([""], dtype=str)),
        ("p_target_u64", np.asarray([[1.0]], dtype=np.float64)),
        ("p_target_u64", np.asarray([[-1]], dtype=np.int64)),
        ("p_target_u64", np.asarray([[1], [2]], dtype=np.uint64)),
        ("p_ref_lower_u64", np.asarray([[1.0]], dtype=np.float64)),
        ("p_ref_upper_u64", np.asarray([[1], [2]], dtype=np.uint64)),
        ("p_ref_sum_u64", np.asarray([[2.0]], dtype=np.float64)),
        ("sample_rate_hz", np.asarray(0.0, dtype=np.float64)),
        ("mask_rule", np.asarray("forged-policy")),
        ("reject_mask", np.asarray([[1]], dtype=np.uint8)),
    ],
)
def test_malformed_current_scalar_or_frame_array_is_refused(
    field: str, malformed: np.ndarray
) -> None:
    product = current_product()
    product[field] = malformed
    with pytest.raises(CurrentProductContractError):
        validate_current_product_identity(product)


def test_empty_checkpoint_cannot_claim_a_consumed_unit() -> None:
    product = current_product()
    for field, dtype in (
        ("frame_index", np.int64),
        ("frame_unit_index", np.int32),
        ("frame_in_unit", np.int32),
        ("fine_threshold_exceedance_frame", np.int64),
        ("fine_threshold_exceedance_bin", np.int64),
    ):
        product[field] = np.asarray([], dtype=dtype)
    for field, dtype in (
        ("p_target_u64", np.uint64),
        ("p_ref_lower_u64", np.uint64),
        ("p_ref_upper_u64", np.uint64),
        ("p_ref_sum_u64", np.uint64),
        ("reject_mask", np.uint8),
        ("valid", np.uint8),
        ("fine_cfar_mode", np.uint8),
        ("fine_threshold_exceedance_count", np.int32),
    ):
        product[field] = np.empty((0, 1), dtype=dtype)
    for field in (
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "normalized_pilot_excess",
        "baseband_power_linear",
        "railed_sample_count",
        "fill_sample_count",
        "railed_sample_total",
        "fine_cfar_location",
        "fine_cfar_scale",
        "fine_cfar_threshold",
        "fine_null_bulk_exceedance_fraction",
    ):
        product[field] = np.empty((0, 1), dtype=np.float64)
    product["fine_power_u64"] = np.empty((0, 0), dtype=np.float32)

    with pytest.raises(CurrentProductContractError, match="must not claim"):
        validate_current_product_identity(product, allow_empty_checkpoint=True)


def test_reference_split_must_match_sum() -> None:
    product = current_product()
    product["p_ref_upper_u64"] = np.asarray([[2]], dtype=np.uint64)
    with pytest.raises(CurrentProductContractError, match="do not match"):
        validate_current_product_identity(product)


def test_reference_split_sum_must_fit_uint64() -> None:
    product = current_product()
    product["p_ref_lower_u64"] = np.asarray(
        [[np.iinfo(np.uint64).max]], dtype=np.uint64
    )
    product["p_ref_upper_u64"] = np.asarray([[1]], dtype=np.uint64)
    product["p_ref_sum_u64"] = np.asarray([[0]], dtype=np.uint64)
    with pytest.raises(CurrentProductContractError, match="exceeds uint64"):
        validate_current_product_identity(product)


def test_sample_count_sum_cannot_wrap() -> None:
    product = current_product()
    product["railed_sample_count"] = np.asarray(
        [[np.iinfo(np.uint64).max]], dtype=np.uint64
    )
    product["fill_sample_count"] = np.asarray([[1]], dtype=np.uint64)
    with pytest.raises(CurrentProductContractError, match="exceeds"):
        validate_current_product_identity(product)


@pytest.mark.parametrize(
    ("status", "terms", "bins", "message"),
    [
        ("enabled", np.zeros((1, 0, 0), dtype=np.uint64), 0, "must have bins"),
        (
            "not_applicable_pilot_out_of_band",
            np.zeros((1, 3, 1), dtype=np.uint64),
            1,
            "shape",
        ),
        ("pending", np.zeros((1, 0, 0), dtype=np.uint64), 0, "empty checkpoint"),
        ("unknown", np.zeros((1, 0, 0), dtype=np.uint64), 0, "must be one of"),
        # The degraded statuses of the retired fine_products=off|auto paths.
        ("disabled_by_option", np.zeros((1, 0, 0), dtype=np.uint64), 0, "must be one of"),
        (
            "kernel_library_lacks_matched_filter_row_projections",
            np.zeros((1, 0, 0), dtype=np.uint64),
            0,
            "must be one of",
        ),
    ],
)
def test_fine_status_must_match_retained_shape(
    status: str, terms: np.ndarray, bins: int, message: str
) -> None:
    product = current_product()
    product["fine_status"] = np.asarray(status)
    product["fine_power_u64"] = terms
    product["fine_num_bins"] = np.asarray(bins, dtype=np.int64)
    with pytest.raises(CurrentProductContractError, match=message):
        validate_current_product_identity(product)


def _out_of_band_product() -> dict[str, np.ndarray]:
    product = current_product()
    product["pilot_in_band"] = np.asarray([0], dtype=np.uint8)
    product["fine_status"] = np.asarray("not_applicable_pilot_out_of_band")
    product["fine_power_u64"] = np.zeros((1, 0, 0), dtype=np.uint64)
    product["fine_num_bins"] = np.asarray(0, dtype=np.int64)
    for field in ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64", "p_ref_sum_u64"):
        product[field] = np.zeros((1, 1), dtype=np.uint64)
    for field in (
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "normalized_pilot_excess",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
    ):
        product[field] = np.full((1, 1), np.nan)
    product["valid"] = np.zeros((1, 1), dtype=np.uint8)
    return product


def test_out_of_band_pilot_without_fine_powers_is_accepted() -> None:
    validate_current_product_identity(_out_of_band_product())


def test_in_band_pilot_must_retain_fine_powers() -> None:
    product = _out_of_band_product()
    product["pilot_in_band"] = np.asarray([1], dtype=np.uint8)
    with pytest.raises(CurrentProductContractError, match="must retain fine"):
        validate_current_product_identity(product)


def test_out_of_band_pilot_must_declare_its_status() -> None:
    product = current_product()
    product["pilot_in_band"] = np.asarray([0], dtype=np.uint8)
    with pytest.raises(CurrentProductContractError, match="out-of-band"):
        validate_current_product_identity(product)


@pytest.mark.parametrize("term", [0, 1, 2])
def test_frame_power_must_be_the_exact_sum_of_its_fine_powers(term: int) -> None:
    # One count added to one fine bin breaks the v6 invariant P_r = sum_f S_r[f]
    # even though every other column is consistent with the stored powers.
    product = current_product()
    product["fine_power_u64"][0, term, 2] += np.uint64(1)
    with pytest.raises(CurrentProductContractError, match="all-bin sum"):
        validate_current_product_identity(product)


def test_frame_power_off_by_one_is_refused_with_consistent_columns() -> None:
    # The stored powers and every column derived from them agree with each
    # other, but not with the fine powers: still refused.
    product = current_product()
    product["p_target_u64"] = np.asarray([[2]], dtype=np.uint64)
    product["coarse_power_ratio"] = np.asarray([[2.0]])
    product["normalized_coarse_power_ratio_db"] = np.asarray([[10.0 * np.log10(2.0)]])
    product["normalized_pilot_excess"] = np.asarray([[1.0]])
    product["pilot_excess_db"] = np.asarray([[0.0]])
    product["reject_mask"] = np.asarray([[1]], dtype=np.uint8)
    with pytest.raises(CurrentProductContractError, match="all-bin sum"):
        validate_current_product_identity(product)


@pytest.mark.parametrize(
    "field",
    [
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "normalized_pilot_excess",
    ],
)
def test_derived_float_column_must_follow_the_fine_sums(field: str) -> None:
    product = current_product()
    product[field] = product[field] + 1.0e-6
    with pytest.raises(CurrentProductContractError, match=field):
        validate_current_product_identity(product)


def test_product_built_from_coarse_columns_from_fine_is_accepted() -> None:
    # A wider synthetic frame set: derive every column from random fine powers
    # with the public function and the validator must accept the result.
    rng = np.random.default_rng(6)
    frames, bins = 5, 8
    fine = rng.integers(0, 1 << 40, size=(frames, 3, bins), dtype=np.uint64)
    fine[3] = 0  # an invalid frame: zero reference power
    product = current_product()
    product["fine_power_u64"] = fine
    product["fine_num_bins"] = np.asarray(bins, dtype=np.int64)
    product["frame_index"] = np.arange(frames, dtype=np.int64)
    product["frame_unit_index"] = np.zeros(frames, dtype=np.int32)
    product["frame_in_unit"] = np.arange(frames, dtype=np.int32)
    for field in (
        "baseband_power_linear",
        "psd_db_reference",
    ):
        product[field] = np.ones((frames, 1))
    for field in ("railed_sample_count", "fill_sample_count"):
        product[field] = np.zeros((frames, 1), dtype=np.uint64)
    product["railed_sample_total"] = np.full((frames, 1), 2, dtype=np.uint64)
    product["psd_frame_db_i16"] = np.zeros((frames, 0), dtype=np.int16)
    derived = coarse_columns_from_fine(product)
    assert tuple(derived) == FINE_DERIVED_FRAME_COLUMNS
    product.update(derived)
    validate_current_product_identity(product)
    assert derived["valid"][:, 0].tolist() == [1, 1, 1, 0, 1]
    for r, field in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        assert [int(v) for v in derived[field][:, 0]] == [
            sum(int(x) for x in fine[f, r]) for f in range(frames)
        ]


def test_nonempty_product_cannot_contain_an_unused_unit() -> None:
    product = current_product()
    product["unit_keys"] = np.asarray(["event", "unused"], dtype=str)
    product["unit_order"] = np.asarray(["event", "unused"], dtype=str)
    product["source_event_keys"] = np.asarray(["event", "unused"], dtype=str)
    product["archive_version"] = np.asarray(["", ""], dtype=str)
    product["unit_git_version_tag"] = np.asarray(
        ["receiver-build", "receiver-build"], dtype=str
    )
    product["unit_input_map_sha256"] = np.asarray(
        ["a" * 64, "a" * 64], dtype=str
    )
    product["unit_collection_server"] = np.asarray(
        ["host-a", "host-a"], dtype=str
    )
    product["unit_scope"] = np.asarray(["triggered", "triggered"], dtype=str)
    product["unit_time0_ctime"] = np.asarray([np.nan, np.nan], dtype=np.float64)
    product["unit_time0_fpga"] = np.asarray([0, 0], dtype=np.uint64)
    product["unit_event_id"] = np.asarray([-1, -1], dtype=np.int64)
    product["unit_delta_time"] = np.asarray(
        [1.0 / 390_625.0, 1.0 / 390_625.0], dtype=np.float64
    )

    with pytest.raises(CurrentProductContractError, match="every consumed unit"):
        validate_current_product_identity(product)


def test_frame_powers_from_fine_are_exact_and_select_rows() -> None:
    # Values near 2**52 per bin: a float64 sum would round, the uint64 sum
    # (checked against Python integers) does not.
    fine = np.zeros((3, 3, 8), dtype=np.uint64)
    fine[:, :, :] = (1 << 52) + 1
    fine[1, 0, 3] += np.uint64(7)
    target, lower, upper = frame_powers_from_fine({"fine_power_u64": fine})
    exact = [sum(int(v) for v in fine[f, 0]) for f in range(3)]
    assert [int(v) for v in target] == exact
    assert [int(v) for v in lower] == [8 * ((1 << 52) + 1)] * 3
    assert target.dtype == lower.dtype == upper.dtype == np.dtype(np.uint64)
    selected, _, _ = frame_powers_from_fine({"fine_power_u64": fine}, np.asarray([2, 1]))
    assert [int(v) for v in selected] == [exact[2], exact[1]]
    mask, _, _ = frame_powers_from_fine({"fine_power_u64": fine}, np.asarray([False, True, False]))
    assert [int(v) for v in mask] == [exact[1]]


def test_frame_powers_refuse_a_wrapped_sum_and_a_metadata_view() -> None:
    big = np.full((1, 3, 4), 1 << 62, dtype=np.uint64)
    with pytest.raises(CurrentProductContractError, match="2\\*\\*64"):
        frame_powers_from_fine({"fine_power_u64": big})
    header_only = np.broadcast_to(np.zeros((), dtype=np.uint64), (2, 3, 4))
    with pytest.raises(CurrentProductContractError, match="metadata-only"):
        frame_powers_from_fine({"fine_power_u64": header_only})
    with pytest.raises(CurrentProductContractError, match="uint64"):
        frame_powers_from_fine({"fine_power_u64": np.zeros((1, 3, 4), dtype=np.int64)})


@pytest.mark.parametrize(
    "status",
    [
        "disabled_by_option",
        "detector_fn_lacks_matched_filter_row_projections",
        "kernel_library_lacks_matched_filter_row_projections",
    ],
)
def test_frame_powers_refuse_a_v5_placeholder_status(status: str) -> None:
    # A v5 product whose fine stage did not run stores an all-zero
    # (N, 3, B) placeholder; its sums are not frame powers.
    product = {
        "fine_power_u64": np.zeros((2, 3, 4), dtype=np.uint64),
        "fine_status": np.asarray(status),
        "target_norm_sq": np.asarray([3], dtype=np.int64),
        "reference_norm_sum_sq": np.asarray([5], dtype=np.int64),
    }
    for helper in (frame_powers_from_fine, pilot_excess_from_fine_powers):
        with pytest.raises(CurrentProductContractError, match="placeholder"):
            helper(product)


def test_pilot_excess_from_fine_powers_is_two_p0_over_reference_over_mu0_minus_one() -> None:
    fine = np.zeros((3, 3, 2), dtype=np.uint64)
    fine[0, :, 0] = (600, 500, 500)   # F = 1.2
    fine[1, :, 1] = (250, 250, 250)   # F = 1.0
    product = {
        "fine_power_u64": fine,       # frame 2: zero reference power
        "target_norm_sq": np.asarray([3], dtype=np.int64),
        "reference_norm_sum_sq": np.asarray([5], dtype=np.int64),
    }
    mu0 = 2 * 3 / 5
    excess = pilot_excess_from_fine_powers(product)
    np.testing.assert_allclose(excess[:2], [1.2 / mu0 - 1.0, 1.0 / mu0 - 1.0], rtol=1e-15)
    assert np.isnan(excess[2])
