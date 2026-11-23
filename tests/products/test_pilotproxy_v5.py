"""Per-pilot product builders shared by the product, reader and characterization tests.

The fixture writes the v5 layout of the products of record (row-domain frame
powers stored beside fine powers whose sums differ from them) and, with
``revision=6``, the current layout (frame powers stored as the fine sums).
The archive cohort acceptance tests that were here went with the v6 archive
pipeline (tag pre-purge-2026-11-23).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pilot_proxy.product_contract import coarse_columns_from_fine
from pilot_proxy.products.reader import MASK_RULE

from pilot_proxy.config.project import default_project


class products:  # noqa: N801 -- stands in for the retired RFIsher product registry
    freq_id = staticmethod(lambda ch: default_project().target_freq_id(str(ch)))


FIXTURE = Path(__file__).parent / "data" / "pilotproxy_v5_fixture.json"


def _fixture_spec() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _decision_contract() -> dict:
    return {
        "active_decision": {
            "implementation": "host_exact_integer_comparison",
            "method": "coarse_normalized_positive_excess",
            "output_field": "reject_mask",
        },
        "fine_candidate_decision": {
            "active": False,
            "calibration_status": "pending_campaign",
            "method": "fine_order_statistic_cfar",
        },
        "fine_measurement": {
            "method": "exact_fine_power_terms",
            "role": "measurement_only_no_scan_time_decision",
            "terms_field": "fine_power_u64",
        },
    }


def _detector_contract(spec: dict, revision: int = 5) -> dict:
    contract = {
        "schema_version": "pilotproxy_detector_contract_v1",
        "coarse_power_ratio_definition": (
            "R_coarse = 2 * P_target / (P_ref_lower + P_ref_upper)"
        ),
        "detector_window_samples": 128,
        "skipped_guard_bins": 1,
        "reference_offset_bins": 2,
        "num_weight_terms": 3,
        "sample_bits_per_component": 4,
        "input_format": "complex_int4_packed_int8",
        "power_accumulator": "uint64",
        "power_accumulator_bits": 64,
        "combine_mode": "all_rows_summed_before_ratio",
        "weight_coordinate_system": "post_spectral_sense_normalization",
        "input_coordinate_system": "post_spectral_sense_normalized",
        "input_preprocessing": {
            "time_reverse_detector_windows_before_kernel": True,
        },
        "mask_source": "normalized_positive_excess_decision",
        "threshold_mode": "none",
        "per_frequency_threshold": False,
        "valid_rule": "p_ref_sum != 0",
        "mask_rule": MASK_RULE,
        "equivalent_mask_rule": (
            "R_coarse > R_null; R_null = "
            "2*target_norm_sq/reference_norm_sum_sq"
        ),
        "fine_reduction": {
            "pad_factor": spec["fine_pad_factor"],
            "guard_fine_bins": spec["fine_guard_fine_bins"],
            "p_fa": spec["fine_p_fa"],
            "designated_bins": spec["fine_designated_bins"],
            "census_excluded_bins": spec["fine_census_excluded_bins"],
        },
    }
    if revision == 6:
        contract["schema_version"] = "pilotproxy_detector_contract_v2"
        contract["coarse_power_ratio_definition"] = (
            "R_coarse = 2 * sum_f S_target[f] / "
            "(sum_f S_ref_lower[f] + sum_f S_ref_upper[f]); "
            "S = fxfft256_v1 fine powers"
        )
        contract["fine_reduction"]["coarse_power_definition"] = (
            "sum_of_fxfft256_fine_powers"
        )
    return contract


def _frame_coordinates(frames: int, units: int):
    frame_unit = np.floor(
        np.arange(frames, dtype=np.float64) * units / frames
    ).astype(np.int32)
    frame_in_unit = np.empty(frames, dtype=np.int32)
    for unit in range(units):
        rows = np.flatnonzero(frame_unit == unit)
        frame_in_unit[rows] = np.arange(rows.size, dtype=np.int32)
    return frame_unit, frame_in_unit


def _unit_times(spec: dict, units: int) -> np.ndarray:
    split = units // 2
    values = []
    for unit in range(units):
        late = unit >= split
        index = unit - split if late else unit
        values.append(
            spec["first_time0_ctime"]
            + (spec["late_era_offset_seconds"] if late else 0.0)
            + (index // 3) * 86164.0905
            + (index % 3) * 3600.0
        )
    return np.asarray(values, dtype=np.float64)


def _write_product(
    path: Path, channel: int, *, frames=None, units=None, revision: int = 5
) -> Path:
    spec = _fixture_spec()
    frame_count = int(spec["frames"] if frames is None else frames)
    unit_count = int(spec["units"] if units is None else units)
    unit_count = min(unit_count, frame_count)
    freq_id = products.freq_id(channel)
    frame_unit, frame_in_unit = _frame_coordinates(frame_count, unit_count)

    lower = np.full((frame_count, 1), spec["p_ref_lower"], dtype=np.uint64)
    upper = np.full((frame_count, 1), spec["p_ref_upper"], dtype=np.uint64)
    reference = lower + upper
    cycle = np.asarray(spec["p_target_cycle"], dtype=np.uint64)
    target = np.resize(cycle, frame_count).reshape(frame_count, 1)
    target_norm = int(spec["target_norm_sq"])
    reference_norm = int(spec["reference_norm_sum_sq"])
    ratio = (target.astype(np.float64) * reference_norm
             / (reference.astype(np.float64) * target_norm))
    excess = ratio - 1.0
    coarse = 2.0 * target.astype(np.float64) / reference.astype(np.float64)
    ratio_db = 10.0 * np.log10(ratio)
    excess_db = np.full((frame_count, 1), np.nan, dtype=np.float64)
    positive = excess > 0.0
    excess_db[positive] = 10.0 * np.log10(excess[positive])
    offset = (
        spec["pilot_below_data_db"]
        - 10.0 * np.log10(spec["dtv_bandwidth_hz"] / spec["bin_enbw_hz"])
        - 10.0 * np.log10(spec["pilot_capture_efficiency"])
    )
    shelf = excess_db + offset
    rejected = np.fromiter(
        (int(num) * reference_norm > target_norm * int(den)
         for num, den in zip(target[:, 0], reference[:, 0])),
        dtype=np.uint8,
        count=frame_count,
    ).reshape(frame_count, 1)

    fine = np.full(
        (frame_count, 3, spec["fine_num_bins"]), 20, dtype=np.uint64)
    bins = np.arange(spec["fine_num_bins"], dtype=np.uint64)
    for frame in range(frame_count):
        fine[frame, 0] += bins
        fine[frame, 1] += bins % 7
        fine[frame, 2] += bins % 11
    fine[:, 0, spec["fine_designated_bins"]] = 10

    source_keys = np.asarray([
        "local:" + json.dumps([
            "fixture.event", f"{channel}-{unit}",
            f"baseband_{channel}_{unit}.h5",
        ], separators=(",", ":"))
        for unit in range(unit_count)
    ])
    unit_keys = np.asarray([
        "local:" + json.dumps([
            "fixture.event", f"{channel}-{unit}",
            f"baseband_{channel}_{unit}_{freq_id}.h5",
        ], separators=(",", ":"))
        for unit in range(unit_count)
    ])
    unit_times = _unit_times(spec, unit_count)
    sample_total = 2 * spec["nfft"] * spec["num_input_streams"]
    kernel = spec["kernel_sha256"]
    source_digest = "e" * 64
    detector_version = (
        "pilot-proxy/fixture "
        f"source={source_digest} kernel=2.3.0 kernel_sha256={kernel} "
        f"pilotproxy_per_pilot_product_v{revision} K=128"
    )
    chime_frequency = 800e6 - freq_id * spec["sample_rate_hz"]
    zeros_u64 = np.zeros((frame_count, 1), dtype=np.uint64)
    columns = dict(
        p_target_u64=target,
        p_ref_lower_u64=lower,
        p_ref_upper_u64=upper,
        p_ref_sum_u64=reference,
        valid=np.ones((frame_count, 1), dtype=np.uint8),
        reject_mask=rejected,
        coarse_power_ratio=coarse,
        normalized_coarse_power_ratio_db=ratio_db,
        normalized_pilot_excess=excess,
        pilot_excess_db=excess_db,
        estimated_data_shelf_snr_db=shelf,
    )
    if revision == 6:
        # The current layout stores the frame powers as the fine sums.
        columns = coarse_columns_from_fine({
            "fine_power_u64": fine,
            "target_norm_sq": np.asarray([target_norm], dtype=np.int64),
            "reference_norm_sum_sq": np.asarray([reference_norm], dtype=np.int64),
            "pilot_below_data_db": np.asarray(spec["pilot_below_data_db"]),
            "bin_enbw_hz": np.asarray(spec["bin_enbw_hz"]),
            "dtv_bandwidth_hz": np.asarray(spec["dtv_bandwidth_hz"]),
            "pilot_capture_efficiency": np.asarray(spec["pilot_capture_efficiency"]),
        })
    np.savez(
        path,
        **columns,
        schema_name=np.asarray(spec["schema_name"]),
        schema_revision=np.asarray(revision, dtype=np.int64),
        schema_version=np.asarray(f"pilotproxy_per_pilot_product_v{revision}"),
        source_event_key_schema_version=np.asarray(
            spec["source_event_key_schema_version"]),
        decision_contract_json=np.asarray(json.dumps(_decision_contract())),
        detector_contract_json=np.asarray(
            json.dumps(_detector_contract(spec, revision))),
        detector_version=np.asarray(detector_version),
        mask_rule=np.asarray(MASK_RULE),
        reference_placement_json=np.asarray(json.dumps({
            "physical_channel": channel,
            "reference_offset_bins": 2,
        })),
        physical_channel=np.asarray([channel], dtype=np.int32),
        freq_id=np.asarray([freq_id], dtype=np.int64),
        pilot_frequency_hz=np.asarray(
            [470.309441e6 + 6e6 * (channel - 14)], dtype=np.float64),
        chime_frequency_hz=np.asarray([chime_frequency], dtype=np.float64),
        pilot_in_band=np.asarray([1], dtype=np.uint8),
        frame_index=np.arange(frame_count, dtype=np.int64),
        frame_unit_index=frame_unit,
        frame_in_unit=frame_in_unit,
        target_norm_sq=np.asarray([target_norm], dtype=np.int64),
        reference_norm_sum_sq=np.asarray([reference_norm], dtype=np.int64),
        pilot_below_data_db=np.asarray(
            spec["pilot_below_data_db"], dtype=np.float64),
        bin_enbw_hz=np.asarray(spec["bin_enbw_hz"], dtype=np.float64),
        dtv_bandwidth_hz=np.asarray(
            spec["dtv_bandwidth_hz"], dtype=np.float64),
        pilot_capture_efficiency=np.asarray(
            spec["pilot_capture_efficiency"], dtype=np.float64),
        fine_status=np.asarray("enabled"),
        fine_num_bins=np.asarray(spec["fine_num_bins"], dtype=np.int64),
        fine_power_u64=fine,
        fine_pad_factor=np.asarray(spec["fine_pad_factor"], dtype=np.int64),
        fine_guard_fine_bins=np.asarray(
            spec["fine_guard_fine_bins"], dtype=np.int64),
        fine_p_fa=np.asarray(spec["fine_p_fa"], dtype=np.float64),
        fine_designated_bins=np.asarray(
            spec["fine_designated_bins"], dtype=np.int64),
        fine_census_excluded_bins=np.asarray(
            spec["fine_census_excluded_bins"], dtype=np.int64),
        source_event_keys=source_keys,
        unit_keys=unit_keys,
        unit_order=unit_keys,
        archive_version=np.asarray(["fixture-v1"] * unit_count),
        unit_git_version_tag=np.asarray(["fixture"] * unit_count),
        unit_input_map_sha256=np.asarray([
            hashlib.sha256(f"input-{channel}-{unit}".encode()).hexdigest()
            for unit in range(unit_count)
        ]),
        unit_collection_server=np.asarray(["local"] * unit_count),
        unit_scope=np.asarray(["local"] * unit_count),
        unit_event_id=np.arange(unit_count, dtype=np.int64) + channel * 1000,
        unit_time0_fpga=np.arange(unit_count, dtype=np.uint64) + 1000,
        unit_time0_ctime=unit_times,
        unit_delta_time=np.full(
            unit_count, 1.0 / spec["sample_rate_hz"], dtype=np.float64),
        sample_rate_hz=np.asarray(spec["sample_rate_hz"], dtype=np.float64),
        nfft=np.asarray(spec["nfft"], dtype=np.int64),
        num_input_streams=np.asarray(
            spec["num_input_streams"], dtype=np.int64),
        detector_window_samples=np.asarray(
            spec["detector_window_samples"], dtype=np.int64),
        sense=np.asarray(spec["sense"], dtype=np.int64),
        rational_overflow_count=np.asarray(0, dtype=np.uint64),
        railed_sample_count=zeros_u64,
        fill_sample_count=zeros_u64,
        railed_sample_total=np.full(
            (frame_count, 1), sample_total, dtype=np.uint64),
        weights_hash=np.asarray(
            hashlib.sha256(f"weights-{channel}".encode()).hexdigest()),
        weight_bank_sha256=np.asarray(spec["weight_bank_sha256"]),
        weight_manifest_sha256=np.asarray(spec["weight_manifest_sha256"]),
        weight_coefficients_sha256=np.asarray(
            spec["weight_coefficients_sha256"]),
    )
    return path


def fine_powers_with_sums(fine, target, lower, upper) -> np.ndarray:
    """Return fine powers with the spectral shape of ``fine`` and exact all-bin sums.

    Under kernel core 3.0.0, the frame powers of a product are the all-bin sums
    of its fine powers, so a fixture with given frame powers must carry fine
    powers that sum to them. Each (frame, term) spectrum is scaled to its wanted
    total by the largest-remainder method. The shape, and so every fine ratio up
    to one factor per frame, is kept, and the sum is exact. A zero-shape
    spectrum with a nonzero total is filled flat.
    """
    fine = np.asarray(fine, dtype=np.int64)
    frames, terms, bins = fine.shape
    want = np.stack([np.asarray(v, dtype=np.int64).reshape(-1)
                     for v in (target, lower, upper)], axis=1)
    if want.shape != (frames, terms):
        raise ValueError("one wanted total per frame and term")
    shape = np.where(fine.sum(axis=2, keepdims=True) > 0, fine, 1)
    total = shape.sum(axis=2, keepdims=True)
    quota = shape * want[:, :, None]
    if np.any(quota < 0) or float(np.max(shape)) * float(np.max(want)) >= 2.0 ** 62:
        raise ValueError("fixture totals are outside the exact int64 range")
    base = quota // total
    remainder = want - base.sum(axis=2)
    order = np.argsort(-(quota % total), axis=2, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(bins)[None, None, :].repeat(frames, 0).repeat(terms, 1), axis=2)
    base += (rank < remainder[:, :, None]).astype(np.int64)
    out = base.astype(np.uint64)
    assert np.array_equal(out.sum(axis=2, dtype=np.uint64), want.astype(np.uint64))
    return out


def _replace(path: Path, **changes) -> None:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True)
                  for name in archive.files}
    values.update(changes)
    np.savez(path, **values)
