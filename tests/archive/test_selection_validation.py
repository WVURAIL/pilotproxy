# coding=utf-8
"""Guards for selection + per-file/per-product validation (review cluster A).

Covers the reviewer's requested cases:
  #1  an omitted selection fails cleanly instead of mixing channels;
  #2  a file whose channel disagrees with the product is a hard error;
and the combiner refusing to stack products with mismatched geometry.

All GPU-free: the channel guard fires before any detector work, so the detector
analyzer needs only a stub kernel / weights / dummy detector_fn to reach it.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.archive.combine import (
    combine_detector_products,
    report_products,
)
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.scan import run_chime_scan
from pilot_proxy.detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
    build_detector_contract,
)
from pilot_proxy.product_contract import (
    PER_PILOT_PRODUCT_SCHEMA_NAME,
    PER_PILOT_PRODUCT_SCHEMA_REVISION,
    PER_PILOT_PRODUCT_SCHEMA_TOKEN,
    SOURCE_EVENT_KEY_SCHEMA_VERSION,
    current_decision_contract_json,
)

K = 128
CH14_HZ = 470.3125e6   # -> ATSC channel 14, CHIME freq_id 844
CH20_HZ = 506.3125e6   # -> a clearly different ATSC channel / freq_id
FREQ_ID14 = 844        # chime_freq_id_from_hz(CH14_HZ)


def _current_detector_contract_json(*, detector_window_samples=K):
    return json.dumps(
        build_detector_contract(
            detector_window_samples=int(detector_window_samples),
            skipped_guard_bins=1,
            reference_offset_bins=2,
            num_weight_terms=3,
            weight_coordinate_system=WEIGHT_COORDINATE_POST_SPECTRAL_SENSE,
            time_reverse_detector_windows_before_kernel=True,
        ),
        sort_keys=True,
    )


def _stub_kernel(k=K):
    specs = SimpleNamespace(K=k, N=3, bits=4, reference_offset_bins=2,
                            as_descriptive_dict=lambda: {})
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "t"))


def _detector_ctx():
    return RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID14], options={
        "detector_fn": lambda **kw: {"results": [], "rational_overflow_count": 0},
        "kernel": _stub_kernel(),
        "weights": np.zeros((3, K), dtype=np.int8),
    })


# -- #1: omitted selection must fail, not silently mix -----------------------

@pytest.mark.parametrize("analyzer_cls", [PilotProxyDetectorAnalyzer])
def test_plan_runs_rejects_empty_selection(analyzer_cls):
    red = analyzer_cls()
    ctx = RunContext(instrument=load_instrument("chime"))
    for empty in (None, "", "   ", []):
        with pytest.raises(ValueError, match="explicit freq_id selection"):
            red.plan_runs(ctx, empty)


def test_run_chime_scan_requires_select(tmp_path):
    for empty in (None, "", []):
        with pytest.raises(SystemExit, match="--select is required"):
            run_chime_scan(input_dir=tmp_path, output_dir=tmp_path / "o",
                           select=empty, analyzer="pilot-proxy-detector", verbose=False)


# -- #2: a file from the wrong channel is a hard error -----------------------

def test_detector_rejects_channel_mismatch():
    red = PilotProxyDetectorAnalyzer()
    red.begin(_detector_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    # a later file that belongs to a different coarse channel must not be absorbed
    with pytest.raises(ValueError, match="Refusing to mix coarse channels"):
        red.consume_file(iter(()), {"f_center_hz": CH20_HZ})


def test_detector_rejects_nfft_mismatch():
    red = PilotProxyDetectorAnalyzer()
    red.begin(_detector_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    with pytest.raises(ValueError, match="nfft"):
        red.consume_file(iter(()), {"f_center_hz": CH14_HZ, "nfft": 8192})


@pytest.mark.parametrize("missing", ["git_version_tag", "input_map_sha256"])
def test_detector_rejects_archive_unit_without_receiver_state(missing):
    red = PilotProxyDetectorAnalyzer()
    red.begin(_detector_ctx(), {"f_center_hz": CH14_HZ, "nfft": 16384})
    meta = {
        "f_center_hz": CH14_HZ,
        "nfft": 16384,
        "unit_key": "archive:unit",
        "scope": "archive.scope",
        "git_version_tag": "receiver-build",
        "input_map_sha256": "a" * 64,
    }
    meta.pop(missing)

    with pytest.raises(ValueError, match=missing):
        red.consume_file(iter(()), meta)


# -- combiner refuses mismatched geometry (F10) ------------------------------

def _write_min_detector_product(
    path,
    *,
    channel,
    nfft,
    n_frames=2,
    unit_keys=(),
    event_keys=None,
    contract=None,
    sample_rate_hz=390_625.0,
):
    events = [str(value) for value in (
        event_keys if event_keys is not None
        else unit_keys if unit_keys
        else [f"event-{index}" for index in range(int(n_frames))]
    )]
    if len(events) != int(n_frames):
        raise ValueError("test fixture requires one frame per event")
    detector_contract = (
        str(contract)
        if contract is not None
        else _current_detector_contract_json()
    )
    shape = (int(n_frames), 1)
    fields = dict(
        schema_name=np.asarray(PER_PILOT_PRODUCT_SCHEMA_NAME),
        schema_revision=np.asarray(PER_PILOT_PRODUCT_SCHEMA_REVISION),
        schema_version=np.asarray(PER_PILOT_PRODUCT_SCHEMA_TOKEN),
        source_event_key_schema_version=np.asarray(
            SOURCE_EVENT_KEY_SCHEMA_VERSION
        ),
        decision_contract_json=np.asarray(current_decision_contract_json()),
        detector_contract_json=np.asarray(detector_contract),
        physical_channel=np.asarray([channel], dtype=np.int32),
        freq_id=np.asarray([800 - channel], dtype=np.int64),
        pilot_frequency_hz=np.asarray([470_000_000.0 + channel]),
        chime_frequency_hz=np.asarray([470_000_000.0 + channel]),
        pilot_in_band=np.asarray([1], dtype=np.uint8),
        nfft=np.asarray(int(nfft), dtype=np.int64),
        sample_rate_hz=np.asarray(float(sample_rate_hz), dtype=np.float64),
        detector_window_samples=np.asarray(K, dtype=np.int64),
        num_input_streams=np.asarray(4, dtype=np.int64),
        sense=np.asarray(-1, dtype=np.int64),
        frame_index=np.arange(int(n_frames), dtype=np.int64),
        p_target_u64=np.ones(shape, dtype=np.uint64),
        p_ref_sum_u64=np.full(shape, 2, dtype=np.uint64),
        p_ref_lower_u64=np.ones(shape, dtype=np.uint64),
        p_ref_upper_u64=np.ones(shape, dtype=np.uint64),
        coarse_power_ratio=np.ones(shape, dtype=np.float64),
        normalized_coarse_power_ratio_db=np.zeros(shape, dtype=np.float64),
        pilot_excess_db=np.full(shape, np.nan, dtype=np.float64),
        estimated_data_shelf_snr_db=np.full(shape, np.nan, dtype=np.float64),
        reject_mask=np.zeros(shape, dtype=np.uint8),
        valid=np.ones(shape, dtype=np.uint8),
        target_norm_sq=np.asarray([1], dtype=np.int64),
        reference_norm_sum_sq=np.asarray([2], dtype=np.int64),
        null_power_ratio=np.asarray([1.0], dtype=np.float64),
        normalized_pilot_excess=np.zeros(shape, dtype=np.float64),
        baseband_power_linear=np.ones(shape, dtype=np.float64),
        railed_sample_count=np.zeros(shape, dtype=np.uint64),
        fill_sample_count=np.zeros(shape, dtype=np.uint64),
        railed_sample_total=np.full(shape, 2 * int(nfft) * 4, dtype=np.uint64),
        integrated_spectrum_before_mask=np.zeros(int(nfft), dtype=np.float64),
        integrated_spectrum_after_mask=np.zeros(int(nfft), dtype=np.float64),
        fine_power_ratio=np.zeros((int(n_frames), 0), dtype=np.float32),
        fine_power_u64=np.zeros((int(n_frames), 0, 0), dtype=np.uint64),
        psd_frame_db_i16=np.zeros((int(n_frames), 0), dtype=np.int16),
        psd_db_reference=np.ones((int(n_frames), 1), dtype=np.float64),
        fine_cfar_location=np.full(shape, np.nan, dtype=np.float64),
        fine_cfar_scale=np.full(shape, np.nan, dtype=np.float64),
        fine_cfar_threshold=np.full(shape, np.nan, dtype=np.float64),
        fine_cfar_mode=np.zeros(shape, dtype=np.uint8),
        fine_threshold_exceedance_count=np.zeros(shape, dtype=np.int32),
        fine_threshold_exceedance_frame=np.asarray([], dtype=np.int64),
        fine_threshold_exceedance_bin=np.asarray([], dtype=np.int64),
        fine_pad_factor=np.asarray(4, dtype=np.int64),
        fine_num_bins=np.asarray(0, dtype=np.int64),
        fine_p_fa=np.asarray(0.001, dtype=np.float64),
        fine_guard_fine_bins=np.asarray(1, dtype=np.int64),
        fine_designated_bins=np.asarray([0], dtype=np.int64),
        fine_census_excluded_bins=np.asarray([], dtype=np.int64),
        fine_status=np.asarray("disabled_by_option"),
        fine_null_bulk_exceedance_fraction=np.full(
            shape, np.nan, dtype=np.float64
        ),
        source_event_keys=np.asarray(events, dtype=str),
        frame_unit_index=np.arange(int(n_frames), dtype=np.int32),
        frame_in_unit=np.zeros(int(n_frames), dtype=np.int32),
        unit_keys=np.asarray(events, dtype=str),
        unit_order=np.asarray(events, dtype=str),
        unit_time0_ctime=np.full(int(n_frames), np.nan, dtype=np.float64),
        unit_time0_fpga=np.zeros(int(n_frames), dtype=np.uint64),
        unit_event_id=np.full(int(n_frames), -1, dtype=np.int64),
        unit_delta_time=np.full(int(n_frames), 1.0 / float(sample_rate_hz)),
        archive_version=np.asarray([""] * int(n_frames), dtype=str),
        unit_git_version_tag=np.asarray([""] * int(n_frames), dtype=str),
        unit_input_map_sha256=np.asarray([""] * int(n_frames), dtype=str),
        unit_collection_server=np.asarray([""] * int(n_frames), dtype=str),
        unit_scope=np.asarray(["local"] * int(n_frames), dtype=str),
        max_chunks_per_file=np.asarray(-1, dtype=np.int64),
        weight_bank_sha256=np.asarray("bank"),
        weight_manifest_sha256=np.asarray("manifest"),
        weights_hash=np.asarray("weights"),
        mask_rule=np.asarray(NORMALIZED_POSITIVE_EXCESS_MASK_RULE),
        detector_version=np.asarray(
            "pilot-proxy/1.0.0 source=test kernel=2.3.0 "
            "kernel_sha256=test pilotproxy_per_pilot_product_v1 K=128"
        ),
        pilot_below_data_db=np.asarray(11.3),
        bin_enbw_hz=np.asarray(3051.7578125),
        dtv_bandwidth_hz=np.asarray(6.0e6),
        pilot_capture_efficiency=np.asarray(1.0),
        rational_overflow_count=np.asarray(0, dtype=np.uint64),
        reference_placement_json=np.asarray("{}"),
    )
    np.savez(path, **fields)


def test_combine_rejects_mismatched_nfft(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384)
    _write_min_detector_product(b, channel=20, nfft=8192)  # mismatched geometry
    with pytest.raises(ValueError, match="disagree on 'nfft'"):
        combine_detector_products([a, b], tmp_path / "out")


def test_combine_rejects_legacy_product_without_namespaced_event_schema(
    tmp_path,
) -> None:
    product_path = tmp_path / "14.npz"
    _write_min_detector_product(product_path, channel=14, nfft=16384)
    with np.load(product_path, allow_pickle=False) as archive:
        legacy = {
            name: archive[name]
            for name in archive.files
            if name != "source_event_key_schema_version"
        }
    np.savez(product_path, **legacy)

    with pytest.raises(
        ValueError, match="missing required field 'source_event_key_schema_version'"
    ):
        combine_detector_products([product_path], tmp_path / "out")


def test_combine_and_report_recompute_namespaced_event_keys(tmp_path) -> None:
    product_path = tmp_path / "14.npz"
    _write_min_detector_product(
        product_path, channel=14, nfft=16384, n_frames=1
    )
    with np.load(product_path, allow_pickle=False) as archive:
        forged = {name: archive[name] for name in archive.files}
    # The current marker is present, but the event key intentionally drops the
    # campaign namespace that remains available in unit_order.
    forged["unit_keys"] = np.asarray(
        ["/campaign-a/baseband_100_786.h5"], dtype=str
    )
    forged["unit_order"] = np.asarray(
        ["/campaign-a/baseband_100_786.h5"], dtype=str
    )
    forged["source_event_keys"] = np.asarray(["baseband_100.h5"], dtype=str)
    np.savez(product_path, **forged)

    with pytest.raises(ValueError, match="namespaced derivation"):
        combine_detector_products([product_path], tmp_path / "out")
    with pytest.raises(ValueError, match="namespaced derivation"):
        report_products([product_path])


def test_authoritative_unit_keys_cannot_be_hidden_by_basename_unit_order(
    tmp_path,
) -> None:
    product_path = tmp_path / "14.npz"
    _write_min_detector_product(
        product_path, channel=14, nfft=16384, n_frames=1
    )
    with np.load(product_path, allow_pickle=False) as archive:
        forged = {name: archive[name] for name in archive.files}
    forged["unit_keys"] = np.asarray(
        ["/campaign-a/baseband_100_786.h5"], dtype=str
    )
    forged["unit_order"] = np.asarray(["baseband_100_786.h5"], dtype=str)
    forged["source_event_keys"] = np.asarray(["baseband_100.h5"], dtype=str)
    np.savez(product_path, **forged)

    with pytest.raises(ValueError, match="exact same unit identities"):
        combine_detector_products([product_path], tmp_path / "out")
    with pytest.raises(ValueError, match="exact same unit identities"):
        report_products([product_path])


@pytest.mark.parametrize(
    "malformation",
    [
        "fractional_schema_revision",
        "fractional_freq_id",
        "boolean_freq_id",
        "fractional_physical_channel",
        "fractional_p_target",
        "negative_p_target",
        "extra_p_target_row",
        "missing_unit_keys",
        "missing_acquisition_time",
        "fractional_frame_unit_index",
        "negative_frame_in_unit",
        "fractional_event_id",
        "fractional_fpga_count",
        "zero_sample_rate",
        "forged_mask_rule",
        "forged_reject_mask",
        "unused_unit",
    ],
)
def test_report_and_combine_fail_closed_on_malformed_current_product(
    tmp_path, malformation: str
) -> None:
    product_path = tmp_path / f"{malformation}.npz"
    _write_min_detector_product(
        product_path, channel=14, nfft=128, n_frames=2
    )
    with np.load(product_path, allow_pickle=False) as archive:
        malformed = {name: archive[name] for name in archive.files}

    replacements = {
        "fractional_schema_revision": (
            "schema_revision",
            np.asarray(1.9, dtype=np.float64),
        ),
        "fractional_freq_id": (
            "freq_id",
            np.asarray([786.9], dtype=np.float64),
        ),
        "boolean_freq_id": ("freq_id", np.asarray([True], dtype=np.bool_)),
        "fractional_physical_channel": (
            "physical_channel",
            np.asarray([14.9], dtype=np.float64),
        ),
        "fractional_p_target": (
            "p_target_u64",
            np.ones((2, 1), dtype=np.float64),
        ),
        "negative_p_target": (
            "p_target_u64",
            -np.ones((2, 1), dtype=np.int64),
        ),
        "extra_p_target_row": (
            "p_target_u64",
            np.ones((3, 1), dtype=np.uint64),
        ),
        "fractional_frame_unit_index": (
            "frame_unit_index",
            np.asarray([0.1, 1.9], dtype=np.float64),
        ),
        "negative_frame_in_unit": (
            "frame_in_unit",
            np.asarray([0, -1], dtype=np.int32),
        ),
        "fractional_event_id": (
            "unit_event_id",
            np.asarray([123.1, 123.9], dtype=np.float64),
        ),
        "fractional_fpga_count": (
            "unit_time0_fpga",
            np.asarray([123.1, 123.9], dtype=np.float64),
        ),
        "zero_sample_rate": (
            "sample_rate_hz",
            np.asarray(0.0, dtype=np.float64),
        ),
        "forged_mask_rule": ("mask_rule", np.asarray("forged-policy")),
        "forged_reject_mask": (
            "reject_mask",
            np.ones((2, 1), dtype=np.uint8),
        ),
    }
    if malformation == "missing_unit_keys":
        malformed.pop("unit_keys")
    elif malformation == "missing_acquisition_time":
        malformed.pop("unit_time0_fpga")
    elif malformation == "unused_unit":
        malformed["unit_keys"] = np.asarray(
            ["event-0", "event-1", "unused"], dtype=str
        )
        malformed["unit_order"] = np.asarray(
            ["event-0", "event-1", "unused"], dtype=str
        )
        malformed["source_event_keys"] = np.asarray(
            ["event-0", "event-1", "unused"], dtype=str
        )
        malformed["archive_version"] = np.asarray(["", "", ""], dtype=str)
        malformed["unit_time0_ctime"] = np.asarray(
            [np.nan, np.nan, np.nan], dtype=np.float64
        )
        malformed["unit_time0_fpga"] = np.asarray([0, 0, 0], dtype=np.uint64)
        malformed["unit_event_id"] = np.asarray([-1, -1, -1], dtype=np.int64)
        malformed["unit_delta_time"] = np.full(
            3, 1.0 / 390_625.0, dtype=np.float64
        )
    else:
        field, value = replacements[malformation]
        malformed[field] = value
    np.savez(product_path, **malformed)

    with pytest.raises(ValueError):
        report_products([product_path])
    with pytest.raises(ValueError):
        combine_detector_products([product_path], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_combine_propagates_measured_sample_rate_to_time_products(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    sample_rate_hz = 1_000.0
    for path, channel in ((a, 14), (b, 20)):
        _write_min_detector_product(
            path,
            channel=channel,
            nfft=128,
            n_frames=2,
            sample_rate_hz=sample_rate_hz,
        )

    out = tmp_path / "out"
    combine_detector_products([a, b], out, chunk_seconds=0.1)
    with np.load(out / "chime_spectrogram_cache.npz") as cache:
        np.testing.assert_allclose(cache["relative_time_s"], [0.0, 0.128])
    with np.load(out / "chime_integrated_spectra.npz") as spectra:
        assert float(spectra["sample_rate_hz"]) == pytest.approx(sample_rate_hz)
    with np.load(out / "chime_reductions_10s.npz") as reductions:
        assert reductions["chunk_index"].tolist() == [0, 1]


def test_combine_aligns_shorter_frame_grid_by_event_identity(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(
        a,
        channel=14,
        nfft=16384,
        n_frames=3,
        unit_keys=("f1", "f2", "f3"),
    )
    _write_min_detector_product(
        b,
        channel=20,
        nfft=16384,
        n_frames=2,
        unit_keys=("f1", "f3"),
    )
    out = tmp_path / "out"
    combine_detector_products([a, b], out)
    with np.load(out / "chime_detector_outputs.npz") as product:
        assert product["frame_index"].size == 2


# -- review round 2 -----------------------------------------------------------

# #1: equal frame counts but different source events must NOT stack as aligned.
def test_combine_intersects_same_count_different_events(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(
        a,
        channel=14,
        nfft=16384,
        n_frames=2,
        event_keys=["baseband_eventA.h5", "baseband_eventB.h5"],
    )
    _write_min_detector_product(
        b,
        channel=20,
        nfft=16384,
        n_frames=2,
        event_keys=["baseband_eventA.h5", "baseband_eventC.h5"],
    )
    out = tmp_path / "out"
    combine_detector_products([a, b], out)
    with np.load(out / "chime_detector_outputs.npz") as product:
        assert product["frame_index"].size == 1


# #9: two coarse channels that resolve to the same ATSC channel must be rejected.
def test_combine_rejects_duplicate_physical_channel(tmp_path):
    a = tmp_path / "399.npz"
    b = tmp_path / "400.npz"
    _write_min_detector_product(a, channel=43, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"])
    _write_min_detector_product(b, channel=43, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"])
    with pytest.raises(ValueError, match="appear in more than one"):
        combine_detector_products([a, b], tmp_path / "out")


# #9: a duplicate freq_id in --select must be rejected.
@pytest.mark.parametrize("analyzer_cls", [PilotProxyDetectorAnalyzer])
def test_plan_runs_rejects_duplicate_freq_id(analyzer_cls):
    red = analyzer_cls()
    ctx = RunContext(instrument=load_instrument("chime"))
    with pytest.raises(ValueError, match="duplicate freq_id"):
        red.plan_runs(ctx, "844,844")


# #7: a callable combine must reject products with different detector contracts.
def test_combine_rejects_contract_mismatch(tmp_path):
    a = tmp_path / "14.npz"
    b = tmp_path / "20.npz"
    _write_min_detector_product(a, channel=14, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"],
                                contract=_current_detector_contract_json(
                                    detector_window_samples=128
                                ))
    _write_min_detector_product(b, channel=20, nfft=16384, n_frames=2,
                                event_keys=["baseband_e.h5", "baseband_f.h5"],
                                contract=_current_detector_contract_json(
                                    detector_window_samples=64
                                ))  # differs
    with pytest.raises(ValueError, match="detector_contract_json"):
        combine_detector_products([a, b], tmp_path / "out")


def test_combine_rejects_noncurrent_contract_before_writing(tmp_path):
    product = tmp_path / "14.npz"
    contract = json.loads(_current_detector_contract_json())
    contract.pop("threshold_mode")
    _write_min_detector_product(
        product,
        channel=14,
        nfft=16384,
        contract=json.dumps(contract, sort_keys=True),
    )
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="current detector contract"):
        combine_detector_products([product], output)

    assert not output.exists()


# #2: the FIRST file must match the requested freq_id rather than only later files.
def test_detector_rejects_first_file_freq_id_mismatch():
    red = PilotProxyDetectorAnalyzer()
    # ctx requests freq_id 844, but the first file's center is a different channel
    with pytest.raises(ValueError, match="requested freq_id 844"):
        red.begin(_detector_ctx(), {"f_center_hz": CH20_HZ, "nfft": 16384})


# #5: an out-of-band pilot must yield an explicitly invalid detector product and
# must NOT invoke the (GPU) kernel.
def test_detector_out_of_band_emits_invalid_without_kernel(tmp_path):
    def _boom(**kw):
        raise AssertionError("detector_fn must not run for an out-of-band pilot")

    ctx = RunContext(instrument=load_instrument("chime"), selection=[400], options={
        "detector_fn": _boom,
        "kernel": _stub_kernel(),
        "weights": np.zeros((3, K), dtype=np.int8),
    })
    red = PilotProxyDetectorAnalyzer()
    # freq_id 400 (643.75 MHz) -> nearest ATSC 43, whose pilot is 559 kHz off-center
    with pytest.warns(RuntimeWarning, match="does not contain"):
        red.begin(ctx, {"f_center_hz": 643.75e6, "nfft": 16384})
    chunk = np.zeros((16384, 4), dtype=np.uint8)
    meta = {
        "f_center_hz": 643.75e6,
        "nfft": 16384,
        "unit_key": "baseband_e_400.h5",
        "scope": "local",
    }
    n = red.consume_file([chunk, chunk], meta)  # must not raise (kernel skipped)
    assert n == 2
    out = tmp_path / "400.npz"
    red.save(str(out))
    got = np.load(out)
    assert int(got["pilot_in_band"][0]) == 0
    assert int(got["physical_channel"][0]) == 43
    assert int(got["freq_id"][0]) == 400
    assert int(np.asarray(got["reject_mask"]).sum()) == 0
    assert int(np.asarray(got["valid"]).sum()) == 0
