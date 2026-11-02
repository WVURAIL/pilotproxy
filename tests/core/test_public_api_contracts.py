"""Retained APIs preserve receiver identity, dimensions, units, and metadata."""

import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.config.frequency_plan import frequency_plan_from_mapping
from pilot_proxy.config.instrument import instrument_readiness
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import coarse_power_ratio_to_db, power_terms_to_coarse_power_ratio
from pilot_proxy.fine_reduction import fine_reduce
from pilot_proxy.integration.detector_core import DetectorCoreProfile
from pilot_proxy.integration.receiver_profile import (
    ReceiverProfile,
    default_reference_receiver_profile,
    receiver_frequency_to_channel,
    receiver_profile_hash,
)
from pilot_proxy.integration.stream_layout import (
    InputStreamMap,
    build_stream_map_for_channel,
    validate_integration_compatibility,
)
from pilot_proxy.kernel import KernelFeatures, KernelSpecs
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH
from pilot_proxy.reference_channelizer import (
    ReferenceChannelizerSpec,
    channelize_real_blocks_to_reference_channels,
)

ROOT = Path(__file__).resolve().parents[2]
CORE_JSON = ROOT / "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"


@pytest.mark.parametrize("nyquist_zone", [1, 2])
def test_profile_json_factories_bind_selection_and_stream_serialization(tmp_path, nyquist_zone):
    document = default_reference_receiver_profile(num_input_streams=2).to_dict()
    document["channelizer"]["frequency_axis"]["nyquist_zone"] = nyquist_zone
    receiver_path = tmp_path / "receiver.json"
    receiver_path.write_text(json.dumps(document))
    profile = ReceiverProfile.from_json(receiver_path)
    core = DetectorCoreProfile.from_json(CORE_JSON)
    selection = receiver_frequency_to_channel(470_309_441.0, profile)
    assert selection.to_dict() == {
        "rf_hz": 470_309_441.0, "coarse_channel_index": 179,
        "coarse_channel_center_hz": 470_312_500.0,
        "fine_bin_offset_hz": -3059.0 if nyquist_zone == 1 else 3059.0,
        "nyquist_zone": nyquist_zone, "requires_time_reversal": nyquist_zone == 2,
    }
    rows = build_stream_map_for_channel(
        num_input_streams=np.int64(2), selection=selection,
        physical_channel=14, dtv_pilot_hz=470_309_441.0,
        feed_ids=["west", 17], polarizations=["X", "Y"],
    )
    assert rows == [
        {"stream_index": i, "feed_index": i, "feed_id": feed, "polarization": pol,
         "selected_coarse_channel": 179,
         "selected_coarse_channel_center_hz": 470_312_500.0,
         "physical_channel": 14, "dtv_pilot_hz": 470_309_441.0}
        for i, (feed, pol) in enumerate([("west", "X"), (17, "Y")])
    ]
    stream_path = tmp_path / "streams.json"
    stream_path.write_text(json.dumps({
        "schema_version": "pilotproxy_stream_map_v1",
        "receiver_profile_id": profile.receiver_profile_id,
        "stream_unit": profile.input_stream_unit, "num_streams": 2, "streams": rows,
    }))
    streams = InputStreamMap.from_json(stream_path)
    validate_integration_compatibility(profile=profile, detector_core=core, stream_map=streams)
    assert streams.streams == rows
    assert core.to_dict() == json.loads(CORE_JSON.read_text())
    assert receiver_profile_hash(profile) == receiver_profile_hash(document)
    assert profile.to_dict() == document


@pytest.mark.parametrize("factory", [DetectorCoreProfile.from_json, ReceiverProfile.from_json])
def test_profile_json_factories_refuse_nonobject_documents(tmp_path, factory):
    path = tmp_path / "profile.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="must contain an object"):
        factory(path)


@pytest.mark.parametrize("count", [True, np.bool_(True), 2.7, "2", 0, -1])
def test_stream_builder_refuses_inexact_or_nonpositive_counts(count):
    selection = receiver_frequency_to_channel(470_309_441.0, default_reference_receiver_profile())
    with pytest.raises(ValueError, match="num_input_streams"):
        build_stream_map_for_channel(num_input_streams=count, selection=selection)


@pytest.mark.parametrize("field", ["feed_ids", "polarizations"])
def test_stream_builder_refuses_label_count_mismatch(field):
    selection = receiver_frequency_to_channel(470_309_441.0, default_reference_receiver_profile())
    with pytest.raises(ValueError, match=f"{field} length must match"):
        build_stream_map_for_channel(num_input_streams=2, selection=selection, **{field: ["X"]})


@pytest.mark.parametrize("scopes", [[], ["local.capture"]])
def test_readiness_missing_fields_matches_loaded_survey_requirements(tmp_path, scopes):
    (tmp_path / "receiver.yaml").write_text(
        "name: receiver\n"
        "band: {f0_mhz: 800.0, bandwidth_mhz: 400.0, n_channels: 1024}\n"
        "nyquist_zone: 2\n"
        f"scopes: {json.dumps(scopes)}\n"
    )
    readiness = instrument_readiness("receiver", directory=str(tmp_path))
    assert readiness.ready is bool(scopes)
    assert readiness.status == ("ready" if scopes else "geometry-only")
    assert readiness.missing() == ([] if scopes else ["scopes"])
    readiness.missing().append("caller-only diagnostic")
    assert readiness.missing() == ([] if scopes else ["scopes"])


def test_frequency_plan_centers_preserve_hertz_and_half_open_edges():
    plan = frequency_plan_from_mapping({"name": "decimal-bands", "bands": [
        {"label": "A", "low_mhz": 100.25, "high_mhz": 101.75, "role": "screened"},
        {"label": "B", "low_mhz": 101.75, "high_mhz": 103.25, "role": "control",
         "target_freq_id": 7},
    ]}, where="fixture")
    first, second = plan.all_bands
    assert [first.center_hz, second.center_hz] == [101_000_000.0, 102_500_000.0]
    assert first.width_hz == second.width_hz == 1_500_000.0
    assert first.contains_hz(first.center_hz)
    assert not first.contains_hz(101_750_000.0)
    assert second.contains_hz(101_750_000.0)
    assert plan.bands("control") == (second,)


@pytest.mark.parametrize("packed,integer", [(True, True), (False, False)])
def test_public_kernel_metadata_matches_loaded_weight_dimensions(packed, integer):
    core = DetectorCoreProfile.from_json(CORE_JSON)
    specs = KernelSpecs(core.detector_window_samples, core.num_weight_terms,
                        core.sample_bits_per_component, core.reference_offset_bins)
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH, expected_kernel=specs)
    header = bank.header
    assert header.n_channels == bank.rom_table.shape[0]
    assert bank.path.stat().st_size == header.header_size + header.n_channels * 3 * 128
    assert specs.as_descriptive_dict() == {
        "detector_window_samples": 128, "num_weight_terms": 3,
        "sample_bits_per_component": 4, "reference_offset_bins": 2,
    }
    features = KernelFeatures(packed, integer, 128, True, False, 4096)
    assert json.loads(json.dumps(features.as_dict())) == {
        "dot_product_path": "dp4a_packed_int8_lanes" if packed else "scalar_integer",
        "use_dp4a": packed,
        "power_accumulator": "uint64_integer" if integer else "float_diagnostic",
        "use_uint64_power_accumulation": integer, "block_threads": 128,
        "use_constant_weight_lanes": True, "use_shared_weight_lanes": False,
        "grid_max_blocks": 4096,
    }


def test_raw_power_ratio_db_preserves_units_shape_and_undefined_values():
    ratio = power_terms_to_coarse_power_ratio(np.array([1, 100, 10_000]), 200)
    np.testing.assert_allclose(coarse_power_ratio_to_db(ratio), [-20, 0, 20], rtol=0, atol=1e-14)
    value = coarse_power_ratio_to_db(0.001)
    assert isinstance(value, float) and value == -30
    values = coarse_power_ratio_to_db([[0.0, -1.0], [np.nan, np.inf]])
    assert values.shape == (2, 2)
    assert np.isnan(values[0]).all() and np.isnan(values[1, 0])
    assert np.isposinf(values[1, 1])


def test_rate_and_fine_bin_properties_match_actual_output_geometry():
    spec = ReferenceChannelizerSpec(n_tap=2, n_sample=32, adc_sample_rate_hz=3200,
                                   band_lower_hz=0, num_channels=16)
    tone = np.cos(2 * np.pi * 5 * np.arange(32) / 32)
    blocks = np.tile(tone, (10, 1))
    output = channelize_real_blocks_to_reference_channels(
        blocks, channel_indices=[4], spec=spec, response=np.ones((2, 32)),
    )
    assert spec.output_sample_rate_hz == 100
    assert output.shape == (1, 9)
    np.testing.assert_allclose(output, 32, atol=2e-6, rtol=0)
    assert output.shape[1] / spec.output_sample_rate_hz == 0.09

    # One integer impulse per stream produces flat fine spectra. Their total
    # divided by the FFT length reproduces the exact row energy.
    rows = np.zeros((3, 12, 2), dtype=np.int32)
    rows[:, ::4, 0] = np.array([3, 1, 2])[:, None]
    reduced = fine_reduce(rows, num_streams=3, windows_per_stream=4, pad_factor=2)
    assert reduced.num_bins == 8
    np.testing.assert_array_equal(reduced.coarse_power_by_term, [27, 3, 12])
    np.testing.assert_array_equal(reduced.fine_power, np.tile([[27], [3], [12]], (1, 8)))
    np.testing.assert_array_equal(reduced.fine_power.sum(axis=1) / reduced.num_bins, [27, 3, 12])
    np.testing.assert_allclose(reduced.fine_power_ratio, 3.6, rtol=1e-7)
