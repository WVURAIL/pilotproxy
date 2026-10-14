"""Pure guards and literal CUDA/NumPy parity for the new paired engine."""

from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.detector_reference import quantize_complex_numpy
from pilot_proxy.testbench.fine_validation_gpu import (
    ALL_STAGES,
    FineValidationEngine,
    arrays_sha256,
    audit_against_cpu,
    integer_bounds,
)


def profile():
    frequency = np.array([0.113, 0.113 + 2 / 128, 0.113 - 2 / 128])
    weights = np.exp(2j * np.pi * frequency[:, None] * np.arange(128))
    return SimpleNamespace(
        ideal_weights=weights, packed_weights=quantize_complex_numpy(weights, 4, 7)
    )


def test_integer_bounds_are_safe_for_full_geometry_and_all_signed_nibbles():
    packed = np.full((3, 128), -120, dtype=np.int8)  # -8-8j
    bounds = integer_bounds(2048, packed)
    assert bounds["row_component_abs_bound"] == 2 * 128 * 7 * 8
    assert bounds["fft_component_abs_bound"] < 2**31
    assert bounds["fine_power_sum_bound"] < 2**63
    assert bounds["coarse_power_sum_bound"] < 2**63
    assert (
        integer_bounds(1, packed)["fine_power_sum_bound"] * 2048
        == bounds["fine_power_sum_bound"]
    )


@pytest.mark.parametrize("streams", [0, 2049, -1, 1.0, True])
def test_invalid_geometry(streams):
    with pytest.raises((ValueError, TypeError)):
        integer_bounds(streams, profile().packed_weights)


@pytest.mark.parametrize(
    "packed", [np.zeros((3, 128)), np.zeros((2, 128), dtype=np.int8)]
)
def test_invalid_weight_contract(packed):
    with pytest.raises(ValueError):
        integer_bounds(1, packed)


def test_output_digest_binds_name_shape_dtype_and_bytes():
    array = np.array([1, 2], dtype=np.uint64)
    first = arrays_sha256({"a": array, "b": array})
    assert first == arrays_sha256({"b": array.copy(), "a": array.copy()})
    for changed in (
        {"a": array},
        {"a": array.reshape(1, 2), "b": array},
        {"a": array.astype(float), "b": array},
        {"a": array + 1, "b": array},
    ):
        assert arrays_sha256(changed) != first


@pytest.fixture
def engine():
    cp = pytest.importorskip("cupy")
    from pilot_proxy.kernel import FStatKernel
    from pilot_proxy.paths import DEFAULT_LIB_PATH

    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(20260909001)
    signal = (rng.normal(size=(128, 128)) + 1j * rng.normal(size=(128, 128))).astype(
        np.complex64
    )
    tone = np.broadcast_to(profile().ideal_weights[0], (128, 128)).astype(np.complex64)
    return FineValidationEngine(
        profile(),
        signal,
        tone,
        gpu=SimpleNamespace(cp=cp, kernel=FStatKernel(DEFAULT_LIB_PATH)),
        num_streams=8,
    )


@pytest.mark.cuda
def test_full_ladder_matches_independent_cpu(engine):
    result = engine.evaluate(20260909002, 0.37, audit=True)
    report = audit_against_cpu(engine, result)
    assert report["passed"]
    assert set(result["powers_by_stage"]) == set(ALL_STAGES)
    assert result["sample_count"] == 8 * 128 * 128
    assert 0 <= result["clip_count"] <= result["sample_count"]
    assert result["physical_certification"] is False


@pytest.mark.cuda
def test_seed_replay_and_paired_null_stages(engine):
    first = engine.evaluate(20260909003, 0)
    second = engine.evaluate(20260909003, 0)
    assert first["output_sha256"] == second["output_sha256"]
    for stage in ALL_STAGES:
        np.testing.assert_array_equal(
            first["powers_by_stage"][stage], second["powers_by_stage"][stage]
        )
    from pilot_proxy.testbench.sensitivity_study import (
        STAGE_ATSC_FLOAT,
        STAGE_IDEAL_TONE_FLOAT,
    )

    np.testing.assert_array_equal(
        first["powers_by_stage"][STAGE_ATSC_FLOAT],
        first["powers_by_stage"][STAGE_IDEAL_TONE_FLOAT],
    )
    assert engine.evaluate(20260909004, 0)["output_sha256"] != first["output_sha256"]
    assert (
        engine.evaluate(20260909003 + 2**63, 0)["output_sha256"]
        == first["output_sha256"]
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "seed,amplitude,audit",
    [
        (True, 1, False),
        (-1, 1, False),
        (2**64, 1, False),
        (1, True, False),
        (1, -1, False),
        (1, float("nan"), False),
        (1, 1, 1),
    ],
)
def test_invalid_frame_arguments(engine, seed, amplitude, audit):
    with pytest.raises((ValueError, TypeError)):
        engine.evaluate(seed, amplitude, audit=audit)


@pytest.mark.cuda
def test_shared_noise_reuse_is_exact_and_inputs_are_not_changed(engine):
    from pilot_proxy.testbench.fine_validation_gpu import draw_noise

    draw = draw_noise(20260909005, num_streams=engine.streams, cp=engine.cp)
    before = engine.cp.asnumpy(draw.samples)
    shared = engine.evaluate(draw.seed, 0.1, noise_draw=draw)
    fresh = engine.evaluate(draw.seed, 0.1)
    assert shared["output_sha256"] == fresh["output_sha256"]
    engine.evaluate(draw.seed, 0.3, noise_draw=draw)
    np.testing.assert_array_equal(engine.cp.asnumpy(draw.samples), before)
    assert (
        engine.evaluate(draw.seed, 0.1, reuse_noise=True)["output_sha256"]
        == fresh["output_sha256"]
    )
    assert (
        engine.evaluate(draw.seed, 0.1, reuse_noise=True)["output_sha256"]
        == fresh["output_sha256"]
    )
    engine.clear_noise_cache()
    assert engine._cached_noise is None
    with pytest.raises(ValueError):
        engine.evaluate(draw.seed + 1, 0.1, noise_draw=draw)
    with pytest.raises(ValueError):
        engine.evaluate(draw.seed, 0.1, noise_draw=draw, reuse_noise=True)


@pytest.mark.cuda
def test_optional_device_q16_epilogue_matches_host_and_preserves_powers(engine):
    from pilot_proxy.fine_reduction import independent_bin_mask

    decision = {
        "anchor_bin": 0,
        "designated_half_width": 2,
        "bulk_mask": independent_bin_mask(256, designated_bins=[0]),
        "cfar_rank": 62,
        "multiplier_q16": 65536,
    }
    base = engine.evaluate(20260909006, 0.1)
    for eta in (1, 65536, 2**64 - 1):
        decision["multiplier_q16"] = eta
        tested = engine.evaluate(20260909006, 0.1, decision=decision)
        assert tested["device_q16_checked"]
        assert tested["device_q16_mask"] in (0, 1)
        assert tested["output_sha256"] == base["output_sha256"]
    decision["bulk_mask"][:] = False
    tested = engine.evaluate(20260909006, 0.1, decision=decision)
    assert tested["device_q16_mask"] == 0


@pytest.mark.cuda
def test_prepared_null_input_matches_fresh_and_refuses_h1(engine):
    from pilot_proxy.testbench.fine_validation_gpu import prepare_null_input

    prepared = prepare_null_input(
        20260909007,
        num_streams=engine.streams,
        cp=engine.cp,
        input_scale=engine.input_scale,
    )
    expected = engine.evaluate(20260909007, 0)
    got = engine.evaluate(20260909007, 0, null_input=prepared, audit=True)
    assert got["output_sha256"] == expected["output_sha256"]
    assert audit_against_cpu(engine, got)["passed"]
    for amplitude, seed in ((0.1, 20260909007), (0, 20260909008)):
        with pytest.raises(ValueError):
            engine.evaluate(seed, amplitude, null_input=prepared)


def test_kernel_packing_width_is_part_of_geometry():
    kernel = SimpleNamespace(
        supports_row_projections=lambda: True,
        supports_fine_powers=lambda: True,
        get_fine_specs=lambda: {
            "windows_per_stream": 128,
            "pad_factor": 2,
            "fine_bins": 256,
        },
        specs=SimpleNamespace(
            detector_window_samples=128, num_weight_terms=3, sample_bits_per_component=8
        ),
    )
    with pytest.raises(ValueError, match="matched-filter geometry"):
        FineValidationEngine(
            profile(),
            np.ones((128, 128)),
            np.ones((128, 128)),
            gpu=SimpleNamespace(cp=None, kernel=kernel),
        )


@pytest.mark.cuda
def test_packing_half_ties_clipping_and_nibbles_match_cpu(engine):
    from pilot_proxy.testbench.fine_validation_gpu import _quantized_inputs

    values = np.array(
        [-7.5, -7.0, -6.5, -0.5, 0.0, 0.5, 6.5, 7.0, 7.5], dtype=np.float32
    )
    rows = np.resize(values + 1j * values[::-1], (128, 128)).astype(np.complex64)
    packed, dequantized, count = _quantized_inputs(
        engine.cp.asarray(rows), 1.0, engine.cp
    )
    expected = quantize_complex_numpy(rows, 4, 1.0)
    np.testing.assert_array_equal(engine.cp.asnumpy(packed), expected)
    from pilot_proxy.detector_reference import unpack_packed_complex

    np.testing.assert_array_equal(
        engine.cp.asnumpy(dequantized), unpack_packed_complex(expected, 4)
    )
    assert count == int(
        np.count_nonzero((np.abs(rows.real) > 7) | (np.abs(rows.imag) > 7))
    )
