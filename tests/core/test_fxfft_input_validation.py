"""Range checks must precede fixed-width casts or absolute values."""
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.fxfft import (
    INPUT_ABS_MAX,
    fine_power_fx,
    fxfft,
    fxfft256,
    fxfft256_scalar,
)


TRANSFORMS = (fxfft256, fxfft)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.uint64, (1 << 64) - 1),
        (np.uint64, 1 << 63),
        (np.int64, -(1 << 63)),
        (np.uint64, INPUT_ABS_MAX + 1),
        (np.int64, INPUT_ABS_MAX + 1),
        (np.int64, -INPUT_ABS_MAX - 1),
    ],
)
def test_out_of_contract_integer_is_rejected_before_conversion(transform, dtype, value):
    x = np.zeros((128, 2), dtype=dtype)
    x[0, 0] = value
    with pytest.raises(OverflowError, match=rf"input exceeds.*max {abs(value)}"):
        transform(x)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.int64, INPUT_ABS_MAX),
        (np.int64, -INPUT_ABS_MAX),
        (np.uint64, INPUT_ABS_MAX),
    ],
)
def test_inclusive_contract_boundary_has_exact_impulse_output(transform, dtype, value):
    x = np.zeros((128, 2), dtype=dtype)
    x[0] = value
    actual = transform(x)
    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, np.full((256, 2), value, dtype=np.int32))


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.bool_, object])
def test_noninteger_input_rejection_is_unchanged(transform, dtype):
    with pytest.raises(TypeError, match="input must be an integer array"):
        transform(np.zeros((128, 2), dtype=dtype))


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_shape_rejection_is_unchanged(transform):
    with pytest.raises(ValueError, match="input must have shape"):
        transform(np.zeros((64, 2), dtype=np.int64))


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_empty_batch_remains_valid(transform):
    actual = transform(np.zeros((0, 128, 2), dtype=np.uint64))
    assert actual.shape == (0, 256, 2)
    assert actual.dtype == np.int32


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_saved_golden_outputs_are_unchanged(transform):
    path = Path(__file__).resolve().parents[1] / "data" / "fxfft256_golden_v1.npz"
    with np.load(path, allow_pickle=False) as golden:
        np.testing.assert_array_equal(transform(golden["inputs"]), golden["outputs"])


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.bool_, object])
def test_scalar_reference_refuses_noninteger_samples(dtype):
    x = np.zeros((128, 2), dtype=dtype)
    x[0, 0] = 0.9 if dtype == np.float64 else 1
    with pytest.raises(TypeError, match="input must be an integer array"):
        fxfft256_scalar(x)


@pytest.mark.parametrize("value", [-INPUT_ABS_MAX, INPUT_ABS_MAX])
def test_scalar_reference_preserves_exact_contract_boundary(value):
    x = np.zeros((128, 2), dtype=np.int64)
    x[0] = value
    assert fxfft256_scalar(x) == [(value, value)] * 256


def test_fine_power_sum_rejects_uint64_overflow():
    # The DC transform is an exact sum: 128 identical complex samples
    # give 2 * (128 * amplitude)**2 per stream without rounded twiddles.
    streams = 512
    rows = np.full((1, streams * 128, 2), INPUT_ABS_MAX, dtype=np.int32)
    expected_dc = streams * 2 * (128 * INPUT_ABS_MAX) ** 2
    assert expected_dc == 1 << 64
    with pytest.raises(OverflowError, match="fine power exceeds uint64"):
        fine_power_fx(rows, num_streams=streams, num_weight_terms=1)


def test_fine_power_high_input_bound_can_still_fit_uint64():
    # A peak-times-stream-count bound exceeds uint64 here, but only 511
    # streams contribute. Check the actual sum before refusing the input.
    streams = 512
    rows = np.full((1, streams * 128, 2), INPUT_ABS_MAX, dtype=np.int32)
    rows[:, -128:] = 0
    powers = fine_power_fx(rows, num_streams=streams, num_weight_terms=1)
    assert powers.dtype == np.uint64
    assert int(powers[0, 0]) == 511 * 2 * (128 * INPUT_ABS_MAX) ** 2


def test_fine_power_deployed_projection_bound_remains_exact():
    streams = 2048
    amplitude = 1 << 14
    rows = np.full((1, streams * 128, 2), amplitude, dtype=np.int32)
    powers = fine_power_fx(rows, num_streams=streams, num_weight_terms=1)
    assert powers.dtype == np.uint64
    assert int(powers[0, 0]) == streams * 2 * (128 * amplitude) ** 2
