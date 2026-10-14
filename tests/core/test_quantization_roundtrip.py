#!/usr/bin/env python3
# coding=utf-8
"""
Verify packed quantization matches the unpacked representation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_reference import (
    matched_filter_row_projections_cpu_reference_packed,
    packed_dtype_for_component_bits,
    quantize_complex_numpy,
    unpack_packed_complex,
)

INT4_COMPONENT_BITS = 4
INT8_COMPONENT_BITS = 8
INVALID_COMPONENT_BITS = 6
UNITY_QUANTIZATION_SCALE = 1.0
INT4_MAX_MAGNITUDE = 7.0
INT8_MAX_MAGNITUDE = 127.0
TEST_MATRIX_SHAPE = (1, 1)


def test_quantize_unpack_roundtrip_4bit_limits() -> None:
    bits = INT4_COMPONENT_BITS
    scale = UNITY_QUANTIZATION_SCALE
    samples = np.asarray(
        [
            [-INT4_MAX_MAGNITUDE - INT4_MAX_MAGNITUDE * 1j,
             -INT4_MAX_MAGNITUDE + INT4_MAX_MAGNITUDE * 1j],
            [INT4_MAX_MAGNITUDE - INT4_MAX_MAGNITUDE * 1j,
             INT4_MAX_MAGNITUDE + INT4_MAX_MAGNITUDE * 1j],
        ],
        dtype=np.complex64,
    )

    max_int = (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1

    r = np.asarray(
        np.clip(np.round(samples.real * scale), -max_int, max_int),
        dtype=np.int32,
    )
    i = np.asarray(
        np.clip(np.round(samples.imag * scale), -max_int, max_int),
        dtype=np.int32,
    )
    expected = np.asarray((r << bits) | (i & mask), dtype=np.int8)

    packed = quantize_complex_numpy(samples, bits, scale)
    assert packed.dtype == np.int8
    np.testing.assert_array_equal(packed, expected)

    unpacked = unpack_packed_complex(packed, bits, dtype=np.int16)
    np.testing.assert_array_equal(unpacked.real.astype(np.int32), r)
    np.testing.assert_array_equal(unpacked.imag.astype(np.int32), i)


def test_quantize_unpack_roundtrip_8bit_limits() -> None:
    bits = INT8_COMPONENT_BITS
    scale = UNITY_QUANTIZATION_SCALE
    samples = np.asarray(
        [
            [-INT8_MAX_MAGNITUDE - INT8_MAX_MAGNITUDE * 1j,
             -INT8_MAX_MAGNITUDE + INT8_MAX_MAGNITUDE * 1j],
            [INT8_MAX_MAGNITUDE - INT8_MAX_MAGNITUDE * 1j,
             INT8_MAX_MAGNITUDE + INT8_MAX_MAGNITUDE * 1j],
        ],
        dtype=np.complex64,
    )

    max_int = (1 << (bits - 1)) - 1
    mask = (1 << bits) - 1

    r = np.asarray(
        np.clip(np.round(samples.real * scale), -max_int, max_int),
        dtype=np.int32,
    )
    i = np.asarray(
        np.clip(np.round(samples.imag * scale), -max_int, max_int),
        dtype=np.int32,
    )
    expected = np.asarray((r << bits) | (i & mask), dtype=np.int16)

    packed = quantize_complex_numpy(samples, bits, scale)
    assert packed.dtype == np.int16
    np.testing.assert_array_equal(packed, expected)

    unpacked = unpack_packed_complex(packed, bits, dtype=np.int16)
    np.testing.assert_array_equal(unpacked.real.astype(np.int32), r)
    np.testing.assert_array_equal(unpacked.imag.astype(np.int32), i)


def test_packed_dtype_invalid_bits_raises() -> None:
    assert packed_dtype_for_component_bits(INT4_COMPONENT_BITS) == np.dtype(np.int8)
    assert packed_dtype_for_component_bits(INT8_COMPONENT_BITS) == np.dtype(np.int16)
    with pytest.raises(ValueError):
        packed_dtype_for_component_bits(INVALID_COMPONENT_BITS)
    with pytest.raises(ValueError):
        quantize_complex_numpy(
            np.zeros(TEST_MATRIX_SHAPE, dtype=np.complex64),
            INVALID_COMPONENT_BITS,
            UNITY_QUANTIZATION_SCALE,
        )
    with pytest.raises(ValueError):
        unpack_packed_complex(
            np.zeros(TEST_MATRIX_SHAPE, dtype=np.int8),
            INVALID_COMPONENT_BITS,
        )


def test_exact_row_projection_reference_matches_direct_complex_dot() -> None:
    samples = np.asarray(
        [[1 + 2j, -3 + 1j, 2 - 2j], [-2 + 0j, 1 - 1j, 3 + 2j]],
        dtype=np.complex64,
    )
    weights = np.asarray(
        [
            [1 + 0j, 2 - 1j, -1 + 2j],
            [-2 + 1j, 1 + 1j, 0 - 1j],
            [1 - 2j, -1 + 0j, 2 + 1j],
        ],
        dtype=np.complex64,
    )
    packed_samples = quantize_complex_numpy(samples, INT4_COMPONENT_BITS, 1.0)
    packed_weights = quantize_complex_numpy(weights, INT4_COMPONENT_BITS, 1.0)

    got = matched_filter_row_projections_cpu_reference_packed(
        packed_samples, packed_weights, INT4_COMPONENT_BITS
    )
    x = unpack_packed_complex(packed_samples, INT4_COMPONENT_BITS)
    w = unpack_packed_complex(packed_weights, INT4_COMPONENT_BITS)
    expected_complex = x @ np.conjugate(w).T
    expected = np.stack(
        (expected_complex.real, expected_complex.imag), axis=0
    ).transpose(2, 1, 0)

    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, expected.astype(np.int32))
