# coding=utf-8
"""Canonical fixed-point constants shared by detector contracts and profiles."""

DEFAULT_DETECTOR_WINDOW_SAMPLES = 128
SUPPORTED_DETECTOR_WINDOW_SAMPLES = frozenset({64, 128})
LOCKED_NUM_WEIGHT_TERMS = 3
LOCKED_SKIPPED_GUARD_BINS = 1
LOCKED_REFERENCE_OFFSET_BINS = LOCKED_SKIPPED_GUARD_BINS + 1
LOCKED_PACKED_COMPLEX_BITS = 8
LOCKED_SAMPLE_BITS_PER_COMPONENT = 4
LOCKED_INPUT_FORMAT = "complex_int4_packed_int8"
LOCKED_POWER_ACCUMULATOR = "uint64"
# Kernel core 3.0.0: each frame power is the all-bin sum of that term's exact
# fixed-point fine power spectrum, P_r = sum_f S_r[f]. The transform length
# cancels in the ratio, so R_coarse keeps its form; the powers are no longer
# the row-domain sums of |z|^2 (detector contract revision 1).
LOCKED_COARSE_POWER_RATIO_DEFINITION = (
    "R_coarse = 2 * sum_f S_target[f] / "
    "(sum_f S_ref_lower[f] + sum_f S_ref_upper[f]); "
    "S = fxfft256_v1 fine powers"
)
LOCKED_COARSE_POWER_DEFINITION = "sum_of_fxfft256_fine_powers"
LOCKED_RAW_PILOT_EXCESS_DEFINITION = (
    "rho_raw = R_coarse - 1 (diagnostic only)"
)
LOCKED_HOST_MASKING_POLICY = "normalized_positive_excess_from_uint64_powers"
DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET = 16
POWER_SUM_ACCUMULATOR_BITS = 64

__all__ = [
    "DEFAULT_DETECTOR_WINDOW_SAMPLES",
    "DOT_PRODUCT_COMPONENT_ACCUMULATOR_BITS_TARGET",
    "LOCKED_COARSE_POWER_DEFINITION",
    "LOCKED_COARSE_POWER_RATIO_DEFINITION",
    "LOCKED_HOST_MASKING_POLICY",
    "LOCKED_INPUT_FORMAT",
    "LOCKED_NUM_WEIGHT_TERMS",
    "LOCKED_PACKED_COMPLEX_BITS",
    "LOCKED_POWER_ACCUMULATOR",
    "LOCKED_RAW_PILOT_EXCESS_DEFINITION",
    "LOCKED_REFERENCE_OFFSET_BINS",
    "LOCKED_SAMPLE_BITS_PER_COMPONENT",
    "LOCKED_SKIPPED_GUARD_BINS",
    "POWER_SUM_ACCUMULATOR_BITS",
    "SUPPORTED_DETECTOR_WINDOW_SAMPLES",
]
