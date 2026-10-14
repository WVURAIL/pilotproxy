# coding=utf-8
"""Weight-norm zero-point regression tests for the positive-excess mask.

int4 quantization of the steering vectors leaves the three weight-term norms
unequal, so under a flat noise floor E[F] = null_power_ratio = 2*target_norm_sq/
reference_norm_sum_sq differs from 1 per channel (~0.985..1.011 across the shipped
ATSC 14-36 bank). Comparing the unnormalized ratio directly with one would pin the H0 mask
fraction toward 0 or 1 per channel. These tests pin the corrected behavior:

* ``weight_term_norms_sq`` matches a brute-force unpack exactly;
* the normalized rule reduces to ``2*p_target > p_ref_sum`` when the
  target-to-reference norm ratio is 1:2, and remains
  exact at the integer cross-multiplication boundary;
* Monte Carlo through the shipped ROM's real int4 weights shows E[F] tracks
  R_null (not 1) and the normalized mask restores an ~50% H0 mask fraction on the
  two most-biased channels;
* exported runtime bundles declare per-channel half-threshold rationals
  ``nt : (nl+nu)`` that the bundle validator cross-checks against the weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.detector_contract import (
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    null_power_ratio_from_weight_norms,
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_reference import (
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

# 4096 rows x 200 trials keeps the Monte Carlo a few seconds while giving
# sigma(mean F) small enough to separate null_power_ratio from 1 by >5 sigma on the two
# most-biased shipped channels (|null_power_ratio - 1| ~ 0.011..0.015 vs SEM ~ 0.0014).
_MC_ROWS = 4096
_MC_TRIALS = 200
_MC_SEED = 20260701
_MOST_BIASED_HIGH_CHANNEL = 18  # largest shipped null_power_ratio
_MOST_BIASED_LOW_CHANNEL = 20   # smallest shipped null_power_ratio


def test_weight_term_norms_sq_matches_bruteforce() -> None:
    rng = np.random.default_rng(1234)
    for bits in (4, 8):
        packed_dtype = np.int8 if bits == 4 else np.int16
        limit = np.iinfo(packed_dtype)
        packed = rng.integers(
            limit.min, limit.max + 1, size=(3, 128), dtype=packed_dtype
        )
        got = weight_term_norms_sq(packed, bits_per_component=bits)
        w = unpack_packed_complex(packed, bits)
        expected = (np.abs(w) ** 2).sum(axis=1)
        assert got == tuple(int(round(v)) for v in expected)
        assert all(isinstance(v, int) for v in got)


def test_normalized_rule_reduces_to_unit_null_for_1_2_norms() -> None:
    # A 1:2 target/reference norm ratio reduces exactly to 2*p_target > p_ref_sum.
    for p_target in range(0, 60):
        for p_ref_sum in (0, 1, 19, 20, 21, 40):
            unit_null_decision = int(
                p_ref_sum != 0 and 2 * p_target > p_ref_sum
            )
            normalized_decision = normalized_positive_excess(
                p_target, p_ref_sum, target_norm_sq=100, reference_norm_sum_sq=200
            )
            assert normalized_decision == unit_null_decision


def test_normalized_rule_exact_boundary() -> None:
    # nt=5, nrs=9 -> mask iff p_target*9 > 5*p_ref_sum, strictly.
    assert normalized_positive_excess(11, 20, target_norm_sq=5, reference_norm_sum_sq=9) == 0
    assert normalized_positive_excess(12, 20, target_norm_sq=5, reference_norm_sum_sq=9) == 1
    # equality is not an excess
    assert normalized_positive_excess(10, 18, target_norm_sq=5, reference_norm_sum_sq=9) == 0
    # invalid reference floor
    assert normalized_positive_excess(10, 0, target_norm_sq=5, reference_norm_sum_sq=9) == 0
    # exactness beyond float precision: (2**60 + 1) * 2 > 2 * 2**60 must mask
    big = 2**60
    assert normalized_positive_excess(
        big + 1, 2 * big, target_norm_sq=1, reference_norm_sum_sq=2
    ) == 1
    assert normalized_positive_excess(
        big, 2 * big, target_norm_sq=1, reference_norm_sum_sq=2
    ) == 0


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("p_target", -1),
        ("p_target", 1 << 64),
        ("p_target", 1.0),
        ("p_target", True),
        ("p_ref_sum", -1),
        ("p_ref_sum", 1 << 64),
        ("p_ref_sum", 1.0),
        ("p_ref_sum", True),
        ("target_norm_sq", 0),
        ("target_norm_sq", -1),
        ("target_norm_sq", 1.0),
        ("target_norm_sq", True),
        ("reference_norm_sum_sq", 0),
        ("reference_norm_sum_sq", -1),
        ("reference_norm_sum_sq", 1.0),
        ("reference_norm_sum_sq", True),
    ],
)
def test_normalized_rule_requires_exact_uint64_domains(
    field: str,
    invalid: object,
) -> None:
    arguments = {
        "p_target": 1,
        "p_ref_sum": 1,
        "target_norm_sq": 1,
        "reference_norm_sum_sq": 1,
    }
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        normalized_positive_excess(**arguments)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("target_norm_sq", 0),
        ("target_norm_sq", 1.0),
        ("target_norm_sq", True),
        ("reference_norm_sum_sq", 0),
        ("reference_norm_sum_sq", 1.0),
        ("reference_norm_sum_sq", True),
    ],
)
def test_null_power_ratio_requires_exact_positive_norms(
    field: str,
    invalid: object,
) -> None:
    arguments = {"target_norm_sq": 1, "reference_norm_sum_sq": 2}
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        null_power_ratio_from_weight_norms(**arguments)


def test_normalized_mask_rule_string_is_current_and_stable() -> None:
    assert NORMALIZED_POSITIVE_EXCESS_MASK_RULE == (
        "valid && (p_target * reference_norm_sum_sq > "
        "target_norm_sq * p_ref_sum)"
    )


def _mc_h0_frames(weights_packed: np.ndarray, *, rows: int, trials: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """White noise through the real int4 path: per-frame (F, p_target, p_ref)."""
    rng = np.random.default_rng(seed)
    conj_w = np.conj(unpack_packed_complex(weights_packed, 4)).T.astype(np.complex64)
    fstats = np.empty(trials, dtype=np.float64)
    p_targets = np.empty(trials, dtype=np.int64)
    p_refs = np.empty(trials, dtype=np.int64)
    k = weights_packed.shape[1]
    for trial in range(trials):
        x = (
            rng.standard_normal((rows, k)) + 1j * rng.standard_normal((rows, k))
        ).astype(np.complex64)
        xq = unpack_packed_complex(
            quantize_complex_numpy(x, 4, 2.0), 4
        ).astype(np.complex64)
        powers = (np.abs(xq @ conj_w) ** 2).sum(axis=0)
        p_targets[trial] = int(round(float(powers[0])))
        p_refs[trial] = int(round(float(powers[1] + powers[2])))
        fstats[trial] = 2.0 * powers[0] / (powers[1] + powers[2])
    return fstats, p_targets, p_refs


@pytest.mark.parametrize(
    "channel", [_MOST_BIASED_HIGH_CHANNEL, _MOST_BIASED_LOW_CHANNEL]
)
def test_shipped_rom_h0_zero_point_and_normalized_mask(channel: int) -> None:
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(channel)
    assert valid and weights is not None
    nt, nl, nu = weight_term_norms_sq(weights)
    nrs = int(nl + nu)
    null_power_ratio = null_power_ratio_from_weight_norms(nt, nrs)
    # The shipped bank's quantized norms are unequal on these channels; that
    # inequality is the entire point of the correction, so guard it.
    assert abs(null_power_ratio - 1.0) > 5e-3, (
        f"channel {channel}: shipped null_power_ratio={null_power_ratio!r} is ~1; pick a channel with a "
        "larger norm imbalance for this regression test"
    )

    fstats, p_targets, p_refs = _mc_h0_frames(
        np.asarray(weights, dtype=np.int8),
        rows=_MC_ROWS,
        trials=_MC_TRIALS,
        seed=_MC_SEED + channel,
    )
    sem = fstats.std(ddof=1) / np.sqrt(len(fstats))
    # E[F] under H0 is null_power_ratio rather than 1: the bias is detected at >5 sigma and the
    # measured mean agrees with null_power_ratio within 6 sigma.
    assert abs(fstats.mean() - null_power_ratio) < 6.0 * sem
    assert abs(fstats.mean() - 1.0) > 5.0 * sem

    normalized_mask = np.asarray(
        [
            normalized_positive_excess(
                int(p_t), int(p_r), target_norm_sq=nt, reference_norm_sum_sq=nrs
            )
            for p_t, p_r in zip(p_targets, p_refs)
        ]
    )
    fraction = float(normalized_mask.mean())
    # P(F > null_power_ratio | H0) ~ 0.5; 200 trials -> binomial sigma 0.035, band = +-4 sigma.
    assert 0.36 <= fraction <= 0.64, (
        f"channel {channel}: normalized H0 mask fraction {fraction:.3f} is not ~0.5"
    )


def test_runtime_bundle_declares_norm_thresholds(tmp_path) -> None:
    import json

    from pilot_proxy.integration.defaults import (
        DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
        DEFAULT_DETECTOR_CORE_PROFILE,
    )
    from pilot_proxy.runtime_bundle import (
        export_runtime_weight_bundle,
        validate_runtime_weight_bundle,
    )

    bundle_dir = tmp_path / "bundle"
    export_runtime_weight_bundle(
        receiver_profile_path=DEFAULT_CHIME_DTV_RECEIVER_PROFILE,
        detector_core_profile_path=DEFAULT_DETECTOR_CORE_PROFILE,
        physical_channels=[_MOST_BIASED_HIGH_CHANNEL],
        weight_coordinate_system="post_spectral_sense_normalization",
        output_dir=bundle_dir,
    )
    pilots = json.loads((bundle_dir / "pilot_profiles.json").read_text())
    row = pilots["profiles"][0]
    assert int(row["positive_excess_half_threshold_num"]) == int(row["target_norm_sq"])
    assert int(row["positive_excess_half_threshold_den"]) == int(row["reference_norm_sum_sq"])
    assert row["reference_norm_sum_sq"] > 0
    assert row["null_power_ratio"] == pytest.approx(
        2.0 * row["target_norm_sq"] / row["reference_norm_sum_sq"]
    )
    report = validate_runtime_weight_bundle(bundle_dir=bundle_dir)
    assert report["valid"] is True, report["errors"]
