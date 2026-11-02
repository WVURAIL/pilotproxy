# coding=utf-8
"""CHORD tone-injection tests of the detector at the signal level.

The bit-exact suites show that every implementation of the detector contract
computes the same integers. They cannot show that the CHORD weight bank is
tuned to the right radio frequencies, because a sign or mapping error would be
reproduced identically by every implementation. These tests check the
frequencies from first principles:

* A tone ``amp * exp(+2j*pi*f*n)`` is synthesized as channelized baseband at
  the exact ATSC pilot RF (``physical_channel_to_pilot_hz``), in the coarse
  channel bound by the bundle. The channel center is
  ``chord_channel_id * 195312.5 Hz`` in the upright frame of Nyquist zone 1,
  the kotekan chord data-product convention.
* The tone is quantized to packed int4, and the declared input preprocessing
  of the bundle (``time_reverse_detector_windows_before_kernel``) is applied.
  The exact integer frame powers (row projections through the frozen
  fxfft256, summed over fine bins, kernel core 3.0.0) and the rational
  half-threshold mask of the deployed kernel are then computed independently
  with numpy integers.
* The pilot tone must assert the mask with the target term dominant on every
  ATSC channel 14-36. A mirrored (conjugated) tone must collapse the target
  response, and this is the check that catches Nyquist-zone and
  window-reversal regressions. Tones at the reference placements of the
  manifest must drive the matching reference term and deassert the mask. This
  holds also for the edge-wrapped upper reference of channel 14 and the
  DC-shifted, edge-wrapped lower reference of channel 21. A weak pilot in
  unit-sigma noise must still assert the mask.

The first end-to-end CHORD integration in kotekan exported bundles with
``time_reverse_detector_windows_before_kernel = false``, derived from the
upright zone 1. The explicit-baseband-frame weight synthesis assumes the
adapter reversal in every zone, so these bundles were blind to upright pilot
tones. Every bit-exact test passed, while a real ATSC pilot would have gone
undetected. The pilot-tone and mirrored-tone tests below fail on that
regression.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_reference import (
    coarse_power_ratio_cpu_reference_packed,
    quantize_complex_numpy,
    unpack_packed_complex,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.runtime_bundle import export_runtime_weight_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
CHORD_PROFILE = CONFIGS_DIR / "receiver_profiles" / "chord_dtv_fengine.json"
PATHFINDER_PROFILE = (
    CONFIGS_DIR / "receiver_profiles" / "chord_pathfinder_dtv_fengine.json"
)
CORE_PROFILE = CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json"
CHORD_BANK = REPO_ROOT / "weights" / "chord_dtv_weights_k64.bin"

COARSE_WIDTH_HZ = 195312.5  # CHORD coarse width == channelized sample rate
K = 64                      # detector window samples
WINDOWS = 128               # synthesized windows per case: one whole fxfft256 stream
BITS = 4
ATSC_CHANNELS = list(range(14, 37))
TONE_AMP = 5.5              # LSB, in the int4 range
DOMINANCE = 100             # required tone-term power dominance factor
FLIP_COLLAPSE = 1000        # flip-collapse requirement cap
FLIP_QUANTIZATION_MARGIN = 4.0  # power headroom for int4 weight leakage


def dirichlet_power_collapse(separation_bins: float) -> float:
    """Return the pilot-to-flip target-power ratio predicted for the unquantized filter.

    The K-tap matched filter responds to a tone ``s`` fine bins away with
    the Dirichlet ratio ``sin(pi*s) / (K*sin(pi*s/K))``. The collapse is
    the inverse of the power ratio, and it is infinite on exact bin multiples.
    """
    s = float(separation_bins)
    numerator = math.sin(math.pi * s)
    if abs(numerator) < 1e-12:
        return float("inf")
    return (K * math.sin(math.pi * s / K) / numerator) ** 2


@pytest.fixture(scope="module")
def chord_bundle(tmp_path_factory) -> dict:
    """Export the deployed CHORD runtime bundle once and parse it."""
    bundle_dir = tmp_path_factory.mktemp("chord_bundle")
    export_runtime_weight_bundle(
        receiver_profile_path=CHORD_PROFILE,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=ATSC_CHANNELS,
        weight_coordinate_system="post_nyquist_zone_normalization",
        output_dir=bundle_dir,
    )
    profiles = json.loads((bundle_dir / "pilot_profiles.json").read_text("utf-8"))
    contract = json.loads((bundle_dir / "detector_contract.json").read_text("utf-8"))
    return {
        "rows": {row["physical_channel"]: row for row in profiles["profiles"]},
        "weights": (bundle_dir / "weights.bin").read_bytes(),
        "time_reverse": bool(
            contract["input_preprocessing"][
                "time_reverse_detector_windows_before_kernel"
            ]
        ),
    }


@pytest.fixture(scope="module")
def chord_bank() -> DetectorWeightBank:
    return DetectorWeightBank(explicit_path=CHORD_BANK)


def synth_packed(
    f_norm: float, amp: float, seed: int, noise_std: float = 0.0
) -> np.ndarray:
    """Return packed int4 windows of ``amp * exp(+2j*pi*f_norm*n)`` plus noise.

    The ``+`` sign is the upright (zone 1) baseband convention of the
    kotekan chord data product. In this convention, a tone above the
    coarse-channel center rotates positively. The sign is fixed here and must
    never be derived from the weight bank.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(WINDOWS * K, dtype=np.float64)
    x = amp * np.exp(2j * np.pi * (f_norm * n + rng.uniform()))
    if noise_std > 0.0:
        x = x + noise_std * (
            rng.standard_normal(WINDOWS * K) + 1j * rng.standard_normal(WINDOWS * K)
        )
    return quantize_complex_numpy(x.reshape(WINDOWS, K), BITS, 1.0)


def deployed_powers(
    packed_samples: np.ndarray, weight_bytes: bytes, time_reverse: bool
) -> list[int]:
    """Return the exact integer P[target, ref_lower, ref_upper] of the deployed chain.

    The integer row projections ``z = x . conj(w)`` are passed through the
    frozen fxfft256 over each 128-window stream, and the fine powers are summed
    over all bins (kernel core 3.0.0). The arithmetic is done here with numpy
    integers, independently of the packed CPU reference.
    """
    from pilot_proxy.fxfft import fine_power_fx

    if time_reverse:
        packed_samples = packed_samples[:, ::-1]
    x = unpack_packed_complex(packed_samples, BITS)
    w = unpack_packed_complex(
        np.frombuffer(weight_bytes, dtype=np.int8).reshape(3, K), BITS
    )
    rows = np.empty((3, x.shape[0], 2), dtype=np.int64)
    for term in range(3):
        z = (x * np.conjugate(w[term])).sum(axis=1)
        rows[term, :, 0] = np.round(z.real).astype(np.int64)
        rows[term, :, 1] = np.round(z.imag).astype(np.int64)
    fine = fine_power_fx(rows, num_streams=x.shape[0] // 128)
    return [sum(int(v) for v in fine[term]) for term in range(3)]


def deployed_mask(powers: list[int], row: dict) -> int:
    num, den = powers[0], powers[1] + powers[2]
    return int(
        den != 0
        and num * row["positive_excess_half_threshold_den"]
        > row["positive_excess_half_threshold_num"] * den
    )


def _channel_case(chord_bundle: dict, channel: int):
    row = chord_bundle["rows"][channel]
    weight_bytes = chord_bundle["weights"][
        row["weight_bank_offset_bytes"] : row["weight_bank_offset_bytes"]
        + row["weight_bank_nbytes"]
    ]
    pilot_hz = physical_channel_to_pilot_hz(channel)  # first principles
    center_hz = row["chord_channel_id"] * COARSE_WIDTH_HZ
    f_norm = (pilot_hz - center_hz) / COARSE_WIDTH_HZ
    return row, weight_bytes, pilot_hz, center_hz, f_norm


def test_bundle_requires_window_time_reversal(chord_bundle) -> None:
    # The explicit-frame contract requires the adapter to reverse each window.
    # If this flag is False, the detection tests below also fail, and this
    # test names the cause directly.
    assert chord_bundle["time_reverse"] is True


def test_pathfinder_bundle_requires_window_time_reversal(tmp_path) -> None:
    export_runtime_weight_bundle(
        receiver_profile_path=PATHFINDER_PROFILE,
        detector_core_profile_path=CORE_PROFILE,
        physical_channels=[14, 21],
        weight_coordinate_system="post_nyquist_zone_normalization",
        output_dir=tmp_path / "bundle",
    )
    contract = json.loads(
        (tmp_path / "bundle" / "detector_contract.json").read_text("utf-8")
    )
    assert (
        contract["input_preprocessing"][
            "time_reverse_detector_windows_before_kernel"
        ]
        is True
    )


@pytest.mark.parametrize("channel", ATSC_CHANNELS)
def test_pilot_lands_in_bound_coarse_channel(chord_bundle, channel) -> None:
    _, _, pilot_hz, center_hz, _ = _channel_case(chord_bundle, channel)
    assert abs(pilot_hz - center_hz) <= COARSE_WIDTH_HZ / 2


@pytest.mark.parametrize("channel", ATSC_CHANNELS)
def test_pilot_tone_asserts_mask(chord_bundle, channel) -> None:
    row, weight_bytes, _, _, f_norm = _channel_case(chord_bundle, channel)
    powers = deployed_powers(
        synth_packed(f_norm, TONE_AMP, 1000 + channel),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    assert deployed_mask(powers, row) == 1
    assert powers[0] >= DOMINANCE * (powers[1] + powers[2])


@pytest.mark.parametrize("channel", ATSC_CHANNELS)
def test_mirrored_tone_is_rejected(chord_bundle, channel) -> None:
    # A conjugated (mirrored) pilot must be rejected, and this is the check
    # that detects Nyquist-zone and window-reversal errors. The achievable
    # collapse depends on the distance between -f and +f on the channelized
    # frequency circle, in K=64 fine bins. The bound is the unquantized
    # Dirichlet prediction with 6 dB of headroom for int4 weight leakage,
    # capped at FLIP_COLLAPSE. For a pilot near the coarse-channel edge
    # (channel 30, offset -96.8 kHz of the +/-97.66 kHz half-width), the
    # flipped tone lands about 0.56 fine bins away, in the main lobe. Only a
    # factor of about 3 is then available, and the bound falls to its floor
    # of 2. A zone regression puts the full pilot response in p_flip and the
    # collapsed response in p_pilot, so the test fails even at the floor.
    row, weight_bytes, _, _, f_norm = _channel_case(chord_bundle, channel)
    p_pilot = deployed_powers(
        synth_packed(f_norm, TONE_AMP, 1000 + channel),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    p_flip = deployed_powers(
        synth_packed(-f_norm, TONE_AMP, 1000 + channel),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    circular = (2.0 * f_norm) % 1.0
    separation_bins = min(circular, 1.0 - circular) * K
    predicted = dirichlet_power_collapse(separation_bins)
    required = min(
        float(FLIP_COLLAPSE),
        max(2.0, predicted / FLIP_QUANTIZATION_MARGIN),
    )
    assert p_flip[0] * required <= p_pilot[0]


@pytest.mark.parametrize("channel", ATSC_CHANNELS)
def test_reference_tones_drive_reference_terms(
    chord_bundle, chord_bank, channel
) -> None:
    # Tones at the reference placements of the manifest (target +/- offset
    # bins) must drive the matching reference term and deassert the mask. The
    # upper reference of channel 14 wraps the coarse-channel edge, and the
    # lower reference of channel 21 is shifted away from DC to -3 bins.
    row, weight_bytes, _, _, f_norm = _channel_case(chord_bundle, channel)
    layout = chord_bank.layout_for_physical_channel(channel)
    lower_off = int(layout["lower_reference_offset_bins"])
    upper_off = int(layout["upper_reference_offset_bins"])

    p_lower = deployed_powers(
        synth_packed(f_norm + lower_off / K, TONE_AMP, 2000 + channel),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    p_upper = deployed_powers(
        synth_packed(f_norm + upper_off / K, TONE_AMP, 3000 + channel),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    assert p_lower[1] >= DOMINANCE * (p_lower[0] + p_lower[2])
    assert deployed_mask(p_lower, row) == 0
    assert p_upper[2] >= DOMINANCE * (p_upper[0] + p_upper[1])
    assert deployed_mask(p_upper, row) == 0


@pytest.mark.parametrize("channel", [14, 21, 36])
def test_weak_pilot_in_noise_detects(chord_bundle, channel) -> None:
    row, weight_bytes, _, _, f_norm = _channel_case(chord_bundle, channel)
    powers = deployed_powers(
        synth_packed(f_norm, 1.0, 4000 + channel, noise_std=1.0),
        weight_bytes,
        chord_bundle["time_reverse"],
    )
    threshold_f = (
        2.0
        * row["positive_excess_half_threshold_num"]
        / row["positive_excess_half_threshold_den"]
    )
    power_ratio = 2.0 * powers[0] / (powers[1] + powers[2])
    assert deployed_mask(powers, row) == 1
    assert power_ratio >= 5.0 * threshold_f


@pytest.mark.parametrize("channel", [14, 21, 36])
def test_integer_powers_match_cpu_reference(chord_bundle, channel) -> None:
    # The independent integer recomputation used above must agree exactly
    # with the repository's own packed CPU reference on the same windows.
    row, weight_bytes, _, _, f_norm = _channel_case(chord_bundle, channel)
    packed = synth_packed(f_norm, 1.0, 4000 + channel, noise_std=1.0)
    if chord_bundle["time_reverse"]:
        packed_for_kernel = np.ascontiguousarray(packed[:, ::-1])
    else:
        packed_for_kernel = packed
    _, sums = coarse_power_ratio_cpu_reference_packed(
        packed_for_kernel,
        np.frombuffer(weight_bytes, dtype=np.int8).reshape(3, K),
        BITS,
    )
    assert [int(value) for value in sums] == deployed_powers(
        packed, weight_bytes, chord_bundle["time_reverse"]
    )
