# PilotProxy method specification

This specification fixes the detector geometry and the statistic used by the
CHIME real-data workflow. We state the operations in the order they are
applied so that a result can be reproduced from the recorded powers and
weight norms.

## Detector geometry

The baseline detector uses:

```text
detector_window_samples = 128
num_weight_terms = 3
skipped_guard_bins = 1
reference_offset_bins = 2
sample_bits_per_component = 4
power_accumulator = uint64
```

The two offset parameters obey
`reference_offset_bins = skipped_guard_bins + 1`. The target term is centered
on the nominal ATSC pilot. The lower and upper reference terms sample the
local non-DTV power on either side of that target.

## Target and reference powers

For each detector row `r`, the kernel returns three powers:

```text
P_target,r
P_ref_lower,r
P_ref_upper,r
```

We sum the powers over rows before forming a ratio:

```text
P_target = sum_r(P_target,r)
P_ref_sum = sum_r(P_ref_lower,r) + sum_r(P_ref_upper,r)
```

This order matters. Averaging per-row F-statistics is not equivalent to the
specified detector.

## F-statistic

The detector compares the target power with the mean of the two reference
powers:

```text
F = 2 * P_target / P_ref_sum
```

Equivalently,

```text
F = P_target / mean(P_ref_lower, P_ref_upper)
```

where the lower and upper powers are the row-summed quantities defined above.
The uncorrected one-bin pilot excess is

```text
rho = F - 1
```

When `P_ref_sum = 0`, the ratio is undefined. Current products store
`valid = 0` and NaN for diagnostic ratios and dB transforms; the binary
decision separately forces `reject_mask = 0`. Numerical zero is not used as
a stand-in for an invalid measurement.

## Norm-corrected positive-excess mask

The CHIME real-data path places its fixed mask threshold at the detector's own
null-hypothesis (`H0`) zero point. Under `H0`, the local spectrum contains no
pilot excess and is treated as flat across the three detector terms. Because
the steering vectors are quantized to int4, their squared norms are not
exactly equal. Therefore,

```text
E[P_term] = sigma^2 * ||w_term||^2
E[F]      = null_power_ratio = 2 * target_norm_sq / reference_norm_sum_sq
```

Here `target_norm_sq = ||w_target||^2` and
`reference_norm_sum_sq = ||w_ref_lower||^2 + ||w_ref_upper||^2`. These are exact
integer norms computed from the packed weights. Independent recomputation
from the shipped ATSC 14--36 manifest gives a `null_power_ratio` range of 0.9853298815 to
1.0111111111. A common threshold at `F = 1` would therefore place different
channels on different sides of their own `H0` zero points.

We compare with `null_power_ratio` using integer powers:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer form of

```text
F > null_power_ratio
```

When `target_norm_sq : reference_norm_sum_sq = 1 : 2`, the comparison reduces to
the earlier `F > 1` rule, `p_target > (p_ref_sum >> 1)`. Products written
under that earlier rule identify it in `mask_rule`.

The corrected pilot excess is

```text
rho_corrected = F / null_power_ratio - 1
```

and is stored per frame as `normalized_pilot_excess`. The CUDA ABI can express
the same comparison through the rational half-threshold
`half_num : half_den = target_norm_sq : reference_norm_sum_sq`. The current
`chime-scan` path reads the uint64 powers and applies the decision on the
host.

Masked frames are omitted from before/after averages. They are not replaced
with zeros.

## v2 time-coherent fine reduction

Kernel core 2.0.0 adds an exact int32 complex row-sum front end: for each
weight term `n` and detector row `m`, stage 1 returns

```text
z[n, m] = sum_k w_term[n, k] * x[m, k]        (exact integer complex)
```

Stage 2 reshapes each term's rows to `[num_streams, windows_per_stream]`,
applies an FFT along the window axis with `pad = 2` zero padding, and sums
incoherently over streams:

```text
S[n, b] = sum_streams |FFT_pad(z[n, stream, :])[b]|^2
F2[b]   = 2 * S_target[b] / (S_ref_lower[b] + S_ref_upper[b])
```

The reduction is coherent in time within a frame and incoherent across
streams and frames. Two identities tie it to the v1 statistic and are
enforced, not assumed: the exact int64 marginal identity (the integer sum
over rows of `|z[n, m]|^2` reproduces the deployed v1 power kernel
bit-for-bit, checked per frame at runtime) and the float Parseval identity
`sum_b S[n, b] = pad * windows * P[n]`, gated at ULP tolerance on the FFT
path.

Detection over fine bins uses a pre-registered null-bulk CFAR: the null is
calibrated on the independent non-designated bins (every `pad`-th bin,
excluding designated transmitter bins with a guard margin and any
census-excluded bins) with location = median and scale = left-side spread
(median minus the 15.87th percentile); when that bulk is itself contaminated
(flag fraction above 0.2), calibration falls back to lower quantiles
(location = P25, scale = (P25 - P2.275)/2) and records the mode. Detections
are declared on all bins against this calibration (any-bin rule), which
removes the census dependence of designated-only rules; the flag rate over
the non-designated independent bins is returned as
`null_bulk_exceedance_fraction` on the `CfarCalibration` from `calibrate_cfar`.
It is not stored in the per-pilot product, which retains only the exact
`fine_power_u64` terms it is recomputed from. Because the same bins fit the
threshold, this is an in-sample threshold diagnostic rather than an independent
measured false-alarm rate. On DTV-occupied channels a real pilot at a nonzero
frequency offset can elevate the fraction; the stored fine spectrum resolves
which.
Stored fields are defined in `PRODUCT_SCHEMA.md`.

### Decision status

The repository currently contains two decision implementations with different
operational status:

- Current CHIME archive products store `reject_mask` from the exact,
  norm-corrected coarse positive-excess comparison `F > null_power_ratio`.
- The floating-point fine reduction is a measurement, not a decision. The scan
  stores only the exact `fine_power_u64` terms; the fine-bin statistic, the
  robust per-frame null-bulk threshold, and any threshold exceedances are
  recomputed from those terms in post-processing and do not replace
  `reject_mask`.
- Kernel core 2.3.0 implements the fixed-point designated-set order-statistic
  CFAR decision and has bit-equality tests against the Python reference. Its
  per-channel runtime-bundle calibration remains `pending_campaign`; a pending
  profile must not enable that rejection path.

Accordingly, “implemented,” “validated,” and “active decision” are separate
claims. A kernel entry point is not an active observing policy until its
calibration status is complete and the receiver integration selects it.

## Dynamic range and frequency capture

The mask remains useful after the reported dB quantities have reached their
effective ceilings. We independently recomputed the following bounds from
the shipped packed weights and a pure-tone response:

- **Int4 weight crosstalk.** Pilot power entering a reference term through
  the quantized weights limits the reported F-statistic to `+39.055` to
  `+62.136` dB across ATSC channels 14--36. The corresponding individual
  reference-term crosstalk is `-68.244` to `-36.730` dB relative to the
  target term and depends on channel.
- **Pilot-frequency offset.** A transmitter offset `df` from the nominal
  pilot places power in the references at exactly `+/-2` fine bins. At
  `|df| = 300 Hz`, this effect limits the reported F-statistic to about
  `+25` to `+28` dB for the shipped weights. The exact value depends on the
  channel and the sign of the offset and is independent of additional pilot
  power.

Both ceilings remain well above `null_power_ratio`. Therefore, they do not change the
positive-excess decision for a strong pilot. They do mean that
`pilot_excess_db` and `estimated_data_shelf_snr_db` should be read as lower bounds for sufficiently
strong, offset transmitters.

For a rectangular `K`-sample window, the target-term capture factor is
`sinc^2(df / 3051.7578125 Hz)`, where
`sinc(x) = sin(pi*x)/(pi*x)`. Independent evaluation gives losses of `0.1385`
dB at `|df| = 300 Hz` and `0.3870` dB at `|df| = 500 Hz`. Apply the factor at
each measured offset. Whether that correction is negligible depends on the
retained calibration-uncertainty budget and should not be assumed here.

## Reference-placement resolver

The resolver starts with `reference_offset_bins = 2` and changes placement
only when the receiver geometry requires it:

- a valid requested reference remains at its requested offset;
- a reference requested past the baseband-frame origin (nu = 0) wraps
  modulo 1 in the periodic coarse-channel FFT coordinate (recorded as
  `edge_wrapped`; under a center-at-DC profile nu = 0 is the coarse-channel
  center/DC, not the channel edge, and the manifest's
  `frame_origin_description` states which);
- a reference that would land on the forbidden coarse-channel DC tone moves
  away from that tone; and
- a target/DC collision fails because the target signal cannot be moved.

A reference requested beyond +-fs/2 of the channel center is not moved; it
aliases to the opposite channel edge and is reported by the manifest's
`*_crosses_channel_edge` fields because it samples the channelizer roll-off.

The manifest records the selected placement. In the shipped ATSC 14--36 bank
(rebuilt for the verified DC-centered frame convention), DTV 14 is the only
adaptive case: its pilot sits one fine bin above the coarse-channel center, so
its lower reference lands one bin below DC with the forbidden DC tone in the
skipped guard between them. No shipped channel collides with, or shifts
around, the forbidden DC tone. DTV 21's upper reference crosses the physical
coarse-channel half-width and is reported as such.

## Supported `K` family

`K = 64` and `K = 128` are both implemented, compiled, and regression-tested.
The CHIME profile and bank select `K = 128`; the full and pathfinder CHORD
profiles and bank select `K = 64`. Both geometries use 128 windows per stream
and therefore the same frozen 256-bin padded fine transform.

`K = 256` is not a supported detector configuration. It would require a new
fixed-point bound, compiled kernel geometry, weight bank, receiver contract,
and validation campaign.
