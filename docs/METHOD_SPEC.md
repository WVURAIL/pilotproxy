# PilotProxy method specification

This specification fixes the detector geometry and the statistic used by the
CHIME real-data workflow. We state the operations in the order they are
applied so that a result can be reproduced from the recorded powers and
weight norms.

The production decision is the rank-based fine-bin comparison described below.
The frame ratio and the positive-excess flag are measurements and diagnostics.
The weight norms set a normalization by the ratio of expected powers, which
differs from the exact expectation of a noisy ratio.

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

First, the exact integer matched-filter row projections `z[n, m]` are formed
for each weight term `n` (target, lower reference, and upper reference) and
detector row `m`. Then the exact fine power spectrum `S[n, f]` of each term is
formed. It is the frozen fixed-point 256-point transform (`fxfft256_v1`) of the
128 projections of each stream, squared and summed over streams (see
"Time-coherent fine reduction" below). The three frame powers are defined from
that spectrum:

```text
P_target    = sum_f S[target, f]
P_ref_lower = sum_f S[ref_lower, f]
P_ref_upper = sum_f S[ref_upper, f]
P_ref_sum   = P_ref_lower + P_ref_upper
```

as exact `uint64` sums over the 256 fine bins. Every power is summed over
streams, windows, and bins before a ratio is formed. An average of per-row or
per-bin power ratios would give a different statistic from the specified
detector.

In kernel cores before 3.0.0, the frame powers were a second sum, of
`|z[n, m]|^2` over detector rows. The schema v5 products of record carry that
row-domain sum in `p_*_u64`, and readers replace it by the sums of the stored
`fine_power_u64` (see `PRODUCT_SCHEMA.md`). With an exact transform of length
`L_F = 256`, the two definitions would differ by the factor `L_F` in every
term, and the factor would cancel in every ratio. The rounded fixed-point
transform, however, does not conserve energy exactly, so the two definitions
agree only up to that rounding. At deployed amplitudes, the archive frames
differ by a median relative `2.75e-6` (p99 `1.65e-5`), and near-empty
(small-amplitude) frames differ by up to about 2 percent.

## Frame power ratio

The detector compares the target frame power with the mean of the two
reference frame powers:

```text
F = 2 * P_target / P_ref_sum
```

Equivalently,

```text
F = P_target / mean(P_ref_lower, P_ref_upper)
```

where the lower and upper powers are the frame powers defined above.
The uncorrected one-bin pilot excess is

```text
rho = F - 1
```

When `P_ref_sum = 0`, the ratio is undefined. Products store `valid = 0` and
NaN for the diagnostic ratios and dB transforms, and the binary decision forces
`reject_mask = 0` on its own path. A numerical zero is never used in place of
an invalid measurement.

## Diagnostic positive-excess mask

The archive diagnostic positive-excess path places its fixed mask threshold at the detector's own
null-hypothesis (`H0`) zero point. Under `H0`, the local spectrum contains no
pilot excess and is treated as flat across the three detector terms. Because
the steering vectors are quantized to int4, their squared norms are not
exactly equal. Therefore,

```text
E[P_term] = sigma^2 * ||w_term||^2
2 E[P_target] / E[P_ref_sum] = null_power_ratio = 2 * target_norm_sq / reference_norm_sum_sq
```

Here, `target_norm_sq = ||w_target||^2` and
`reference_norm_sum_sq = ||w_ref_lower||^2 + ||w_ref_upper||^2`. These are exact
integer norms computed from the packed weights. By independent recomputation
from the shipped manifest for ATSC channels 14 to 36, we find that
`null_power_ratio` ranges from 0.9853298815 to 1.0111111111. A common
threshold at `F = 1` would therefore place different channels on different
sides of their own `H0` zero points.

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
the plain `F > 1` rule, `p_target > (p_ref_sum >> 1)`. Products made under that
rule record it in `mask_rule`.

The corrected pilot excess is

```text
rho_corrected = F / null_power_ratio - 1
```

and is stored per frame as `normalized_pilot_excess`. The comparison is
applied on the host in exact integer arithmetic from the uint64 frame powers.
The runtime bundle records it as the rational half-threshold
`half_num : half_den = target_norm_sq : reference_norm_sum_sq`. It is applied
on the host because `P_target * half_den` does not fit `uint64` at the
fine-sum scale.

Masked frames are dropped from the before and after averages and are never
replaced with zeros.

## Time-coherent fine reduction

The front end, present since kernel core 2.0.0, is an exact int32 complex row
sum. For each weight term `n` and detector row `m`, stage 1 returns

```text
z[n, m] = sum_k w_term[n, k] * x[m, k]        (exact integer complex)
```

In stage 2, the rows of each term are reshaped to
`[num_streams, windows_per_stream]`, the frozen fixed-point transform
`fxfft256_v1` is applied along the window axis with `pad = 2` zero padding, and
the result is summed incoherently over streams:

```text
S[n, b] = sum_streams |fxfft256(z[n, stream, :])[b]|^2      (exact uint64)
F2[b]   = 2 * S_target[b] / (S_ref_lower[b] + S_ref_upper[b])
P[n]    = sum_b S[n, b]                                      (frame power)
```

The reduction is coherent in time over one frame and incoherent over streams
and frames. The frame powers of the previous sections are the all-bin sums of
the same `S`, so the coarse ratio and the fine statistic come from one
spectrum, with no second power path. For the fused entry points, `P[n]` is
reduced on the device, and for the staged `PilotProxy_Compute_FinePowers_U64`
output it is summed on the host. Both must equal
`fxfft.fine_power_fx(z).sum(axis=-1)` exactly, and this equality is the gate
(`tests/kernel`, and the per-frame product invariant in `PRODUCT_SCHEMA.md`).

The float prototype `fine_reduction.fine_reduce` uses `numpy.fft` or
`cupy.fft` instead of the fixed-point transform. Its float Parseval identity
`sum_b S[n, b] = pad * windows * sum_m |z[n, m]|^2` is a property of that
float prototype only, gated at ULP tolerance in its own tests.

Detection over the fine bins uses a pre-registered null-bulk constant
false-alarm rate (CFAR) test. The null is calibrated on the independent
non-designated bins, which are every `pad`-th bin with the designated
transmitter bins, their guard margin, and any census-excluded bins removed. The
location is the median, and the scale is the left-side spread (the median minus
the 15.87th percentile). When that bulk is itself contaminated (flag fraction
above 0.2), the calibration falls back to lower quantiles (location = P25,
scale = (P25 - P2.275)/2) and records the mode.

Detections are declared on all bins against this calibration (the any-bin
rule), which removes the census dependence of rules that test only the
designated bins. The flag rate over the non-designated independent bins is
returned as `null_bulk_exceedance_fraction` on the `CfarCalibration` from
`calibrate_cfar`. This rate is not stored in the per-pilot product, which keeps
only the exact `fine_power_u64` terms it is recomputed from. Since the same
bins fit the threshold, the rate is an in-sample threshold diagnostic and
gives no independent measure of the false-alarm rate. On DTV-occupied
channels, a real pilot at a nonzero frequency offset can raise the fraction,
and the stored fine spectrum shows which case applies. The stored fields are
defined in `PRODUCT_SCHEMA.md`.

### Decision status

The decision paths in the repository have different operational status:

- CHIME archive products store `reject_mask` from the exact, norm-corrected
  coarse positive-excess comparison `F > null_power_ratio` on the frame powers.
  In schema v6, these are the fine-power sums. The stored mask of a schema v5
  product used the row-domain powers, and readers recompute it from the fine
  sums.
- The floating-point fine reduction is a measurement and makes no decision. The
  scan stores only the exact `fine_power_u64` terms. The fine-bin statistic,
  the robust per-frame null-bulk threshold, and any threshold exceedances are
  recomputed from those terms in post-processing and leave `reject_mask`
  unchanged.
- Kernel core 4.0.0 implements the fixed-point designated-set order-statistic
  CFAR decision, with bit-equality tests against the Python reference. Its
  per-channel runtime-bundle calibration is still `pending_campaign`, and a
  pending profile must never enable that rejection path.

Thus, a decision path can be implemented and validated and still be inactive.
A kernel entry point becomes an active observing policy only when its
calibration status is complete and the receiver integration selects it.

## Dynamic range and frequency capture

The mask remains useful after the reported dB quantities have reached their
effective ceilings. We independently recomputed the following bounds from
the shipped packed weights and a pure-tone response.

Pilot power that enters a reference term through the quantized int4 weights,
known as weight crosstalk, limits the reported local-reference power ratio to
between `+39.055` and `+62.136` dB over ATSC channels 14 to 36. The crosstalk of
an individual reference term is between `-68.244` and `-36.730` dB relative to
the target term and depends on the channel.

A transmitter offset `df` from the nominal pilot places power in the
references at exactly `+/-2` fine bins. At `|df| = 300 Hz`, this effect limits
the reported local-reference power ratio to about `+25` to `+28` dB for the
shipped weights. The exact value depends on the channel and the sign of the
offset and is independent of additional pilot power.

Both ceilings are far above `null_power_ratio`, so they do not change the
positive-excess decision for a strong pilot. For sufficiently strong, offset
transmitters, however, `pilot_excess_db` and `estimated_data_shelf_snr_db`
should be read as lower bounds.

For a rectangular `K`-sample window, the target-term capture factor is
`sinc^2(df / 3051.7578125 Hz)`, where
`sinc(x) = sin(pi*x)/(pi*x)`. By independent evaluation, we find losses of
`0.1385` dB at `|df| = 300 Hz` and `0.3870` dB at `|df| = 500 Hz`. The factor
must be applied at each measured offset. The size of this correction relative to the
retained calibration-uncertainty budget is not assumed here.

## Reference-placement resolver

The resolver starts with `reference_offset_bins = 2` and changes placement
only when the receiver geometry requires it:

- A valid requested reference stays at its requested offset.
- A reference requested past the baseband-frame origin (nu = 0) wraps modulo 1
  in the periodic coarse-channel FFT coordinate and is recorded as
  `edge_wrapped`. Under a center-at-DC profile, nu = 0 is the coarse-channel
  center (DC) and not the channel edge, and the `frame_origin_description` of
  the manifest states which convention applies.
- A reference that would land on the forbidden coarse-channel DC tone moves
  away from that tone.
- A target and DC collision fails, because the target signal cannot be moved.

A reference requested more than fs/2 from the channel center is not moved. It
aliases to the opposite channel edge and is reported in the
`*_crosses_channel_edge` fields of the manifest, because it samples the
channelizer roll-off.

The manifest records the selected placement. In the shipped bank for ATSC
channels 14 to 36, built for the verified DC-centered frame convention, DTV 14
is the only adaptive case. Its pilot is one fine bin above the coarse-channel
center, so its lower reference lands one bin below DC, with the forbidden DC
tone in the skipped guard between them. No shipped channel collides with, or
shifts around, the forbidden DC tone. The upper reference of DTV 21 crosses the
physical coarse-channel half-width and is reported as crossing.

## Supported `K` family

`K = 64` and `K = 128` are both implemented, compiled, and regression-tested.
The CHIME profile and bank select `K = 128`, and the full and Pathfinder CHORD
profiles and bank select `K = 64`. Both geometries use 128 windows per stream
and therefore the same frozen 256-bin padded fine transform.

`K = 256` is not a supported detector configuration. It would require a new
fixed-point bound, compiled kernel geometry, weight bank, receiver contract,
and validation campaign.
