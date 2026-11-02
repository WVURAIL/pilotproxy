# Statistical names and compatibility fields

This page defines the names used in the paper calculations. Symbols are local
to the equations below, and serialized names that differ are listed.

| Quantity | Definition | Current implementation |
|---|---|---|
| Fine power `S_r[f]` | Fixed-point FFT power summed over inputs | `fine_power_u64` |
| Frame power `P_r` | Sum over every fine bin | `product_contract.frame_powers_from_fine` |
| Frame ratio `F` | `2 P_0 / (P_1 + P_2)` | `coarse_power_ratio` |
| Flat-noise normalization `mu_0` | Target/reference integer weight-norm ratio | `null_power_ratio_from_weight_norms` |
| Normalized ratio `Q` | `F / mu_0` | `normalized_coarse_power_ratio` |
| Signed pilot excess `e` | `Q - 1` | `normalized_pilot_excess` |
| Fine ratio `T[f]` | `2 S_0[f] / (S_1[f] + S_2[f])` | Reconstructed from exact fine powers |
| Detection score `w` | Maximum marker ratio divided by bulk order statistic | `scores.required_eta_q16_by_rank` and `fine_decision` |
| Rank `rho` | One-based bulk order statistic | `rho`; C ABI `cfar_rank` is zero-based |
| Threshold `eta` | Dimensionless limit on `w` | `eta_q16 / 65536` |
| Retention `kappa` | Equal-event probability kept | `threshold_rule.sweep` |
| Shelf proxy `x` | Signed retained mean excess times `C_shelf`, floored after averaging | Sweep `x`; uncertainty in `se` |
| Bias ratio `r` | `a abs(R_DV)` | `bias_over_sigma` |

## Power ratios and the F distribution

A suitably normalized ratio of independent Gaussian power sums can follow an
F distribution. In the real detector, however, the reference norms are unequal,
the inputs are correlated, the transform is fixed point, and the score is
normalized by a rank. With these properties, the F null law is no longer
guaranteed. The measured quantities are therefore called *power ratios*, and the
detection
quantity is called the *rank-based fine score*. The ideal F models used in
validation hold only under their stated conditions, and they do not calibrate
the production false-alarm rate.

## Native names before core 4

Detector core 4 and package 1.1.0 renamed the native interface. Consumers
built against an earlier core, such as a Kotekan vendor pin, map as follows:

| Before core 4 | Core 4 |
|---|---|
| `FStatKernel` | `PilotProxyKernel` |
| `FStat_*`, `FStatHandle` | `PilotProxy_*`, `PilotProxyHandle` |
| `FSTAT_*` build macros | `PILOTPROXY_*` |
| `libfstatistic.so` | `libpilotproxy.so` |
| `cuda/f_statistic.{h,cu}` | `cuda/pilotproxy_detector.{h,cu}` |
| `cuda/f_statistic_reference.{h,cpp}` | `cuda/pilotproxy_reference.{h,cpp}` |
| `fstat_ref_*` | `pilotproxy_ref_*` |

A consumer moves to core 4 by rebuilding with `make build-kernel`, updating its
imports and native calls, and refreshing its vendor pin. The earlier names are
not kept as aliases, and the wrapper rejects a library of another core major.
The integer arithmetic, the weights, frame-power schema v6, and the threshold
test are the same in both cores. The weight-file format is marked by the
`FSTATWGT1` magic bytes, which are unchanged, so the binary banks and their
hashes are also unchanged. Frozen evidence keeps the filenames and versions it
was made with.

Injection-recovery reports use `power_ratio` (schema v2), and result summaries
use `power_ratio_histogram_png` (schema v2). A consumer reads the declared
schema of a report before it uses those fields. Clean-IQ metadata uses
`pilotproxy_atsc_clean_iq_v1`, and frozen records may carry
`fstat_atsc_clean_iq_v1`.

## Averaging and uncertainty

The frame proxy keeps negative excesses. The floor at zero is applied to the
retained mean only, and individual frames are never floored. The value
`x_hi = x + 2 se` is a working upper end, and its simultaneous confidence is not
certified.


## Names retained by existing files and callers

| Name in files and callers | Meaning or replacement |
|---|---|
| `deployed*` decision fields | Assigned conditional thresholds; not installed hardware |
| `fine_score_z63` | Rank-63 fine score for the declared export geometry |

Schema-v5 archives are read through their stored fine spectra, because the
row-domain powers stored in v5 differ from the frame powers. Other frozen
records keep their schemas and meanings. Readers accept only the documented
versions (see [PRODUCT_SCHEMA.md](PRODUCT_SCHEMA.md)).

In RFIsher, the fitted damping variance is denoted `s_NL`, and `sigma_nl`
denotes the physical damping length. The `_Pres` row stores a residual bias
projection and must be removed before the fitted Fisher block is inverted. Frozen
matrices that label the variance derivative `sigma_NL` keep their recorded
schema.

## Power and threshold units

The shelf proxy x is dimensionless. The simple model assigns temperature
Tsys*x and distributes its squared amplitude uniformly over the supported
forecast modes. The channel table supplies power_scale in
a = power_scale*x**2, with a in mK^2. This is a model convention, not a
measured sky calibration. RFIsher schema-v3 response_dv is per mK^2.

The selected word maximizes retention subject to both the upper first-order
bias ratio being at most one and the upper additive power being no greater
than power_limit. That required per-channel limit comes from a finite-model
check; it has no default and must not be copied from an unchecked linear
one-sigma tolerance. The CLI accepts the limits as a JSON object through
--power-limits, mapping channel numbers to positive powers in mK^2.

These are conditional, in-sample selections. The conversion's range and
uncertainty and an independent evaluation of the frozen configuration are
needed before reporting a physical operating result.
