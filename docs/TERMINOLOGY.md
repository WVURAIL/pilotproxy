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
| Amplitude ratio `gamma` | Calibrated redundant-stack amplitude / archive shelf proxy | `STACK_AMPLITUDE_RATIO` |
| Effective temporal gain `G` | Assumed `G_int / mean_rho` in two stationary families | `temporal_gain`, `minimum_family_gain` |
| Forecast strength `a` | `(gamma x_hi)^2 N G` | `threshold_rule.strength` |
| Bias ratio `r` | `a abs(R_DV)` | `bias_over_sigma` |
| Calendar cost | `calendar_factor / kappa`, factor 1 by default | `decide`, `decide_guarded` |

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

## Averaging and coherence

The frame proxy keeps negative excesses. The floor at zero is applied to the
retained mean only, and individual frames are never floored. The value
`x_hi = x + 2 se` is a working upper end, and its simultaneous confidence is not
certified (see [RETAINED_PROXY_CONFIDENCE_BAND.md](RETAINED_PROXY_CONFIDENCE_BAND.md)).

`gamma` covers both the spatial phase cancellation of calibrated redundant
stacking and the transfer from the archive to the short captures. It therefore
differs from the fraction of RFI that is intrinsically coherent. `G` is an
assumed budget. It is neither a measured lifetime nor a universal lower bound,
and its minimum over the model families does not qualify a cross-day threshold.

## Names retained by existing files and callers

| Name in files and callers | Meaning or replacement |
|---|---|
| `coherent_fraction.json`, `coherent_fraction*` keys | Stack-to-proxy amplitude ratio `gamma` |
| Removed `COHERENT_FRACTION*` Python imports | Use `STACK_AMPLITUDE_RATIO*` |
| `coherent_gain.json`, `coherent_gain_by_band` | Assumed effective temporal gain |
| Removed `coherent_gain()`, `admissible_min_gain()` | Use `temporal_gain()`, `minimum_family_gain()` |
| Removed `strength(..., c=...)` keyword | Use `strength(..., stack_amplitude_ratio=...)` for `gamma` |
| `deployed*` decision fields | Assigned conditional thresholds. The name does not mean installed hardware |
| `finite_budget` decision field | Total nominal-strength cap, with the transfer allowances added |
| `coherence_limit_s` | Full-era effective-gain-time diagnostic. It is neither a physical phase lifetime nor a joint guard limit |
| `fine_score_z63` | Rank-63 fine score for the declared export geometry |
| Cache knee rule v1 | Authentication of the saved cache; current decisions use v2 |

Schema-v5 archives are read through their stored fine spectra, because the
row-domain powers stored in v5 differ from the frame powers. Other frozen
records keep their schemas and meanings. Readers accept only the documented
versions (see [PRODUCT_SCHEMA.md](PRODUCT_SCHEMA.md)).

In RFIsher, the fitted damping variance is denoted `s_NL`, and `sigma_nl`
denotes the physical damping length. The `_Pres` row stores a residual bias
projection and must be removed before the fitted Fisher block is inverted. Frozen
matrices that label the variance derivative `sigma_NL` keep their recorded
schema.
