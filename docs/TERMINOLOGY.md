# Statistical names and compatibility fields

These names describe the current paper calculations. Symbols are local to the
equations below; legacy serialized names are explicitly identified.

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

## Is this an F statistic?

A suitably normalized ratio of independent Gaussian power sums can have an
F distribution. The real detector's unequal reference norms, correlated inputs,
fixed-point transform and rank normalization do not establish that null law.
Call the measured quantities **power ratios** and the detection quantity the
**rank-based fine score**. Ideal F models in validation are conditional models,
not production false-alarm calibrations.

## Native interface migration

Package 2 introduced the native rename; package 3 removes the remaining
misleading Python aliases. Core 4.0.0 uses these descriptive names:

| Previous interface | Current interface |
|---|---|
| `FStatKernel` | `PilotProxyKernel` |
| `FStat_*`, `FStatHandle` | `PilotProxy_*`, `PilotProxyHandle` |
| `FSTAT_*` build macros | `PILOTPROXY_*` |
| `libfstatistic.so` | `libpilotproxy.so` |
| `cuda/f_statistic.{h,cu}` | `cuda/pilotproxy_detector.{h,cu}` |
| `cuda/f_statistic_reference.{h,cpp}` | `cuda/pilotproxy_reference.{h,cpp}` |
| `fstat_ref_*` | `pilotproxy_ref_*` |

Rebuild with `make build-kernel`, update imports and native consumers, and
refresh kotekan's vendor pin. Old library symbols and Python class names
are not aliases. The wrapper rejects another core major. Integer arithmetic,
weights, frame-power schema v6 and threshold comparison are unchanged.
The `FSTATWGT1` magic bytes still identify the existing weight-file format;
changing a label does not require rewriting these binary data or their hashes.
Historical evidence retains the filenames and versions actually used.

New injection-recovery reports use `power_ratio` in schema v2, and new
result summaries use `power_ratio_histogram_png` in schema v2. Dated reports
are unchanged; read their declared schema before consuming those fields.
Clean-IQ metadata uses `pilotproxy_atsc_clean_iq_v1` in place of the former
`fstat_atsc_clean_iq_v1` token.

## Averaging and coherence

The frame proxy keeps negative excesses. Only the retained **mean** is floored
at zero; flooring individual frames was a development method and is not the
paper's estimator. `x_hi = x + 2 se` is a working upper end, not a certified
simultaneous confidence bound.

`gamma` includes spatial phase cancellation in calibrated redundant stacking
and the archive-to-capture transfer. It is not the fraction of RFI that is
intrinsically coherent. `G` is not a measured lifetime or a universal lower
bound. Its family minimum does not qualify a cross-day threshold.

## Names retained by existing files and callers

| Legacy name | Meaning / current name |
|---|---|
| `coherent_fraction.json`, `coherent_fraction*` keys | Stack-to-proxy amplitude ratio `gamma` |
| Removed `COHERENT_FRACTION*` Python imports | Use `STACK_AMPLITUDE_RATIO*` |
| `coherent_gain.json`, `coherent_gain_by_band` | Assumed effective temporal gain |
| Removed `coherent_gain()`, `admissible_min_gain()` | Use `temporal_gain()`, `minimum_family_gain()` |
| Removed `strength(..., c=...)` keyword | Use `strength(..., stack_amplitude_ratio=...)` for `gamma` |
| `deployed*` decision fields | Assigned conditional thresholds; not installed hardware |
| `finite_budget` decision field | Combined nominal-strength cap, including transfer allowances |
| `coherence_limit_s` | Full-era effective-gain-time diagnostic; not a physical phase lifetime or joint guard limit |
| `fine_score_z63` | Rank-63 fine score for the declared export geometry |
| Cache knee rule v1 | Authentication of the saved cache; current decisions use v2 |

Schema-v5 archives are supported through their stored fine spectra; their old
row-domain powers are not the current frame powers. Other frozen records retain
their original schemas and meanings. Current readers accept only documented
versions, not arbitrary old layouts. See [PRODUCT_SCHEMA.md](PRODUCT_SCHEMA.md).

RFIsher names its fitted damping **variance** `s_NL`; `sigma_nl` denotes the
physical damping length. Its `_Pres` row stores a residual bias projection
and must be removed before inverting the fitted Fisher block. Frozen older
matrices labeled the variance derivative `sigma_NL`; those records retain
their recorded schema.
