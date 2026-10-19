# Fine-reduction product fields

These fields extend the current per-pilot product described in
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md). They are measurements and diagnostics;
the product's active rejection decision remains the coarse norm-corrected
positive-excess rule recorded by `decision_contract_json`. Since schema v6
(kernel core 3.0.0) that rule is formed from the all-bin sums of these fine
powers, so the fine terms are mandatory for every in-band pilot.

## Product-wide fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_status` | str | `enabled` for every in-band pilot, `pending` for an empty checkpoint, or `not_applicable_pilot_out_of_band` (zero frame powers, NaN ratios, `valid = 0`). There is no disabled or degraded status: the frame powers are the fine-power sums, so a scan without fine powers has no frame powers and fails. `fine_products` accepts only `on`. |
| `fine_num_bins` | int64 | Padded fine-bin count; zero when no fine terms exist. |
| `fine_pad_factor` | int64 | Zero-padding factor of the window-axis transform. |
| `fine_designated_bins` | int64[] | Designated transmitter bins used by the diagnostic analysis. |
| `fine_census_excluded_bins` | int64[] | Fine bins excluded from the diagnostic null bulk. |
| `fine_guard_fine_bins` | int64 | Guard margin around designated bins. |
| `fine_p_fa` | float64 | Requested diagnostic tail probability. |

## Per-frame fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_power_u64` | uint64 `(N, 3, fine_num_bins)` when enabled; `(N, 0, 0)` for a pilot out of band | Exact deployed-statistic power terms per fine bin — target, lower reference, upper reference — from the frozen fixed-point transform. Their all-bin sums are the frame powers `p_target_u64`, `p_ref_lower_u64`, `p_ref_upper_u64`. |

That row is the whole per-frame fine product: apart from the frame powers
(their exact all-bin sums, stored under the coarse field names and checked
against them on every write and read), the scan stores nothing derived from
the fine terms. The float power ratio, the null-bulk location,
scale, threshold and threshold-estimation mode, the null-bulk exceedance
fraction, and the ragged exceedance list are not written by the scan. They are
recomputed in post-processing from `fine_power_u64`, at whatever operating point
is calibrated there; the scan applies no fine decision of its own. Earlier
schema revisions stored those derived arrays, and `fine_power_ratio` still
appears under that spelling when reading the archived survey products.

The current contract fails closed on status/shape disagreement. An enabled
product has a positive bin count and all three terms for every frame. A pending
status is valid only for an empty checkpoint. An out-of-band pilot has
`fine_num_bins = 0` and shape `(N, 0, 0)`; zeros are never substituted for
missing measurements.

One caveat carries over to the recomputation. The null-bulk exceedance fraction
is an in-sample fraction of the same independent null-bulk bins used to estimate
the frame threshold, so it is not an independently measured false-alarm rate.

## Frame powers from the fine terms

For `fine_status = enabled`, every frame satisfies, in exact integer
arithmetic,

```text
p_r_u64[:, 0] == fine_power_u64[:, r, :].sum(axis=1)      r = target, lower, upper
p_ref_sum_u64 == p_ref_lower_u64 + p_ref_upper_u64
```

The writer (`archive/detector.py`), `validate_current_product_identity`,
`products/reader.py` and `tools/audit_per_pilot.py` all enforce it, and every
derived coarse column must equal `coarse_columns_from_fine(product)`. This
invariant replaces the row-domain marginal identity of kernel cores 2.x, under
which the integer sum of squared row projections reproduced a separately
computed coarse power. That branch is removed: the coarse and fine
measurements now come from one spectrum, and the diagnostic fine threshold is
still not the active rejection policy. The padded fixed-point transform itself
remains gated by its golden vectors and its Parseval drift bound
(`tests/core/test_fxfft256.py`); it does not conserve energy exactly, so the
frame powers are not `256 * sum_m |z|^2`.
