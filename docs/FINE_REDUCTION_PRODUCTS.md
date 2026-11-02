# Fine-reduction product fields

These fields extend the current per-pilot product described in
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md). They are measurements and
diagnostics. The active rejection decision of the product is the coarse
norm-corrected positive-excess rule recorded by `decision_contract_json`. In
schema v6 (kernel core 3.0.0), that rule is formed from the all-bin sums of
these fine powers, and the fine terms are therefore mandatory for every
in-band pilot.

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
| `fine_power_u64` | uint64 `(N, 3, fine_num_bins)` when enabled; `(N, 0, 0)` for a pilot out of band | Exact deployed-statistic power terms per fine bin (target, lower reference, and upper reference) from the frozen fixed-point transform. Their all-bin sums are the frame powers `p_target_u64`, `p_ref_lower_u64`, `p_ref_upper_u64`. |

That row is the whole per-frame fine product. Its exact all-bin sums are stored
as the frame powers under the coarse field names, and the two are checked
against each other on every write and read. Nothing else derived from the fine
terms is stored by the scan. The float power ratio, the null-bulk location, scale, threshold, and
threshold-estimation mode, the null-bulk exceedance fraction, and the ragged
exceedance list are recomputed in post-processing from `fine_power_u64`, at the
operating point calibrated there. The scan applies no fine decision of its own.
When the archived survey products are read, the ratio is exposed under the
spelling `fine_power_ratio`.

The contract fails closed when the status and the shape disagree. An enabled
product has a positive bin count and all three terms for every frame. A pending
status is valid only for an empty checkpoint. An out-of-band pilot has
`fine_num_bins = 0` and shape `(N, 0, 0)`. Zeros are never substituted for
missing measurements.

The recomputed null-bulk exceedance fraction is an in-sample fraction of the
same independent null-bulk bins used to estimate the frame threshold, and so it
is not an independently measured false-alarm rate.

## Frame powers from the fine terms

For `fine_status = enabled`, every frame satisfies, in exact integer
arithmetic,

```text
p_r_u64[:, 0] == fine_power_u64[:, r, :].sum(axis=1)      r = target, lower, upper
p_ref_sum_u64 == p_ref_lower_u64 + p_ref_upper_u64
```

This identity is enforced by the writer (`archive/detector.py`),
`validate_current_product_identity`, `products/reader.py`, and
`tools/audit_per_pilot.py`, and every derived coarse column must equal
`coarse_columns_from_fine(product)`. The coarse and fine measurements come
from one spectrum. The active rejection policy is the coarse rule, and the fine
threshold is a diagnostic. The padded fixed-point transform is gated by its
golden vectors and its Parseval drift bound (`tests/core/test_fxfft256.py`).
Since the rounded transform does not conserve energy exactly, the frame powers
differ from `256 * sum_m |z|^2`.
