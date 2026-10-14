# Fine-reduction product fields

These fields extend the current per-pilot product described in
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md). They are measurements and diagnostics;
the product's active rejection decision remains the coarse norm-corrected
positive-excess rule recorded by `decision_contract_json`.

## Product-wide fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_status` | str | `enabled`, `pending` for an empty checkpoint, `disabled_by_option`, `not_applicable_pilot_out_of_band`, or an explicit capability-failure status. `fine_products=on` fails instead of degrading. |
| `fine_num_bins` | int64 | Padded fine-bin count; zero when no fine terms exist. |
| `fine_pad_factor` | int64 | Zero-padding factor of the window-axis transform. |
| `fine_designated_bins` | int64[] | Designated transmitter bins used by the diagnostic analysis. |
| `fine_census_excluded_bins` | int64[] | Fine bins excluded from the diagnostic null bulk. |
| `fine_guard_fine_bins` | int64 | Guard margin around designated bins. |
| `fine_p_fa` | float64 | Requested diagnostic tail probability. |

## Per-frame fields

| Field | Type | Meaning |
| --- | --- | --- |
| `fine_power_u64` | uint64 `(N, 3, fine_num_bins)` when enabled; `(N, 0, 0)` otherwise | Exact deployed-statistic power terms per fine bin — target, lower reference, upper reference — from the frozen fixed-point transform. |

That row is the whole per-frame fine product: the scan stores the exact terms
and nothing derived from them. The float power ratio, the null-bulk location,
scale, threshold and threshold-estimation mode, the null-bulk exceedance
fraction, and the ragged exceedance list are not written by the scan. They are
recomputed in post-processing from `fine_power_u64`, at whatever operating point
is calibrated there; the scan applies no fine decision of its own. Earlier
schema revisions stored those derived arrays, and `fine_power_ratio` still
appears under that spelling when reading the archived survey products.

The current contract fails closed on status/shape disagreement. An enabled
product has a positive bin count and all three terms for every frame. A pending
status is valid only for an empty checkpoint. Every other no-measurement status
has `fine_num_bins = 0` and shape `(N, 0, 0)`; zeros are never substituted for
missing measurements.

One caveat carries over to the recomputation. The null-bulk exceedance fraction
is an in-sample fraction of the same independent null-bulk bins used to estimate
the frame threshold, so it is not an independently measured false-alarm rate.

The exact integer sum of squared matched-filter row projections reproduces the
coarse power terms bit-for-bit. The padded fixed-point transform is separately
gated by its Parseval identity and golden vectors. These identities connect the
coarse and fine measurements without making the diagnostic fine threshold the
active rejection policy.
