# Per-pilot product schema

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 6
schema_version = "pilotproxy_per_pilot_product_v6"
source_event_key_schema_version = "pilotproxy_namespaced_source_event_key_v1"
detector_contract = "pilotproxy_detector_contract_v2"
```

Schema v6 is the product written by kernel core 3.0.0. It keeps every field
name of v5 and redefines the frame powers: `p_target_u64`, `p_ref_lower_u64`
and `p_ref_upper_u64` are the exact all-bin sums of `fine_power_u64`
(`P_r = sum_f S_r[f]`), `p_ref_sum_u64 = lower + upper`, and the derived
columns (`coarse_power_ratio`, `normalized_coarse_power_ratio_db`,
`normalized_pilot_excess`, `pilot_excess_db`, `estimated_data_shelf_snr_db`,
`reject_mask`, `valid`) are formed from those sums exactly as before. The
exact invariant, enforced by the writer, `validate_current_product_identity`,
`products/reader.py` and `tools/audit_per_pilot.py`, is

```text
fine_status == "enabled":  p_r_u64[:, 0] == fine_power_u64[:, r, :].sum(axis=1)   (exact integers)
                           every derived column == coarse_columns_from_fine(product)
```

(integers and flags exactly; floats to `rtol = 4e-13`, `atol = 1e-12`, NaN in
the same places). Fine powers are mandatory for every in-band pilot
(`fine_status = enabled`, or `pending` for an empty checkpoint); the only other
status is `not_applicable_pilot_out_of_band`, with zero powers, NaN ratios,
`valid = 0`, shape `(N, 0, 0)` and `fine_num_bins = 0`. `fine_products=off` and
`auto` are refused.

## Schema v5: the products of record

The archive campaign of record (`chime_pilots_rebuild_20260829`, kernel core
2.3.0) and the channel 37 control (`ch37_control_491_20260924`) are schema v5
products with detector contract revision 1. In them `p_*_u64` are the retired
row-domain powers `sum_m |z[n, m]|^2`. They are never edited and never
migrated or copied. Every v5 product of record carries `fine_power_u64` for
every frame, so the current definition is computable on read:

- `products/reader.py` and `products/acceptance.py` open v5 (detector contract
  v1) and v6 (contract v2). A v5 view replaces the stored row-domain columns
  by `coarse_columns_from_fine(product)` and records
  `frame_power_definition = "recomputed_from_fine_power_u64 (v5 stored
  row-domain powers ignored)"`; a v6 view asserts that the stored columns
  equal them. A cohort mixing v5 and v6 products is refused.
- `product_contract.frame_powers_from_fine`,
  `pilot_excess_from_fine_powers` and `coarse_columns_from_fine` are the one
  route from an `.npz` to the frame powers and every derived column; the
  analysis modules (`threshold_rule.load_frames`, `archive_health`) read
  through them. The score caches bind the unchanged v5 files by path and size.
- Resume and combine accept only v6: a v5 checkpoint cannot be resumed under
  kernel core 3.0.0.

On the channel 30 product (598.npz) the recomputation changes the valid and
reject flags on 0 frames and moves `Q` by a median relative `-1.2e-6` (max
`1.1e-4`).

The v5 all-zero `(N, 3, B)` fine-term placeholder (written when the fine
stage did not run, under `fine_status` `disabled_by_option` or a
`*_lacks_matched_filter_row_projections` status) is not accepted in v6: its
sums are not frame powers. Resume refuses such a checkpoint, the frame-power
helpers refuse such a v5 product, and an out-of-band pilot stores exactly
`(N, 0, 0)` with zero bins.

The launch contract was tightened before the first archive cohort to require
per-unit source scope and receiver-state fields. Early v5 smoke products that
lack `unit_scope`, `unit_git_version_tag`, `unit_input_map_sha256`, or
`unit_collection_server` are invalid and must be regenerated.

Revisions before v5 are not accepted, converted, repaired, or resumed. Delete
them and regenerate from authoritative inputs with the current checkout.

Detailed arrays are listed in
[`PER_PILOT_PRODUCT_FIELDS.md`](PER_PILOT_PRODUCT_FIELDS.md) and
[`FINE_REDUCTION_PRODUCTS.md`](FINE_REDUCTION_PRODUCTS.md).

## Decision contract

`decision_contract_json` separates three roles:

1. `active_decision`: `coarse_normalized_positive_excess`, implemented as
   an exact host integer comparison and stored in `reject_mask`;
2. `fine_measurement`: `exact_fine_power_terms` --- the scan measures and
   stores the exact terms and applies no fine decision of its own;
3. `fine_candidate_decision`: `fine_order_statistic_cfar`, implemented and
   bit-tested since kernel core 2.3.0 but inactive (`"active": false`) while
   runtime calibration is `pending_campaign`.

It also records `measurements`, the coarse and fine local-reference power
ratios that post-processing recomputes, and (v6) `frame_powers`: the
definition `sum_of_fxfft256_fine_powers`, the three frame-power fields, their
source `fine_power_u64`, and the invariant above.

Only the coarse rule produces the stored `reject_mask`. Nothing in the fine
surface contributes to it.

## Fine null-bulk diagnostic

The null-bulk exceedance fraction is the fraction of the same independent
null-bulk bins used to estimate a frame's diagnostic threshold that exceed that
threshold. It is an in-sample diagnostic, not an independently measured
false-alarm probability or false-alarm rate.

It is **not a stored field**. Since schema v3 the product stores only the
exact `fine_power_u64` terms (and, since v6, their all-bin sums as the frame
powers), so this fraction --- like every other fine diagnostic --- is
recomputed in post-processing at whatever operating point is calibrated
there.

## Resume and combine boundary

Resume and combine require the exact schema identity, decision contract,
geometry, timing identity, weights, and detector provenance. A mismatch is an
error. Current runtime code contains no aliases, adapters, repair paths, or
fallback readers for pre-release products.

The source-event identity version is required and is versioned independently of
the enclosing product schema, which is at revision 6. It gates the change from
historical basename-only keys to keys that retain the complete archive or
campaign namespace. Products without this field cannot be combined or resumed
because identical basenames in different campaigns are not the same
acquisition.
