# Per-pilot product schema

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 6
schema_version = "pilotproxy_per_pilot_product_v6"
source_event_key_schema_version = "pilotproxy_namespaced_source_event_key_v1"
detector_contract = "pilotproxy_detector_contract_v2"
```

Schema v6 is the product written by kernel core 3.0.0. It keeps every field
name of v5 and redefines the frame powers. The fields `p_target_u64`,
`p_ref_lower_u64`, and `p_ref_upper_u64` are the exact all-bin sums of
`fine_power_u64` (`P_r = sum_f S_r[f]`), `p_ref_sum_u64 = lower + upper`, and
the derived columns (`coarse_power_ratio`, `normalized_coarse_power_ratio_db`,
`normalized_pilot_excess`, `pilot_excess_db`, `estimated_data_shelf_snr_db`,
`reject_mask`, `valid`) are formed from those sums with the same formulas as
in v5. The exact invariant is enforced by the writer,
`validate_current_product_identity`, `products/reader.py`, and
`tools/audit_per_pilot.py`:

```text
fine_status == "enabled":  p_r_u64[:, 0] == fine_power_u64[:, r, :].sum(axis=1)   (exact integers)
                           every derived column == coarse_columns_from_fine(product)
```

Integers and flags must match exactly, and floats must match to
`rtol = 4e-13` and `atol = 1e-12`, with NaN in the same places. Fine powers are
mandatory for every in-band pilot (`fine_status = enabled`, or `pending` for an
empty checkpoint). The only other status is `not_applicable_pilot_out_of_band`,
with zero powers, NaN ratios, `valid = 0`, shape `(N, 0, 0)`, and
`fine_num_bins = 0`. `fine_products=off` and `auto` are refused.

## Schema v5: the products of record

The archive campaign of record (`chime_pilots_rebuild_20260829`, kernel core
2.3.0) and the channel 37 control (`ch37_control_491_20260924`) are schema v5
products with detector contract revision 1. In these products, `p_*_u64` are
the row-domain powers `sum_m |z[n, m]|^2`. The products are never edited,
migrated, or copied. Every v5 product of record carries `fine_power_u64` for
every frame, so the current definition is computed on read:

- `products/reader.py` and `products/acceptance.py` open v5 (detector contract
  v1) and v6 (contract v2). A v5 view replaces the stored row-domain columns
  by `coarse_columns_from_fine(product)` and records
  `frame_power_definition = "recomputed_from_fine_power_u64 (v5 stored
  row-domain powers ignored)"`. A v6 view asserts that the stored columns
  equal them. A cohort with both v5 and v6 products is refused.
- `product_contract.frame_powers_from_fine`,
  `pilot_excess_from_fine_powers`, and `coarse_columns_from_fine` are the one
  route from an `.npz` to the frame powers and every derived column, and the
  analysis modules (`threshold_rule.load_frames`, `archive_health`) read
  through them. The score caches bind the unchanged v5 files by path and size.
- Resume and combine accept only v6, so a v5 checkpoint cannot be resumed
  under kernel core 3.0.0.

On the channel 30 product (598.npz) the recomputation changes the valid and
reject flags on 0 frames and moves `Q` by a median relative `-1.2e-6` (max
`1.1e-4`).

A v5 product whose fine stage did not run, under `fine_status`
`disabled_by_option` or a `*_lacks_matched_filter_row_projections` status,
holds an all-zero `(N, 3, B)` fine-term placeholder. Its sums are not frame
powers, so v6 does not accept it. Resume refuses such a checkpoint, the
frame-power helpers refuse such a v5 product, and an out-of-band pilot stores
exactly `(N, 0, 0)` with zero bins.

The launch contract requires per-unit source scope and receiver-state fields.
Early v5 smoke products, written before this requirement, that lack
`unit_scope`, `unit_git_version_tag`, `unit_input_map_sha256`, or
`unit_collection_server` are invalid and must be regenerated.

Revisions before v5 are not accepted, converted, repaired, or resumed. Delete
them and regenerate from authoritative inputs with the current checkout.

Detailed arrays are listed in
[`PER_PILOT_PRODUCT_FIELDS.md`](PER_PILOT_PRODUCT_FIELDS.md) and
[`FINE_REDUCTION_PRODUCTS.md`](FINE_REDUCTION_PRODUCTS.md).

## Decision contract

`decision_contract_json` defines three roles:

1. `active_decision`: `coarse_normalized_positive_excess`, implemented as
   an exact host integer comparison and stored in `reject_mask`.
2. `fine_measurement`: `exact_fine_power_terms`. The scan measures and
   stores the exact terms and applies no fine decision of its own.
3. `fine_candidate_decision`: `fine_order_statistic_cfar`, implemented and
   bit-tested since kernel core 2.3.0 but inactive (`"active": false`) while
   runtime calibration is `pending_campaign`.

The contract also records `measurements`, the coarse and fine local-reference
power ratios recomputed in post-processing, and, in v6, `frame_powers`. This
field holds the definition `sum_of_fxfft256_fine_powers`, the three frame-power
fields, their source `fine_power_u64`, and the invariant above.

The stored `reject_mask` is produced by the coarse rule alone, and nothing in
the fine surface contributes to it.

## Fine null-bulk diagnostic

The null-bulk exceedance fraction is the fraction of independent null-bulk bins
that exceed the diagnostic threshold of a frame, where the same bins are used to
estimate that threshold. It is an in-sample diagnostic, and so it cannot serve
as an independently measured false-alarm probability or false-alarm rate.

The fraction is not stored. From schema v3 on, the product stores only the
exact `fine_power_u64` terms, and from v6 on also their all-bin sums as the
frame powers. This fraction, like every other fine diagnostic, is therefore
recomputed in post-processing at the operating point calibrated there.

## Resume and combine boundary

Resume and combine require the exact schema identity, decision contract,
geometry, timing identity, weights, and detector provenance. A mismatch is an
error. Runtime code contains no aliases, adapters, repair paths, or fallback
readers for other products.

The source-event identity version is required and is versioned independently of
the enclosing product schema, which is at revision 6. It requires keys that keep
the complete archive or campaign namespace, and basename-only keys are refused.
Products without this field cannot be combined or resumed, because identical
basenames in different campaigns can belong to different acquisitions.
