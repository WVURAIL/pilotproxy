# Per-pilot product schema

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 5
schema_version = "pilotproxy_per_pilot_product_v5"
source_event_key_schema_version = "pilotproxy_namespaced_source_event_key_v1"
```

This is the only supported per-pilot product contract. Earlier schema
revisions are not accepted, converted, repaired, or resumed. Delete them and
regenerate from authoritative inputs with the current checkout.

One v5 migration is explicit: an older no-measurement checkpoint may carry an
all-zero fine-term placeholder. Resume accepts only the exact zero form and
normalizes it to `(N, 0, 0)` with zero bins. Nonzero placeholders are invalid,
and current writers never emit placeholders.

The launch contract was tightened before the first archive cohort to require
per-unit source scope and receiver-state fields. Early v5 smoke products that
lack `unit_scope`, `unit_git_version_tag`, `unit_input_map_sha256`, or
`unit_collection_server` are invalid and must be regenerated. The token remains
v5 because no production cohort started under the earlier field set.

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
   bit-tested in kernel core 2.3.0 but inactive (`"active": false`) while
   runtime calibration is `pending_campaign`.

It also records `measurements`, the coarse and fine local-reference power
ratios that post-processing recomputes.

Only the coarse rule produces the stored `reject_mask`. Nothing in the fine
surface contributes to it.

## Fine null-bulk diagnostic

The null-bulk exceedance fraction is the fraction of the same independent
null-bulk bins used to estimate a frame's diagnostic threshold that exceed that
threshold. It is an in-sample diagnostic, not an independently measured
false-alarm probability or false-alarm rate.

It is **not a stored field**. Schema v3 stores only the exact
`fine_power_u64` terms, so this fraction --- like every other fine diagnostic
--- is recomputed in post-processing at whatever operating point is calibrated
there.

## Resume and combine boundary

Resume and combine require the exact schema identity, decision contract,
geometry, timing identity, weights, and detector provenance. A mismatch is an
error. Current runtime code contains no aliases, adapters, repair paths, or
fallback readers for pre-release products.

The source-event identity version is required and is versioned independently of
the enclosing product schema, which is at revision 5. It gates the change from
historical basename-only keys to keys that retain the complete archive or
campaign namespace. Products without this field cannot be combined or resumed
because identical basenames in different campaigns are not the same
acquisition.
