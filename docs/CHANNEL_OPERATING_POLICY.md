# Channel operating thresholds

The deployment compiler turns the threshold rule's decisions
(`tools/threshold_rule.py`, `decisions.json`) into the channel operating
policy kotekan reads. Each ATSC channel 14 to 36 gets one assignment at its
receiver frequency ID:

- **deployed** (outcome `useful`): a finite threshold at the decided Q16 word
  (`deployed_word`: the knee, or the loosest word below it that meets the
  tolerance);
- **not deployed** (outcome `not useful`): the **reject-all limit of the same
  threshold family**, which permanently masks that frequency. It is encoded as
  `threshold.kind: reject_all`, never as `eta_q16: 0`: the frozen detector
  uses a strict comparison, zero scores would not fire at zero, and zero is
  not a legal runtime multiplier. The row keeps the rule's reason.
- **not deployed and not masked**: a not-useful decision whose reason is
  `DTV without a matching pilot` (channel 19). The detector has no pilot to act
  on there and would never fire, so the channel is `threshold.kind: pass_all`,
  listed under `unmasked_freq_ids` and absent from both consumer lists; its
  residual DTV is left to the consumer's other flaggers.

A decision outside channels 14 to 36 (the channel 37 control band) is not a
DTV channel; the review lists it under `not_in_policy` and assigns nothing.
The compiler changes no decision and infers no threshold beyond them.

## Commands

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir review
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir complete-review --require-complete
pilot-proxy validate-operating-policy --policy-dir review
```

The output directory must be new. Compilation reads the declared inputs and
writes `policy.json` and a hash manifest. Validation authenticates the manifest
and recompiles the review from the pinned sources. It does not modify a runtime
bundle, activate a telescope configuration, or rerun physical experiments.

### Complete synthetic rehearsal

```sh
python tools/make_operating_policy_example.py --output-dir /tmp/policy-example
pilot-proxy validate-runtime-weight-bundle --bundle-dir /tmp/policy-example/bundle
pilot-proxy compile-operating-policy --plan /tmp/policy-example/policy_plan.json --output-dir /tmp/policy-review --require-complete
pilot-proxy validate-operating-policy --policy-dir /tmp/policy-review
```

The generator copies the CHIME receiver profile and the core profile, writes
synthetic decisions (channel 14 not deployed, channel 29 deployed at
Q16 = 65536, and the channel 37 control band), exports a bundle for channels 14
and 29, and marks channel 29's profile as carrying its word. The review lists
`permanent_mask_freq_ids: [844]` and `require_fine_freq_ids: [614]`. This is
command rehearsal only; the decisions qualify no physical channel. Preserve the
generated directory at its original absolute path: validation authenticates
pinned source paths as well as file hashes.

## Plan contract

The top-level schema is `pilotproxy_channel_operating_policy_plan_v2`. JSON
duplicate keys, nonfinite values, unknown fields and missing fields are refused.
File references are `{ "path": "...", "sha256": "<lowercase SHA256>" }`; paths
are resolved relative to the plan. Every referenced file must still be available
when validating the review.

| Field | Required content |
| --- | --- |
| `receiver_profile` | Hash-bound strict receiver profile with a `metadata.channel_id_map`. |
| `receiver_channel_id_namespace` | `chime_freq_id` (the CHIME upgrade) or `chord_freq_id`, matching the receiver profile and the bundle. |
| `decisions` | The threshold rule's `decisions.json`. |
| `bundle_files` | File references for all five runtime bundle files, retaining their names in one directory. |
| `era` | Nonempty `id`, `scope`, and `evidence` file references: the era the decisions were made on. |
| `consumer` | `name`, `version`, `build_sha256`, `capabilities`, and `evidence` file references. |

For every channel 14 to 36 of the decisions the compiler checks:

1. The decision is `useful` with a positive uint64 `deployed_word`, or
   `not useful` with a nonempty `reason`.
2. The runtime bundle passes its validator, matches the receiver, and has
   exactly one profile for the channel. Its receiver frequency ID maps through
   the receiver's exact offset and grid to a centre inside the channel's six
   MHz, and stays below the consumer's exclusive limit of 12,288.
3. On `chime_freq_id`, that ID is the project's CHIME `freq_id` for the band
   (`Project.target_freq_id`: 844 for channel 14 down to 506 for channel 36);
   the CHIME profile's map gives `freq_id = coarse_channel_index + 1`.
4. A deployed channel whose bundle profile is `calibrated` carries exactly the
   decided word (`cfar_multiplier_q16`); its anchor, width, bulk mask and rank
   are copied into the binding. A deployed channel whose profile is still
   `pending_campaign` is listed in `pending_freq_ids`: the decision has a word,
   and the bundle does not carry it yet.

An [archive operating-curve candidate](ARCHIVE_POLICY_CANDIDATE.md) can carry
all numerical fields and the decided word while remaining explicitly
`archive_review_only`. Its imported detector-frame selection is not qualified
for a consumer's masking granularity or physical science use. Such profiles are
listed in `review_only_candidate_freq_ids`; the review carries their bindings
but does not produce a complete deployment configuration.

Every bound source is rehashed before the review is returned; conflicting
identities for the same path are refused.

## Consumer and release boundary

The declared consumer name is `Kotekan DtvRfiMask`. Its version and build digest
must be supplied, with evidence and all of these explicit capability tokens:

- `dtv_fine_support_v1`
- `dtv_permanent_mask_freq_ids_v1`
- `dtv_require_fine_freq_ids_v1`

An absent capability causes refusal. Checking a declared name, version, digest,
and capability list does **not** verify that a binary implements them. The
output keeps `consumer_capability_implementation_verified: false` and lists the
remaining runtime acceptance work.

The review exports disjoint `permanent_mask_freq_ids` and
`require_fine_freq_ids`, exact `required_threshold_bindings`, and
`pending_freq_ids`. `permanent_mask_freq_ids` is the consumer encoding of the
reject-all threshold limit, not a new scientific category. Missing or invalid
measurements remain validity conditions; the threshold limit must not
manufacture valid data or retained exposure.

`deployment_configuration_complete` is true only when no deployed channel is
pending and none has archive-review-only qualification. Only then is `kotekan_parameters` populated with the two consumer
lists; otherwise it is null. All reviews retain `activation_authorized: false`
and `science_certification: false`.

Before activation, independently check the named consumer implementation, live
receiver IDs and input state, exact bundle and threshold bindings, era, full-rate
fine support, validity handling, and the intended frequency coverage. A release
owner must approve activation. No review output changes a live configuration.

`tests/core/test_operating_policy.py` compiles the frozen decisions of
2026-10-12 (`tests/data/threshold_decisions_2026-10-12.json`) on the CHIME
receiver profile: eight channels go to the permanent mask, channel 19 is left
unmasked and fourteen go to their words.
