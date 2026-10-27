# Channel operating thresholds

The purpose of the deployment compiler is to turn the decisions of the
threshold rule (`tools/threshold_rule.py`, `decisions.json`) into the channel
operating policy that kotekan reads. Each ATSC channel from 14 to 36 receives
one assignment at its receiver frequency ID:

- deployed (outcome `useful`): a finite threshold at the decided Q16 word
  (`deployed_word`, which is the knee or the loosest word below it that meets
  the tolerance).
- not deployed (outcome `not useful`): the reject-all limit of the same
  threshold family, which permanently masks that frequency. It is encoded as
  `threshold.kind: reject_all`. The word `eta_q16: 0` cannot encode it,
  because the frozen detector uses a strict comparison, zero scores would not
  fire at zero, and zero is not a legal runtime multiplier. The row keeps the
  reason given by the rule.
- not deployed and not masked: a not-useful decision whose reason is
  `DTV without a matching pilot` (channel 19). The detector has no pilot to act
  on there and would never fire, so the channel is `threshold.kind: pass_all`,
  listed under `unmasked_freq_ids`, and absent from both consumer lists. Its
  residual DTV is left to the other flaggers of the consumer.

A decision for a channel other than 14 to 36 (the channel 37 control band) is
not a DTV decision. The review lists it under `not_in_policy` and assigns
nothing to it. The compiler changes no decision and infers no threshold of its
own.

## Commands

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir review
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir complete-review --require-complete
pilot-proxy validate-operating-policy --policy-dir review
```

The output directory must be new. Compilation reads the declared inputs and
writes `policy.json` and a hash manifest. Validation authenticates the manifest
and recompiles the review from the pinned sources. Neither step modifies a
runtime bundle, activates a telescope configuration, or reruns physical
experiments.

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
and 29, and marks the profile of channel 29 as carrying its word. The review
lists `permanent_mask_freq_ids: [844]` and `require_fine_freq_ids: [614]`. This
is a rehearsal of the commands, and its decisions qualify no physical channel.
Keep the generated directory at its original absolute path, since validation
authenticates the pinned source paths as well as the file hashes.

## Plan contract

The top-level schema is `pilotproxy_channel_operating_policy_plan_v2`.
Duplicate JSON keys, nonfinite values, unknown fields, and missing fields are
refused. File references are `{ "path": "...", "sha256": "<lowercase SHA256>" }`,
with paths resolved relative to the plan. Every referenced file must still be
available when the review is validated.

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
   the exact offset and grid of the receiver to a center in the 6 MHz of the
   channel, and it stays below the exclusive limit of 12288 set by the
   consumer.
3. On `chime_freq_id`, that ID is the CHIME `freq_id` of the project for the
   band (`Project.target_freq_id`: 844 for channel 14 down to 506 for channel
   36). The map of the CHIME profile gives
   `freq_id = coarse_channel_index + 1`.
4. A deployed channel whose bundle profile is `calibrated` carries exactly the
   decided word (`cfar_multiplier_q16`), and its anchor, width, bulk mask, and
   rank are copied into the binding. A deployed channel whose profile is still
   `pending_campaign` is listed in `pending_freq_ids`, since the decision has a
   word that the bundle does not carry yet.

An [archive operating-curve candidate](ARCHIVE_POLICY_CANDIDATE.md) can carry
all numerical fields and the decided word while it stays explicitly
`archive_review_only`. Its imported detector-frame selection is qualified
neither for the masking granularity of a consumer nor for physical science use.
Such profiles are listed in `review_only_candidate_freq_ids`. The review
carries their bindings, but it cannot produce a complete deployment
configuration from them.

Every bound source is rehashed before the review is returned. Conflicting
identities for the same path are refused.

## Consumer and release boundary

The declared consumer name is `Kotekan DtvRfiMask`. Its version and build digest
must be supplied, with evidence and all of these explicit capability tokens:

- `dtv_fine_support_v1`
- `dtv_permanent_mask_freq_ids_v1`
- `dtv_require_fine_freq_ids_v1`

A missing capability causes refusal. The compiler checks the declared name,
version, digest, and capability list, and it has no way to verify that a binary
implements them. The output therefore keeps
`consumer_capability_implementation_verified: false` and lists the runtime
acceptance work that remains.

The review exports disjoint `permanent_mask_freq_ids` and
`require_fine_freq_ids`, exact `required_threshold_bindings`, and
`pending_freq_ids`. `permanent_mask_freq_ids` is the consumer encoding of the
reject-all threshold limit and adds no scientific category. Missing or invalid
measurements remain validity conditions, and the threshold limit must never
turn them into valid data or retained exposure.

`deployment_configuration_complete` is true only when no deployed channel is
pending and none has archive-review-only qualification. Only then is
`kotekan_parameters` populated with the two consumer lists, and otherwise it is
null. Every review keeps `activation_authorized: false` and
`science_certification: false`.

Before activation, independently check the named consumer implementation, live
receiver IDs and input state, exact bundle and threshold bindings, era, full-rate
fine support, validity handling, and the intended frequency coverage. A release
owner must approve activation. No review output changes a live configuration.

`tests/core/test_operating_policy.py` compiles the frozen decisions in
`tests/data/threshold_decisions_2026-10-12.json` on the CHIME
receiver profile. Eight channels go to the permanent mask, channel 19 is left
unmasked, and fourteen go to their words.
