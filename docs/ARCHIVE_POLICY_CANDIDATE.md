# Archive operating-curve candidate bundles

The archive candidate builder imports already selected operating words and their frozen score geometry into a separate numerical bundle. It does not choose a new threshold or claim that an archive word is ready for telescope use.

Run it from the current checkout with an explicit source path:

```sh
PYTHONPATH=src python -m pilot_proxy.runtime_candidate \
  --plan /absolute/path/candidate-plan.json \
  --output-dir /absolute/path/new-candidate-directory
```

The output directory must be new. Existing bundles and paper evidence are not modified.

## Input plan

The plan schema is `pilotproxy_archive_candidate_plan_v1`. It has exactly these fields:

- `schema_version`: the schema token above.
- `decisions`: a `{ "path": "...", "sha256": "..." }` reference to the threshold rule's decision record.
- `bundle_files`: references for the five files of an unpopulated exported runtime bundle. Their original filenames must remain together in one directory.
- `score_metadata`: a mapping from each finite-threshold television channel, as a string, to its hash-bound `ch<N>.json` score metadata. Exactly the finite channels must be present. Channel 37 remains outside the policy.

All relative paths resolve from the plan directory. Each score metadata file must be the channel file in `decisions.inputs.scores`; the builder also checks the adjacent NPZ against `npz_sha256`. Retain the original input paths for later reproduction.

The score metadata supplies the anchor, contiguous designated set, guarded bulk bins, zero-based rank, archive epochs, and original product identity. The builder imports those values without refitting them. At the chosen word it replays the exact supported keep rule and checks retained-frame count, retained-event count, and the original event-weighted retained fraction against the decision record. It also checks equal frame weights within each event and equal original event mass and matches the decision's frame length to the bundle geometry. This verifies the imported selection; it does not independently rerun the Fisher forecast or the threshold search.

For every finite channel, the original CHIME product must still be available. Its complete SHA-256 and size are checked, together with physical channel, receiver frequency ID, exact packed-weight hash, integer weight norms, sample format, coordinate system, and preprocessing. The product must declare and contain the exact fine-power terms. Missing evidence or a mismatch stops the build. A matching channel number alone is insufficient.

## Provenance and qualification

Legacy calibrated profiles still require a finite `null_quantile` strictly between zero and one. An operating-curve candidate instead has `null_quantile: null` and a `threshold_selection` object with:

- `method: "operating_curve_candidate_v1"`;
- `selection_granularity: "detector_frame"`;
- `qualification: "archive_review_only"`;
- the decision-record, score-metadata, and score-population SHA-256 identities;
- `decision_channel` and exact integer `decision_word_q16`, matching the numerical profile.

Those hashes must also appear in `source_product_sha256`; the archive product hash and named epochs are retained. No null-rejection probability is invented for an operating-curve threshold.

The runtime field `status: "calibrated"` means that the numerical arguments are populated. The separate candidate qualification explicitly withholds scientific and operational acceptance. The operating-policy compiler carries these exact bindings in its review, but leaves `deployment_configuration_complete: false` and `kotekan_parameters: null`; `--require-complete` refuses them. Do not load an archive-review candidate as an active release bundle.

## Outputs and remaining work

`bundle/` contains the five runtime files with updated checksums. `candidate_review.json` records the verified input identities, per-channel weight and coordinate checks, decision-population replay, output hashes, and source-code identities. `source_plan.json` preserves the submitted plan. The review has no activation authorization, science certification, or verified consumer implementation.

The imported words describe detector-frame selection. CHIMEv1's pair-expanded mask requires a separate policy evaluation with the actual frame alignment, incomplete-pair handling, retention, retained proxy, and sampling guard. Current receiver/input conditions, physical pilot-to-band transfer, temporal covariance, loaded runtime behavior, and release approval also remain separate acceptance questions. A complete candidate record does not establish any of them.
