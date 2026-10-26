# Archive operating-curve candidate bundles

The purpose of the archive candidate builder is to import operating words that were already selected, together with their frozen score geometry, into a new numerical bundle. No new threshold is chosen, and the imported words are qualified for archive review only.

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
- `score_metadata`: a mapping from each finite-threshold television channel, as a string, to its hash-bound `ch<N>.json` score metadata. The finite channels must all be present, and no others. Channel 37 is not part of the policy.

All relative paths resolve from the plan directory. Each score metadata file must be the channel file named in `decisions.inputs.scores`, and the adjacent NPZ is checked against `npz_sha256`. Keep the original input paths for later reproduction.

The score metadata supplies the anchor, the contiguous designated set, the guarded bulk bins, the zero-based rank, the archive epochs, and the original product identity. These values are imported without refitting. At the chosen word, the exact supported keep rule is replayed, and the retained-frame count, the retained-event count, and the original event-weighted retained fraction are compared with the decision record. The builder also checks that the frames of each event have equal weight, that the original events have equal mass, and that the frame length of the decision matches the bundle geometry. These checks verify the imported selection. The Fisher forecast and the threshold search are not rerun by the builder.

For every finite channel, the original CHIME product must still be available. Its complete SHA-256 and size are checked, together with physical channel, receiver frequency ID, exact packed-weight hash, integer weight norms, sample format, coordinate system, and preprocessing. The product must declare and contain the exact fine-power terms. Missing evidence or a mismatch stops the build. A matching channel number alone is insufficient.

## Provenance and qualification

Null-quantile calibrated profiles require a finite `null_quantile` strictly between zero and one. An operating-curve candidate instead has `null_quantile: null` and a `threshold_selection` object with:

- `method: "operating_curve_candidate_v1"`;
- `selection_granularity: "detector_frame"`;
- `qualification: "archive_review_only"`;
- the decision-record, score-metadata, and score-population SHA-256 identities;
- `decision_channel` and exact integer `decision_word_q16`, matching the numerical profile.

Those hashes must also appear in `source_product_sha256`. The archive product hash and the named epochs are kept as well. An operating-curve threshold carries no null-rejection probability.

The runtime field `status: "calibrated"` means that the numerical arguments are populated. Scientific and operational acceptance are withheld by the candidate qualification, which is recorded apart from that status. The operating-policy compiler carries these exact bindings in its review but leaves `deployment_configuration_complete: false` and `kotekan_parameters: null`, and `--require-complete` refuses them. An archive-review candidate must never be loaded as an active release bundle.

## Outputs and remaining work

`bundle/` contains the five runtime files with updated checksums. `candidate_review.json` records the verified input identities, per-channel weight and coordinate checks, decision-population replay, output hashes, and source-code identities. `source_plan.json` preserves the submitted plan. The review has no activation authorization, science certification, or verified consumer implementation.

The imported words describe the selection of detector frames. The pair-expanded mask of CHIMEv1 requires its own policy evaluation with the actual frame alignment, incomplete-pair handling, retention, retained proxy, and sampling guard. Acceptance also needs evidence on the current receiver and input conditions, the physical pilot-to-band transfer, temporal covariance, loaded runtime behavior, and release approval. A complete candidate record covers none of these.
