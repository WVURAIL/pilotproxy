# DTV detector operator handoff

The purpose of this procedure is to commission the DTV detector on one receiver, from the build to the masked 10-second visibility product. It covers a receiver-specific commissioning threshold. Certification of BAO safety and selection of the science-optimal threshold are not covered by this handoff.

The Kotekan implementation is in `develop` from pull request 1677 (merge
`6b923e51`). The pull request exports `dtv_mask`,
`dtv_powers` and the optional `dtv_fine_support` rank-validity rows, and its
`DtvRfiMask` stage applies invalid rank support and the permanent-mask
(reject-all) limit together with the detector mask. Use the
[channel operating policy guide](CHANNEL_OPERATING_POLICY.md) to record a finite
threshold or that limit for each declared channel and receiver-frequency scope.
Incomplete assignments remain explicit and cannot form a complete deployment
configuration. The policy compiler does not activate a telescope configuration.

A release requires a calibrated bundle, a verified telescope configuration, an acceptance record, and a site restart and rollback procedure. An exported bundle marked `pending_campaign` and a receiver profile marked `example_requires_data_product_verification` do not meet these requirements. Development bundles are kept apart from accepted releases.

[Archive operating-curve candidates](ARCHIVE_POLICY_CANDIDATE.md) are a review product of their own. They preserve saved words and exact numerical geometry with no null-quantile interpretation, and their `archive_review_only` qualification stops the operating-policy compiler from producing a complete deployment configuration. Such a candidate is never a release bundle, and its detector-frame policy still needs evaluation at the actual masking granularity of the consumer.

## 1. Build and check the implementation

Use the accepted Kotekan revision recorded in the release manifest. Build with `USE_CUDA=ON` and `WITH_TESTS=ON` on a supported CUDA device. Follow the checkout's `docs/sphinx/compiling/general.rst`, `README.md`, and `docs/sphinx/dev/dev_pilotproxy_testing.rst`. Record the complete build options, compiler/CUDA versions, binary hash, and `external/pilotproxy/VENDOR.json`.

From that checkout, with Python virtual-environment support installed, run the CI entry point with the absolute build-directory path:

```sh
bash tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh "$KOTEKAN_BUILD"
```

First, set `KOTEKAN_BUILD` to the test build directory of the release manifest. The script creates a temporary virtual environment with pinned dependencies, installs the PilotProxy revision pinned by `VENDOR.json`, and exports and validates development bundles for CHORD and CHIME. The pipeline and integration tests are run with timeouts and an ephemeral loopback REST port, and the environment and generated files are then removed. The script does not calibrate the telescope. Preserve its log and the list of any skipped tests. The sustained-workload command and the required environment variables are given in the validation instructions above. Run the Pathfinder geometry on the intended hardware before acceptance.

## 2. Freeze the telescope and threshold

The release must name the actual receiver, the frequency assignment of each node and GPU, the rendered configuration, and the data population. The Pathfinder example uses 8192 channelized samples per block at 195312.5 samples/s (41.94304 ms), 64-sample detector windows, 128 windows per stream, and 128 input slots. The CHIME upgrade uses a different profile, and its constants must never be copied into the Pathfinder configuration. Verify the real sample encoding, frequency axis, stream order, and preprocessing against captured telescope data.

Record which slots are live, missing, flagged, or connected to RFI antennas, and which are actually summed by the detector. The detector sums all configured slots and does not read the bad-input mask of the correlator. A configured slot is not necessarily a live independent input, so the release population must be confirmed from current telemetry. The [receiver profile](../configs/receiver_profiles/chord_pathfinder_dtv_fengine.json) and [stream map](../configs/stream_maps/chord_pathfinder_dish_pol_example.json) are examples until that verification is complete.

Freeze these bundle fields for every bound detector row of CH14 to CH36. CH37 stays unbound as a control channel.

| Item | Required record |
|---|---|
| Receiver binding | Physical channel, receiver frequency ID, weights and preprocessing |
| Fine score | `decision_version`, `anchor_bin`, `designated_half_width`, `bulk_mask_words_hex`, zero-based `cfar_rank` |
| Threshold | Exact integer `cfar_multiplier_q16`: the threshold rule's deployed word for the channel |
| Validity | Required usable bulk population, excluded reference lines, and applicable receiver/input conditions |

The target is the contiguous anchor-centered set supported by the runtime API, and an arbitrary search over the native FFT would be a different detector. The comparison is strict. A frame is rejected when a marker-bin ratio exceeds the Q16 multiplier times the selected bulk ratio, and a frame at equality is kept.

In pull request 1677, the reference mask must exclude every bin no more than max(h, 2) padded bins from the anchor, with circular wraparound, where h is the designated half-width. For h of 2 or less, this is two padded bins (one native bin) on each side. The declared zero-based rank may be any integer below the frozen reference population, and runtime validity further requires more usable positive-denominator references than that rank. Pending bundles leave the rank unset. The 125-bin or 126-bin bulk of the archive characterization meets this exclusion. However, its fixed rank 62 and that bulk form an experimental profile of their own and are not requirements of a generic runtime bundle. Qualify the actual runtime geometry and the rank together, since an archive threshold cannot be swapped for a telescope threshold.

Choose the commissioning rejection target and confidence requirement before calibration. Use distinct data sets for development, calibration, and evaluation. Freeze the anchor, exclusions, rank, and threshold before examining evaluation results. Qualify each receiver and channel condition that is claimed. CH37 is a control channel, and a null on CH37 does not show a null on every channel. Synthetic qualification must stay labeled as synthetic. Account for temporal dependence before binomial confidence bounds are used on telescope observations.

The calibration captures need a commissioning bundle. The [capture replay](CAPTURE_FINE_REPLAY.md) accepts only captures whose bound profiles are `calibrated`, while every exported bundle is `pending_campaign`, so the calibration and evaluation captures are recorded with a commissioning bundle instead. To build one, copy the exported bundle, set the anchor, designated half-width, guarded bulk mask, rank, and a provisional Q16 multiplier of each bound channel, and mark the profile `calibrated`. The validator then requires three provenance fields: a nonempty `epochs` list, at least one `source_product_sha256`, and a `null_quantile` strictly between 0 and 1. A bundle whose provenance is only a note is refused.

A commissioning bundle records the commissioning epoch in `epochs`, the sha256 of the files its provisional values came from (at least the exported `pilot_profiles.json`) in `source_product_sha256`, and the null quantile implied by the rejection target chosen above in `null_quantile`. It also adds a `note` such as "provisional commissioning multiplier, not a calibration". `tools/make_operating_policy_example.py` fills the same fields for its synthetic bundle. Rewrite `sha256sums.txt`, run `validate-runtime-weight-bundle`, and run the bundle only with `dtv_apply_mask: false`. The keep boundary exported by the replay (`required_multiplier_q16`) does not depend on the multiplier, so the provisional value biases nothing and sets only the recorded shadow mask bit. A commissioning bundle is never a release bundle.

Bind only channels that carry a calibrated or commissioning profile. A bound channel whose profile is still `pending_campaign` stops the whole kotekan instance at its first voltage frame on any node where that channel binds, and on the F-engine template this stops the correlator as well. An uncalibrated channel can be handled in two ways. It can be dropped from the exported bundle, so that it stays unbound with mask zero and support `(-1, -1)`. It can also be listed in the `dtv_permanent_mask_freq_ids` of that node, so that the detector does not evaluate it and the applied mask rejects it outright. That list may name only receiver IDs present on the node.

Each channel's word, or its permanent mask, comes from the threshold rule (`tools/threshold_rule.py`, its `decisions.json`). Compile the decisions against the release bundle with the deployment compiler ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)):

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir policy_review --require-complete
pilot-proxy validate-operating-policy --policy-dir policy_review
```

Unassigned channels become `dtv_permanent_mask_freq_ids`. An assigned channel needs its bundle profile `calibrated` at the decided word. Until then, the review lists the channel as pending and is not complete. Neither command activates a bundle.

## 3. Validate the release bundle and configure the fine path

In the matching PilotProxy environment, validate the release bundle:

```sh
pilot-proxy validate-runtime-weight-bundle --bundle-dir "$DTV_RELEASE_BUNDLE"
```

Set `DTV_RELEASE_BUNDLE` to the immutable bundle directory recorded in the manifest. Preserve the result and the hashes of all bundle files. Each requested profile must have `fine_calibration.status: calibrated` with its accepted constants and provenance. Schema validation checks the form of the calibration and says nothing about its quality. Integration-test bundles marked `calibrated` are synthetic fixtures and must never become release bundles.

The existing F-engine template uses these keys:

```yaml
dtv_enabled: true
dtv_apply_mask: false
dtv_runtime_bundle_dir: <absolute release bundle directory>
```

For CHORD, the detector command uses `detector_window_samples: 64` and the pipeline uses `num_times: 8192`. For CHIME, these are 128 and 16384. The stage derives the detector block size from the selected core, and each GPU frame must contain exactly one block.

Keep the detector on CUDA stream 3 and its output synchronization and all three `cudaOutputData` commands on stream 4, with `num_cuda_streams: 5`. The copies of `dtv_mask`, `dtv_powers` and `dtv_fine_support` must follow `cudaSyncOutput` on that stream so they wait for the detector. An output copy left on the default copy stream can read the initial fill before the detector writes its result.

A bound channel needs a calibrated profile (section 2). Inspect the startup binding records, where each expected pilot must be bound with the correct receiver ID. On a node with no bound pilots, a zero mask means that the detector is disabled, and it says nothing about the data being clean.

These keys are verified in Kotekan `config/fengine/chord.j2` and `config/fengine/include/dtv_chord.j2`. They do not form a complete Pathfinder overlay. A rendered and reviewed overlay for the actual Pathfinder configuration is a release requirement, and the include must never be pasted into an unverified live graph.

## 4. Accept shadow operation, then applied operation

Before acquisition, the site owners must freeze the acceptance limits and the responsible reviewer in the release manifest. Begin with recorded masks and `dtv_apply_mask: false`. In this mode, n2k has no dependency on the detector outputs. The detector, however, still holds voltage-ring frames while it processes them, so a slow detector can back-pressure the voltage producer and delay n2k. Record-only operation therefore does not guarantee independent progress. Keep the binary, bundle, input population, and configuration fixed during the segment. Record the frame and timestamp coverage, packet loss, GPU cadence and headroom, binding state, mask fractions, and synchronized samples of the voltage, mask, and power products. Recompute the sampled decisions from the saved voltages with the exact bundle, and require exact agreement of the masks and coarse powers. Confirm the pilot location, the reference occupancy, and the held-out operating bound under the stated conditions.

The pull request exports `dtv_mask`, `dtv_powers`, and, when `dtv_fine_support_name` is set, `dtv_fine_support`. `dtv_mask` is int8 per frequency. `dtv_powers` is uint64 with 3 frame powers per frequency, each the all-bin sum of the fine powers of the stage, and a stage that vendors a core before 3.0.0 must be re-vendored before its powers match a replay. `dtv_fine_support` (int32, 2 per frequency) records `(rank_valid, n_bulk)` for every evaluated frame, where `(-1, -1)` means unevaluated, `(0, n_bulk)` invalid rank support, and `(1, n_bulk)` valid rank support. Full-rate fine scores are not exported, and sampled-voltage replay recovers them only for the sampled frames. An invalid bulk produces mask zero, so a zero mask must be read together with the support rows before a frame is counted as having passed a valid fine test. Logs and packet-loss monitoring are needed as well, but they cannot replace this validity check.

`DtvRfiMask` folds the detector mask into the aligned RFImask frame. With `fine_support_buf` configured, it clears every time bit of a frequency with invalid rank support. `permanent_mask_freq_ids` (the reject-all limit) clears a frequency regardless of the detector score, and `require_fine_freq_ids` names the frequencies that must carry an evaluated fine decision. Both the correlation and the valid-sample counts must consume its output (`dtv_apply_mask: true` in `dtv_chord.j2`). Rank support covers only the rank, and input health, packet completeness, and physical calibration need evidence of their own. The [complete synthetic policy rehearsal](CHANNEL_OPERATING_POLICY.md#complete-synthetic-rehearsal) exercises the portable handoff and approves no live configuration.

Applied release requires a validated way to mark or exclude every invalid detector decision. This can be a dedicated validity path, or an existing input-health gate whose coverage of the exact invalid conditions of the fine decision has been shown and recorded. Sampling and logs cannot show that coverage. An unresolved invalid-support path blocks applied release even when sampled replay passes, although shadow commissioning can still measure the gap.

Use `pilot-proxy replay-fine-capture --plan capture_plan.json --output-dir new_replay` and `pilot-proxy validate-fine-capture-replay --replay-dir new_replay` for the capture-specific fine path. The [capture replay guide](CAPTURE_FINE_REPLAY.md) supplies the pinned native format, contracts, failure interpretation, calibration export, and a reproducible synthetic example. The replay preserves invalid decisions and checks the masks and coarse powers exactly. Kotekan's `tests/verify_pilotproxy_pipeline.py` checks the fine masks and exact coarse powers of the stage on a small synthetic fixture. Actual telescope replay with a verified receiver state is still a release requirement, since the synthetic examples test only the plumbing of the commands.

Apply masking only after the shadow acceptance is signed and the site-approved configuration shows that both the correlation and the valid-sample counts consume the same `DtvRfiMask` output. The F-engine template selects this path with `dtv_apply_mask: true`, and the Pathfinder overlay must show its equivalent. Validate the block alignment and the resulting masked and unmasked 10-second visibility and count products. Save this acceptance apart from the DSP calibration of the detector. The acceptance leaves the BAO ruling of each channel unchanged.

## 5. Stop, recalibrate, or roll back

Stop acceptance on an unexpected binding or path, a replay mismatch, frame misalignment, inadequate processing headroom, unexplained loss, an invalid-support problem, or a failed operating bound. Also stop using the old calibration after a material change of the receiver or input population, a pilot leaving its marker bins, or reference contamination that the calibrated conditions do not cover. Keep the evidence. A threshold adjusted on the evaluation segment can never count as a pass.

Use the approved procedure of the site to restore the previous binary, configuration, and bundle, with DTV application disabled. A bundle change requires a restart and a new recorded run, since a live bundle swap is not supported. Kotekan's `README.md` documents `sudo systemctl stop kotekan` and `sudo systemctl start kotekan` for an installed `kotekan` service, but the actual telescope service unit and the authorized restart window must be recorded before release. These entries must come from the site and never from the development checkout.

Recalibration creates a new versioned bundle and repeats independent evaluation and replay. The release package must name the responsible operator, maintainer, escalation contact, exact service and configuration paths, stop, restore, and start commands, and the rollback bundle. Until those site-specific entries are populated, the release is not ready for deployment.

## Minimal telescope data request

Provide synchronized native voltage samples and detector outputs for CH14 to CH36 and CH37, with contiguous blocks preserved and any sampling or decimation recorded. The request covers:

- the raw packing convention and the dimensions `[time, frequency, polarization, dish]`
- `coarse_freq`, `fpga_seq_num`, the sample interval, and the UTC mapping
- frame and ring offsets
- input identities and their live, missing, flagged, or RFI-antenna status
- packet-loss counts and the gain and quantization configuration
- binary, configuration, and bundle hashes, and the binding logs
- coarse powers, masks, and GPU timing

Independent calibration and evaluation epochs are needed with their null-label evidence, together with on-signal epochs for detection checks. For applied acceptance, also keep the matching 10-second visibilities, valid-sample counts and weights, and flags. This handoff needs no data downstream of that product.

## Release sidecar

Ship a small human-readable manifest next to the bundle. It contains:

- the release ID, date, and owners
- the source, binary, vendor, and bundle hashes
- the receiver profile and input population
- the rendered configuration and startup bindings
- the threshold criterion and calibration domain
- the source-data hashes and split definitions
- the per-channel thresholds, sample counts, confidence results, and limitations
- the test, replay, soak, and 10-second acceptance evidence
- the logging and data locations
- the verified site restart and rollback procedure

Commissioning qualification and conditional BAO results are kept in fields of their own. Store an example of a successful run and an example of a failed or invalid-frame run, so that the next operator knows what each looks like.

The implementation is in `src/pilot_proxy/runtime_bundle.py` and `src/pilot_proxy/fine_decision.py` of this repository, and in `lib/cuda/cudaPilotProxyDetector.cpp`, `config/fengine/include/dtv_chord.j2`, and `tests/test_pilotproxy_integration.py` of Kotekan. Record the reviewed revisions and the evidence paths in the release manifest, since a path alone does not pin a version.
