# DTV detector operator handoff

This procedure ends at the masked 10-second visibility product. It covers a receiver-specific commissioning threshold; it does not certify BAO safety or select the science-optimal threshold.

The Kotekan implementation is in `develop` from pull request 1677 (merge
`6b923e51`). The pull request exports `dtv_mask`,
`dtv_powers` and the optional `dtv_fine_support` rank-validity rows, and its
`DtvRfiMask` stage combines invalid rank support and the permanent-mask
(reject-all) limit with the detector mask. Use the
[channel operating policy guide](CHANNEL_OPERATING_POLICY.md) to record a finite
threshold or that limit for each declared channel and receiver-frequency scope.
Incomplete assignments remain explicit and cannot form a complete deployment
configuration. The policy compiler does not activate a telescope configuration.

**Release requires a calibrated bundle, verified telescope configuration, acceptance record, and site restart/rollback procedure.** An exported bundle marked `pending_campaign` and a receiver profile marked `example_requires_data_product_verification` do not meet those requirements. Keep development bundles separate from accepted releases.

[Archive operating-curve candidates](ARCHIVE_POLICY_CANDIDATE.md) are a separate review product. They can preserve saved words and exact numerical geometry without assigning a null-quantile interpretation, but their `archive_review_only` qualification prevents the operating-policy compiler from producing a complete deployment configuration. They are not release bundles; detector-frame policy still needs evaluation at the actual consumer masking granularity.

## 1. Build and check the implementation

Use the accepted Kotekan revision recorded in the release manifest. Build with `USE_CUDA=ON` and `WITH_TESTS=ON` on a supported CUDA device. Follow the checkout's `docs/sphinx/compiling/general.rst`, `README.md`, and `docs/sphinx/dev/dev_pilotproxy_testing.rst`. Record the complete build options, compiler/CUDA versions, binary hash, and `external/pilotproxy/VENDOR.json`.

From that checkout, with Python virtual-environment support installed, run the CI entry point with the absolute build-directory path:

```sh
bash tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh "$KOTEKAN_BUILD"
```

Set `KOTEKAN_BUILD` to the release manifest's test build directory first. This script creates a temporary virtual environment with pinned dependencies, installs the PilotProxy revision pinned by `VENDOR.json`, and exports and validates **development** bundles for CHORD and CHIME. It runs the pipeline and integration tests with timeouts and an ephemeral loopback REST port, then removes the environment and generated files. It does not calibrate the telescope. Preserve its log and any skipped tests. The sustained-workload command and required environment variables are in the validation instructions above; run the Pathfinder geometry on the intended hardware before acceptance.

## 2. Freeze the telescope and threshold

The release must name the actual receiver, node/GPU frequency assignment, rendered configuration, and data population. The current Pathfinder example uses 8192 channelized samples per block at 195312.5 samples/s (41.94304 ms), 64-sample detector windows, 128 windows per stream, and 128 input slots. CHIME's upgrade uses a different profile; its constants must not be copied here. Verify the real sample encoding, frequency axis, stream order, and preprocessing against captured telescope data.

Record which slots are live, missing, flagged, or connected to RFI antennas, and which are actually summed by the detector. The detector currently sums all configured slots; it does not consume the correlator's bad-input mask. Do not equate configured slots with live independent inputs; current telemetry must establish the release population. The [receiver profile](../configs/receiver_profiles/chord_pathfinder_dtv_fengine.json) and [stream map](../configs/stream_maps/chord_pathfinder_dish_pol_example.json) are examples until that verification is complete.

For every bound CH14–36 detector row, freeze these bundle fields. CH37 remains an unbound comparison control:

| Item | Required record |
|---|---|
| Receiver binding | Physical channel, receiver frequency ID, weights and preprocessing |
| Fine score | `decision_version`, `anchor_bin`, `designated_half_width`, `bulk_mask_words_hex`, zero-based `cfar_rank` |
| Threshold | Exact integer `cfar_multiplier_q16`: the threshold rule's deployed word for the channel |
| Validity | Required usable bulk population, excluded reference lines, and applicable receiver/input conditions |

The target is the contiguous anchor-centered set supported by the runtime API. An arbitrary native-FFT search is a different detector. The comparison is strict: reject when a marker-bin ratio exceeds the Q16 multiplier times the selected bulk ratio; equality is kept.

In pull request 1677, the reference mask must exclude every bin within max(h, 2) padded bins of the anchor, where h is the designated half-width, including circular wraparound; at h of 2 or less that is two padded bins (one native bin) on each side. The declared zero-based rank may be any integer below the frozen reference population; runtime validity additionally requires more usable positive-denominator references than that rank. Pending bundles leave the rank unset. The archive characterization's 125/126-bin bulk meets this exclusion, but its fixed rank 62 and that bulk are a separate experimental profile, not generic runtime-bundle requirements. Qualify the actual runtime geometry and rank together; an archive threshold is not interchangeable with a telescope threshold.

Choose the commissioning rejection target and confidence requirement before calibration. Use separate development, calibration, and evaluation data. Freeze the anchor, exclusions, rank, and threshold before examining evaluation results. Qualify each receiver/channel condition being claimed; CH37 is a comparison control, not automatic proof of a null for every channel. Synthetic qualification must remain labeled synthetic. Account for temporal dependence before using binomial confidence bounds on telescope observations.

**Commissioning bundle for the calibration captures.** The [capture replay](CAPTURE_FINE_REPLAY.md) accepts only captures whose bound profiles are `calibrated`, and every exported bundle is `pending_campaign`. Record the calibration and evaluation captures with a commissioning bundle instead. Copy the exported bundle, set each bound channel's anchor, designated half-width, guarded bulk mask, rank and a provisional Q16 multiplier, and mark the profile `calibrated`. The validator then requires three provenance fields: a non-empty `epochs` list, at least one `source_product_sha256` and a `null_quantile` strictly between 0 and 1, so a bundle whose provenance is only a note is refused. A commissioning bundle records the commissioning epoch in `epochs`, the sha256 of the files its provisional values came from (at least the exported `pilot_profiles.json`) in `source_product_sha256`, and the null quantile that the rejection target chosen above implies as `null_quantile`, and adds a `note` such as "provisional commissioning multiplier, not a calibration". `tools/make_operating_policy_example.py` fills the same fields for its synthetic bundle. Rewrite `sha256sums.txt`, run `validate-runtime-weight-bundle`, and run the bundle only with `dtv_apply_mask: false`. The keep boundary the replay exports (`required_multiplier_q16`) does not depend on the multiplier, so the provisional value biases nothing; it sets only the recorded shadow mask bit. A commissioning bundle is never a release bundle.

Bind only channels that carry a calibrated or commissioning profile. A bound channel whose profile is still `pending_campaign` stops the whole kotekan instance at its first voltage frame on any node where that channel binds; on the F-engine template this stops the correlator too. Leave an uncalibrated channel unbound, by leaving it out of the exported bundle (it then has mask zero and support `(-1, -1)`), or list it in that node's `dtv_permanent_mask_freq_ids`, so that the detector does not evaluate it and the applied mask rejects it outright. That list may name only receiver IDs present on the node.

Each channel's word, or its permanent mask, comes from the threshold rule (`tools/threshold_rule.py`, its `decisions.json`). Compile the decisions against the release bundle with the deployment compiler ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)):

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir policy_review --require-complete
pilot-proxy validate-operating-policy --policy-dir policy_review
```

Unassigned channels become `dtv_permanent_mask_freq_ids`. An assigned channel needs its bundle profile `calibrated` at the decided word; until then the review lists it as pending and is not complete. Neither command activates a bundle.

## 3. Validate the release bundle and configure the fine path

In the matching PilotProxy environment, validate the release bundle:

```sh
pilot-proxy validate-runtime-weight-bundle --bundle-dir "$DTV_RELEASE_BUNDLE"
```

Set `DTV_RELEASE_BUNDLE` to the immutable bundle directory recorded in the manifest. Preserve the result and hashes of all bundle files. Each requested profile must have `fine_calibration.status: calibrated` with its accepted constants and provenance. Schema validation does not establish calibration quality. Integration-test bundles marked `calibrated` are synthetic fixtures and must not become release bundles.

The existing F-engine template uses these keys:

```yaml
dtv_enabled: true
dtv_apply_mask: false
dtv_runtime_bundle_dir: <absolute release bundle directory>
```

For CHORD, the detector command uses `detector_window_samples: 64` and the pipeline uses `num_times: 8192`. For CHIME, these are 128 and 16384. The stage derives the detector block size from the selected core; each GPU frame must contain exactly one block.

Keep the detector on CUDA stream 3 and its output synchronization and all three `cudaOutputData` commands on stream 4, with `num_cuda_streams: 5`. The copies of `dtv_mask`, `dtv_powers` and `dtv_fine_support` must follow `cudaSyncOutput` on that stream so they wait for the detector. An output copy left on the default copy stream can read the initial fill before the detector writes its result.

A bound channel needs a calibrated profile (section 2). Inspect the startup binding records: each expected pilot must be bound with the correct receiver ID. Zero masks on a node with no bound pilots mean “detector disabled,” not measured clean data.

These keys are verified in Kotekan `config/fengine/chord.j2` and `config/fengine/include/dtv_chord.j2`. They are **not a complete Pathfinder overlay**. A rendered, reviewed overlay for the actual Pathfinder configuration is a release requirement; do not paste the include into an unverified live graph.

## 4. Accept shadow operation, then applied operation

Before acquisition, the site owners must freeze the acceptance limits and responsible reviewer in the release manifest. Begin with recorded masks and `dtv_apply_mask: false`. This mode gives n2k no dependency on the detector outputs, but the detector still holds voltage-ring frames while processing. A slow detector can back-pressure the voltage producer and delay n2k; record-only operation does not guarantee independent progress. Keep the binary, bundle, input population, and configuration fixed during the segment. Record frame/timestamp coverage, packet loss, GPU cadence and headroom, binding state, mask fractions, and synchronized sampled voltage/mask/power products. Recompute sampled decisions from the saved voltages with the exact bundle and require exact mask and coarse-power agreement. Confirm pilot location, reference occupancy, and the held-out operating bound under the stated conditions.

The pull request exports `dtv_mask` (int8, frequency), `dtv_powers` (uint64, frequency × 3 frame powers: the all-bin sums of the stage's fine powers; a stage vendoring a core before 3.0.0 must be re-vendored before its powers match a replay) and, when `dtv_fine_support_name` is set, `dtv_fine_support` (int32, frequency × 2), recording `(rank_valid, n_bulk)` for every evaluated frame: `(-1, -1)` means unevaluated, `(0, n_bulk)` invalid rank support, and `(1, n_bulk)` valid rank support. It does **not** export full-rate fine scores; sampled-voltage replay can recover them only for sampled frames. An invalid bulk produces mask zero, so a zero mask alone is not proof that the frame passed a valid fine test; read it with the support rows. Logs and packet-loss monitoring are necessary but do not replace that validity check.

`DtvRfiMask` folds the detector mask into the aligned RFImask frame. With `fine_support_buf` configured it clears every time bit of a frequency with invalid rank support; `permanent_mask_freq_ids` (the reject-all limit) clears a frequency independently of the detector score, and `require_fine_freq_ids` names the frequencies that must carry an evaluated fine decision. Both correlation and valid-sample counts must consume its output (`dtv_apply_mask: true` in `dtv_chord.j2`). Rank support alone does not establish input health, packet completeness, or physical calibration. The [complete synthetic policy rehearsal](CHANNEL_OPERATING_POLICY.md#complete-synthetic-rehearsal) exercises the portable handoff without approving a live configuration.

**Applied release requires a validated way to mark or exclude every invalid detector decision.** This can be a dedicated validity path or an existing input-health gate whose coverage of the exact fine-decision invalid conditions has been demonstrated and recorded. Sampling and logs do not establish that coverage. An unresolved invalid-support path blocks applied release, even when sampled replay passes; shadow commissioning can still measure the gap.

Use `pilot-proxy replay-fine-capture --plan capture_plan.json --output-dir new_replay` and `pilot-proxy validate-fine-capture-replay --replay-dir new_replay` for the capture-specific fine path. The [capture replay guide](CAPTURE_FINE_REPLAY.md) supplies the pinned native format, contracts, failure interpretation, calibration export, and a reproducible synthetic example. It preserves invalid decisions and checks masks and coarse powers exactly. Kotekan's `tests/verify_pilotproxy_pipeline.py` checks the stage's fine masks and exact coarse powers on a small synthetic fixture. Actual telescope replay with verified receiver state is still a release requirement; synthetic examples establish only the command's plumbing.

Apply masking only after the shadow acceptance is signed and the site-approved configuration demonstrates that **both correlation and valid-sample counts consume the same combined mask**. The F-engine template selects this path with `dtv_apply_mask: true`; the Pathfinder overlay must demonstrate its equivalent. Validate block alignment and the resulting masked/unmasked 10-second visibility and count products. Save this acceptance independently of the detector's DSP calibration. It does not change a channel's BAO ruling.

## 5. Stop, recalibrate, or roll back

Stop acceptance on an unexpected binding/path, replay mismatch, frame misalignment, inadequate processing headroom, unexplained loss, invalid-support problem, or a failed operating bound. Also stop using the old calibration after a material receiver/input-population change, a pilot leaving its marker bins, or reference contamination outside the calibrated conditions. Keep the evidence; do not adjust the threshold on the evaluation segment and call it a pass.

Use the site's approved procedure to restore the previous binary, configuration, and bundle, with DTV application disabled. Bundle changes require a restart and a new recorded run; there is no promised live bundle swap. Kotekan's `README.md` documents `sudo systemctl stop kotekan` and `sudo systemctl start kotekan` for an installed `kotekan` service, but the actual telescope service/unit and authorized restart window must be recorded before release. Do not infer those from the development checkout.

Recalibration creates a new versioned bundle and repeats independent evaluation and replay. The release package must name the responsible operator, maintainer, escalation contact, exact service/config paths, stop/restore/start commands, and the rollback bundle. Until those site-specific entries are populated, the release is not ready for deployment.

## Minimal telescope data request

Provide synchronized native voltage samples and detector outputs for CH14–36 plus CH37, preserving contiguous blocks and recording any sampling/decimation. Include the raw packing convention; dimensions `[time, frequency, polarization, dish]`; `coarse_freq`; `fpga_seq_num`; sample interval; UTC mapping; frame/ring offsets; input identities and live/missing/flagged/RFI-antenna status; packet-loss counts; gain/quantization configuration; binary/config/bundle hashes; binding logs; coarse powers; masks; and GPU timing. Include independent calibration and evaluation epochs with their null-label evidence, and on-signal epochs for detection checks. For applied acceptance also retain the corresponding 10-second visibilities, valid-sample counts/weights, and flags. No data after that product is required by this handoff.

## Release sidecar

Ship a small human-readable manifest next to the bundle containing: release ID/date and owners; source/binary/vendor/bundle hashes; receiver profile and input population; rendered configuration and startup bindings; threshold criterion and calibration domain; source-data hashes and split definitions; per-channel thresholds, sample counts, confidence results, and limitations; test/replay/soak/10-second acceptance evidence; logging/data locations; and the verified site restart/rollback procedure. Keep commissioning qualification and conditional BAO results in separate fields. Store an example successful run and a failed or invalid-frame example so the next operator knows what each looks like.

Implementation references: this repository’s `src/pilot_proxy/runtime_bundle.py` and `src/pilot_proxy/fine_decision.py`; Kotekan’s `lib/cuda/cudaPilotProxyDetector.cpp`, `config/fengine/include/dtv_chord.j2`, and `tests/test_pilotproxy_integration.py`. Record the reviewed revisions and evidence paths in the release manifest; paths alone are not a version pin.
