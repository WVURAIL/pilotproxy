# DTV detector operator handoff

The purpose of this procedure is to commission the DTV detector on one receiver, from the build to the masked 10-second visibility product. It covers a receiver-specific commissioning threshold. Certification of BAO safety and selection of the science-optimal threshold are not covered by this handoff.

On Kotekan `develop`, the detector runs in every Pathfinder X-engine node (`config/chord/pathfinder.j2`, which includes `config/chord/dtv.j2`), and the recv node writes its products (`config/chord/recv.j2`). REST switches turn the mask (`/dtv_mask_enabled`) and the recording (`/dtv_write_enabled`) on and off until the next restart. `DtvRfiMask` decides each channel from the detector's statistic at the channel's word, and folds that decision and the permanent-mask (reject-all) limit into the RFI mask that the correlators and their valid-sample counts read. This handoff describes fine decision v3 (kotekan#1924). Until it merges, Kotekan `develop` runs fine decision v1 and v2 bundles, which carry an anchor, a search half-width, reference bins and a rank, and whose invalid rank support `DtvRfiMask` also clears. The Pathfinder runs a v1 bundle. On the CHIME upgrade, the detector runs detect only. Recording, calibration and testing are in [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md), and this handoff covers acceptance and release. Use the [channel operating policy guide](CHANNEL_OPERATING_POLICY.md) to record a finite threshold or the reject-all limit for each declared channel and receiver-frequency scope. Incomplete assignments remain explicit and cannot form a complete deployment configuration. The policy compiler does not activate a telescope configuration.

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

Record which slots are live, missing, flagged, or connected to RFI antennas, and which are actually summed by the detector. On CHORD, the detector packs every input flagged by `bufferBadInputs`, the RFI antennas among them, as zero, and on CHIME it does the same since kotekan#1880. A configured slot is not necessarily a live independent input, so the release population must be confirmed from current telemetry. The [receiver profile](../configs/receiver_profiles/chord_pathfinder_dtv_fengine.json) and [stream map](../configs/stream_maps/chord_pathfinder_dish_pol_example.json) are examples until that verification is complete.

Freeze these bundle fields for every bound detector row of CH14 to CH36. CH37 is bound with the status `control`, so its fine powers are recorded for the words, and it is never masked.

| Item | Required record |
|---|---|
| Receiver binding | Physical channel, receiver frequency ID, weights and preprocessing |
| Fine score | `decision_version` `fine_decision_v3`, which carries no anchor, half-width, reference bins or rank |
| BAO word | Exact integer `power_limit_multiplier_q16` of a channel that passed the BAO test |
| Default word | Exact integer `cfar_multiplier_q16` of every channel: the control's 99% word scaled to the channel |
| Validity | Summed inputs, the control capture, and applicable receiver/input conditions |

The detector takes the largest target power `n_peak` over all 256 fine bins and the median reference power `d_rho` of the 128 even bins, and publishes the pair in `dtv_stat`. A consumer keeps a frame when `n_peak * 2^16 <= word * d_rho` (`pilotproxy_v3_keep`), so a frame at equality is kept. It uses the BAO word where the bundle carries one, and the default word otherwise. A row with `d_rho = 0` has no decision and is masked.

The bundle fixes no search geometry, so a pilot anywhere in the channel is searched, and the fine profile holds only the words. Qualify the words on the telescope that uses them, since an archive word cannot be swapped for a telescope word.

Choose the commissioning rejection target and confidence requirement before calibration. Use distinct data sets for development, calibration, and evaluation. Freeze the words before examining evaluation results. Qualify each receiver and channel condition that is claimed. CH37 is a control channel, and a null on CH37 does not show a null on every channel. Synthetic qualification must stay labeled as synthetic. Account for temporal dependence before binomial confidence bounds are used on telescope observations.

A first run needs a commissioning bundle. `pilot-proxy calibrate-bundle` builds one from captures, or from the telescope parameters alone with a simulated control, and fills the provenance the validator requires (Step 2 of [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md)). Its report gives the source of each word and of the control word, and a word from simulated white noise is a detect-only setting, not a calibration. A commissioning bundle is never a release bundle.

Bind only channels that carry a calibrated or commissioning profile. A bound channel whose profile is still `pending_campaign` stops the whole kotekan instance at its first voltage frame on any node where that channel binds, and on the Pathfinder this stops the correlators of that node as well. An uncalibrated channel can be handled in two ways. It can be dropped from the exported bundle, so that it stays unbound and its `dtv_stat` row holds `[0, 2^64 - 1]`, which every word keeps. It can also be listed in the `permanent_mask_freq_ids` of the detector and `DtvRfiMask`, so that the detector does not evaluate it and the applied mask rejects it outright. That list may name only receiver IDs present on the node, and the Pathfinder template does not set it yet.

Every deployed word comes from a calibration run on the telescope where it is used. The archive values, including the threshold rule (`tools/threshold_rule.py`) run on the CHIME archive, characterize the method and are not deployed. Nothing is yet calibrated for the CHIME upgrade or CHORD. Each telescope starts from a detect-only recording of its own with the control, several days long. Its first part calibrates and the days after test, the calibration is the shortest first part whose words keep passing, and a run with the new bundle then confirms the masks (Step 4 of [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md)). On CHIME, `chime-first-56005bb7` records this run. With `--threshold power_limit` and the forecast table of the telescope, the run gives the BAO word of each channel that passes the BAO test (Step 2). It also gives the default word of every channel, the control's 99% word scaled to the channel, which does not depend on the channel's own pilot. Compile the decisions of the BAO test against the release bundle with the deployment compiler ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)):

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json --output-dir policy_review --require-complete
pilot-proxy validate-operating-policy --policy-dir policy_review
```

Unassigned channels become `dtv_permanent_mask_freq_ids`. An assigned channel needs its bundle profile `calibrated` at the decided word. Until then, the review lists the channel as pending and is not complete. Neither command activates a bundle.

A decision of `useful` means the channel passed the BAO test at its word, so a default word does not make one. A channel that fails carries the reject word 1 as its BAO word and is masked whole, by the policy and by the bundle itself, unless the threshold rule found DTV without a matching pilot there. A commissioning bundle with no BAO words, such as the Pathfinder's, skips the compiler. It is deployed after `test-bundle` and `qualify-bundle`, and the stage reads each word from the bundle.

## 3. Validate the release bundle and configure the fine path

In the matching PilotProxy environment, validate the release bundle:

```sh
pilot-proxy validate-runtime-weight-bundle --bundle-dir "$DTV_RELEASE_BUNDLE"
```

Set `DTV_RELEASE_BUNDLE` to the immutable bundle directory recorded in the manifest. Preserve the result and the hashes of all bundle files. Each requested profile must have `fine_calibration.status: calibrated` with its accepted constants and provenance. Schema validation checks the form of the calibration and says nothing about its quality. Integration-test bundles marked `calibrated` are synthetic fixtures and must never become release bundles.

On the Pathfinder, these variables are set when `config/chord/pathfinder.j2` and `config/chord/recv.j2` are rendered:

```yaml
dtv_runtime_bundle_dir: <absolute release bundle directory>
dtv_apply_mask: false
dtv_write_enabled: false
```

`dtv_apply_mask` and `dtv_write_enabled` give the starting values of `/dtv_mask_enabled` and `/dtv_write_enabled`, and `tools/pathfinder_dtv_switch.py` switches them on a list of nodes at one shared time. It refuses to switch the mask on where a node's bundle is detect-only, since Kotekan stops on that update. The detector runs whenever the bundle loads. On CHORD, the detector command uses `detector_window_samples: 64` and the pipeline `num_times: 8192`. On CHIME, these are 128 and 16384. The stage derives the detector block size from the selected core, and each GPU frame must contain exactly one block.

A bound channel needs a calibrated profile (section 2). Inspect the startup binding records, where each expected pilot must be bound with the correct receiver ID. On a node with no bound pilots, a zero mask means that the detector is disabled, and it says nothing about the data being clean.

The F-engine template (`config/fengine/include/dtv_chord.j2`) is a test configuration. Its keys and streams differ from the Pathfinder graph, and it must never be pasted into a live configuration.

## 4. Accept shadow operation, then applied operation

Before acquisition, the site owners must freeze the acceptance limits and the responsible reviewer in the release manifest. Begin with recorded masks and `dtv_apply_mask: false`. In this mode, n2k has no dependency on the detector outputs. The detector, however, still holds voltage-ring frames while it processes them, so a slow detector can back-pressure the voltage producer and delay n2k. Record-only operation therefore does not guarantee independent progress. Keep the binary, bundle, input population, and configuration fixed during the segment. Record the frame and timestamp coverage, packet loss, GPU cadence and headroom, binding state, mask fractions, and synchronized samples of the voltage, mask, and power products. Recompute the sampled decisions from the saved voltages with the exact bundle, and require exact agreement of the masks, the coarse powers and, for v3 channels, the `dtv_stat` pairs. CHORD has no baseband dump yet (2026-11), so on the Pathfinder this replay waits for one. Confirm the pilot location, the reference occupancy, and the held-out operating bound under the stated conditions.

The stage exports `dtv_mask`, `dtv_stat`, `dtv_powers`, `dtv_fine_support` and, for the bound channels, `dtv_fine_powers`. On the Pathfinder, `dtv_mask` and `dtv_stat` are sent in every mode, and the others only when the templates are rendered with `dtv_calibration_mode: true` (kotekan#1894), so a run used for calibration needs it. The recv node writes one file of each sent product per NUMA half and window of about 43 s, under `/mnt/cs00/data/dtv/acq_<start time>/`. `dtv_stat` is uint64 with the pair `[n_peak, d_rho]` per frequency, and a row the detector does not evaluate holds `[0, 2^64 - 1]`, which every word keeps. `dtv_mask` is int8 per frequency, and the row of a v3 channel stays 0, since `DtvRfiMask` decides it from `dtv_stat`. `dtv_powers` is uint64 with 3 frame powers per frequency, each the all-bin sum of the fine powers of the stage, and a stage that vendors a core before 3.0.0 must be re-vendored before its powers match a replay. `dtv_fine_support` (int32, 2 per frequency) records the rank support `(rank_valid, n_bulk)` of a v1 or v2 channel and holds `(-1, -1)` for a v3 channel. The fine powers give the exact fine score of every frame of a bound channel, so a recorded run can be rescored offline with any word. A frame with `d_rho = 0` has no decision and is masked, so a masked frame must be read together with its `dtv_stat` row before it is counted as a detection. Logs and packet-loss monitoring are needed as well, but they cannot replace this check.

`DtvRfiMask` folds the detector decision into the aligned RFImask frame. For a v3 channel it reads `stat_buf` and applies `pilotproxy_v3_keep` at the channel's `power_limit_multiplier_q16`, or at `cfar_multiplier_q16` where there is none, and `mask_control` masks the control like a bound channel. A v3 bundle needs `stat_buf`. For a v1 or v2 channel with `fine_support_buf` configured, it clears every time bit of a frequency with invalid rank support. `permanent_mask_freq_ids` (the reject-all limit) clears a frequency regardless of the detector score, and `require_fine_freq_ids` names the frequencies that must carry an evaluated fine decision. Both the correlation and the valid-sample counts must consume its output. On the Pathfinder, `pathfinder.j2` gives the merged mask to every correlator and to `RfiMaskSum`, and `N2Accumulate` records the share of each visibility frame with the mask on as `frac_dtv_mask_enabled`. The decision covers only the DTV test, and input health, packet completeness, and physical calibration need evidence of their own. The [complete synthetic policy rehearsal](CHANNEL_OPERATING_POLICY.md#complete-synthetic-rehearsal) exercises the portable handoff and approves no live configuration.

Applied release requires a validated way to mark or exclude every invalid detector decision. On the Pathfinder, `DtvRfiMask` masks a v3 frame with `d_rho = 0`, and clears a v1 or v2 frequency with invalid rank support from `fine_support_buf`. That path must be shown on recorded data. Sampling and logs cannot show that coverage. An unresolved invalid-support path blocks applied release even when sampled replay passes, although shadow commissioning can still measure the gap.

Use `pilot-proxy replay-fine-capture --plan capture_plan.json --output-dir new_replay` and `pilot-proxy validate-fine-capture-replay --replay-dir new_replay` for the capture-specific fine path. The [capture replay guide](CAPTURE_FINE_REPLAY.md) supplies the pinned native format, contracts, failure interpretation, calibration export, and a reproducible synthetic example. The replay preserves invalid decisions. It checks the masks and the coarse powers exactly, and with stat files every `dtv_stat` row. It cannot yet read a capture recorded by a current Kotekan build (see the guide). Kotekan's `tests/verify_pilotproxy_pipeline.py` checks the fine masks and exact coarse powers of the stage on a small synthetic fixture. Actual telescope replay with a verified receiver state is still a release requirement, since the synthetic examples test only the plumbing of the commands.

Apply masking only after the shadow acceptance is signed and the site-approved configuration shows that both the correlation and the valid-sample counts consume the same `DtvRfiMask` output. On the Pathfinder, this is `/dtv_mask_enabled`, and its `frac_dtv_mask_enabled` in the visibility files shows when it was on. Validate the block alignment and the resulting masked and unmasked 10-second visibility and count products. Save this acceptance apart from the DSP calibration of the detector. The acceptance leaves the BAO ruling of each channel unchanged.

## 5. Stop, recalibrate, or roll back

Stop acceptance on an unexpected binding or path, a replay mismatch, frame misalignment, inadequate processing headroom, unexplained loss, an invalid-decision problem, or a failed operating bound. Also stop using the old calibration after a material change of the receiver or input population, or reference contamination that the calibrated conditions do not cover. Keep the evidence. A threshold adjusted on the evaluation segment can never count as a pass.

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

The implementation is in `src/pilot_proxy/runtime_bundle.py` and `src/pilot_proxy/fine_decision.py` of this repository, and in `lib/cuda/cudaPilotProxyDetector.cpp`, `lib/stages/DtvRfiMask.cpp`, `config/chord/dtv.j2`, `config/chord/recv.j2`, `config/chime/dtv.j2` and `tests/test_pilotproxy_integration.py` of Kotekan. Record the reviewed revisions and the evidence paths in the release manifest, since a path alone does not pin a version.
