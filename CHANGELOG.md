# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- New calibrations write fine decision v3, which Kotekan runs since kotekan#1924. `calibrate-bundle`, `plan-run` and
  `process-run` write v2 or v1 only when the plan or `--decision-version` names it. `export-runtime-weight-bundle`
  writes its pending profiles under the same version, with no search geometry for v3, and
  `export_runtime_weight_bundle` takes `decision_version`. A v3 profile needs a receiver channel-id namespace, so a
  bundle of `reference_800mhz_pfb`, which has none, must name v2 or v1. `process-run`'s `summary.md` names the
  decision version on its first line. The operating-policy example is a v1 bundle.
- Removed the September evidence packages `docs/evidence/ca_os_archive_bulk_2026-09-29` and
  `ch37_rank_injection_2026-09-30`, and their index `F1_INVENTORY.md`. They chose the bulk mask and the CFAR rank
  of fine decisions v1 and v2, which v3 does not use, and nothing reads them. Under frame-health gate v2 their bulk
  statistics move only in the last printed digit. Git history holds them, and RFIsher a281112 reproduces their
  ledgers.
- Kernel core 4.2.1: `kernel_accumulate_row_sums` is gone, and a single channel launches the batched row-sum
  kernel at batch 1. Output is unchanged (117 GPU tests and the CUDA test binary pass on an RTX 5000 Ada), and the
  times are within the run-to-run spread of 4.2.0. Kotekan vendors core 4.2.0 until its copy is updated.
- `tools/threshold_rule.py` requires `--table`, the RFIsher channel table of the telescope, so the CHIME table is no
  longer used when it is left out.
- `CHIME_DERIVED.md` lists every `docs/evidence` entry, under Code-only cut or the new Not CHIME data, and
  `tests/core/test_chime_derived.py` fails on one that is in neither. Its Inline quotes add `METHOD_SPEC.md` and DD001's
  coarse operating points.
- The power-limit test divides the corrected estimate and its margin by the pilot passband factor `r_c`, as the
  dissertation's BAO test does. A power-limit plan (and the `--power-limit` file) must give `pilot_passband` for every
  channel, the control included; for CHIME it is `r_c` of the dissertation's `pilot_passband.json`. Without it the test
  read each ratio low by `r_c^2`. `limit_test` takes `pilot_passband`, and its `x` and `upper_shelf` are on the sky.
  The report and `bundle.decisions.json` record the factor.
- Removed the v6 archive pipeline (`src/pilot_proxy/archive/`, `products/acceptance.py`) and the commands
  `chime-survey`, `chime-inventory`, `chime-scan`, `chime-combine`, `audit-archive-health` and `products
  check-archive`, with the CANFAR scripts, `setup_env.sh`, `launch_gpu_session.py`, the archive freeze and monitor
  scripts, `tools/audit_local_archive_run.py`, the `archive` pip extra and its docs. The archive scans of record were
  made with an earlier release; tag `pre-purge-2026-11-23` holds the code, and every path is in
  `tests/data/removed_paths.txt`. The `archive-integration` CI job keeps its name and runs the current-identity
  checks.
- Removed the injection workflow: `inject-pilot-tone`, `analyze-injection-recovery`, `chime/injection.py`,
  `chime/injection_recovery.py` and `characterization/injection.py`.
- `archive_health.py` keeps only the v2 frame-health gate (`evaluate_frame_health`); the archive health audit is
  removed. `validate_identifier` moved to `pilot_proxy.config.instrument`, and `chime/baseband_format.py` reads the
  instrument from there. UG001 and DD001 no longer describe the removed commands, and their PDFs are rebuilt.
- The archive frame-health gate is `pilotproxy_archive_frame_health_gate_v2`. It adds the reason
  `fill_samples_above_one_percent`, which excludes a frame whose `fill_sample_count` is more than 1% of
  `railed_sample_total`, so a frame that only partly overlaps absent baseband is no longer counted as healthy. The
  rule applies to per-pilot v5/v6 products and refuses one that lacks the fill count; archived products store no fill
  count and are unaffected. The histogram and fine-histogram exporters refuse ledgers, preflights and coarse exports
  written under the v1 gate, which must be regenerated, and a partly filled valid frame makes the exact spectrum
  correction unavailable.
- `test-bundle` accepts a masked fraction when `|bias| <= std`, the criterion of the dissertation's BAO test. Frames are
  grouped by the UTC hour of their time. The bias is the test's mean masked fraction less the calibration's, both
  weighted by frames, and `std` is the standard deviation of the calibration's hourly fractions at the same word.
  The rule applies to each channel at the word the stage applies and to the control at the control word, over its
  night frames with the night option. With `std` 0 only a bias of 0 passes, and a calibration of fewer than two
  hours fails. Captures that record no frame time (`chord_baseband`) are binned by capture, and the report says so.
  `calibrate-bundle` records `masked_by_hour` for each channel and for the control, and `applied_masked_by_hour` at
  a BAO word. A report without them fails and asks for a recalibration. Both checks stay blocking. They used to be a
  binomial tail on the control and a bootstrap interval over captures, which treat the frames as independent; on
  Pathfinder Run 1 they failed every calibration length. `masked_interval` and `applied_masked_interval` are no
  longer written, and `knee_interval_q16` only for v1 and v2 channels. `PATHFINDER_RECALIBRATION.md` asks for
  whole sidereal days of calibration, since Taurus A transits once per sidereal day.
- `tests/core/test_default_word_pin.py` pins the v3 default words against the dissertation
  (`data/assembly/default_words.json` at 03ca6bb): the control's night and all-frame 99th percentiles and the 24
  default words, from the control's keep words of the v3 score cache (`tests/data/v3_control_ch37_keep_words.npz`,
  withheld from the code-only cut). `bundle_calibration.control_quantile_q16` is the quantile step that
  `calibrate-bundle` uses.

- Removed `runtime_candidate` (the archive operating-curve candidate builder), its tests and
  `docs/ARCHIVE_POLICY_CANDIDATE.md`; nothing called it. The operating-policy compiler no longer has archive-review
  branches (`review_only_candidate_freq_ids`, `candidate_configuration_complete`), and the bundle validator refuses
  `provenance.threshold_selection`, which only the candidate builder wrote.
- `calibrate-bundle` computes null scores for v1 and v2 only. With no control, the v3 first word is the quantile of
  the exact keep words of the simulated white-noise powers, as with a control (32 inputs: 122738 to 122739 Q16).
  `simulated_null_scores` is replaced by `simulated_powers`, and `null_scores` refuses v3.
- Fixes from the 2026-11-22 review:
  - CHORD and Pathfinder weights in raw input coordinates are the mirror of the post bank; they were the post bank,
    blind to an upright pilot. `pack_channelized_streams_for_detector` takes a required `time_reverse_windows` flag
    (the bundle's `time_reverse_detector_windows_before_kernel`) in place of `nyquist_zone`, so CHORD packing
    reverses in zone 1. CHIME banks and packing are unchanged.
  - The operating policy compiles a no-pilot channel at the word the stage applies (`reject_all` at the reject
    word 1, otherwise finite) and lists it under `no_pilot_freq_ids`, instead of recording it as unmasked; a pending
    no-pilot channel leaves the policy incomplete. `expected_masked_fraction` is removed.
  - `calibrate-bundle` leaves out RFI dishes whose `dish_label` is stored as bytes, and refuses a plan whose control
    channel is not also under `channels`, which gave a bundle without the channel-37 row.
  - Capture replay accepts bundles with the channel-37 control row and scores a bound control like a pilot.
  - Good-input counts drop the `-1` rows the writer fills for a mask frame that has not arrived.
  - Plans and score caches of a run with untimed windows follow recording order, not `fpga_seq_num`.
  - The era rule no longer opens an instrument-change boundary when the input map returns after a one-month
    excursion; `characterize archive` eras are named the monthly-state rule (`monthly-state-eras-v3`), not the
    dissertation's era rule.
  - `PRODUCT_SCHEMA.md` and `products/reader.py` name the writer of the v5 products of record (pilot-proxy
    2.0.0.dev0, core 2.3.0, in the private development repository), not the code on main.
  - `inject-pilot-tone` works in slabs of 8192 rows; peak memory was about 50 times the baseband dataset.
  - `analyze_dtv_run.py --ranges` has no default (the default file was deleted); `calibration_length.py` takes
    `--night`, as `process-run` does; `prepare_fine_validation_waveforms.py` runs the generator from the checkout
    and records each source's sha256.
  - Removed with no callers: `characterize spectrogram`, the anchor estimator, bootstrap and contrast writers in
    `anchors.py` (the marker bins and bulk mask stay), the residual health views, and `tools/fused_real_receipt.py`,
    whose input has had no producer since `d3a3749`.
  - Tests that could not fail now can, and a golden v3 bundle (`tests/data/v3_bundle_golden/`) pins the applied
    word and role of each channel.
- Removed code and data that neither the live system nor the dissertation uses. Every removed path is listed in
  `tests/data/removed_paths.txt`, and tag `pre-purge-2026-11-23` (63e72e8) holds them:
  - the LimeSDR/SDR bench: transfer sweep, live handoff, antenna controls, smoke and reference captures,
    distribution reference, OTA protocol, the SDR upgrade adapter and streaming model, and the vendored LimeSuite
    headers;
  - the synthetic SNR and estimator-transfer bench: commands `evaluate-snr`, `plot-results`, `summarize-results`,
    `quantize` and `bench`, the testbench harness, AWGN and quantize modules, `current_geometry_sensitivity.py`,
    `audit_settled_waveform.py`, `run_estimator_transfer.sh` and `examples/quickstart.sh`. `evaluate_snr.py` and
    `sensitivity_study.py` keep only what the pinned benchmark drivers import, whose bytes are unchanged;
  - the survey stress and durability tests, and the estimator-transfer freezer with its test;
  - the `chime-run` and `chime-inspect` commands, the batch runner and segmented input; `chime-scan --source
    local` covers staged directories. The analyzer's weight-manifest and reference-placement checks moved into
    `archive/detector.py`;
  - seven early `docs/evidence` studies (mask/residual frontier, M scaling, fine-gain Monte Carlo and its
    correction, SDR bench, and the first digital and OTA estimator-transfer freezes).
- `combine_pathfinder_dtv_run.py` combines an acquisition again when recv has added files since it was combined
  (windows written after a quiet gap in the same acquisition). `combine.json` now lists the files it combined
  (`sources`). If files it combined before are gone, combining again would drop their windows, so the tool refuses
  and exits 1. An acquisition combined by the older tool, whose `combine.json` has no file list, is skipped as
  before. A combined file is reopened after it is combined again in the same process.
- `pathfinder_dtv_switch.py` refuses to switch the mask on at a node whose DtvRfiMask bundle is detect-only (a
  calibrated channel with no `mask_qualification` record), before any POST: Kotekan stops on such an update. It
  reads the `pilot_profiles_path` in each node's `/config`, or a copy given with `--bundle-dir`. An
  `http.client` error no longer aborts the switch loop.
- `scripts/canfar`: `canfar_supervise.sh` ends at the run's terminal state as `scan_scope.json` records it (every
  unit completed or quarantined, none failed, capped or unprocessed), not on an "incomplete requested scope" line
  in the log tail, which is also printed when retryable units are left. `canfar_closeout.sh` checks each channel's
  scope the same way before it copies a product, and stops if a verification step fails.
- `tools/audit_per_pilot.py` maps channel 37 to freq_id 491 (the control) and takes `--bank` for products made
  with another weight bank, such as the control's `chime_dtv_weights_k128_r14_37.bin`.
- The archive tests place channel 15 at 476.171875 MHz, the centre of freq_id 829; 476.3125 MHz was off the CHIME
  grid. The annotation of `reference_800mhz_pfb.json` gives its ascending channel formula.
- `inject-pilot-tone` dithers the tone it adds: each component gets `floor(tone + U)`, with `U` uniform on [0, 1)
  from the seeded generator. Rounding the integer data plus the tone added `round(tone)`, so a tone below 0.5 LSB
  left the file unchanged and a larger one came out as a staircase (1.38, 1.54 and 0.64 times `a^2` at `a` = 0.6,
  0.8 and 1.5). Each manifest entry adds `realized_amplitude_lsb` and the added
  `dither_variance_lsb2_per_component` (at most 1/4). A zero-amplitude copy is still byte-identical.
- `tools/audit_local_archive_run.py` accepts the `storage_service` that `chime-scan` writes in each execution
  record (null, or the replica it used) and refuses any other extra key. Before, the closeout failed on every
  complete run.
- Every site that applies a fine_decision_v3 word uses the word the stage applies (`v3_word`): the power-limit
  word where calibration wrote one, and the default word otherwise. `compile-operating-policy` binds a v3 channel
  or control at that word, so a v3 control and a v3 channel with no power-limit word now compile. It used to
  require `power_limit_multiplier_q16` on every deployed v3 channel and on a v3 control. `run-spectrogram`,
  `run-scores` and the `process-run` summary keep v3 frames at that word and name it in `word_q16`,
  `current_threshold` and `masked_summary.csv`. The summary also gives that word's `eta` and `word_source`, and
  its scale is still that of the default word. `analyze_dtv_run.py` matches archive ranges at that word. For
  bundles with no power-limit word, and for v1 and v2 bundles, `run-spectrogram`, `run-scores`, `process-run` and
  `analyze_dtv_run.py` give the same outputs as before.
- A fine_decision_v3 channel with no passing BAO word, or with no captures, carries the reject word 1 as its
  `power_limit_multiplier_q16` (`REJECT_WORD_Q16`) instead of no power-limit word. DtvRfiMask and the relay apply
  that word, and it keeps only a frame with no target power, so they mask the channel whole without a permanent
  list, which the Pathfinder templates do not pass. Before, both decided such a channel at its default word.
- A channel whose calibration captures never showed its pilot gets the reason `pilot not seen in the calibration
  captures`. It used to get `pilot not seen`, the 2026-10-12 wording of the threshold rule's `DTV without a
  matching pilot`, so `compile-operating-policy` left it unmasked (`pass_all`); it is now `reject_all`.
- `test-bundle` finds shared test and calibration data by identity as well as by path. `calibrate-bundle` records
  `calibration_capture_ids` and the control's `capture_ids`: a recorded window by its time, sequence number and
  frequencies, any other capture by its size and the SHA-256 of its first and last MiB. A copy, a hard link, or the
  same windows read per window, combined or from a several-frame file no longer pass as new test data. A report
  without the identities is compared by path alone.
- `tools/calibration_length.py` gives the fine decision v3 words with no floor or anchor: each channel's default
  word and, with `--power-limit` (the `power_limit` threshold), its power-limit word. With `--power-limit` each
  part of a split needs the `min_captures` that a power-limit word must keep, 30 unless the power-limit plan sets
  another. The v3 default word needs no captures of the channel, and with no power limit the tool sets 30 for the
  masked fractions. The v1 and v2 output without `--power-limit` is unchanged.
- The Pathfinder run tools no longer pass a fine_decision_v3 run they could not read. `check_pathfinder_dtv_run.py`
  reports a v3 channel with no `dtv_stat` rows as an error (exit 1) instead of reading its all-zero `dtv_mask`
  rows as 0% masked, and counts its windows without `dtv_stat`. It and `compare_pathfinder_dtv_mask.py` refuse
  a directory with `dtv_stat` files when `--bundle-dir` is not given, and the check stops when two different
  windows share an `fpga_seq_num`, as acquisitions on either side of an F-engine resync do, instead of letting
  one replace the other. `compare_pathfinder_dtv_mask.py` compares the
  bundle's v3 channels by default, counts a block read from both a per-window and a combined file once, and exits 1
  when a check fails or a frequency could not be compared. `analyze_dtv_run.py` reads a not-evaluated v3 row
  `[0, 2^64 - 1]` as untested, not as a valid test, reports unequal frame counts instead of failing in a
  reshape, and takes `--domain` for a two-domain CHIME run, whose two streams it used to read as one.
  `pathfinder_preflight.py` requires `dtv_stat` to be sent, received and written when the detector writes it
  (kotekan#1924). The `test` extra installs `hdf5plugin`, so the run-tool tests no longer skip in CI.
- `replay-fine-capture` checks fine_decision_v3 captures against `dtv_stat`, listed in the plan's optional
  `raw_files.stat`; a v3 plan without it is refused. A v3 row's recorded pair must equal `fine_stat_v3` of the
  reference fine powers bit for bit, and its `dtv_mask` row must be 0, as the detector writes it. The decision at the
  applied word (`masks_from_stat` at `v3_word`, `d_rho = 0` masked) is exported as `expected_decision`, and
  `bundle_multiplier_q16` of a v3 row is that word. The replay used to compare the recorded mask with the v3
  decision at the default word, so every frame flagged at that word was a mismatch, a `d_rho = 0` frame was expected
  kept, and the pair was never read. The reader still takes only the pinned metadata ABI, which predates
  `dtv_stat`, and only `rawFileWrite` files with `chord_freq_id` bundles. CHORD writes `dtv_stat` with
  `hdf5FileWrite`, so the command cannot yet replay a recorded v3 capture.
- `test-bundle` checks each channel's masked fraction at the word the stage applies: for a v3 channel with a
  power-limit word, that word (`v3_word`), against the interval `calibrate-bundle` now also records at it
  (`applied_masked_interval`). It used to test the default word only, so a qualified BAO word had no
  out-of-sample test. A report without the interval fails the check and asks for a recalibration.
- Bundle validation refuses fine_decision_v3 rows when `receiver_channel_id_namespace` is null, as Kotekan does
  at startup (DtvRfiMask maps v3 words by receiver_channel_id).
- `calibrate-bundle` takes an optional night site, `"default_word": {"night": {"latitude_deg": ...,
  "longitude_deg_east": ...}}`, and then sets the control word over the control frames recorded with the Sun below
  the horizon there (`sun.py`), as CHIME's 21 cm analyses use the data. In the day the Sun is common to every input
  and raises the control's scores; on the CHIME archive the all-frame 99th percentile is 1.716 and the night one
  1.133. The power-limit corrections still read every control frame, the report's control record gives the site and
  the night frame count, and `test-bundle` counts the control's false alarms over its night frames. `plan-run` and
  `process-run` write the block with `--night LATITUDE LONGITUDE`. Frame times come from each `dtv_fine_powers`
  window and each `chime_baseband` capture (`time0_ctime`); `chord_baseband` captures record none and are refused.
- Removed files nothing used: the v1-geometry rescoring CLI `detectors/narrowband_marker/rescore.py` and its
  tests, `scripts/check_scan.py`, three SDR tools (`audit_sdr_streaming_throughput.py`,
  `benchmark_sdr_upgrade_streaming.py`, `plot_sdr_antenna_controls.py`) and the two standalone figure wrappers.
  `tests/data/removed_paths.txt` names the last commit that holds each.
- `bundle_calibration.null_level` is removed. `bundle_calibration.weight_norms` gives a channel's weight norms, and
  `detector_contract.null_power_ratio_from_weight_norms` their null level `mu_0`.
- The power-limit test of `calibrate-bundle` takes only the control frames that the decision decides (keep word
  below `MAX_SCORE`), and a word at which the control keeps no frame gets an infinite selection-shift error, so it
  cannot pass. Both match the dissertation, and neither changes an archive outcome.
  `power_limit_word.ratio_excess` forms the excess with the products' `pilot_excess_from_fine_powers` from the
  weight norms, so the two agree exactly and float fine powers are refused.
- A power-limit plan must give `curvature` for every calibrated channel and `margin` (0 where measured as none);
  `calibrate-bundle` and `plan-run` refuse a plan without them. Both used to default to zero, which on the archive
  would pass channels 21, 29 and 32 under the worst case. With the dissertation's offset differences and route
  term, the test reproduces the dissertation's worst-case test to 1e-13 (`docs/evidence/bao_word_crosscheck_2026-11-22`).
- `docs/evidence/v3_timing_laptop_2026-11-21` records the v3 core timing on the laptop RTX 5000 Ada (core 4.2.0,
  no correlator load), and the DD quotes it. The deployment GPUs are not yet timed under v3.
- `qualify-bundle` refuses a test report with no passing masked-fraction check of a calibrated channel, so a channel
  the test plan left out is not qualified untested.
- Under v3, the pilot-seen gate of `calibrate-bundle` counts the frames whose keep word passes the channel's default
  word, not the unscaled control word, so a channel with a scale above 1 no longer reads its noise as the pilot. A
  channel with no captures in a power-limit plan gets a `not useful` decision (`no captures`), so the operating
  policy masks it whole.
- The scale of the v3 default word is written `lambda_c` in the docs, as in the dissertation.
- The operating-policy compiler requires the consumer capability `dtv_stat_v3_v1` when the bundle holds a
  `fine_decision_v3` profile: the consumer reads `dtv_stat` and applies the v3 words (kotekan#1924). It used to
  ask only for the v1/v2 capabilities.
- `test-bundle` fails the masked-fraction check of a channel calibrated with no captures, with a message, in place
  of a KeyError on its missing `masked_interval`.
- The power-limit test of `calibrate-bundle` forms its excess as the frame-level ratio `e = F / mu_0 - 1`
  (`power_limit_word.ratio_excess`, the products' normalized pilot excess), with each channel and the control at
  the null level of its own weights, as the dissertation does. It used to divide by a median of the bin ratios,
  under v3 over the even bins farther than 96 fine bins from the pilot; `shelf_reference`, `SHELF_HALF_WIDTH` and
  `median_excess` are removed. The report gives `estimate`, `mu0` and the control's `control_mu0` with each test.
- Under v3, a channel that fails the power-limit test of `calibrate-bundle` carries no `power_limit_multiplier_q16`,
  so Kotekan and the relay decide it at its default word, and the operating policy masks it whole. It used to carry
  the unscaled control word, which they would have applied in place of the default word. The report gives
  `power_limit_word_q16: null` and the reason in `power_limit_word_source` (`none, power limit: <reason>`).
- `tools/prepare_fine_validation_cache.py --channels` builds the fine-validation cache of a channel outside the
  record campaign, such as the channel-37 control. Its case indices continue the record order (37 takes 644-671), so
  `validate_fine_detector.engine_for` loads them unchanged. `--path-map` relocates the recorded preparation paths,
  which stay checked by SHA256, and `--prior-preflight` names the reserved seed ledgers. The new
  `tools/supplement_fine_campaign_plan.py` derives a sealed campaign plan for such a cache from the record plan,
  with the record's protocol and the whole 0.75 dB level lattice in `levels_db`.
- Under v3, the power-limit test of `calibrate-bundle` reads the control at `floor(u / lambda_c)`, where `lambda_c` is the
  channel scale of the default word, so a control frame counts as kept at the channel word `u` when its keep word
  is at most `floor(u / lambda_c)` (`bundle_calibration.control_words_at`). The default word and the selection shift
  then map the control onto the channel the same way, as the dissertation's v3 rerun does. The null of the control
  (all its frames) is unchanged, and the report gives `control_word_scale` with the corrections.
- The v3 default word of `calibrate-bundle` is the control's word at `null_quantile` (99%), from its exact keep
  words, scaled to each channel's null level `mu_0,c g_c / (mu_0,37 g_37)` and rounded up to Q16, in place of the
  knee. `mu_0` comes from the stored weight norms (`bundle_calibration.weight_norms`), and the passband factor `g` from the
  optional plan block `"default_word": {"passband_factor": {channel: g}}` (1 where not given). A v3 channel with no
  captures gets the same word. The report gives `control_quantile_q16`, `reference_level` and each channel's `mu0`,
  `passband_factor` and `scale`. On the archive it reproduces the dissertation's v3 default words exactly. v1 and v2
  bundles keep the knee and floor.
- `combine_pathfinder_dtv_run.py` also combines `dtv_stat`, the `[n_peak, d_rho]` rows that a fine decision v3 run records,
  so a v3 run can be combined like the others.
- The Pathfinder tools read fine decision v3 runs. `fine_decision.masks_from_stat` turns recorded
  `[n_peak, d_rho]` rows into dtv_mask decisions at a word (exact at the keep boundary; unevaluated rows are kept,
  missing blocks are -1, `d_rho = 0` is rejected), and `run_files.v3_words` gives each v3 channel's N2 word. With
  `--bundle-dir`, `check_pathfinder_dtv_run.py`, `compare_pathfinder_dtv_mask.py` (with `--mask-control` for the
  control) and `analyze_dtv_run.py` decide the v3 channels from `dtv_stat`, whose dtv_mask rows hold 0.
- Kernel core 4.1.1 allocates the completion counter of `PilotProxy_Compute_FineStatV3_U64` in
  `PilotProxy_Create`, so that no compute entry allocates device memory.
- Fine decision v3 is added beside v1 and v2, as kernel core 4.1.0 and, with no anchor or search width, 4.2.0.
  The kernel writes the statistic `[n_peak, d_rho]` per frame in place of a mask: `n_peak` is the largest
  `2 S_t[b]` of all 256 fine bins, so the padded bins catch a pilot between two independent bins, and `d_rho` the
  median reference power `S_lo + S_up` of the 128 even bins (rank 63, zeros included), with no bulk mask or rank.
  A v3 bundle block carries no `anchor_bin` or `designated_half_width` (the validator refuses them), and the v3
  null gives one score per frame. The consumer applies its word with
  `pilotproxy_v3_keep` (`cuda/pilotproxy_decision.h`), which keeps a frame when
  `n_peak * 2**16 <= word * d_rho` in exact 128-bit products and returns 1 to keep and 0 to flag,
  as Kotekan's RFI mask does; `d_rho = 0` means no decision. The entry point is
  `PilotProxy_Compute_FineStatV3_U64`, and the Python reference is `fine_stat_v3`,
  `fine_keep_v3` and `fine_keep_word_v3`.
- Runtime bundles can carry fine decision v3. A `fine_decision_v3` block names no
  `bulk_mask_words_hex`, `cfar_rank` or `marker_guard_bins`, and the validator refuses them. It carries
  `cfar_multiplier_q16`, the default word, and may carry `power_limit_multiplier_q16`. `calibrate-bundle`
  writes v3 when the plan names it (`plan-run` and `process-run` take `--decision-version`): with the `power_limit` threshold the tested
  word becomes the power-limit word, and the default word is written beside it. `test-bundle`, the run
  scores, the spectrogram, the capture reference and replay and the operating policy read the v3 geometry
  with `calibration_geometry`, and the operating policy checks the power-limit word. New calibrations still
  write v2 until Kotekan runs v3.
- Runtime bundles carry fine decision v2. `calibrate-bundle` writes `fine_decision_v2` unless
  the plan names `decision_version`, and its words, floor and null scores follow that version.
  The bundle validator, `test-bundle`, the run scores, the capture reference and replay, and the
  operating policy read each channel's version. Bundles of `fine_decision_v1` stay valid.
- Fine decision v2 is added beside v1. It reads the largest target bin of the search
  against the median of the reference powers of the background bins,
  `2 S_t[b] * 2**16 > multiplier_q16 * den_r`. The Python reference is
  `fine_mask_decision_v2`, `fine_required_multiplier_q16_v2` and
  `required_multipliers_by_rank_v2`, the kernel entry points are
  `PilotProxy_Compute_FusedFineMaskV2_U64` and its `WithSupport` form, and
  `final_score_arrays` takes `decision_version=2`. Fine decision v1, the bundles and the
  detector contract are unchanged.
- `calibrate-bundle` (the plan key `threshold`), `plan-run` and `process-run` (`--threshold`) take a threshold choice. `knee` is
  the default. `power_limit` gives each channel the word that keeps the most frames
  under a checked residual power, the BAO acceptance test with the RFIsher table. It
  uses the same fine powers, measures the null and the selection shift on the control,
  and writes `bundle.decisions.json` for the operating-policy compiler.

## 1.1.0, November 13, 2026

Tagged `v1.1.0` at 6d3032a. The version string dates from October 24.

- `plan-run` splits a detect-only run into a calibration plan and a test plan.
- `run-spectrogram` plots the fine powers of a detect-only run with and without the mask.
- `calibrate-bundle` accepts the control channel as a bound channel, so a run records
  it for the next calibration.
- `calibrate-bundle` and `test-bundle` read the `dtv_fine_powers` files of a detect-only
  run, so calibration no longer needs baseband voltages, and empty capture lists give a
  first bundle from the telescope parameters.
- `calibrate-bundle` sets each channel's word at the knee of its kept fraction, over
  the 65-bin search of record, with a floor from the control channel. Channels with
  no pilot or fewer than 30 captures take the floor. `test-bundle` compares each
  masked fraction with the calibration, and its plan takes `channels` in place of
  `strong` and `quiet`.
- Bundle manifests record profile file names, so a bundle holds no local paths.
- The CHIME era lists moved out of the repository. `projects/chime_atsc/eras/`
  holds empty stand-ins, and archive characterization needs the real lists copied in.
- `pilot-proxy calibrate-bundle` builds a detect-only commissioning bundle from
  captures, and `pilot-proxy test-bundle` checks a bundle on other captures.
  `docs/RUNNING_THE_DETECTOR.md` describes the capture, calibrate, test and deploy
  loop. `tools/extract_baseband_channels.py` and `tools/analyze_dtv_run.py` support it.
- **Breaking threshold-model change:** selection now maximizes retention under
  both an upper first-order bias criterion and a checked finite-power limit.
  It uses RFIsher schema-v3 tables and the declared model normalization
  `a = power_scale*x**2` in mK^2. This replaces the knee, stack-amplitude ratio,
  temporal-gain and recent-year selection rules. Observing-time cost is
  `1 / kappa`; the earlier duty factor and 20-year cutoff are removed.
- `tools/threshold_rule.py` requires `--power-limits`, a JSON mapping of channel
  numbers to checked positive additive powers in mK^2. It removes
  `--policy-budget`, `--finite-budget` and `--calendar-factor`, and writes
  `pilotproxy-power-decisions-v3`. Update readers for the new fields; the old
  knee, gain and recent-year diagnostics are absent. Channel 37 remains a
  background control and is excluded from operating-word selection.
- The Python threshold API requires `power_scale=` for `bias_over_sigma` and
  `decide`, and `power_limit=` for `decide`. The old strength/gain helpers and
  `decide_guarded` are removed. Score-cache construction now takes
  `--metadata`, `--output` and optional `--current-word`, and writes
  `pilotproxy-score-cache-v2`; it no longer derives a knee or accepts the
  earlier working/export manifests. Use the recorded source revision when
  reproducing the older model and its decisions.
- Remove obsolete confidence-band, capture-transfer, mask-frontier,
  transmitter-census and power-era study code. The commands
  `analyze-transmitter-census`, `characterize power-eras` and
  `characterize era-boundaries` are removed. Dated evidence and configured
  receiver-state intervals remain available.
- Fix numerical edge cases, input and schema validation, CLI failures,
  report publication and retry behavior. Add regression coverage for archive,
  receiver, runtime and statistical workflows. Inventory compatibility now
  uses explicit legacy cases and the retained v5 inventory, removing the
  obsolete July-inventory dependency. CUDA arithmetic and shipped weights
  are unchanged.
- Detector core 4.0.0 renames the native interface: the Python class to
  `PilotProxyKernel`, the C API to `PilotProxy_*`, macros to `PILOTPROXY_*`,
  the native library to `libpilotproxy.so`, and source files to
  `pilotproxy_detector.*` / `pilotproxy_reference.*`. Consumers must be
  rebuilt, since the earlier names are not aliases. Arithmetic and product
  schema v6 are unchanged. `docs/TERMINOLOGY.md` maps the names.
- `STACK_AMPLITUDE_RATIO*`, `temporal_gain` and `minimum_family_gain` replace
  `COHERENT_FRACTION*`, `coherent_gain` and `admissible_min_gain`. The third
  argument of `strength` is `stack_amplitude_ratio`, including keyword calls.
- Injection-recovery and result-summary schema v2 use `power_ratio` and
  `power_ratio_histogram_png`. Existing reports and weight bytes are unchanged.
- The threshold rule defaults to the paper's ideal calendar year, `1 / kappa`.
  To reproduce the duty-based times and decisions of the frozen records, pass
  `calendar_factor=LEGACY_CALENDAR_FACTOR` (CLI: `--calendar-factor
  6.574444324363269`). The paper's outcomes and words are the same under the
  ideal-year convention. Saved records are not rewritten.
- `--policy-budget` accepts the combined nominal-strength caps, and
  `--finite-budget` is an alias. Empty recent-year support rejects the
  candidate instead of raising an indexing error. Invalid budgets and time
  factors are refused.
- The CHIME receiver profile states its channelizer response as
  `sinc_hamming`: CHIME's F-engine is a 4-tap, 2048-point PFB with a
  sinc-Hamming window (Amiri et al. 2024, arXiv:2408.00172). The CHORD
  profiles keep `implementation_specific`.
- `channelizer.frequency_axis.nyquist_zone` replaces `spectral_sense`. The
  channel-internal spectrum is mirrored when the zone is even: CHIME samples
  400-800 MHz at 800 MS/s in zone 2 (mirrored), CHORD samples in zone 1, and
  the reference testbench mixes its band down to zone 1. The detector weight
  coordinate is `post_nyquist_zone_normalization` (input
  `post_nyquist_zone_normalized`), and `--nyquist-zone` replaces
  `--spectral-sense` in `evaluate-snr`, `quantize` and the study tools. In
  Python, `ReceiverProfile.nyquist_zone`, the `nyquist_zone=` keywords and
  `apply_nyquist_zone_to_detector_matrix` replace the `spectral_sense` forms.
- Older profiles, weight manifests, runtime bundles, products and command
  lines that use `spectral_sense` or the `post_spectral_sense_*` names are
  still read, and a manifest's receiver-profile digest of the old
  serialization still binds its profile. Nothing writes the old names.
  Runtime bundles keep every key Kotekan reads.
- The archive and validation studies share one exact sorted-rank scoring
  routine, and JSON validation and capture-file hashing share helpers. The
  independent counting reference and the CUDA arithmetic are unchanged.
  Invalid UTF-8 JSON is reported as a validation error.
- The capture-ratio study driver moves to
  `tools/archive/derive_stack_amplitude_ratio.py`.
- The shipped weight manifests are regenerated, and both weight banks are
  byte-identical. The manuals use the current detector terminology and show
  the implemented Kotekan mask path, and the documentation describes the
  current code. Incremental builds track the CUDA twiddle header. No result
  changes.
- Provenance records name the rewritten RFIsher and RadioFisher commits. No
  result changes.

## 1.0.0, October 22, 2026

Initial public release: detector core 3.0.0 and per-pilot product schema v6.

- Frame powers are the all-bin sums of the exact fine powers,
  `P_r = sum_f S_r[f]`, with `S_r` the uint64 output of the frozen
  `fxfft256_v1` summed over streams. The coarse ratio `F = 2 P_0 / (P_1 + P_2)`
  and the fine statistic come from one spectrum. Detector contract revision 2
  (`pilotproxy_detector_contract_v2`) states this as
  `coarse_power_definition = sum_of_fxfft256_fine_powers`.
- The kernel has no row-domain power branch. The fused entries'
  `d_power_out` carries the fine-power sums, and the staged path sums the
  fine-power entry on the host. Both equal
  `fxfft.fine_power_fx(rows).sum(axis=-1)` exactly. `PilotProxyKernel` refuses a
  library of another core major before binding any entry.
- Per-pilot products (`pilotproxy_per_pilot_product_v6`) keep the exact fine
  terms `fine_power_u64` (`[N, 3, 256]` uint64) with the unsummed
  `p_ref_lower_u64` / `p_ref_upper_u64`, the per-frame PSD
  `psd_frame_db_i16`, and per-frame `railed_sample_count` and
  `fill_sample_count` against `railed_sample_total = 2 * nfft * streams`.
  Fine products are mandatory for every in-band pilot. The writer, the
  validator, `products/reader.py` and `tools/audit_per_pilot.py` enforce
  `p_r[:, 0] == fine_power_u64[:, r, :].sum(axis=1)` exactly.
- Readers open v5 (contract revision 1) and v6 (contract revision 2) products.
  A v5 view replaces the stored row-domain columns by `coarse_columns_from_fine`
  and says so in `frame_power_definition`. The threshold rule and the
  archive-health gate read the excess and ratios through
  `product_contract.frame_powers_from_fine`, `pilot_excess_from_fine_powers`
  and `coarse_columns_from_fine`. Earlier product schemas are refused.
- CHIME run outputs of detector contract revision 1 are validated as records.
  The CHIME plots and `injection_recovery` refuse them, and runtime weight
  bundles must carry contract revision 2.
- `chime-run` stores `fine_power_u64` and the reference split in
  `chime_detector_outputs.npz`, and refuses a detector result whose powers are
  not the all-bin sums of its fine powers.
- Schema tokens are `pilotproxy_result_schema_v2`,
  `pilotproxy_detector_core_profile_v3` and
  `pilotproxy_snr_validation_report_v2`.
- `characterize era-boundaries --line-identity` also cuts an era where the
  pilot line's identity changes: a steady line that jumps by more than the
  marker half-width in one month, or a steady line that becomes unsteady,
  backed by a level step larger than any step the same line makes on that
  channel. Drift of one bin per month never cuts.
