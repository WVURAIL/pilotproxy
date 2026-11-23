# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Every site that applies a fine_decision_v3 word uses the word the stage applies (`v3_word`): the power-limit
  word where calibration wrote one, and the default word otherwise. `compile-operating-policy` binds a v3 channel
  or control at that word, so a v3 control and a v3 channel with no power-limit word now compile; it used to
  require `power_limit_multiplier_q16` on every v3 row. `run-spectrogram`, `run-scores` and the `process-run`
  summary keep v3 frames at that word and name it in `word_q16`, `current_threshold` and `masked_summary.csv`, and
  `analyze_dtv_run.py` matches archive ranges at it. Bundles with no power-limit word, and v1 and v2 bundles, give
  the same outputs as before.
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
