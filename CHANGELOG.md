# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 3.0.1, October 23, 2026

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
  serialization still binds its profile; nothing writes the old names.
  Runtime bundles keep every key Kotekan reads.
- The shipped weight manifests are regenerated; both weight banks are
  byte-identical. No result changes.

## 3.0.0, October 23, 2026

- Remove the unused `COHERENT_FRACTION*`, `coherent_gain`, and
  `admissible_min_gain` Python aliases. Use `STACK_AMPLITUDE_RATIO*`,
  `temporal_gain`, and `minimum_family_gain`. The third argument of `strength`
  is now `stack_amplitude_ratio`, including keyword calls.
- Share exact sorted-rank scoring between the archive and validation studies;
  retain the independent counting reference and unchanged CUDA arithmetic.
- Share JSON validation and capture-file hashing helpers. Invalid UTF-8 JSON
  is reported as a validation error.
- Correct detector terminology in both manuals and show the implemented
  Kotekan mask path. Preserve the superseded interface design in `docs/archive/`.
- Track the CUDA twiddle header in incremental builds and fix standalone
  WVU diagram color initialization.

## Handoff cleanup, October 23, 2026

- Package 2.0.0 / core 4.0.0 renames the Python interface to
  `PilotProxyKernel`, C API to `PilotProxy_*`, macros to `PILOTPROXY_*`,
  native library to `libpilotproxy.so`, and source files to
  `pilotproxy_detector.*` / `pilotproxy_reference.*`. Rebuild consumers;
  the removed names are not aliases. Arithmetic and product schema v6 are
  unchanged. See `docs/TERMINOLOGY.md` for the migration table.
- Injection-recovery and result-summary schema v2 use `power_ratio` and
  `power_ratio_histogram_png`; historical reports and weight bytes stay intact.
- `STACK_AMPLITUDE_RATIO`, `temporal_gain` and `minimum_family_gain` replace
  misleading coherence names in active code. Old imports remain aliases.
- The threshold rule defaults to the paper's ideal calendar year, `1 / kappa`.
  To reproduce historical duty-based times and decisions, pass
  `calendar_factor=LEGACY_CALENDAR_FACTOR` (CLI: `--calendar-factor
  6.574444324363269`). Recorded paper outcomes and words are unchanged under
  the ideal-year convention. Saved records are not rewritten.
- `--policy-budget` accepts the combined nominal-strength caps;
  `--finite-budget` remains an alias. Empty recent-year support rejects the
  candidate instead of raising an indexing error. Invalid budgets and time
  factors are refused.
- Installation detail is in `docs/USAGE.md`; current equations and tests are
  indexed in `docs/PAPER_CODE_MAP.md`. The old capture-ratio study driver is
  under `tools/archive/` and the tools catalog distinguishes historical work.

## Kernel core 3.0.0, per-pilot product schema v6 --- frame powers are fine-power sums

The three frame powers are now defined as the all-bin sums of the exact fine
powers, `P_r = sum_f S_r[f]`, with `S_r` the uint64 output of the frozen
`fxfft256_v1` summed over streams. The coarse ratio
`F = 2 P_0 / (P_1 + P_2)` keeps its form (the transform length cancels), and
the coarse ratio and the fine statistic now come from one spectrum.

Kernel (core 3.0.0, ABI break): the row-domain branch is removed.
`FStat_Compute_Powers_U64`, `FStat_Compute_Powers`,
`FStat_Compute_DiagnosticFloat` and the coarse rational-half mask entries are
gone, `FStat_Create` and `FStat_Create_Batch` no longer take a float output
buffer, and the fused entries' `d_power_out` carries the fine-power sums. The
Python wrappers follow (`create_raw(rows, d_in)`; `compute_powers*`,
`compute_diagnostic_float` and `compute_numden_mask_*` are removed). The staged
path sums `FStat_Compute_FinePowers_U64` on the host; both equal
`fxfft.fine_power_fx(rows).sum(axis=-1)` exactly.

Products: schema v6 (`pilotproxy_per_pilot_product_v6`) keeps every field
name; `p_target_u64`, `p_ref_lower_u64` and `p_ref_upper_u64` are the fine
sums and every derived column follows from them. The writer,
`validate_current_product_identity`, `products/reader.py` and
`tools/audit_per_pilot.py` enforce `p_r[:, 0] == fine_power_u64[:, r, :].sum(axis=1)`
exactly. Fine products are mandatory for every in-band pilot:
`fine_products` accepts only `on`, and the only status without fine terms is
`not_applicable_pilot_out_of_band`. Detector contract revision 2
(`pilotproxy_detector_contract_v2`) states the new definition and replaces
`v1_marginal_identity` with `coarse_power_definition =
sum_of_fxfft256_fine_powers`.

The v5 products of record are not edited or migrated. Readers open v5
(contract v1) and v6 (contract v2); a v5 view replaces the stored row-domain
columns by `coarse_columns_from_fine` and says so in
`frame_power_definition`. New in `product_contract`: `frame_powers_from_fine`,
`pilot_excess_from_fine_powers`, `coarse_columns_from_fine`. The threshold
rule and the archive-health gate read the excess and the ratios through them.
Re-deriving the decisions of record this way changes no word, outcome or
reason.

Removed: `fine_reduction.exact_coarse_power_by_term`,
`check_coarse_power_marginal_identity` and the
`coarse_power_ratio_from_marginal` field, and the per-frame marginal tripwire.
The float `fine_reduce` keeps its float Parseval identity as a property of the
float prototype only. `detector_reference.coarse_power_ratio_cpu_reference_packed`
now runs rows -> fxfft256 -> exact sums, so the CPU-only sweep keeps the
kernel's semantics. Tools and integrations that formed `sum |z|^2`
(`capture_reference`, the SDR upgrade adapter and streaming adapter,
`testbench/fine_validation_gpu`, `evaluate_snr`, `fused_real_receipt`,
`measure_fine_gain`, `mask_residual_frontier`, `characterize_capture_transfer`)
now use the fine-power sums; their receipt and shard schemas were bumped where
the stored values changed.

`PilotProxyKernel` refuses a library whose core major is not 3 before binding any
entry (a 2.x `FStat_Create` takes an output buffer the 3.0.0 binding lacks).
Schema tokens whose meaning changed are bumped: the result schema to
`pilotproxy_result_schema_v2`, the detector-core profile to
`pilotproxy_detector_core_profile_v3` (its v2 token is a retired name; the
profile drops `mag_squared_accumulator_bits_target`, the int32 `|z|^2`
register that went with the row-domain branch) and the `evaluate-snr` report to
`pilotproxy_snr_validation_report_v2`; `load_evaluation` and the clean-term
conditioning in `plot_results` refuse the earlier revisions.

`chime-run` now stores the fine powers of its frame powers in
`chime_detector_outputs.npz`: `fine_power_u64` of shape
`(num_frames, num_pilots, 3, 256)` and the reference split `p_ref_lower_u64`,
`p_ref_upper_u64`. The runner refuses a detector result whose powers are not
the all-bin sums of its fine powers; the writer and `validate-products` check
that identity exactly for every contract-revision-2 output (a `chime-scan`
combine carries the split and leaves the fine powers in its per-pilot
products).

The fine-validation tools (`reduce_fine_validation.py`,
`report_fine_null_validation.py`, `report_fine_sensitivity.py`) form the
frame-level ratios from `fixed_fine.sum(-1)` and `float_fine.sum(-1)`; the raw
shards' coarse members are authenticated but no longer read.
`mask_residual_frontier.py --stage rederive` re-expresses a stored frontier
study (such as `docs/evidence/mask_residual_frontier_2026-09-07`) under the
fine-sum frame power after reproducing its stored excess bit for bit from its
own stored powers, and the report stage accepts a stored study re-reported on
another machine (same identity apart from the software block and input paths).

CHIME run outputs written before core 3.0.0 (detector contract revision 1,
among them the 2026 matched-capture reductions) carry row-domain powers and no
fine powers, so they cannot be re-expressed. `validate-products` validates
them as records and reports `detector_contract_revision` 1 with a
`record_note`; the CHIME plots and `injection_recovery` refuse them, and
runtime weight bundles must carry contract revision 2. The frame-power helpers
refuse a v5 product whose `fine_status` marks an all-zero fine placeholder.
The archive-health spectrum correction checks the stored integrated spectra
with the stored `valid` and `reject_mask` they were accumulated with (on a v5
product the row-domain decision) and reports the frames on which that mask
differs from the current one.


## Era boundaries by the pilot line's identity

`characterize era-boundaries --line-identity` also cuts an era where the pilot
line's identity changes: a steady line (monthly offsets within half the marker half-width)
that jumps by more than the marker half-width in one month, or a steady line
that becomes unsteady, backed by a level step larger than any step the same
line makes on that channel. Drift of one bin per month never cuts, nor does a
hand-over between the two offsets of a two-state line, nor an edge of a
reversing block. Without the flag the on/off list of 2026-10-14 reproduces byte
for byte. The era list of record moves to
`projects/chime_atsc/eras/line_identity_2026-10-17/`: channels 17, 18, 20 and
34 start later (2022-10, 2024-08, 2022-09, 2019-08); the other twenty are
unchanged.


## Per-pilot product schema v5 --- railed and fill counts per frame

`chime-scan` now records `railed_sample_count` and `fill_sample_count` per
frame against a shared `railed_sample_total = 2 * nfft * streams`. A 4-bit
component on either rail (nibble 0 or 15) has been clipped by the ADC or the
F-engine requantizer rather than measured; the `0x00` byte is the archive's
missing-data fill signature and is tallied separately, so a frame partially
overlapping absent baseband is not misrecorded as saturated. Neither count can
be recovered later because no raw samples are stored.

Schema v4 (railed counts without the fill split) existed for one day and no
cohort was produced with it; v4 snapshots are rejected like any other
development snapshot.

Measured on real CHIME baseband (freq_id 506): median 795 railed of 67,108,864
components per frame (0.0012%), worst frame 18,948 (0.028%).


## Per-pilot product schema v3 --- retain the per-frame PSD

`chime-scan` now records `psd_frame_db_i16`, the per-frame power spectrum with
feeds summed, as int16 in dB about a per-product reference.

The analyzer already computed this array every frame and threw it away, so the
change costs no compute. It closes the limitation `archive_health` documents:
the v1 gate cannot apply a new frame mask, window, or threshold to the two
archived accumulators, because the per-frame spectra were gone.

Measured on real CHIME data: 0.005 dB maximum round-trip error, and summing the
retained frames reproduces the integrated spectrum to 7.7e-4. About 18 kB per
frame after deflate.


## Per-pilot product schema v2 --- retain the exact fine terms

`chime-scan` now records `fine_power_u64` (`[N, 3, 256]` `uint64`, the frozen
`fxfft256` terms) plus the unsummed `p_ref_lower_u64` / `p_ref_upper_u64`.

v1 retained only the float32 `fine_power_ratio`, which is neither the deployed
statistic nor invertible to it, so no exact Q16 fine decision could be replayed
from an archived product. v1 products are rejected rather than migrated: the
missing integers cannot be reconstructed without reprocessing raw baseband.

Adds about 4.4 kB/frame compressed.


## Unreleased

This is the development line for PilotProxy 2.0. It deliberately removes
pre-2.0 Python aliases, duplicated command entry points, and receiver-specific
names from receiver-neutral contracts. Existing products must be regenerated;
the stricter validators do not reinterpret incomplete legacy products.

### Added

- `pilot-proxy-control`, a datatrawl analyzer for non-pilot control freq_ids
  (`pilotproxy_control_product_v1`): per-frame `baseband_power_linear` in the
  detector's native units, a per-frame K=128 rectangular coarse power marginal
  (the unit-modulus analogue of the deployed statistic, so
  `F(b) = 2*S[b] / (S[b-2] + S[b+2])` is recoverable offline at any bin,
  including a virtual-pilot bin on channels with no transmitter), one
  full-resolution integrated spectrum, and the same freq_id-stripped
  `source_event_keys` the combine step joins on. CPU-only by default
  (`--set gpu=1` optional and excluded from the resume fingerprint); the
  deployed detector remains the only mask authority.

### Changed

- Standardized programmatic frame-size naming on `frame_size_samples` while
  preserving the `--frame-size-samples` spelling and persisted format tokens.
- Centralized the PSD encoding step and fine-decision version instead of
  repeating their literal values in runtime writers.
- Made the test extra self-contained for plotting tests and expanded the
  configured Ruff check to the complete repository.
- Reference-placement diagnostics now say what the `edge_wrapped` bookkeeping
  flag means physically (`frame_origin_description`: under a center-at-DC
  profile the wrap crosses the coarse-channel center/DC, not the channel edge)
  and report references requested beyond +-fs/2 of the channel center
  (`*_crosses_channel_edge`, `channel_edge_notes`); `list-channels` prints the
  corresponding `NOTE`s, including for manifests written before these fields
  existed. Weight values and placements are unchanged; regenerating a manifest
  changes its hash only.
- Tightened receiver, stream-map, detector-core, runtime-bundle, and product
  compatibility validation.
- Replaced the receiver-specific CHIME detector-contract identity with the
  receiver-neutral `pilotproxy_detector_contract_v1` contract.
- Centralized duplicated detector preprocessing and schema identities.
- Updated generated specifications and operational documentation to describe
  the current product contract.
- Separated historical provenance from supported runtime behavior.

### Removed

- Pre-2.0 BAO-named Python/CLI aliases, detector wrapper aliases, and the
  `pilot_proxy.datatrawl_plugins` forwarding namespace.
- Hidden no-op detection flags that were accepted but never affected results.
- Unreferenced runtime and analysis helpers, stale compatibility constants,
  and the unused pandas dependency.
- The receiver-specific CHIME detector-contract token and its public builder
  aliases. Use `pilotproxy_detector_contract_v1`, `build_detector_contract`,
  and `validate_detector_contract`.
- Result-summary helpers from the `pilot_proxy.testbench` package root. Import
  them from `pilot_proxy.testbench.summarize_results`.
- The duplicate `scripts/fulldepth_and_subsets.py` analysis entry point. The
  historical paper provenance remains archived, but it is not a supported
  runtime command.
- Ambiguous SNR-summary aliases `num_positive_excess_frames` and
  `positive_excess_fraction`. The replacement columns state that they count
  finite data-shelf SNR estimates among detector-valid frames.

### Fixed

- CUDA tests now unpack the availability result correctly instead of treating
  its `(available, reason)` tuple as a truth value.
- Runtime invariants remain active under optimized Python, and the CANFAR
  launcher now exits unsuccessfully when a session misses its startup timeout.
- `chime-baseband-packed` now classifies open/schema failures as an
  unreadable unit via datatrawl's `unreadable_file()` (quarantine + continue),
  matching the built-in `chime-baseband` reader, instead of crashing an entire
  archive scan on one corrupt file.
- `tests/datatrawl` inventory fixtures write full current-schema rows
  (`scope`, `name`) required by datatrawl's fail-closed inventory validation,
  and the event-keyed combine test expects the source's logical unit key
  (scope/event/name triple) rather than the legacy URI form.

- Propagated CUDA API failures to Python callers.
- Prevented debug CUDA builds from being reused as release artifacts.
- Preserved receiver-specific sample rates throughout combined products.

## 1.0.0

- Initial public release.
