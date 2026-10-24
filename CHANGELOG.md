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
  byte-identical. Documentation describes the current code and drops
  development history. No result changes.

## 3.0.0, October 23, 2026

- Remove the unused `COHERENT_FRACTION*`, `coherent_gain`, and
  `admissible_min_gain` Python aliases. Use `STACK_AMPLITUDE_RATIO*`,
  `temporal_gain`, and `minimum_family_gain`. The third argument of `strength`
  is now `stack_amplitude_ratio`, including keyword calls.
- Share exact sorted-rank scoring between the archive and validation studies;
  the independent counting reference and the CUDA arithmetic are unchanged.
- Share JSON validation and capture-file hashing helpers. Invalid UTF-8 JSON
  is reported as a validation error.
- Correct detector terminology in both manuals and show the implemented
  Kotekan mask path.
- Track the CUDA twiddle header in incremental builds and fix standalone
  WVU diagram color initialization.

## 2.0.0, October 23, 2026

- Detector core 4.0.0 renames the Python interface to `PilotProxyKernel`, the
  C API to `PilotProxy_*`, macros to `PILOTPROXY_*`, the native library to
  `libpilotproxy.so`, and source files to `pilotproxy_detector.*` /
  `pilotproxy_reference.*`. Rebuild consumers; the earlier names are not
  aliases. Arithmetic and product schema v6 are unchanged. `docs/TERMINOLOGY.md`
  maps the names.
- Injection-recovery and result-summary schema v2 use `power_ratio` and
  `power_ratio_histogram_png`; existing reports and weight bytes are unchanged.
- `STACK_AMPLITUDE_RATIO`, `temporal_gain` and `minimum_family_gain` replace
  the coherence names; the coherence names remain aliases in this release.
- The threshold rule defaults to the paper's ideal calendar year, `1 / kappa`.
  To reproduce the duty-based times and decisions of the frozen records, pass
  `calendar_factor=LEGACY_CALENDAR_FACTOR` (CLI: `--calendar-factor
  6.574444324363269`). The paper's outcomes and words are the same under the
  ideal-year convention. Saved records are not rewritten.
- `--policy-budget` accepts the combined nominal-strength caps;
  `--finite-budget` is an alias. Empty recent-year support rejects the
  candidate instead of raising an indexing error. Invalid budgets and time
  factors are refused.
- The capture-ratio study driver moves to
  `tools/archive/derive_stack_amplitude_ratio.py`.

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
  fine-power entry on the host; both equal
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
  and says so in `frame_power_definition`; the threshold rule and the
  archive-health gate read the excess and ratios through
  `product_contract.frame_powers_from_fine`, `pilot_excess_from_fine_powers`
  and `coarse_columns_from_fine`. Earlier product schemas are refused.
- CHIME run outputs of detector contract revision 1 are validated as records;
  the CHIME plots and `injection_recovery` refuse them, and runtime weight
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
