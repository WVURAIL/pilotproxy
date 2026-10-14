# Changelog

All notable changes to PilotProxy are recorded here. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
