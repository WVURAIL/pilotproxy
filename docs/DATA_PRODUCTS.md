# PilotProxy Data Products

This document defines the canonical files produced by the CHIME real-data
workflows. `chime-run` writes the products directly from one staged directory.
`chime-scan` first writes one authoritative `<freq_id>.npz` product per coarse
channel and then combines those products on a shared event/frame identity. The
two paths use the same canonical detector, cache, reduction, table, and plotting
formats where those formats apply.

Unless stated otherwise, `mask = 1` means that the frame is rejected and
`valid = 1` means that the reference denominator is nonzero.

The current per-pilot schema and its explicit active/diagnostic/candidate
decision contract are defined in `PRODUCT_SCHEMA.md`. Its fine-reduction
products (`fine_power_u64`, the exact `uint64` target, lower, and upper
terms per fine bin from the frozen `fxfft256`, plus the fine scalars
`fine_status`, `fine_num_bins`, `fine_pad_factor`, `fine_guard_fine_bins`,
`fine_p_fa`, `fine_designated_bins`, and `fine_census_excluded_bins`) live
only in the authoritative per-pilot `<freq_id>.npz` files. The null-bulk
calibration and the detection list are not stored; they are recomputed
offline from `fine_power_u64`. A `chime-run` output also stores the fine
powers of its frame powers in `chime_detector_outputs.npz` (below); a
`chime-scan` combine does not copy them, and analyses of fine detections read
the per-pilot products directly.

## JSON Files

### `run_config.json`

This file records the run-level detector contract and provenance. Both workflows
use:

- `schema_version = pilotproxy_chime_run_config_v1`;
- `detector_contract`;
- `physical_channels`;
- `mask_policy`;
- `reference_placement_summary` when placement metadata are available.

The staged `chime-run` path also records fields such as `input_dir`,
`output_dir`, `weight_coordinate`, and file-level `provenance`. The combined
`chime-scan` path instead records `source = chime-scan`,
`freq_id_by_pilot`, and `detector_provenance_by_pilot`. Consumers should use
the schema and named fields rather than assume that the two producers have
byte-identical JSON.

### `stats.json`

This file records the detector geometry and run statistics. Shared fields
include:

- `schema_version = pilotproxy_chime_stats_v1`;
- `detector_contract`;
- `num_frames` and `num_pilots`;
- `num_input_streams`;
- `windows_per_stream`;
- `mask_policy`;
- `reference_placement_summary` when available.

`chime-run` also records `detector_rows_per_frame`, `kernel_specs`,
`weight_coordinate`, and `null_power_ratio_by_channel`. A `chime-scan` combine records
`combine_alignment`, `rational_overflow_count_by_pilot`, and any cross-build
provenance notes. The combined scan keeps `null_power_ratio` in
`chime_detector_outputs.npz` rather than copying it into `stats.json`.

The `detector_contract` states the coordinate convention through:

- `weight_coordinate_system`;
- `input_coordinate_system`;
- `input_preprocessing.time_reverse_detector_windows_before_kernel`.

### `input_manifest.json`

This file records the HDF5 inputs using one of two explicit document shapes:

- the staged runner writes `pilotproxy_chime_input_manifest_v1` with
  `input_dir`, `absolute_time_used`, and discovered `datasets` metadata;
- the archive combine writes receiver-neutral
  `pilotproxy_scan_input_manifest_v1` with `source`, `physical_channels`, and
  the `input_files` unit keys collected from the per-pilot products.

The distinct identities prevent consumers from interpreting one manifest shape
as the other.

### `product_validation.json`

`pilot-proxy validate-products --output-json ...` writes this report; it is not
created merely by running the detector. Its principal fields are:

- `valid`;
- `num_errors`;
- `errors`;
- `detector_contract_revision`: 2 for a current run. A run written before
  kernel core 3.0.0 carries revision 1 (row-domain frame powers, no fine
  powers to recompute them from); it is validated as a record and the report
  adds a `record_note`. The CHIME plots and `injection_recovery` refuse such a
  run rather than analyse it under the current definition.

### `.pilotproxy-combine-generation.json`

`chime-scan`/`chime-combine` publishes this hidden generation manifest last.
It contains a unique generation ID and SHA-256 digest for every canonical file
in that combined output set. A failed publication that is rolled back also
advances the generation ID. `validate-products` reads the identity before and
after validation, checks the file digests, and refuses a run if publication
overlapped its reads. The hidden `.pilotproxy-combine-publish.json` journal is
present only while a transaction needs completion or recovery and also makes
validation fail closed. A kernel-backed exclusive publish lock serializes
writers across recovery, journal preparation, and canonical replacement; a
second writer cannot replace or remove the active writer's journal.

Direct `chime-run` outputs predate and do not require the combine-generation
manifest; in the absence of both combine metadata files they retain their
existing validation behavior.

## Detector NPZ

### `chime_detector_outputs.npz`

This is the canonical frame-by-pilot detector product.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `frame_index` | `(num_frames,)` | `int64` | frame | Contiguous positional frame index |
| `p_target_u64` | `(num_frames, num_pilots)` | `uint64` | power | Target frame power: the all-bin sum of the target term's exact fine powers |
| `p_ref_lower_u64` | `(num_frames, num_pilots)` | `uint64` | power | Lower reference frame power: the all-bin sum of the lower term's exact fine powers |
| `p_ref_upper_u64` | `(num_frames, num_pilots)` | `uint64` | power | Upper reference frame power: the all-bin sum of the upper term's exact fine powers |
| `p_ref_sum_u64` | `(num_frames, num_pilots)` | `uint64` | power | `p_ref_lower_u64 + p_ref_upper_u64` |
| `fine_power_u64` | `(num_frames, num_pilots, 3, 256)` | `uint64` | power | `chime-run` only: the exact `fxfft256` fine powers (target, lower, upper) summed over input streams, whose all-bin sums are the three frame powers |
| `coarse_power_ratio` | `(num_frames, num_pilots)` | `float64` | unitless | `2*p_target/p_ref_sum` |
| `normalized_coarse_power_ratio_db` | `(num_frames, num_pilots)` | `float64` | dB | `10*log10(F/null_power_ratio)` |
| `pilot_excess_db` | `(num_frames, num_pilots)` | `float64` | dB | One-bin pilot-excess PNR |
| `estimated_data_shelf_snr_db` | `(num_frames, num_pilots)` | `float64` | dB | Estimated ATSC data-shelf SNR; finite only where its transform is defined |
| `valid` | `(num_frames, num_pilots)` | `uint8` | 0/1 | `p_ref_sum != 0` |
| `mask` | `(num_frames, num_pilots)` | `uint8` | 0/1 | `1 = reject` under the recorded mask rule |
| `target_norm_sq` | `(num_pilots,)` | `int64` | unitless | Exact `||w_target||^2` of the int4 weights |
| `reference_norm_sum_sq` | `(num_pilots,)` | `int64` | unitless | Exact `||w_ref_lo||^2 + ||w_ref_up||^2` |
| `null_power_ratio` | `(num_pilots,)` | `float64` | unitless | `2*target_norm_sq/reference_norm_sum_sq`, the weight-norm H0 reference |
| `normalized_pilot_excess` | `(num_frames, num_pilots)` | `float64` | unitless | `F/null_power_ratio - 1`, or NaN for invalid frames |

The current mask is the norm-corrected positive-excess comparison:

```text
valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the integer form of `F > null_power_ratio`. The frame powers are
`P_r = sum_f S_r[f]` over the exact `fxfft256`
fine powers (`METHOD_SPEC.md`). A `chime-run` output stores those fine powers
in `fine_power_u64`, so every frame power can be re-formed from them: the
writer and `validate-products` require `p_target_u64`, `p_ref_lower_u64` and
`p_ref_upper_u64` to equal `fine_power_u64[..., r, :].sum()` exactly and
`p_ref_sum_u64` to equal their reference sum. A `chime-scan` combine stores
the reference split but leaves the fine powers in its per-pilot products (a
combined copy would hold `num_frames x num_pilots x 768` words). Outputs of
detector contract revision 1 (detector cores before 3.0.0) carry row-domain
powers `sum_m |z|^2` in the frame-power fields and neither the
split nor the fine powers; `chime-run` outputs identify them by
`kernel_version` and `coarse_power_ratio_definition` in `stats.json`, and
`chime-scan` combines by the per-pilot product schema (v5 or v6). Because
`normalized_coarse_power_ratio_db` is referenced to `null_power_ratio`, the
null and the positive-excess boundary both sit at exactly 0 dB. The current
product contract requires the exact powers, all norm-related arrays, schema
identity, and decision contract. Products that do not meet it are not accepted;
regenerate them from authoritative inputs.

For `chime-scan`, `num_frames` is the event/frame intersection retained by the
combine. The source per-pilot products can contain additional frames that were
not common to every channel.

## Spectrogram Cache NPZ

### `chime_spectrogram_cache.npz`

This cache carries the frame-level baseband power and the matching detector
mask used by the plotting functions.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `baseband_power_linear` | `(num_frames, num_pilots)` | `float64` | power | Mean non-coherent baseband power for the frame |
| `baseband_power_db` | `(num_frames, num_pilots)` | `float64` | dB | `10*log10(baseband_power_linear)` where power is positive |
| `mask` | `(num_frames, num_pilots)` | `uint8` | 0/1 | Detector rejection mask copied from the detector product |
| `valid` | `(num_frames, num_pilots)` | `uint8` | 0/1 | Detector validity copied from the detector product |
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `frame_index` | `(num_frames,)` | `int64` | frame | Contiguous positional frame index |
| `relative_time_s` | `(num_frames,)` | `float64` | s | `frame_index*nfft/sample_rate_hz` |

`relative_time_s` is accumulated data time. It does not restore gaps between
separate archive events and should not be interpreted as wall-clock time.

## Integrated Spectra NPZ

### `chime_integrated_spectra.npz`

The `chime-scan` combine writes a reporting stack of the integrated spectra in
the per-pilot products. The analyzer uses a rectangular-window FFT, sums
`|FFT|^2` over input streams, and accumulates over frames. `before` includes
valid frames; `after` includes valid frames with `reject_mask = 0`. Therefore
`before - after` is the accumulated spectrum of the rejected frames.

| Array | Shape | Dtype | Units | Meaning |
|---|---:|---|---|---|
| `schema_version` | scalar | `str` | — | `pilotproxy_chime_integrated_spectra_v1` |
| `physical_channel` | `(num_pilots,)` | `int32` | channel | ATSC physical channel |
| `pilot_frequency_hz` | `(num_pilots,)` | `float64` | Hz | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(num_pilots,)` | `float64` | Hz | CHIME coarse-channel center |
| `freq_id` | `(num_pilots,)` | `int64` | id | CHIME coarse-channel identifier when recorded |
| `integrated_spectrum_before_mask` | `(num_pilots, nfft)` | `float64` | power | Sum over valid frames |
| `integrated_spectrum_after_mask` | `(num_pilots, nfft)` | `float64` | power | Sum over valid, kept frames |
| `masked_fraction_by_channel` | `(num_pilots,)` | `float64` | 0..1 | Rejected valid frames divided by valid frames; NaN when none are valid |
| `sample_rate_hz` | scalar | `float64` | Hz | Shared per-channel sample rate, or NaN when timing metadata are unavailable |
| `nfft` | scalar | `int64` | bins | FFT length recorded by the per-pilot products |

Bin `k` maps to baseband frequency as:

```text
((k + nfft//2) % nfft - nfft//2) * sample_rate_hz / nfft
```

The authoritative copy remains in each `_per_pilot/<freq_id>.npz`; see
`PER_PILOT_PRODUCT_FIELDS.md`. Integrated spectra are accumulated before terminal
event intersection, so they represent each pilot's full processed frame set.
The canonical frame arrays can represent a smaller all-channel intersection.
This distinction is recorded by the per-pilot products and
`stats.json.combine_alignment`.

## Reductions NPZ

### `chime_reductions_10s.npz`

This file groups the canonical frame arrays into approximately 10 s of
contiguous data time. The grouping uses `frame_index`, the analysis frame length,
and the sample rate recorded by the products; it does not use the per-file
absolute-time axis.

All arrays below have a leading dimension of `num_chunks`. Arrays that also
vary by pilot have shape `(num_chunks, num_pilots)`.

- `chunk_index`, `chunk_start_frame`, `chunk_stop_frame`;
- `input_power_mean`, `cleaned_power_mean`;
- `mask_fraction`, `mask_fraction_valid`, `mask_fraction_total`;
- `unmasked_count`, `total_count`, `valid_count`, `invalid_count`;
- `masked_count_valid`, `unmasked_count_valid`;
- `normalized_coarse_power_ratio_db_median`, `normalized_coarse_power_ratio_db_p95`, `normalized_coarse_power_ratio_db_max`;
- `estimated_data_shelf_snr_db_median`, `estimated_data_shelf_snr_db_p95`, `estimated_data_shelf_snr_db_max`.

`cleaned_power_mean` is the mean over valid frames with `mask = 0`; it is NaN
when a chunk contains no such frame. The term “cleaned” is a product-field name,
not a claim that all interference has been removed.

## Frame Identity NPZ

### `chime_frame_identity.npz`

An event-keyed `chime-scan` combine writes the identities retained in the
canonical stack:

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `frame_event_key` | `(num_frames,)` | `str` | Source event identity with the per-channel `freq_id` token removed |
| `frame_in_unit` | `(num_frames,)` | `int64` | Frame position within the source file |

The event key, per-unit index, and frame-within-unit identity are mandatory.
Products without those tags are rejected rather than positionally aligned.

## Tables

Tables are written under `tables/`:

- `mask_summary_by_pilot.csv` is written by the detector/combine path;
- `spectrum_before_after.csv` is written by `chime-run` and regenerated by the
  baseband spectrum plot;
- `normalized_coarse_power_ratio_summary_by_channel.csv` and
  `data_shelf_snr_histogram_summary.csv` are written by `chime-plot` or
  `chime-run --plot`.

The data-shelf SNR summary is derived from the norm-corrected pilot excess and therefore uses the same null coordinate as `reject_mask`.

## Figures

`pilot-proxy chime-plot` or `chime-run --plot` writes these figures under
`figures/`:

- `data_shelf_snr_histogram_by_channel.png`;
- `coarse_power_ratio_survival_by_channel.png`;
- `normalized_coarse_power_ratio_db_spectrogram.png`;
- `baseband_spectrogram.png`;
- `baseband_spectrum_before_after_mask.png`;
- `mask_spectrogram.png`.

PNG is the default. Setting `PILOT_PROXY_FIGURE_FORMATS=png,pdf` adds PDF copies
without changing the numerical products.
