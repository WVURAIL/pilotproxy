# Per-pilot product field reference

This document defines the fields of the per-pilot product, the only supported
public product of this kind. The product identity and the decision semantics
are defined in [`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md), and the arrays specific
to the fine reduction are listed in
[`FINE_REDUCTION_PRODUCTS.md`](FINE_REDUCTION_PRODUCTS.md).

```text
schema_name = "pilotproxy_per_pilot_product"
schema_revision = 6
schema_version = "pilotproxy_per_pilot_product_v6"
source_event_key_schema_version = "pilotproxy_namespaced_source_event_key_v1"
```

Schema v6 keeps every v5 field name and redefines the value of the frame
powers. The fields `p_target_u64`, `p_ref_lower_u64`, and `p_ref_upper_u64` are
the exact all-bin sums of `fine_power_u64`, and the derived columns follow from
them. The v5 products of record store row-domain powers in the same fields,
and readers recompute those columns from the stored fine powers (see
[`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md)).

The `pilot-proxy-detector` analyzer writes one product for one selected receiver
coarse channel. `reject_mask = 1` means reject. Event-keyed frame identity,
exact weight norms, detector provenance, and the decision contract are
mandatory. Products missing them are invalid and must be regenerated.

In the tables below, `N` is the number of detector frames and `U` is the number
of consumed input files.

## Identity and geometry

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `freq_id` | `(1,)` | `int64` | CHIME coarse-channel identifier |
| `physical_channel` | `(1,)` | `int32` | Nearest ATSC physical channel |
| `pilot_in_band` | `(1,)` | `uint8` | `1` when the selected coarse channel contains the nominal ATSC pilot |
| `pilot_frequency_hz` | `(1,)` | `float64` | ATSC pilot RF frequency |
| `chime_frequency_hz` | `(1,)` | `float64` | CHIME coarse-channel center |
| `nfft` | scalar | `int64` | Analysis frame and FFT length used for this product |
| `sample_rate_hz` | scalar | `float64` | Positive acquisition sample rate; finite `unit_delta_time` values must equal its reciprocal |
| `detector_window_samples` | scalar | `int64` | Receiver-selected CUDA detector window `K` (128 for CHIME; 64 for the current CHORD profiles) |
| `num_input_streams` | scalar | `int64` | Input feed/polarization streams summed |
| `sense` | scalar | `int64` | Nyquist-zone sign: `+1` for an odd (upright) zone, `-1` for an even (mirrored) zone |
| `source_event_key_schema_version` | scalar | `str` | Required namespaced event-identity contract; basename-only products are rejected |

The schema does not fix `nfft` to one value. The software profile and the
tests use 16384 samples, the frame of the planned CHIME engine upgrade. The
active acquisition value must be recorded from the run, and a current-frame
value of 12288 samples remains provisional until it is independently verified.
Any accepted value must be divisible by `K`.

If the nominal ATSC pilot does not fall within half a coarse-channel bandwidth,
the analyzer sets `pilot_in_band = 0`. It still emits one row per input frame,
but sets `valid = 0`, `reject_mask = 0`, and the integer detector powers to
zero. Derived detector values and `baseband_power_linear` are NaN, and neither
integrated spectrum receives the frame.

## Per-frame detector output (length `N`)

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `frame_index` | `(N,)` | `int64` | Zero-based positional frame counter |
| `p_target_u64` | `(N, 1)` | `uint64` | Target frame power `sum_f fine_power_u64[:, 0, f]` |
| `p_ref_sum_u64` | `(N, 1)` | `uint64` | Lower plus upper reference frame power |
| `p_ref_lower_u64` | `(N, 1)` | `uint64` | Lower reference frame power `sum_f fine_power_u64[:, 1, f]`, kept unsummed |
| `p_ref_upper_u64` | `(N, 1)` | `uint64` | Upper reference frame power `sum_f fine_power_u64[:, 2, f]`, kept unsummed |
| `fine_power_u64` | `(N, 3, 256)` when enabled; `(N, 0, 0)` for a pilot out of band | `uint64` | Exact fine-power terms from the frozen `fxfft256`; never a zero placeholder |
| `psd_frame_db_i16` | `(N, 16384)` | `int16` | Per-frame PSD, feeds summed, in dB about `psd_db_reference` |
| `coarse_power_ratio` | `(N, 1)` | `float64` | `2*p_target/p_ref_sum` |
| `normalized_coarse_power_ratio_db` | `(N, 1)` | `float64` | `10*log10(F)` |
| `pilot_excess_db` | `(N, 1)` | `float64` | One-bin pilot-excess PNR |
| `estimated_data_shelf_snr_db` | `(N, 1)` | `float64` | Estimated ATSC data-shelf SNR |
| `valid` | `(N, 1)` | `uint8` | `p_ref_sum != 0` |
| `reject_mask` | `(N, 1)` | `uint8` | `1 = discard` under the recorded positive-excess rule |
| `normalized_pilot_excess` | `(N, 1)` | `float64` | `F/null_power_ratio - 1`, or NaN when invalid |
| `target_norm_sq` | `(1,)` | `int64` | Exact `||w_target||^2` of the int4 weights |
| `reference_norm_sum_sq` | `(1,)` | `int64` | Exact `||w_ref_lo||^2 + ||w_ref_up||^2` |
| `baseband_power_linear` | `(N, 1)` | `float64` | Mean non-coherent baseband power for the frame |
| `railed_sample_count` | `(N, 1)` | `uint64` | 4-bit components on either rail (nibble 0 or 15), counted from the raw frame; components of `0x00` fill bytes are excluded and counted below |
| `fill_sample_count` | `(N, 1)` | `uint64` | Components in `0x00` bytes, the archive's missing-data fill signature (a genuinely doubly-negative-railed sample is indistinguishable and lands here) |
| `railed_sample_total` | `(N, 1)` | `uint64` | Shared denominator for both counts: `2 * nfft * streams` |
| `psd_db_step_per_code` | `()` | `float64` | dB per `psd_frame_db_i16` code; `0.01` |

Saturation of the ADC or F-engine is recorded per frame because it cannot be
recovered later. The product stores no raw samples (5.8 MB against 8.0 GB for
one pre-flight channel). `baseband_power_linear == 128` catches only a frame
that is railed everywhere, so partial saturation needs these counts. The counts
treat real and imaginary parts as distinct components and are written for
invalid frames too, since clipping is a property of the samples and does not
depend on the detection. The `0x00` byte is the documented missing-data fill
signature of the archive, the same signature that `archive_health` uses to
exclude whole frames. Its components are therefore counted in `fill_sample_count`
instead of being recorded as clipping. Without this split, a frame that
partially overlaps absent baseband would be permanently misrecorded as
saturated. On real CHIME baseband (freq_id 506), we find a median of 795 of
67108864 components (0.0012%), with the worst frame at 18948 (0.028%), a
frame-to-frame spread of 24 times.

`null_power_ratio` is not written to the product. It is exactly
`2*target_norm_sq/reference_norm_sum_sq` from two fields above, and a stored
copy could disagree with its own inputs. The formulas below that name it still
hold, with `null_power_ratio` computed from the norms.

### Fine-decision contract (scalars, one per product)

These scalars are written whenever the fine terms are computed. The scan
applies no fine decision itself. The scalars record the contract that the terms
were produced under, so that post-processing can decide and replay exactly.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `fine_status` | `()` | `str` | `enabled` for an in-band pilot; `pending` only for an empty checkpoint; `not_applicable_pilot_out_of_band` otherwise |
| `fine_num_bins` | `()` | `int64` | Fine transform length actually used (`256` when enabled, otherwise `0`) |
| `fine_pad_factor` | `()` | `int64` | Zero-padding factor of the fine transform (`2`) |
| `fine_guard_fine_bins` | `()` | `int64` | Guard bins excluded either side of the designated window (`1`) |
| `fine_p_fa` | `()` | `float64` | Declared per-bin false-alarm rate (`0.001`) |
| `fine_designated_bins` | `(D,)` | `int64` | Designated-window bin indices |
| `fine_census_excluded_bins` | `(E,)` | `int64` | Bins excluded from the census; empty when none |
| `decision_contract_json` | `()` | `str` | The whole contract as JSON, for exact replay |

A v5 product whose fine stage did not run stores an all-zero `(N, 3, B)`
placeholder. Its sums are not frame powers, so v6 has no such form. Resume
refuses a checkpoint that carries it, the frame-power helpers refuse such a
v5 product, and an out-of-band pilot stores exactly `(N, 0, 0)` with zero
bins.

### Per-unit receiver state (length equal to `unit_order`)

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `unit_scope` | `(U,)` | `str` | Source archive disposition; local inputs declare `local` |
| `archive_version` | `(U,)` | `str` | Raw HDF5 `archive_version`, or empty when absent |
| `unit_git_version_tag` | `(U,)` | `str` | Raw HDF5 `git_version_tag`, or empty when absent |
| `unit_input_map_sha256` | `(U,)` | `str` | Typed-content SHA-256 of HDF5 `index_map/input`, or empty when absent |
| `unit_collection_server` | `(U,)` | `str` | Raw HDF5 `collection_server`, or empty when absent |

`unit_scope` has one value per unit, so it must be indexed by unit and never
by frame number. It is populated from the inventory row of the source. Every
unit of a current product has a nonempty value, and a `--source local` run
uses the declared value `local`.

The input-map digest hashes the dtype, shape, and C-order values of the
dataset. The dtype signature records field names, offsets, item size, subarray
shape, and primitive dtype strings and ignores library-specific dtype metadata.
The canonical compact JSON header is prefixed by its unsigned eight-byte
big-endian length and followed by the C-order value bytes, and the complete
stream is hashed with SHA-256. This digest allows for receiver input
configurations to be distinguished without copying the full map into every
product.

General local products may record empty receiver attributes when the source
file does not contain them. Archive smoke, rehearsal, checkpoint, and closeout
gates require nonempty `unit_scope`, `unit_git_version_tag`, and
`unit_input_map_sha256` for every unit. The receiver configuration is the tuple
of `archive_version`, `unit_git_version_tag`, and `unit_input_map_sha256`,
while the scope is acquisition bookkeeping and the collection server is
provenance. The tuple marks the receiver-configuration boundary, so it can be
intersected with the station era, which is derived on its own. For a common
event, combine also requires `unit_scope`, `archive_version`,
`unit_git_version_tag`, and `unit_input_map_sha256` to agree over channels
when those values are present.

`weight_coefficients_sha256` `()` `str` is listed under Provenance below.

The product stores `p_target_u64` and `p_ref_sum_u64` without converting them to
a thresholded statistic. In a v6 product, they are the all-bin sums of the
stored `fine_power_u64`, so a reader can also re-form them from the fine terms.
`products/reader.py` does exactly that and refuses any disagreement. Thus, an
alternative F threshold or dB calibration can be recomputed from the same
detector pass, provided the required calibration constants are also used, and
the CUDA detector does not need to be rerun.

### Per-frame PSD (schema v3)

The analyzer runs one 16384-point FFT over all 2048 feeds in every frame and
sums its power. Schemas v1 and v2 folded that array into
`integrated_spectrum_*` and discarded the per-frame result.
`archive_health.SPECTRAL_LIMITATION` states the consequence: the gate "cannot
apply an arbitrary new frame mask, FFT window, or threshold to the archived
spectra."

Keeping the per-frame PSD therefore costs storage and no extra computation. It
is decoded with:

```python
psd = psd_db_reference * 10.0 ** (psd_frame_db_i16 / 1000.0)
psd[psd_frame_db_i16 == psd_db_invalid_code] = numpy.nan
```

Frames that never reached the transform carry `psd_db_invalid_code`
(`-32768`). This code lies past the clipped code range by design, so it cannot
be confused with a measured level.

The int16 encoding in dB was chosen from measurement. On a real channel 36
capture, the PSD spans only 18.3 dB, so steps of 0.01 dB give a round-trip
error of at most 0.005 dB (0.0025 dB median). This error is far below the
precision needed to locate peaks or place detector cell boundaries, and the
encoding is half the size of `float32` and a third of its deflated size. `float16` cannot be used, because
the values are of order 10^8 to 10^9 and every bin overflows.

The sum of the stored frames reproduces `integrated_spectrum_before_mask` to
7.7e-4 relative, the quantization limit. The accumulators that the product
still carries can therefore be derived from the per-frame PSD.

### Exact fine terms (schema v2)

`fine_power_u64` is the sufficient statistic for every fine decision. It holds
the three exact `uint64` power terms per fine bin (target, lower reference, and
upper reference) produced by the frozen fixed-point `fxfft256`, the transform
used by the deployed kernel.

Schema v1 kept only `fine_power_ratio`, one `float32` per bin, formed with a
floating-point numpy FFT. Two pieces of information were lost, and neither can
be recovered without reprocessing the raw archive. First, the three `uint64`
values were collapsed into their ratio. The frozen decision
`fine_mask_decision` requires `uint64 [3, 256]` and rejects any other dtype, so
a v1 product cannot replay the deployed decision at any threshold. Second, the
float FFT and `fxfft256` disagree by up to about 5e-5 relative, so the stored
ratio is not the deployed quantity even in principle.

With `fine_power_u64` stored, every quantity downstream of the transform can be
recomputed exactly offline. These are the ratio, the CFAR location, scale, and
threshold, the null bulk, the designated-set decision, and any future `eta` or
Q16 multiplier. The terms cannot be recovered later, and a rescan takes weeks.

The cost is about 4.4 kB per frame after npz deflation (6144 B raw), roughly
3.3 GB for an archive of about 750000 frames. Observed values need 46 bits at
the 2048-stream geometry, so `uint64` is both necessary and sufficient. The
all-bin sums, the frame powers, are bounded by
`256 * 2 * R^2 * 128 * streams * 1.001`, where `R` is the row-sum component
bound in `cuda/config.h`. The bound is about `2^55` at 2048 streams, which
leaves 9 bits of `uint64` to spare.

In schema v6, the fine terms are also the source of the frame powers, so they
are mandatory for every in-band pilot, and `fine_products` accepts only `on`.

`p_ref_lower_u64` and `p_ref_upper_u64` are kept for the same reason at the
coarse level: the sum alone cannot show that only one reference window was
contaminated.

All four coarse power fields are exact `uint64 (N, 1)` arrays. The product
contract checks each split in Python integer arithmetic and rejects overflow,
any row where lower plus upper does not equal `p_ref_sum_u64`, and (v6, fine
status `enabled`) any row where a frame power differs from the exact all-bin
sum of that term's `fine_power_u64`. A backend that omits one of these terms
fails instead of writing a placeholder.

### Mask convention

The current `reject_mask` compares the target and reference powers after
correcting for unequal quantized weight norms. A valid frame is rejected when:

```text
reject_mask = valid && (p_target_u64 * reference_norm_sum_sq
                        > target_norm_sq * p_ref_sum_u64)
            = valid && (F > null_power_ratio)
```

The integer cross multiplication is the recorded `mask_rule`. It avoids a
floating-point threshold decision and uses the weight-norm flat-floor reference
`null_power_ratio = 2*target_norm_sq/reference_norm_sum_sq` instead of assuming
`null_power_ratio = 1`. The analyzer recomputes this bit from the exact stored
powers and norms, and a mask bit supplied by the backend is never taken as
authoritative.

Reporting can derive `keep_mask = 1 - reject_mask`, and the per-pilot product does
not store a second copy.

## Integrated power spectra

For each full analysis frame, the analyzer computes a rectangular-window FFT of
the raw samples, sums `|FFT|^2` over input streams, and accumulates the result.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `integrated_spectrum_before_mask` | `(nfft,)` | `float64` | Sum over valid frames |
| `integrated_spectrum_after_mask` | `(nfft,)` | `float64` | Sum over valid frames with `reject_mask = 0` |

Therefore, `before - after` is the accumulated spectrum of frames rejected by
the stored mask. The arrays are raw accumulated power, and normalization by
frame count or input-stream count is a reporting choice.

Bin `k` maps to baseband frequency by:

```text
((k + nfft//2) % nfft - nfft//2) * fs / nfft
```

where `fs = 1 / unit_delta_time` when every contributing unit has the same
finite sample period. If the periods differ, one shared frequency axis is not
defined, and the canonical combine rejects that mixture.

The production path uses CuPy when a GPU runtime is available, while tests can
use NumPy. Both implement the same FFT and float64 feed-sum accumulation, but
their results can differ by normal floating-point roundoff. The stored spectra
contain accumulated power and no per-frame complex FFT values. A different
window or mask threshold therefore requires a new spectral pass, since those
choices cannot be reconstructed from the two accumulated arrays.

## Absolute-time axis

The packed HDF5 reader copies timing and event attributes from each file into a
per-unit table aligned with `unit_order`. Two per-frame arrays give the unit of
each frame and the position of the frame in that unit, so no absolute
timestamp is stored per frame.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `unit_time0_ctime` | `(U,)` | `float64` | File start UNIX time, or NaN when absent |
| `unit_time0_fpga` | `(U,)` | `uint64` | FPGA count at file start, or 0 when absent |
| `unit_event_id` | `(U,)` | `int64` | CHIME event identifier, or `-1` when absent |
| `unit_delta_time` | `(U,)` | `float64` | Sample period in seconds, or NaN when absent |
| `frame_unit_index` | `(N,)` | `int32` | Unit index `u` for each frame |
| `frame_in_unit` | `(N,)` | `int32` | Zero-based frame position in unit `u` |

For frame `f`, compute wall time with:

```python
u = frame_unit_index[f]
t = unit_time0_ctime[u] + frame_in_unit[f] * nfft * unit_delta_time[u]
```

If a synthetic or incomplete file lacks the root attributes, the analyzer stores
`NaN / 0 / -1 / ""` and continues. Any LST or wall-time analysis must first
exclude those missing values.

## Provenance and calibration

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `weights_hash` | scalar | `str` | SHA-256 of the selected packed weight profile |
| `weight_bank_sha256` | scalar | `str` | SHA-256 of the complete weight bank, or empty for injected weights |
| `weight_manifest_sha256` | scalar | `str` | SHA-256 of the adjacent manifest, or empty when unavailable |
| `weight_coefficients_sha256` | scalar | `str` | SHA-256 of the weight coefficients alone, excluding header fields, so a header-only change does not read as a different bank |
| `detector_version` | scalar | `str` | Package, source-tree, kernel, schema, and `K` identity string |
| `mask_rule` | scalar | `str` | Integer rejection rule used for this product |
| `reference_placement_json` | scalar | `str` | Selected reference-placement metadata encoded as JSON |
| `rational_overflow_count` | scalar | `uint64` | Accumulated fixed-point overflow telemetry |
| `max_chunks_per_file` | scalar | `int64` | Per-file cap, or `-1` when uncapped |
| `detector_contract_json` | scalar | `str` | Full detector contract encoded as JSON |
| `pilot_below_data_db` | scalar | `float64` | Pilot-to-data-shelf calibration constant |
| `bin_enbw_hz` | scalar | `float64` | Detector-bin equivalent noise bandwidth |
| `dtv_bandwidth_hz` | scalar | `float64` | Assumed DTV bandwidth |
| `pilot_capture_efficiency` | scalar | `float64` | Pilot capture-efficiency factor |

The file also stores keys needed for resume and channel alignment:

| Array | Shape | Meaning |
|---|---:|---|
| `unit_keys` | `(U,)` | Sorted set of committed archive unit keys |
| `unit_order` | `(U,)` | Unit keys in analyzer consumption order |
| `source_event_keys` | `(U,)` | Namespaced event identity derived from the aligned `unit_order` entry and `freq_id`; combine recomputes and verifies every value |

`unit_keys` and `unit_order` must each be unique and contain the same exact set.
Every nonempty unit must own at least one frame through `frame_unit_index`, and
the only allowed zero-frame resume checkpoint also has zero units. These rules
keep an unused unit out of the completed-unit set of a resume.

Resume checks the schema, `freq_id`, frame cap, detector geometry, calibration,
weights, mask rule, detector contract, and reference placement. A source-tree
hash or package-version change makes the full `detector_version` differ and
stops the resume, even when the remaining geometry tokens match. This prevents
one checkpoint from appending frames produced by a second implementation and
then relabeling the earlier frames. Start a clean output directory for the new
build.

---

## Derived in reporting (not stored)

The frame-level fields support these downstream products without another CUDA
detector pass:

- `keep_mask = 1 - reject_mask`;
- local-reference power ratio histograms and survival curves from the raw integer powers;
- alternative frame-level thresholds from the raw powers and norms;
- masked fraction from `reject_mask` and `valid`;
- baseband before/after summaries from `baseband_power_linear` and a selected
  frame mask;
- per-frame wall time, and then LMST where the timing attributes and telescope
  longitude are available. The completed-archive v1 audit uses the DRAO
  longitude of -119.6175° east-positive, publishes its UTC-as-UT1 GMST
  polynomial, and records the frames whose sample period prevents time and
  exposure recovery.

The fine array has two retrospective roles. Geometry predicts a broad
+/-30-bin acquisition neighborhood. In that neighborhood, the v1 archive audit
may accept a narrow +/-2-bin line anchor independently in each provisional UTC
quarter, when its health-filtered persistence and uniqueness tests pass. The
audit records refusals and external-line sentinels. Neither derived object
changes the stored coarse `reject_mask`, and the broad acquisition window is
never a final calibrated designation.

The stored integrated spectra support the spectrum before/after comparison for
the mask used during the run. They do not preserve enough information to apply a
new frame threshold or FFT window after the fact.

The health repair of the completed survey is one narrow, proven exception, and
it gives no general remasking capability. A frame with
`baseband_power_linear == 128` must consist entirely of decoded `(-8,-8)`
samples, which are the native CHIME offset-binary raw byte `0x00`, or
equivalently the two's-complement byte `0x88` of the detector input after
repacking. Its rectangular-FFT contribution is therefore reconstructible and
can be subtracted from the accumulated spectra. Detector-invalid rows were
never accumulated (see [`ARCHIVE_HEALTH_REPAIR.md`](ARCHIVE_HEALTH_REPAIR.md)).
Any other new frame exclusion still requires the original HDF5 data for a
spectral correction.

Full 6 MHz DTV-channel mask expansion is not part of this product. It requires a
larger pass over the neighboring coarse channels after the pilot detector has
been validated.
