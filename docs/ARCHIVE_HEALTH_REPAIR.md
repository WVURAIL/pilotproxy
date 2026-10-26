# Completed-archive health and fine-diagnostic repair

This document defines the first retrospective repair of the completed
23-channel CHIME pilot survey. The source products are left unchanged. The
repair writes a versioned inclusion mask, a reason-coded exclusion ledger,
recomputed scalar and fine summaries, and the one correction of the integrated
spectra that the retained data can prove exactly.

```bash
pilot-proxy audit-archive-health \
  --products-dir /path/to/_per_pilot \
  --product-archive /path/to/_per_pilot.zip \
  --inventory-archive /path/to/chime-pilots.zip \
  --kernel-library /path/to/libpilotproxy-2.1.0-c85f50dd.so \
  --source-repository /path/to/pilot-proxy \
  --source-commit 94b1de0e07bdbabb7e544aff22b62ef866e9cf0c \
  --output-dir /path/to/archive-health-v1 \
  --dissertation-style \
  --expect-products 23 \
  --expect-excluded-frames 182 \
  --expect-invalid-frames 4 \
  --expect-ceiling-frames 178
```

Use `--no-plots` for a run that writes only machine-readable output. The four
`--expect-*` arguments assert the known counts of the completed survey and are
never tuned as health thresholds. A different input set fails these assertions
instead of silently inheriting the counts.

The optional source-history pair rehashes `src/pilot_proxy` at each supplied
clean Git commit, using the same algorithm that is stamped into the products.
In the attached completed archive, commit `94b1de0e...` reproduces the source
hash in 21 products. The other two products carry a second build hash that does
not map to the supplied clean revision. That hash is recorded as unmatched, and
the audit never assumes that it came from a clean commit.

## Versioned, fail-closed frame gate

The gate identity is
`pilotproxy_archive_frame_health_gate_v1`. Before it returns an inclusion mask,
the gate validates the required fields, shapes, binary flags, frame-to-unit
indices, source-event keys, and the fine array. Missing or malformed evidence
is an error and is never read as a healthy row.

The completed products contain 750461 frames. The v1 union excludes 182 and
retains 750279:

| Reason code | Rows | Interpretation |
| --- | ---: | --- |
| `detector_invalid` | 4 | The detector denominator was zero. The analyzer did not add these rows to either integrated spectrum. |
| `detector_powers_all_zero` | 4 | Both detector power terms are zero. These are the same four rows, so this count must not be added again. |
| `baseband_power_at_negative_full_scale_ceiling` | 178 | The decoded frame-mean sample power is exactly its complex-int4 maximum, 128. |

The gate also fails closed on non-finite or out-of-range baseband power,
non-finite valid-row coarse F, and non-finite or negative valid-row fine F. No
further row of the completed survey triggers these guards.

Every excluded row appears once in `archive_exclusion_ledger.jsonl`. A record
holds all triggered reason codes, the product hash, physical channel,
`freq_id`, frame index, frame-within-source index, event ID, source-unit key,
and derived UTC. Its stable `ledger_key` contains the gate version, source
event, frame, and reasons. Reason counts can overlap, and
`excluded_unique_frames` is always the count of the union.

## Full-scale power and the native byte

Each native CHIME byte stores two offset-binary four-bit components. Decoding
maps each component to the integer range -8 through +7. A complex sample has
power at most

```text
(-8)^2 + (-8)^2 = 128.
```

`baseband_power_linear` is the arithmetic mean of those non-negative sample
powers over every time sample and input stream in one frame. A mean equals the
upper bound only if every term equals the upper bound. Thus, a mean of exactly
128 proves that every decoded sample is `(-8, -8)`.

The byte value of this sample depends on the encoding. In the native CHIME
offset-binary format, the raw byte is `0x00`. After the lossless repack to the
two's-complement detector input, the same sample is the byte `0x88`, and that
value is correct only after the repack.

## Exact correction of the integrated spectra

In general, the two accumulated spectra of an NPZ cannot be remasked, since
they hold no per-frame complex FFTs. The 178 ceiling rows are a special case,
because the retained scalar determines their entire sample array.

For a constant `(-8,-8)` frame, the unnormalized rectangular FFT is zero away
from bin 0. At bin 0, the production expression of the analyzer is
`abs(complex64 FFT)**2`, followed by a float64 sum over streams. The
complex64 absolute value rounds before it is squared. For the recorded
`nfft=16384` and 2048 input streams, the exact production contribution is

```text
34,359,736,320 per stream
70,368,739,983,360 per frame.
```

An RTX/CuPy reproduction of the analyzer operation gives that value at bin 0
and exact zeros in every other bin. The infinite-precision expression
`2048 * 16384^2 * 128` is slightly larger and must not be substituted for the
recorded complex64 arithmetic.

For each of the 178 ceiling rows, the production value is subtracted once from
the before-mask spectrum. For each of the 118 ceiling rows whose stored coarse
decision kept the frame, it is also subtracted once from the after-mask
spectrum.

The remaining 60 ceiling rows were already rejected by the stored mask and
therefore never entered the after spectrum. The four detector-invalid rows
entered neither spectrum, so their exact spectral correction is zero. The
result is published as `health_corrected_integrated_spectra.npz`, with the
counts and DC correction value beside every channel. Publication refuses any
product with an excluded valid row whose samples are not reconstructible.

As a second invariant, each corrected spectrum sum is compared with
`nfft^2 * num_input_streams * sum(baseband_power_linear)` over the same
health-included frames. We find that the completed products agree to a
relative error of `2.1e-7` in the worst case, below the recorded `5e-7` bound
for the roundoff of the production complex64 FFT and absolute square. Every
corrected spectral bin is non-negative.

This correction is limited to the ceiling rows, and per-frame spectra in
general are not restored. A new arbitrary threshold, mask, FFT window, or
dilation still requires the source HDF5 files.

## Retrospective fine repair

The archived fine-F arrays are kept, but the stored designated set of the run
was fine bin 0, and that field is not reused as evidence. A pilot need not lie
at envelope bin 0. Its center predicted from geometry is the residual between
its RF frequency and the nearest coarse detector grid point, after the recorded
Nyquist-zone sign (`sense`) is applied.

The repair publishes two objects, so that the broad window is never taken as a
final designation:

1. `predicted_acquisition_neighborhood` is the anchor predicted from geometry,
   +/-30 bins, modulo the 256-bin grid. It is a broad prior for acquisition and
   diagnosis, and it does not localize the line empirically.
2. `measured_epoch_line_anchor` is estimated independently in retrospective,
   provisional UTC calendar quarters. These quarters are a reproducible
   partition of the archive and carry no authority as transmitter or station
   epochs. In each quarter, every health-included fine-F row is divided by its
   row median, and candidate pilot bins are restricted to the predicted
   acquisition neighborhood. A candidate is accepted only with at least 30
   usable frames, a persistence fraction (normalized ratio at least 1.5) of at
   least 0.10, and a persistence margin of at least 0.02 over the strongest
   competitor that lies in the acquisition neighborhood and more than 5 bins
   from the candidate. A candidate no more than two bins from the acquisition
   boundary is also refused when a stronger persistent sentinel lies past that
   boundary. This rule keeps the shoulder of a line past the boundary from
   being mislabeled as the pilot, as on channel 23. An accepted line receives
   the narrow +/-2-bin candidate window used by the calibration tools.

The strongest line not in the acquisition neighborhood is exported as a
sentinel. It may mark an instrument line or an unrelated spectral line, but
it can never redefine the target pilot anchor. Each rejected quarter records its
evidence and refusal reasons and then falls back explicitly to the broad
predicted acquisition diagnostic. Circular deltas, measured minus predicted,
use the interval `[-N/2,N/2)`, with positive values toward increasing
fine-array index.

The recomputed independent CFAR null bulk excludes the broad predicted
acquisition neighborhood, every accepted measured-line window, the stored guard
bins, and the stored census exclusions. The machine summary exports the rate
over the predicted neighborhood and the rate over the selected epoch window or
fallback under different names. Both rates are retrospective float diagnostics
derived from the stored fine-F array. They leave `reject_mask` unchanged and
are not a deployed or prospectively calibrated fine decision.

## Inventory and binary evidence

The optional evidence arguments bind independent attachments to the result:

- every NPZ member in `_per_pilot.zip` must hash identically to the audited
  product;
- the supplied kernel library SHA-256 must equal the `kernel_sha256` token in
  every product;
- every processed and excluded source-unit key must join `inventory.jsonl`;
- inventory-minus-processed units must equal the surviving quarantine ledger
  when both archives are supplied.

For the completed evidence set, `_per_pilot.zip` reproduces all 23 NPZs. The
supplied kernel hash
`c85f50ddf898517bc0101d1882c854c3df70b09f0ab0b58803dc32f59e3c6d12`
matches every product. The inventory contains 170377 unique target-frequency
objects in 9214 events. The products consume 170374 units, and the difference
of three units is exactly the three-row quarantine. All 9214 inventory events
join processed products, because each quarantined event has other usable
frequency units.

The per-pilot unit axes contain 4692 units with no complete stored frame. Over
all channels, 8983 distinct events have at least one stored frame, and 8980
have at least one v1 health-included frame. For health-filtered frame-level
analyses, the event count is 8980, and the broader inventory coverage of 9214
events must never be used in its place.

In total, the inventory catalogs 28237618443352 bytes of discovered objects.
This is a catalog size and was not measured in transfer. The attachments contain no
transfer or performance log, so network bytes, peak storage, GPU time, wall
time, and throughput are unavailable. The frame-health science needs none of
these operational metrics.

The surviving enumeration accounts for 16327 scope events. Of these, 6140
outrigger-labeled events were excluded, which leaves 10187 in the target survey
scope. Of those, 10184 appear in the completed-event list, and three remain in
the attempts record, each with a recorded attempt count of 2. The completed set
partitions into 9214 events with target-frequency inventory rows, 107
explicitly recorded aged-out accepted empty events, and 863 events derived by
set subtraction as having no common target path. Since no direct per-event status
ledger survived for those 863 events, the last category is marked as derived.

Neither ZIP contains raw HDF5 data or a run or performance log. The CADC keys
of the inventory locate the original units, and with them all 71 unique source
files behind the 182 exclusions. Retrieving those files is left to a
root-cause study of its own.

## Health-filtered observing exposure

Each channel summary also exports frame counts and rectangular-frame exposure
seconds after the v1 gate in all of these reproducible coordinates:

- UTC calendar month and UTC hour of day;
- `America/Vancouver` civil calendar month and hour, with daylight-saving
  conversion supplied by the IANA time-zone database;
- local meteorological season, defined as DJF, MAM, JJA, and SON from the local
  civil month; and
- DRAO local mean sidereal hour (LMST), distinct from local civil time, local
  solar time, and local apparent sidereal time (LAST).

Frame time is
`unit_time0_ctime[u] + frame_in_unit * nfft * unit_delta_time[u]`, and frame
duration is `nfft * unit_delta_time[u]`. The complete rectangular duration of a
frame is credited to the bin that holds its start, and a frame is never split
at a bin boundary. Coverage counters record any missing timing metadata
explicitly.

For LMST, the longitude is -119.6175° in the east-positive convention. The
dependency-free transform uses

```text
JD = unix_utc_seconds / 86400 + 2440587.5
T = (JD - 2451545.0) / 36525
GMST_deg = 280.46061837 + 360.98564736629 (JD - 2451545.0)
           + 0.000387933 T^2 - T^3 / 38710000
LMST_hours = ((GMST_deg - 119.6175) mod 360) / 15.
```

UTC is used as a proxy for UT1, because the DUT1 and Earth-orientation values
were not retained. This approximation is acceptable for one-hour exposure bins
and too coarse for precision astrometry. The hard-coded tests cover the J2000
anchor and modern values from the USNO Sidereal Time API, and the declared
implementation stays no more than 0.01 sidereal second from those vectors.

In the completed archive, 2489 stored frames, all otherwise health-included,
belong to units without a finite positive `unit_delta_time`. The temporal
profile is therefore marked `partial`, with 747972 stored and 747790
health-included frames that have reproducible time and duration coordinates.
The 2489 unresolved frames stay in the scalar science denominators and are
kept out of the temporal exposure seconds.

Since the NPZs do not record the triggered or scheduled class of a unit, the
class is never guessed. When the inventory attachment is supplied, `unit_order` joins
exactly to its `scope` field. The attached inventory maps
`chime.event.baseband.raw` to `triggered_event` and
`chime.scheduled.baseband.raw` to `scheduled`. The machine summary exports the
inventory and product unit counts, the stored and health-included frame counts,
and the exposure seconds for both classes. For the same units with missing
durations, these exposure seconds are labeled partial. The discovered catalog
bytes are reported apart from the transferred bytes, which are unavailable.

## Outputs and their scientific names

`archive_health_summary.json` contains per-channel scalar distributions,
histograms, Wilson 95% binomial intervals, predicted/measured fine-anchor
evidence and refusals, health-filtered temporal exposure, spectrum correction
provenance, and evidence joins. `archive_exclusion_ledger.jsonl`
contains the excluded rows. `health_corrected_integrated_spectra.npz` contains
the exactly corrected spectral sums and denominators.

When plotting is enabled, every manifest path is a portable POSIX path relative
to the release root. Each deterministic
`figures/channel_XX_fid_YYYY/` directory receives:

- `health_filtered_histograms.pdf`, a vector PDF containing four
  health-filtered scalar histograms;
- `relative_time_averaged_spectra.pdf`, a vector PDF containing the exactly
  v1-corrected relative before/after time averages;
- `fine_f_utc_monthly_heatmap.png`, a 300-DPI raster heatmap with the broad
  predicted anchor and the accepted provisional-quarter measured anchors
  labeled; and
- `channel_XX_diagnostic_atlas.pdf`, a one-page dissertation atlas containing
  all three diagnostic sections. Its text, histograms, and spectra are vector
  objects, and the heatmap is an embedded raster image.

`diagnostic_manifest.json` gives the release-relative path, SHA-256, format, rendering
mode, channel, and `freq_id` for every asset. The 23 atlas PDFs are intended as
the direct appendix inputs. The 69 individual figure files support alternative
layouts without rerunning the science analysis.

For dissertation release builds, pass `--dissertation-style`. This option
fails closed unless LaTeX can render embedded Latin Modern T1 text, and it
never substitutes DejaVu silently. The PDF writer suppresses `CreationDate` and
`ModDate`, and the manifest records the style contract, so repeated builds from
the same products have deterministic content and metadata. Set the standard
`SOURCE_DATE_EPOCH` environment variable to pin `created_utc` in the machine
summary. The summary records the source of that timestamp, the environment or
the wall clock. Attachment provenance is a portable identity of basename, byte
count, and SHA-256, with no operator-local absolute path.

The spectrum is relative power with no absolute calibration. The fine-F
heatmap shows the detector statistic, and no raw voltages enter it. Since the
NPZs contain neither a flux or temperature calibration nor per-frame voltage
FFTs, they cannot support an absolutely calibrated PSD or a raw-voltage
spectrogram.

## Residual consumers

The released `rfisher` APIs accept product paths and load their own frame
arrays. If only the surrounding report code were filtered, the residual floor,
correlation time, threshold sweep, and mask-fraction routines would silently
read the excluded rows again.

To prevent this, `temporary_residual_health_views` applies the same v1 gate and
writes short-lived NPZ views. Each view is incomplete by design and contains
only the coarse-policy and residual columns used by RFIsher, the source product
hash, and the health counts. The fine-decision fields are dropped, and the views
never replace the canonical survey product. A science-side consumer routes every
call through these views and applies `evaluate_frame_health` to its per-frame
policy denominators.

On channels 22 and 24, every frame kept by the stored coarse rule is one of the
full-scale rows excluded by v1. With no health-included kept or null frame left
under the stored mask, the empirical residual floor and the threshold sweep are
unavailable. A consumer records that status and takes the conservative excision
action, and it must never describe the two coherence brackets as agreeing. To
reopen threshold optimization for either channel, new clean or transmitter-off
calibration data are needed, or an independently justified synthetic floor.
Further manipulation of these NPZs cannot replace that data.

The audit leaves the frozen summaries in `data/provenance/`
(`dissertation_summary_v3.json` and its predecessors) unchanged. Their figures
were made before the health repair and are not health-corrected.
