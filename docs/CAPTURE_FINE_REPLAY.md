# Capture-specific fine replay and score export

The purpose of `replay-fine-capture` is to recompute the submitted fine
decision from synchronized native voltage, mask, and coarse-power files. It
exports the exact decimal Q16 keep boundary, the usable reference count, the
validity, and the source frame provenance. The runtime bundle is left
unchanged, and no operating threshold is chosen.

Install this branch in its own environment and run:

```sh
python -m pip install -e '.[calibration]'
pilot-proxy replay-fine-capture --plan capture_plan.json --output-dir replay
pilot-proxy validate-fine-capture-replay --replay-dir replay
```

For validation, the capture files and the contracts must both stay available.
The output directory must be new. Identity, geometry, and configuration errors
stop the command before results are written. A mask or power mismatch is
preserved in the output and returns a nonzero exit status. Invalid fine support
is a recorded outcome and is never treated as a file-format error. Those rows
remain in the export, and the report states `invalid_support`. A successful
exit therefore does not mean that all decisions were valid.

## Reproducible synthetic example

From this checkout, choose new directories:

```sh
python tools/make_capture_replay_example.py --output-dir /tmp/fine-capture-example
pilot-proxy replay-fine-capture \
  --plan /tmp/fine-capture-example/plan.json \
  --output-dir /tmp/fine-capture-replay
pilot-proxy validate-fine-capture-replay --replay-dir /tmp/fine-capture-replay
```

The example has two detector blocks, one bound pilot frequency, and one
unbound control frequency. Of the two blocks, the first is a seeded
synthetic draw, and the second is zero voltage, which produces invalid reference support. The
expected outcome is exact agreement for all four frequency and block
comparisons, two score rows, one invalid row, and status `invalid_support`.
The invalid row has `valid: false`, `n_bulk: 0`, `required_multiplier_q16: null`,
and recorded mask zero. It is never replaced by a numerical score or silently
removed.

`--valid-only` makes both blocks synthetic random draws, to test the
calibration-file handoff. These tiny fixtures cannot qualify an operating
point. In the example, the saved products come from the CPU reference, and the
binary and configuration evidence is labeled synthetic. Exact arithmetic and
the refusal paths are checked by the component tests.

## Supported capture format

This command supports `rawFileWrite` files from the little-endian, 64-bit CHORD
metadata ABI inspected at Kotekan commit
`c81def5518bb86b03afa837d6c57655ef48c548d`. Each raw frame contains a little-endian
32-bit metadata size, 148888 metadata bytes, and the expected payload bytes.
The reader checks the metadata layout limits, array type/name/dimensions,
dimension names/scaling/strides, zero array offset, coarse frequency IDs and
order, and FPGA sequence/downsampling. Unsupported layouts are refused.

Voltage is `[T,F,P,D]`, two polarizations, one offset-binary byte per complex
sample: real in the high nibble, imaginary in the low nibble. Decoding
subtracts eight from each nibble, with no requantization, nibble swap, or
inferred gain. Streams follow `s = p * num_dishes + d`. If the exact bundle
requires reversal, only the samples in each detector window are reversed. The
receiver and weight coordinate system must agree with that setting.

The plan specifies K=64 or K=128 and exactly 128 windows per stream, so the
detector block has 8192 or 16384 samples. Voltage frames must divide a complete
block, and the supplied capture must contain whole, contiguous blocks. A mask
frame contains `[F]` int8 values, and a power frame contains the frame powers
as `[F,3]` little-endian uint64 values. The FPGA sequence of a mask or power
frame is that of the first voltage sample in the block, and its downsampling is
the input downsampling times the block length. The replay expects the frame
powers to be the all-bin sums of the exact fine powers
(`capture_reference.expected_products`). A capture recorded by a stage that
vendors a core before 3.0.0 carries row-domain sums and fails power parity,
while its masks can still agree.

The submitted loader binds physical channels 14 to 36. Unbound frequencies,
such as a captured CH37 control, must have zero saved masks and coarse powers,
and they receive parity checks but no fine score. Every bound captured profile
must be in the export with `fine_calibration.status: calibrated`. That label is
checked as configuration only and is never accepted as scientific evidence.

The native metadata field `stream_ids` labels the received packets, while the
ordered detector input slots come from the input-state contract. Both the stream
IDs and the loss counters are kept as provenance. The input-state contract supplies
the physical input identities, since the mapping, the independence of live
inputs, and the receiver gain cannot be inferred from packet IDs.

## Plan and contract files

The generator creates a complete, inspectable example of every file below.
All JSON is UTF-8. Duplicate keys, nonfinite values, changed hashes, repeated
raw file paths, repeated profiles, and missing capture members are refused.
Each file reference is `{"path": "relative/or/absolute/path", "sha256": "..."}`
with a full lowercase SHA256. References are relative to the JSON file that
holds them, and population evidence references are relative to the plan.

The plan has schema `pilotproxy_capture_fine_replay_plan_v1` and contains:

| Field | Meaning |
|---|---|
| `capture_id` | Human-readable capture identity, with no claim of independence |
| `first_fpga_seq_num`, `expected_input_frames` | Exact start and complete expected input population |
| `receiver_profile` | Existing receiver-profile JSON, pinned by byte hash |
| `packing_contract` | Geometry, frequency, packing, and clock contract |
| `input_state_contract` | Fixed ordered inputs, gains, quantization, preprocessing, and validity interval |
| `producer` | Recorded runtime producer and configuration contract |
| `bundle_files` | All five runtime bundle members, individually pinned, retaining their names in one directory |
| `raw_files` | Explicit ordered arrays named `voltage`, `masks`, and `powers`; no globs or inferred sorting |
| `profiles` | One `{physical_channel, profile_id}` entry per bound captured profile, with optional population declaration |

Packing schema `pilotproxy_capture_packing_v1` specifies `layout: TFPD`,
`sample_encoding: int4x2_swapped_withoffset`, `voltage_tap: pre_dtv_mask`,
`pre_detector_transform: none`, `num_times`, `num_dishes`, `num_polarizations`,
`detector_window_samples`, `windows_per_stream`, `frequency_ids`,
`sample_rate_hz`, `nyquist_zone`, and
`time_reverse_detector_windows_before_kernel`. Its ordered `frequency_map`
contains `receiver_channel_id`, `coarse_channel_index`, and `rf_center_hz` for
each frequency. The tool checks these against the receiver and bound profiles.

Packing `timing` contains `time_downsampling_fpga` and `utc_reference`, with
`fpga_seq_num`, `unix_seconds`, and `tick_period_seconds`. Decimal clock values
are checked with rational arithmetic against the sample rate. Keep the same
clock reference for calibration and evaluation. The actual start of each
capture is given in its plan, and the packing contract stays the same.

Input-state schema `pilotproxy_capture_input_state_v1` records
`input_population_id`, ordered `streams` (`stream_index`, `polarization_index`,
`dish_index`, `input_id`, `status`), a pinned `gain_evidence` file, and descriptive
`quantization` and `preprocessing` fields. Status is `live`, `missing`,
`flagged`, or `rfi_antenna`, and all configured slots still contribute to the
replay. The half-open interval `valid_from_fpga_seq_num` through
`valid_until_fpga_seq_num` must cover the entire capture.

Producer schema `pilotproxy_capture_producer_v1` records a full `source_revision`
and pinned `binary`, `configuration`, and `vendor_manifest` files. Its `binding`
records the byte hashes of the receiver, packing, input state, and every bundle
file. Its `detector` object must exactly declare `enabled: true`,
`samples_per_detector_frame`, `num_input_streams`, and ordered `frequency_ids`.
These acquisition records are supplied as inputs. The command checks
their consistency and the product parity. It does not parse an arbitrary
rendered configuration into that declaration, and it cannot show which binary
ran on the hardware. Preserve the startup bindings and the independent review
of the release and configuration with the evidence.

## Calibration handoff and output interpretation

The output contains `replay.json`, `manifest.json`, `producer.json`,
`profile_CH*.json`, and `scores_CH*.json`. The producer receipt binds the
runtime producer and the exact installed reference sources and environment.
Every score keeps the raw file paths, byte offsets, metadata hashes, timing and
frequency identity, recorded and expected masks and coarse powers, support, and
the exact Q16 boundary. A valid always-masked boundary is written as the
decimal string `18446744073709551616`, and an invalid row has a null boundary.

The optional `population` declaration of a profile supplies `role`,
`population_id`, `null_law`, `sampling`, pinned `null_evidence` and
`sampling_design`, and an ordered `units` array with `block_index`, `trial_id`,
and `independent_unit_id` for every captured block. The null and sampling
objects follow the score-population contract
(`src/pilot_proxy/runtime_calibration.py`). The exporter never invents an IID
claim or a transmitter-off label. Unique unit names do not make the units
independent, and a background population is never relabeled as verified
noise.

The exported `trial_id` is derived from native block content, timing, clock
reference, and frequency order, so renaming the capture or supplied trial does
not make the same block a new observation. The supplied trial name is kept as
`declared_trial_id`, and the independent-unit identity is kept as supplied
provenance.

When the saved products agree, the declaration produces `population_CH*.json`.
Invalid rows are written to it, and the population validator refuses them.
When product parity fails, the diagnostic scores remain, but no population file
is emitted. Invalid or mismatching rows must never be edited out to claim
qualification.

Calibration and evaluation must use the same score profile, the same fixed
packing and input state, and the same producer. Use one fixed clock reference
and one state-validity interval that covers both independent captures, so that
only their plans, raw files, and units differ. Point the calibration plan at the
exported profile and population files with their exact hashes, and then follow
the proposal and held-out validation procedure. The replay does not cover
threshold selection, validation of full-rate invalid handling, the ten-second
visibility endpoint, or the authorization of applied masks.
