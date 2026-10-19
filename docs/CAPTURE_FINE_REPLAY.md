# Capture-specific fine replay and score export

`replay-fine-capture` recomputes the submitted fine decision from synchronized
native voltage, mask, and coarse-power files. It exports the exact decimal Q16
keep boundary, usable reference count, validity, and source frame provenance.
It does not change a runtime bundle or choose an operating threshold.

Install this branch in its own environment and run:

```sh
python -m pip install -e '.[calibration]'
pilot-proxy replay-fine-capture --plan capture_plan.json --output-dir replay
pilot-proxy validate-fine-capture-replay --replay-dir replay
```

Both the capture files and contracts must remain available for validation. The
output directory must be new. Identity, geometry, and configuration errors stop
before results are written. A mask or power mismatch is preserved in the output
and returns a nonzero exit status. Invalid fine support is a recorded outcome,
not a file-format error: those rows remain in the export and the report states
`invalid_support`. A successful exit does not mean all decisions were valid.

## Reproducible synthetic example

From this checkout, choose new directories:

```sh
python tools/make_capture_replay_example.py --output-dir /tmp/fine-capture-example
pilot-proxy replay-fine-capture \
  --plan /tmp/fine-capture-example/plan.json \
  --output-dir /tmp/fine-capture-replay
pilot-proxy validate-fine-capture-replay --replay-dir /tmp/fine-capture-replay
```

The example has two detector blocks, one bound pilot frequency and one unbound
comparison frequency. The first block is a seeded synthetic draw; the second
is zero voltage, producing invalid reference support. Expected outcome: exact
agreement for all four frequency/block comparisons, two score rows, one invalid
row, and status `invalid_support`. The invalid row has `valid: false`,
`n_bulk: 0`, `required_multiplier_q16: null`, and recorded mask zero. It is never
replaced by a numerical score or silently removed.

`--valid-only` makes both blocks synthetic random draws for testing the
calibration-file handoff. These tiny fixtures cannot qualify an operating point.
The example's saved products come from the CPU reference, not a GPU or telescope
run. Its binary/configuration evidence is explicitly synthetic. Component tests
separately check exact arithmetic and refusal paths.

## Supported capture format

This command supports `rawFileWrite` files from the little-endian, 64-bit CHORD
metadata ABI inspected at Kotekan commit
`c81def5518bb86b03afa837d6c57655ef48c548d`. Each raw frame contains a little-endian
32-bit metadata size, 148888 metadata bytes, and the expected payload bytes.
The reader checks the metadata layout limits, array type/name/dimensions,
dimension names/scaling/strides, zero array offset, coarse frequency IDs and
order, and FPGA sequence/downsampling. Unsupported layouts are refused.

Voltage is `[T,F,P,D]`, two polarizations, one offset-binary byte per complex
sample: real in the high nibble, imaginary in the low nibble. Decoding subtracts
eight per nibble. There is no requantization, nibble swap, or inferred gain.
Streams follow `s = p * num_dishes + d`. If the exact bundle requires reversal,
only the samples within each detector window are reversed. The receiver and
weight coordinate system must agree with that setting.

The plan specifies K=64 or K=128 and exactly 128 windows per stream. Thus the
detector block has 8192 or 16384 samples. Voltage frames must divide a complete
block; the supplied capture must contain whole, contiguous blocks. A mask frame
contains `[F]` int8 values; a power frame contains `[F,3]` little-endian uint64
values, the frame powers. The replay expects the kernel core 3.0.0 definition,
the all-bin sums of the exact fine powers (`capture_reference.expected_products`);
a capture recorded by a stage that vendors an older core carries the retired
row-domain sums and fails power parity while its masks can still agree. Their FPGA sequence identifies the first voltage sample in the block,
and their downsampling is the input downsampling times the block length.

The submitted loader binds physical channels 14–36. Unbound frequencies,
including a captured CH37 control, must have zero saved masks and coarse powers;
they receive parity checks but no fine score. Every bound captured profile must
be included in the export and have `fine_calibration.status: calibrated`.
That label is checked as configuration, not accepted as scientific evidence.

Native metadata `stream_ids` identify received packets, not the ordered detector
input slots. They and loss counters are retained as provenance. The separate
input-state contract supplies the physical input identities; the command cannot
infer that mapping, live-input independence, or receiver gain from packet IDs.

## Plan and contract files

The generator creates a complete, inspectable example of every file below.
All JSON is UTF-8. Duplicate keys, nonfinite values, changed hashes, repeated
raw file paths, repeated profiles, and missing capture members are refused.
Each file reference is `{"path": "relative/or/absolute/path", "sha256": "..."}`
with a full lowercase SHA256. References are relative to the JSON containing
them; population evidence references are relative to the plan.

The plan has schema `pilotproxy_capture_fine_replay_plan_v1` and contains:

| Field | Meaning |
|---|---|
| `capture_id` | Human-readable capture identity; it does not establish independence |
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
`sample_rate_hz`, `spectral_sense`, and
`time_reverse_detector_windows_before_kernel`. Its ordered `frequency_map`
contains `receiver_channel_id`, `coarse_channel_index`, and `rf_center_hz` for
each frequency. The tool checks these against the receiver and bound profiles.

Packing `timing` contains `time_downsampling_fpga` and `utc_reference`, with
`fpga_seq_num`, `unix_seconds`, and `tick_period_seconds`. Decimal clock values
are checked with rational arithmetic against the sample rate. Keep the same
clock reference across calibration and evaluation; their actual capture starts
belong in the plans, not in separate packing contracts.

Input-state schema `pilotproxy_capture_input_state_v1` records
`input_population_id`, ordered `streams` (`stream_index`, `polarization_index`,
`dish_index`, `input_id`, `status`), a pinned `gain_evidence` file, and descriptive
`quantization` and `preprocessing` fields. Status is `live`, `missing`, `flagged`,
or `rfi_antenna`; all configured slots still contribute to replay. The half-open
interval `valid_from_fpga_seq_num` through `valid_until_fpga_seq_num` must cover
the entire capture.

Producer schema `pilotproxy_capture_producer_v1` records a full `source_revision`
and pinned `binary`, `configuration`, and `vendor_manifest` files. Its `binding`
records the byte hashes of the receiver, packing, input state, and every bundle
file. Its `detector` object must exactly declare `enabled: true`,
`samples_per_detector_frame`, `num_input_streams`, and ordered `frequency_ids`.
These are supplied acquisition records. The command checks consistency and
product parity; it does not parse arbitrary rendered configuration into that
declaration or establish what binary ran on the hardware. Preserve startup
bindings and the independent release/configuration review with the evidence.

## Calibration handoff and output interpretation

The output contains `replay.json`, `manifest.json`, `producer.json`,
`profile_CH*.json`, and `scores_CH*.json`. The producer receipt binds both the
runtime producer and exact installed reference sources/environment. Every score
retains raw file paths, byte offsets, metadata hashes, timing/frequency identity,
recorded/expected masks and coarse powers, support, and the exact Q16 boundary.
Valid always-masked boundaries remain the decimal string `18446744073709551616`.
They are distinct from invalid rows with a null boundary.

A profile's optional `population` declaration supplies `role`, `population_id`,
`null_law`, `sampling`, pinned `null_evidence` and `sampling_design`, and an
ordered `units` array containing `block_index`, `trial_id`, and
`independent_unit_id` for **every** captured block. The null and sampling objects
follow the score-population contract (`src/pilot_proxy/runtime_calibration.py`). The exporter never
invents an IID claim or a transmitter-off label. Unique unit names are not proof
of independence. A background population remains background, not verified noise.

The exported `trial_id` is derived from native block content, timing, clock
reference, and frequency order, so renaming the capture or supplied trial does
not make the same block a new observation. The supplied trial name remains
`declared_trial_id`; the independent-unit identity remains supplied provenance.

When saved products agree, the declaration produces `population_CH*.json`.
Invalid rows are included; the population validator will refuse them. When product parity fails, diagnostic scores remain but no population file
is emitted. Do not edit out invalid or mismatching rows to claim qualification.

Calibration and evaluation must use the same score profile, fixed packing/input
state, and producer. Use one fixed clock reference and state-validity interval
covering both independent captures. Only their plans/raw files/units differ.
Point the calibration plan at the exported profile and population files with
their exact hashes, then follow the separate proposal and held-out validation
procedure. Replay does not select a threshold, validate full-rate invalid
handling, establish a ten-second visibility endpoint, or authorize applied masks.
