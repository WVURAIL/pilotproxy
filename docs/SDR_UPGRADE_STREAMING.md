# CPU streaming SDR upgrade adapter

The purpose of `pilot_proxy.integration.sdr_upgrade_streaming.StreamingDigitalAdapter`
is to process a continuous SDR stream with the same arithmetic as the batch
adapter in `sdr_upgrade_adapter.py`. It consumes one contiguous 2 MHz complex
digital stream and emits the same single-input detector geometry: 390625 Hz
after 25/128 resampling, 16384 samples per frame, 128 rows of 128 samples, three
packed projectors, and integer complex row projections. The adapter has no
hardware API and no physical RF mapping. It does not replicate the input into an
array, and it makes no thermal-null claim.

## Use and ownership

```python
from pilot_proxy.integration.sdr_upgrade_adapter import DigitalAdapterConfig
from pilot_proxy.integration.sdr_upgrade_streaming import StreamingDigitalAdapter

adapter = StreamingDigitalAdapter(
    DigitalAdapterConfig(2_000_000, mixer_hz=100_000., target_bin=0, sample_scale=5.),
    first_input_index=0,
)
for index, iq in contiguous_chunks:
    for frame in adapter.push(iq, input_start_index=index):
        consume(frame)
final = adapter.finish()
for frame in final["frames"]:
    consume(frame)
save_receipt(final["receipt"])
```

The index is mandatory and must equal `next_input_index`. A chunk can have any
length. A single sample is accepted, and an empty complex array is a no-op.
Nonfinite, non-complex, multidimensional, or incorrectly indexed inputs fail
before any sample is consumed. The adapter is not thread safe, so one owner must
serialize the calls. Returned arrays belong to the caller and never alias the
retained buffers.

The internal input block is fixed at 8192 samples (4.096 ms), which adds a
bounded batching latency before an otherwise complete frame is emitted. The
retained state is an 8192-sample input buffer, at most 455 mixed history
samples, one 16384-sample output frame buffer (at most 16383 filled between
calls), the fixed FIR and weight arrays, and the counters. Work arrays are
bounded by the fixed block. The input chunks and the returned list of complete
frames belong to the caller and use memory in proportion to their sizes. A
caller that needs a low peak memory submits bounded chunks and consumes the
frames promptly. Processing throughput has not been qualified.

## Continuity, boundaries and time

`finish()` drains the final short input block. Only outputs with complete FIR
support are accepted, so no zero-padded ring-down reaches the detector. Complete
frames are emitted, and the incomplete frame is counted and then discarded. A
finished segment rejects any later `push` or `finish` call until an explicit
reset.

`gap(missing_input_samples)` requires a known positive count. The old segment is
finished first, and a new segment is then started at the expected source index
plus that count. The caller must consume the frames returned at the boundary and
keep its receipt. `reset(first_input_index=..., reason=...)` performs the same
boundary drain but declares a new source-index origin, with no missing count
inferred. It can be used for an unknown transport loss, a clock restart, or a
new recording. A reset after `finish()` returns `None`, because that boundary was
already emitted.

Every gap or reset restarts the mixer phase, the resampler lattice, and the
frame origin. The old FIR history and any incomplete frame are discarded, and no
artificial sample or frame spans the gap. Phase coherence across segments is
therefore not available. Ordinary chunks in one segment preserve all of these
states.

For the segment-local untrimmed output index `j`, the sample center has the
exact rational time `(source_origin*25 + j*128 - 4096) / 50000000` seconds
relative to digital source index zero. The first index with full support is 64,
which corresponds to 81.92 us after the start of the segment. The integer
numerator and denominator fields are authoritative, and the floating seconds
field is given for convenience. This time is a label derived from the digital
sample index. It is neither a hardware timestamp nor evidence of SDR clock
continuity. Segment IDs distinguish explicit source-index resets that reuse
earlier numerical indices.

## Arithmetic contract

Each segment uses the frozen batch FIR, mixer expression, scale, symmetric
four-bit ties-to-even quantizer, packed weights, and CPU integer projection.
Each local polyphase call begins at an input index divisible by 128, which
preserves the original resampling lattice. Every partition of a segment into
ordinary chunks feeds the same fixed internal blocks, resulting in deterministic
streaming outputs that are bitwise equal for a given segment on a fixed
numerical runtime.

The floating outputs of the batch and streaming paths can differ at the roundoff
level, because the local numerical kernels see different array extents. The
qualification compares their complex absolute error with
`2e-12 * max(1, max(abs(input)))` and checks selected scalar FIR convolutions
independently. This tolerance is an engineering value confirmed by test, and it
is not proven for every floating input. The phase is computed from the batch's
absolute segment-local mixer-index expression, so no phase drift accumulates
from chunk to chunk. For extremely long records, the phase still has the
precision limits of floating-point evaluation. Indices above `2**53 - 1` in one
segment are refused.

Identical packed samples and integer projections are verified for each qualified
fixture. Each frame exposes `minimum_unclipped_quantizer_half_step_distance`,
the smallest distance from an unsaturated scaled component to a half-integer.
The release compares this margin with the largest measured scaled batch error.
Near an exact half-step, one ulp can change the ties-to-even rounding between
the batch and streaming paths, so packed equality is asserted only for the
qualified fixtures. An adversarial half-step test also verifies that the
streaming output does not depend on the chunk boundaries.

Zero target and reference powers give `undefined_both_zero` with NaN, and a
positive target with zero references gives `infinite`. Neither status is
accepted as noise, as a finite statistic, or as a qualified detection. Saturated
components are counted. The ratio is `2*target/(lower+upper)` with no
weight-norm adjustment, formed from the same frame powers as in the batch
adapter (the all-bin sums of the `fxfft256` fine powers).

## Qualification

The continuity and boundary checks are run with
`PYTHONPATH=src python -m pytest tests/core/test_sdr_upgrade_streaming.py -q`.
A release is written with
`PYTHONPATH=src python tools/validate_sdr_upgrade_streaming.py --output NEW_DIR`,
which refuses to overwrite an existing output. The release holds six
deterministic records of noise, tones, zeros, and saturation, five chunk
partitions of each record (one of them made only of single-sample chunks),
independent scalar FIR and packed-integer projection checks, the rounding
margin of each record, a unit-test receipt, source snapshots, and a SHA256
manifest.

Real SDR transport, qualified references of radio noise and steady signals, the
clock and RF mapping, the physical filter transfer, and live Pathfinder
integration each need their own measurements. This CPU companion confirms the
digital state handling only.
