# Live antenna RX to CPU host handoff

The purpose of `tools/run_sdr_live_handoff.py` is to measure a short live
handoff from the receiver to the CPU host through regular files. It uses the
unchanged qualified native LimeSDR reference worker and the existing CPU
streaming digital adapter. USB-call latency, physical RF calibration, a
thermal-noise distribution, sustained operation, and live Pathfinder readiness
are not covered by this measurement.

An engineering smoke test accepts two seconds of RX. Before the primary
acquisition, the primary plan fixes three accepted intervals of four seconds
each. Four seconds is the maximum of the unchanged capture helper, and no native
limit is extended. Each attempt uses a 500 MHz LO, a 2 MHz complex sample rate,
an RX gain of 30, and a 1.5 MHz RX LPF on the receiver with the antenna
connected. In the native mode `noise`, RX is accepted with TX disabled, and
register evidence is read before and after. No TX stream or payload is started.
Before the accepted interval, the existing LMS_Init internal gain calibration
and LPF tuning still run.

Processing uses the fixed 100 kHz digital mixer, target bin 0, sample scale 500,
and the single-input upgrade output geometry of 16384 samples. All of these
choices are digital, and none of them calibrates the RF frequency response, the
absolute power, or a telescope array. Every ratio status and clipping count is
kept.

## Publication, ownership and bounded state

The native worker writes the returned samples and the accepted sample spans to
regular files and then emits a newline-terminated record to `rx-chunks.jsonl`. A
partial JSON line counts as unpublished. For each published accepted span, the
wrapper verifies the offset and count, the exact hardware sample timestamps, the
continuity, the status and counters, finite data, and the overload limits, and
only then reads exactly those bytes. Publication only makes a chunk visible. No
per-chunk fsync is made for durability, and the final qualification comes later.
Since
the worker publishes before its own final checks, every live result is
provisional until the native acceptance, the complete byte count, and the
cleanup succeed.

A bounded queue holds at most 32 chunks of at most 8192 complex64 samples (2 MiB
of payload). Each payload slice is copied by the reader before it is queued, and
the single consumer then owns the copy. A queue overflow fails the attempt and
stops the native helper. On disk, the regular file spool is not bounded by the
queue. Its accepted payload for four seconds is 64 MB, and the native startup
and qualified capture files are also kept. Backlog reports distinguish validated
committed samples from accepted-file bytes that are visible before publication.
Divided by 2 MHz, the sample backlog gives the duration of the buffered samples,
and it is not a measured latency.

The consumer owns one StreamingDigitalAdapter, which is fed explicit hardware
sample indices with a new segment origin for every independently initialized
record. Missing samples are never fabricated, and captures are never stitched
together. In a qualified record, the unchanged worker refuses a transport gap.
Any failure ends the attempt, keeps the partial evidence, and leaves the later
attempts unattempted.

## Timing and concurrency evidence

The wrapper records local monotonic times for the observation of a completed
row, the enqueue, the start and end of service, and the production of each
frame. After service, the file size, sample backlog, and queue occupancy are
recorded with their own times. These clocks are never equated with RF arrival or with the
completion of a USB receive call. A pidfd refers to the actual native child,
which avoids ambiguity from PID reuse. Each successful attempt must produce at
least one frame while the native child is observed alive and while the accepted
file is still below its fixed final sample count. Evidence that the worker is
alive is not enough by itself, since that period can cover cleanup after the
capture. The final publication and the first observed complete file size are
recorded as two distinct events. Service completion of the last input is measured
from the observed final publication. After the helper exits, the final boundary
processing of the adapter runs, and its own recorded interval covers the native
cleanup and the wait for EOF. That longer interval is not a queue-drain latency,
and no hardware stop timestamp is invented from it.

The CPU service times and the observed backlog are descriptive. The queue bound
is an implementation limit and has not been calibrated as a deadline. Each
attempt has a host budget of 40 seconds in the wrapper, and the native helper
keeps its existing finite deadlines. The CPU consumer and its publication reader
share one pinned logical CPU, while the qualified native worker keeps the
affinity of the invoking process and its existing I/O threads. Each numerical
library is limited to one thread. Other work on the host can still interfere
with the measurement.

## End-of-record validation

After the native worker completes and its cleanup is confirmed, all accepted
bytes are rehashed, and each byte is verified to have been consumed once. Next,
the same bytes are replayed through the streaming adapter in independent
8192-sample partitions, and the exact float, packed, and projection hashes and
the output fields of every frame are compared. Finally, the frozen batch
reference is run, with float agreement checked to an absolute tolerance of 2e-12
and packed and projection equality checked exactly for these records. The
minimum quantizer half-step distance is kept, and floating-point quantization
parity is claimed for these records only.

The final incomplete output frame is discarded, with a receipt of its support.
All expected complete frames, the clipping counts, the finite, infinite, and
undefined ratios, and the failed attempts stay visible. No fitting, rescaling,
interval selection, retry, or gate change follows a primary outcome. Smoke
attempts are development runs and are kept apart from the primary records.

## Commands

The commands below run with `PYTHONPATH=src` and
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

```
python tools/run_sdr_live_handoff.py prepare NEW_OUTPUT --kind smoke \
  --build-manifest /path/to/sdr_reference_transport_2026-09-09/build-v1/build.json \
  --serial 0x1d423d9108f273
python tools/run_sdr_live_handoff.py run NEW_OUTPUT \
  --hardware-authorized --rf-confined-authorized
```

The three fixed four-second primary records use a new output, prepared on its
own with `--kind primary`. Preparation does not open the hardware. A launch is
refused when the frozen source, plan, or runtime has changed, or when the output
was launched before. No `--transmit` option exists in this wrapper. As release
evidence, the original capture receipts, all chunk publications, the host
timing, the boundary receipts, and the offline parity are kept. All source
identities are checked again after the attempted records, and a failed integrity
check is kept in the record. SIGINT and SIGTERM stop the owned helper and
preserve the failure state, and an attempt that needed forced cleanup never
qualifies.
