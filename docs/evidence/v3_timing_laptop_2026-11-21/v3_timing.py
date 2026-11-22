"""Time the detector core under fine decision v3 (PilotProxy_Compute_FineStatV3_U64) at the CHIME geometry.

The forms are those of the dissertation's A40 and L40S records (wide-time-2026-11-06, l40s-time-2026-11-20), with
the v3 statistic in place of the 65-bin mask:
  stat/n1        one channel, one FineStatV3 call per frame (powers out)
  seq_stat/n16   16 handles, 16 sequential FineStatV3 calls per frame, the stage path
  batch_stat/n16 one batch handle of 16 channels, one call per frame
Each form is timed over N_FRAMES frames with perf_counter and a device synchronize, N_REPEATS times with the order
rotated, after one warm-up call. In paced mode, seq_stat/n16 is launched once per 41.94 ms frame for DURATION_S
seconds, as beside the correlator load.

usage: v3_timing.py idle  LIB OUT.json [N_REPEATS]
       v3_timing.py paced LIB OUT.json DURATION_S
Needs the pilotproxy source tree (core 4.2.0 or later) on PYTHONPATH and LIB built from it.
"""
import json
import platform
import sys
import time

import cupy as cp
import numpy as np
from pilot_proxy.kernel import PilotProxyKernel

MODE, LIB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K, STREAMS, WINDOWS, TERMS, FINE_BINS = 128, 2048, 128, 3, 256
ROWS = STREAMS * WINDOWS
N_FRAMES = 50
FRAME_MS = WINDOWS * K * 2.56 / 1000.0              # 41.94304 ms at the CHIME sample rate

k = PilotProxyKernel(LIB)
if not k.supports_fine_stat_v3():
    raise SystemExit(f"{LIB} has no PilotProxy_Compute_FineStatV3_U64")
rng = np.random.default_rng(20261018)
weights = [rng.integers(0, 256, size=(TERMS, K), dtype=np.uint8).astype(np.int8) for _ in range(16)]
cp.random.seed(20261018)
d_in = cp.random.randint(0, 256, size=(16, ROWS, K), dtype=cp.int32).astype(cp.uint8).view(cp.int8)
seq = [k.create_raw(ROWS, d_in[i].data.ptr) for i in range(16)]
batch = k.create_raw_batch(ROWS, 16, d_in.data.ptr)
fine = cp.zeros((16, TERMS, FINE_BINS), dtype=cp.uint64)
stat = cp.zeros((16, 2), dtype=cp.uint64)
power = cp.zeros((16, TERMS), dtype=cp.uint64)
cp.cuda.Device().synchronize()


def stat_n1():
    k.compute_fine_stat_v3_u64(seq[0], weights[0].ctypes.data, int(fine[0].data.ptr), int(stat[0].data.ptr),
                               int(power[0].data.ptr))


def seq_stat_n16():
    for i in range(16):
        k.compute_fine_stat_v3_u64(seq[i], weights[i].ctypes.data, int(fine[i].data.ptr), int(stat[i].data.ptr),
                                   int(power[i].data.ptr))


def batch_stat_n16():
    k.compute_fine_stat_v3_u64(batch, weights[0].ctypes.data, int(fine.data.ptr), int(stat.data.ptr),
                               int(power.data.ptr))


FORMS = {"stat/n1": stat_n1, "seq_stat/n16": seq_stat_n16, "batch_stat/n16": batch_stat_n16}
meta = dict(lib=LIB, gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), host=platform.node(),
            streams=STREAMS, k=K, frame_ms=FRAME_MS, decision="fine_decision_v3",
            entry="PilotProxy_Compute_FineStatV3_U64")

if MODE == "idle":
    repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    for f in FORMS.values():
        f()
    cp.cuda.Device().synchronize()
    times = {name: [] for name in FORMS}
    names = list(FORMS)
    for r in range(repeats):
        for name in names[r % len(names):] + names[:r % len(names)]:
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            for _ in range(N_FRAMES):
                FORMS[name]()
            cp.cuda.Device().synchronize()
            times[name].append((time.perf_counter() - t0) / N_FRAMES * 1e3)
    out = dict(meta, mode="idle", frames_per_measurement=N_FRAMES, repeats=repeats,
               forms={n: dict(median_ms=float(np.median(v)), p16_ms=float(np.percentile(v, 16)),
                              p84_ms=float(np.percentile(v, 84)), share_of_frame=float(np.median(v)) / FRAME_MS)
                      for n, v in times.items()})
elif MODE == "paced":
    duration = float(sys.argv[4])
    seq_stat_n16()
    cp.cuda.Device().synchronize()
    times, late = [], 0
    t_next = time.perf_counter()
    t_end = t_next + duration
    while t_next < t_end:
        now = time.perf_counter()
        if now < t_next:
            time.sleep(t_next - now)
        t0 = time.perf_counter()
        seq_stat_n16()
        cp.cuda.Device().synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        t_next += FRAME_MS / 1e3
        if time.perf_counter() > t_next:
            late += 1
    a = np.array(times)
    out = dict(meta, mode="paced", form="seq_stat/n16", frames=len(a), median_ms=float(np.median(a)),
               p99_ms=float(np.percentile(a, 99)), max_ms=float(a.max()), late_frames=late,
               share_of_frame=float(np.median(a)) / FRAME_MS, max_share_of_frame=float(a.max()) / FRAME_MS)
else:
    raise SystemExit("mode must be idle or paced")
if k.last_error():
    raise SystemExit(k.last_error())
for h in seq + [batch]:
    k.destroy(h)
print(json.dumps({key: v for key, v in out.items() if key != "lib"}, indent=1))
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
