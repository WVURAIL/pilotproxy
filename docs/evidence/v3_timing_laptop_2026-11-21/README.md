# Fine decision v3 core timing on the laptop RTX 5000 Ada — 2026-11-21

`PilotProxy_Compute_FineStatV3_U64` timed with core 4.2.0 (tree of 8f0c8d5, `make -C cuda SM=89`) at the CHIME
geometry: 2048 streams, K = 128, 128 windows, frame 41.94 ms. Host: the RTX 5000 Ada laptop GPU under WSL2, with
no correlator load. This is not a deployment GPU. The A40 and L40S runs, idle and beside the Kotekan load, are
still to do.

| Form | What it calls | Median | p16–p84 | Share of frame |
|---|---|---|---|---|
| `stat/n1` | one channel | 0.395 ms | 0.372–0.428 ms | 0.9% |
| `seq_stat/n16` | 16 calls in sequence (the stage path) | 5.99 ms | 5.96–6.31 ms | 14.3% |
| `batch_stat/n16` | 16 channels in one batch call | 4.29 ms | 4.26–4.74 ms | 10.2% |

Idle: 50 frames per measurement, 30 repeats, order rotated, after one warm-up call (`idle.json`).

Paced: `seq_stat/n16` once per 41.94 ms frame for 30 s, 716 frames (`paced_idle.json`): median 11.86 ms (28.3%),
p99 23.65 ms, maximum 52.08 ms (1.24 frames), 1 late frame. The paced median is twice the idle one because the
laptop GPU clocks down between frames.

## Reproduce

    make -C cuda SM=89
    PYTHONPATH=src python docs/evidence/v3_timing_laptop_2026-11-21/v3_timing.py idle  cuda/libpilotproxy.so idle.json 30
    PYTHONPATH=src python docs/evidence/v3_timing_laptop_2026-11-21/v3_timing.py paced cuda/libpilotproxy.so paced_idle.json 30

Needs CuPy. The inputs are random int8 samples and weights.
