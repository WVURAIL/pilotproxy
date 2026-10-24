# Can CA-CFAR replace OS-CFAR for the fine decision?

29 September 2026. Release of record: `results/archive_no_split_2026-09-24_r2`, the release the dissertation vendors. Code: RFIsher a281112, pilot-proxy bdfcdce and RadioFisher 12a1616, cloned into `clones/`; no `.py` file differs from the release's `source_snapshots`. Definitions follow ch06: D is the anchor +/- 2 bins; B is the selector's bulk of 125 or 126 alternate bins (no product marks any census bins); Z(rho) = max_D T / T_(rho); Z_CA = max_D T / mean_B T over the same bulk.

## Decision

**No: CA should not replace OS.** If the goal is one parameter, use OS at a fixed median rank, rho = 63 (zero-based `cfar_rank` 62). It optimises over eta alone, like CA. It gives up at most 0.012 dB against CA in clean noise, and it is immune to the contamination this archive has. No verdict changes under any of the three rules: no rule has a feasible point on any channel, and the verdicts come from the capture ruling.

The premise that "the references are well behaved" holds for the median, not for the mean:

- **Bulk mean (part 1).** Under the ideal law, 99.9 percent of frames have a mean within 0.6 percent of the median, and none is more than 0.9 percent above it. On the archive, 7 to 49 percent of frames are inflated by more than 5 percent on the quiet channels, and 97 to 100 percent on the seven proxy-high channels. A 5 percent shift is about 2.5 standard deviations of Z, and the inflated frames are mostly the pilot-on frames.
- **Frontiers (part 2).** CA's least residual is worse on all seven channels with an r_tol:
  - 1.13 to 9.0 times the OS rank search's;
  - 1.01 to 7.3 times OS at rho = 63;
  - 1.13 to 7.6 times the rank search on the held-out evaluation block.

  The median rank alone comes within 1.006 to 1.68 times the rank search (0.97 to 1.45 times held out).
- **Channel 35 (part 3).** Each score's two modes are set by where the carrier sits, so at the separating eta both scores call carrier frames clean at the same rates. CA pushes carriers that drifted out of D toward the noise: at eta = 5 it calls 13 percent (peak in B) and 36 percent (peak elsewhere) of carrier frames clean, against 0.1 and 0.2 percent for Z(63).
- **Bench (part 4).** CA is 0.012 dB more sensitive than OS at rho = 63 in clean noise. One bulk bin at +10 dB costs CA 2.3 dB, one at +20 dB 9.0 dB, three at +30 dB 23.0 dB. OS at rho = 63 loses at most 0.03 dB in every case.

**Implementation.** The deployed decision is bit-exact because the reference is a single bulk bin's ratio: the kernel compares num[bin] * 2^16 * den_r > multiplier * num_r * den[bin] in 192-bit integers and finds the rank by pairwise cross-multiplication. A mean of 125 ratios of uint64 has no fixed-width exact form; its common denominator runs to thousands of bits (this study's exact boundary needed Python big integers). CA in the kernel would mean one of:

- floating point, which ends ch06's bit-for-bit reproduction claim;
- a different statistic, a ratio of summed powers (not tested, and just as exposed to one strong bin);
- big-integer arithmetic on the GPU.

## 1. Bulk behaviour on the archive (`bulk_behaviour.py`, `data/bulk/`, `figs/bulk_mean_over_median`)

**Method.** For every channel and every current-era frame (release era and split; counts match the ledger):

- r = mean_B T / median_B T;
- u = max_B T / median_B T (the upper-tail share).

**Ideal law.** i.i.d. F(4096, 8192), 4e5 simulated frames per bulk size. At |B| = 125:

- r: median 1.0003, 99th percentile 1.0046, 99.9th 1.0060, maximum 1.0089;
- u: median 1.071, 99th percentile 1.107, 99.9th 1.123.

**Explanations tested for frames with r > 1.05.** The mean is recomputed with each candidate removed:

- the near-anchor bulk bins (within +/- 7 bins; on an odd anchor these are the pilot's Dirichlet shoulders);
- persistent bins, above the ideal single-bin 0.999 point in more than half the frames (none is census-marked);
- the frame's own top 3 bins.

"Pilot on" means max_D T > 10 x median_B T.

| ch | state | anchor | \|B\| | med r | p99 r | r>1.05 | r>1.25 | med u | p99 u | >1.05 w/o near | w/o near+pers | w/o top3 | pers bins | pilot-on | >1.05 on/off |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 14 | low | 6 e | 125 | 1.0185 | 11.42 | 0.393 | 0.226 | 2.19 | 911 | 0.304 | 0.304 | 0.193 | 1 | 0.115 | 0.92/0.33 |
| 15 | high | 241 o | 126 | 1.2567 | 6.37 | 0.971 | 0.527 | 23.38 | 395 | 0.742 | 0.141 | 0.309 | 15 | 0.111 | 0.75/1.00 |
| 16 | low | 211 o | 126 | 1.0068 | 2.59 | 0.230 | 0.077 | 1.43 | 119 | 0.112 | 0.112 | 0.068 | 0 | 0.039 | 0.87/0.20 |
| 17 | high | 202 e | 125 | 9.5131 | 35.07 | 0.994 | 0.994 | 459.7 | 3852 | 0.994 | 0.105 | 0.994 | 45 | 0.886 | 1.00/0.95 |
| 18 | low | 172 e | 125 | 1.0334 | 9.29 | 0.422 | 0.156 | 2.82 | 542 | 0.326 | 0.326 | 0.176 | 2 | 0.145 | 0.82/0.35 |
| 19 | low | 154 e | 125 | 1.0024 | 1.38 | 0.072 | 0.015 | 1.12 | 28 | 0.057 | 0.057 | 0.017 | 0 | 0.020 | 0.43/0.06 |
| 20 | low | 141 o | 126 | 1.0843 | 3.70 | 0.690 | 0.124 | 8.84 | 234 | 0.148 | 0.111 | 0.081 | 8 | 0.339 | 0.65/0.71 |
| 21 | low | 126 e | 125 | 1.0040 | 2.89 | 0.149 | 0.051 | 1.22 | 117 | 0.140 | 0.140 | 0.052 | 0 | 0.016 | 0.90/0.14 |
| 22 | high | 106 e | 125 | 1.3631 | 14.95 | 0.997 | 0.811 | 10.22 | 952 | 0.996 | 0.081 | 0.992 | 30 | 0.995 | 1.00/0.54 |
| 23 | low | 122 e | 125 | 1.0050 | 3.10 | 0.261 | 0.107 | 1.30 | 127 | 0.254 | 0.254 | 0.094 | 0 | 0.016 | 0.79/0.25 |
| 24 | high | 66 e | 125 | 6.7044 | 14.37 | 1.000 | 1.000 | 157.7 | 828 | 1.000 | 0.020 | 1.000 | 51 | 1.000 | 1.00/- |
| 25 | low | 47 o | 126 | 1.0066 | 2.66 | 0.227 | 0.070 | 1.44 | 109 | 0.183 | 0.183 | 0.066 | 0 | 0.022 | 0.87/0.21 |
| 26 | low | 16 e | 125 | 1.0059 | 3.14 | 0.122 | 0.033 | 1.36 | 139 | 0.094 | 0.094 | 0.044 | 1 | 0.026 | 0.77/0.10 |
| 27 | low | 246 e | 125 | 1.0130 | 5.16 | 0.291 | 0.095 | 1.77 | 297 | 0.222 | 0.222 | 0.110 | 0 | 0.100 | 0.74/0.24 |
| 28 | low | 225 o | 126 | 1.0500 | 4.37 | 0.500 | 0.099 | 3.50 | 249 | 0.143 | 0.120 | 0.101 | 7 | 0.443 | 0.29/0.66 |
| 29 | low | 202 e | 125 | 1.0021 | 2.19 | 0.076 | 0.030 | 1.11 | 78 | 0.067 | 0.067 | 0.029 | 0 | 0.012 | 0.91/0.07 |
| 30 | high | 201 o | 126 | 8.1020 | 14.47 | 0.998 | 0.967 | 180.9 | 520 | 0.998 | 0.077 | 0.990 | 54 | 0.966 | 1.00/0.94 |
| 31 | high | 172 e | 125 | 8.5527 | 20.23 | 0.994 | 0.994 | 504.7 | 2126 | 0.994 | 0.064 | 0.994 | 45 | 0.985 | 1.00/0.62 |
| 32 | low | 141 o | 126 | 1.0034 | 2.50 | 0.120 | 0.038 | 1.18 | 106 | 0.091 | 0.091 | 0.043 | 0 | 0.019 | 0.96/0.10 |
| 33 | low | 128 e | 125 | 1.0490 | 2.05 | 0.485 | 0.054 | 2.33 | 68 | 0.411 | 0.285 | 0.172 | 8 | 0.042 | 0.86/0.47 |
| 34 | low | 90 e | 125 | 1.0416 | 2.50 | 0.343 | 0.042 | 4.02 | 107 | 0.293 | 0.176 | 0.047 | 2 | 0.026 | 0.56/0.34 |
| 35 | high | 107 o | 126 | 4.5970 | 8.80 | 0.985 | 0.984 | 225.5 | 638 | 0.985 | 0.015 | 0.984 | 38 | 0.980 | 1.00/0.28 |
| 36 | ambig. | 62 e | 125 | 1.0461 | 7.83 | 0.484 | 0.216 | 2.93 | 402 | 0.429 | 0.344 | 0.260 | 11 | 0.938 | 0.49/0.46 |

**What explains it:**

- **Proxy-high channels (15, 17, 22, 24, 30, 31, 35): an unmarked persistent pedestal.**
  - *Extent:* 15 to 54 bulk bins sit above the ideal 0.999 point in most frames, typically at offsets -26 to -60 from the anchor.
  - *With the pedestal and near-anchor bins removed:* the share inflated by more than 5 percent falls to 2 to 14 percent, and by more than 25 percent to 0 to 4 percent.
  - *Near-anchor bins or top 3 alone:* almost no effect, because the pedestal is wide.
- **Odd anchors with a strong pilot (15, 20, 28): the pilot's shoulders.** Removing the +/- 7-bin neighbourhood cuts the inflated share from 0.971 / 0.690 / 0.500 to 0.742 / 0.148 / 0.143. The frame's largest bulk bin sits within 7 bins of the anchor in 35, 84 and 69 percent of frames.
- **Proxy-low channels: sparse, sporadic bins in the pilot-on frames.**
  - *Not the near-anchor or persistent bins:* channel 21 falls only from 0.149 to 0.140.
  - *The frame's own top 3 bins:* they carry 77 to 87 percent of the excess (21 falls to 0.052).
  - *Pilot on against off:* inflated in 74 to 96 percent of pilot-on frames (14, 16, 18, 21, 23, 25, 26, 27, 29, 32), against 6 to 35 percent when the pilot is off.
- **The OS median** is untouched until more than half the bulk is contaminated.

## 2. Frontiers (`ca_frontier.py`, `compare_frontiers.py`, `replay63.py`)

**Adapted copy.** The release's preparation code already accepts a single key {1: requirements}, since it requires every rank 1..max(key). What cannot take it is `select_operating_point`, which always passes the bundle's rank columns, and `replay_on_block`. `make_selection_ca.py` writes `selection_ca.py`, a copy of the release's `selection.py` (diff in `selection_ca.diff`). It changes:

- the two relative imports, which become absolute;
- a new option `cfar='os'|'ca'` on those two functions (named `cfar` because the function already has a local called `score`);
- with `cfar='ca'`, preparation receives {1: Z_CA boundaries}; every other input is the same.

**Exact CA boundary** (`ca_score.py`, big-integer sums). It equals ceil(2^16 Z_CA) on all 250,699 frames checked, with no off-by-one cases.

**Proof, before any CA run.** On all seven channels, both the unmodified release function and the adapted copy with `cfar='os'` reproduce, byte for byte:

- the `selection.csv` row;
- `channels/chNN/operating_points.csv`;
- `channels/chNN/operating_point.csv`;
- the ledger's surface summary and `frontier_points`.

The two full surfaces (673,470 to 2,421,226 pairs) also have identical SHA-256 digests.

**Least residual (R = r_sys / r_tol).** The held-out column replays each point on the evaluation block.

| ch | r_tol | OS best: rho, R, m | OS 63: R, m | CA: R, m | CA/best | CA/63 | 63/best | feasible | held-out R: best, 63, CA |
|---:|---:|---|---|---|---:|---:|---:|---|---|
| 18 | 0.01196 | 2, 214.5, 0.958 | 224.6, 0.950 | 1130, 0.614 | 5.27 | 5.03 | 1.047 | 0,0,0 | 213.5, 233.2, 1322 |
| 19 | 0.01196 | 1, 7731, 0.975 | 9434, 0.952 | 3.24e4, 0.158 | 4.20 | 3.44 | 1.220 | 0,0,0 | 1.00e4, 1.39e4, 2.33e4 |
| 21 | 0.00851 | 8, 1.02e4, 0.968 | 1.25e4, 0.916 | 9.15e4, 0.113 | 9.00 | 7.32 | 1.230 | 0,0,0 | 1.23e4, 1.79e4, 9.38e4 |
| 22 | 0.00851 | 1, 4.90e6, 0.460 | 8.24e6, 0.004 | 8.30e6, 0.000 | 1.69 | 1.01 | 1.681 | 0,0,0 | 3.99e6, 5.09e6, 5.22e6 |
| 31 | 0.01583 | 94, 1.36e5, 0.989 | 1.43e5, 0.989 | 4.44e5, 0.980 | 3.26 | 3.11 | 1.050 | 0,0,0 | 5.40e5, 5.21e5, 6.59e5 |
| 32 | 0.01583 | 3, 4818, 0.990 | 5655, 0.839 | 1.97e4, 0.357 | 4.08 | 3.48 | 1.174 | 0,0,0 | 5155, 5810, 2.78e4 |
| 33 | 0.01583 | 34, 690.8, 0.158 | 694.8, 0.127 | 785.7, 0.114 | 1.14 | 1.13 | 1.006 | 0,0,0 | 134.1, 135.2, 151.4 |

**Surface sizes and envelope comparisons.**

| ch | OS pairs / frontier pts | CA pairs / frontier pts | CA reaches the OS-best R at any mask | OS mask for CA's least R | least R at m <= 0.5: OS, OS 63, CA |
|---:|---|---|---|---:|---|
| 18 | 1,231,451 / 3,222 | 9,856 / 753 | no | 0.108 | 413, 461, 1247 |
| 19 | 673,470 / 1,277 | 5,395 / 441 | no | 0.080 | 1.25e4, 1.34e4, 3.24e4 |
| 21 | 1,161,990 / 3,543 | 8,986 / 860 | no | 0.016 | 1.23e4, 1.34e4, 9.15e4 |
| 22 | 2,421,226 / 1,997 | 19,315 / 1 | no | 0.000 | 4.90e6, 8.24e6, 8.30e6 |
| 31 | 1,285,897 / 3,699 | 10,254 / 2,534 | no | 0.977 | 6.91e5, 6.99e5, 7.05e5 |
| 32 | 1,039,980 / 2,891 | 8,148 / 720 | no | 0.069 | 6165, 6264, 1.97e4 |
| 33 | 1,539,961 / 1,027 | 12,190 / 590 | no | 0.035 | 690.8, 694.8, 785.7 |

**Why CA is worse.** On 18, 19, 21 and 32 the CA curve turns back on itself: below its least-residual point, masking more raises the residual. At a low multiplier CA keeps the frames with an inflated bulk mean, which are the pilot-on, contaminated frames.

**Channel 22.** CA's only frontier point is keeping everything (8.30e6 is the unmasked R). The rank search reaches 4.90e6 at rho = 1.

**Feasibility.** Unchanged. Both families are refused by the drift screen, as in the release: insufficient support on 18 and 19, drift on the others. Every point here is a diagnostic.

**Figures.** `figs/ca_frontier_ch{18,19,21,22,31,32,33}` and `figs/ca_frontier_overview`.

## 3. Channel 35's drift (`drift35.py`, `figs/drift_ch35`)

**Setup.** Current era, 29,767 frames; anchor 107 (odd); D = 105 to 109; |B| = 126. The carrier peak is the largest T within +/- 30 bins. "No carrier" means a peak below 10 x the bulk median; there is a clean gap between 9.6 and 10.8.

**Modes.** Z(63) has modes at 39 and 298, with the valley at 125. Z_CA has modes at 6.4 and 57, with the valley at 29.

**Fraction of carrier frames called clean.**

| peak | frames (calibration / evaluation share) | med mean/median | med Z63 | med ZCA | valley: Z63 / ZCA | eta=5 | eta=10 | ideal 1e-3 (1.10) |
|---|---|---:|---:|---:|---|---|---|---|
| in D | 9,880 (0.447 / 0.217) | 3.15 | 314 | 98.9 | 0.0015 / 0.0008 | 0 / 0 | 0 / 0 | 0 / 0 |
| in B | 10,542 (0.270 / 0.439) | 4.66 | 59.1 | 13.1 | 0.672 / 0.684 | 0.0013 / 0.133 | 0.0066 / 0.389 | 0 / 0.0008 |
| elsewhere (odd bins) | 8,894 (0.255 / 0.343) | 5.57 | 37.2 | 6.31 | 0.984 / 0.989 | 0.0017 / 0.356 | 0.0083 / 0.767 | 0 / 0.0012 |
| no carrier | 451 (0.029 / 0.002) | 1.00 | 1.03 | 1.03 | 1 / 1 | 1 / 1 | 1 / 1 | 0.960 / 0.962 |

- **At the separating eta the scores agree.** Carriers inside D are flagged, and drifted carriers are called clean at 67 to 68 percent (peak in B) and 98 to 99 percent (elsewhere), under both scores. That is a window problem, not a reference problem.
- **They differ in the drifted carriers' distance from the noise.** The carrier inflates the mean it is divided by (median 4.7 to 5.6 times the bulk median), so Z_CA drops to 13 and 6.3, against 59 and 37 for Z(63). 1 percent of these frames sit below 2.4 and 1.8 on Z_CA.
- **AUC for separating in-D from out-of-D frames:** CA 0.992, OS 0.978. CA's edge comes only from compressing the out-of-D carriers toward the noise.

## 4. Synthetic bench (`bench.py`, `figs/bench_ca_os`)

**Model:**

- **Bins:** ideal i.i.d. F(4096, 8192); the numerator is drawn exactly, with common random numbers across all cases.
- **Geometry:** |D| = 5; the bulk is 125 bins at an even anchor, 126 at an odd one.
- **Pilot:** a tone on the anchor with a rectangular Dirichlet response. On the odd anchor its shoulders reach the bulk: 0.045 at +/- 3, 0.19 summed.
- **Thresholds:** each score's own clean-null 1e-3 point from 4e6 frames: CA 1.0999, OS 63 1.1005, OS 94 1.0810, OS 113 1.0638.

**SNR loss at Pd = 0.9, in dB, against clean OS 63** (-8.63 dB per bin). Even anchor; the odd anchor agrees within 0.1 dB.

| contamination | CA | OS 63 | OS 94 | OS 113 |
|---|---:|---:|---:|---:|
| none | -0.01 | 0 | +0.01 | +0.03 |
| 1 bin at +10 dB | +2.27 | +0.01 | +0.02 | +0.07 |
| 2 bins at +10 dB | +3.71 | +0.02 | +0.04 | +0.11 |
| 3 bins at +10 dB | +4.83 | +0.03 | +0.06 | +0.18 |
| 1 bin at +20 dB | +8.95 | +0.01 | +0.02 | +0.07 |
| 3 bins at +20 dB | +13.32 | +0.03 | +0.06 | +0.18 |
| 1 bin at +30 dB | +18.49 | +0.01 | +0.02 | +0.07 |
| 3 bins at +30 dB | +23.04 | +0.03 | +0.06 | +0.18 |

**Clean-noise loss at |B| = 125** (4e5 frames, 10 batches):

- OS 63 needs 0.0119 +/- 0.0009 dB more than CA at Pd = 0.5, and 0.0114 +/- 0.0003 dB at Pd = 0.9.
- OS 94 needs 0.018 dB more, OS 113 0.041 dB.
- The loss is small because the reference's spread (0.0024 for the mean, 0.0030 for the median) is a tenth of the spread of max_D T.

**Other effects:**

- **False alarms:** under contamination CA's actual false-alarm rate drops to 0, meaning it is desensitised.
- **Shoulders:** they leave Pd near threshold unchanged, but compress median Z_CA / Z(63) to 0.99 at +10 dB, 0.89 at +20 dB and 0.45 at +30 dB (0.22 on the channel 35 archive frames).

## 5. Recommendation

**Use OS at a fixed median rank, rho = 63, if the goal is one parameter; do not switch to CA.**

| | CA | OS rho = 63 | OS rank search |
|---|---|---|---|
| parameters | eta | eta | rho, eta |
| clean SNR at Pd = 0.9 vs CA | 0 | +0.011 dB | +0.01 to +0.04 dB |
| loss, 1 / 3 bins at +10 dB | 2.3 / 4.8 dB | 0.01 / 0.03 dB | at most 0.07 / 0.18 dB |
| archive frames with the reference shifted > 5 percent | 7 to 100 percent | none (median) | as OS 63 |
| least R vs rank search, calibration / held out | 1.13 to 9.0 / 1.13 to 7.6 times | 1.006 to 1.68 / 0.97 to 1.45 times | 1 |
| feasible points | 0 | 0 | 0 |
| bit-exact in kernel | no fixed-width form | yes (rank 62) | yes |

**Rank search against a fixed median rank.** The rank search buys 1.05 to 1.68 times in least residual on the calibration block, and less held out; on channel 31 the median rank is better held out. It also costs 125 times the surface and one more tuned degree of freedom. Keeping it is defensible if the thesis wants the full frontier, but no verdict needs it.

**No verdict changes.** There are zero feasible points under all three rules, and the verdicts come from the capture ruling.

**What a switch would touch** (read-only checks):

- **Kernel and kotekan PR 1677** (`kotekan-detector-pr`, 1d137db5).
  - *What the code does now:*
    - `cudaPilotProxyDetector.cpp` reads `cfar_rank` and `cfar_multiplier_q16` from the bundle's `fine_calibration` ("fine_decision_v1") and checks the rank against the bulk.
    - It calls `FStat_Compute_FusedFineMask_U64`. The vendored `external/pilotproxy/f_statistic.cu` (byte-identical to upstream 2d0381ed) finds the rank by counting with 128/192-bit cross-multiplication.
    - `testPilotProxyDetector.cpp` holds a CPU reference of the same rank selection.
  - *A fixed median rank:* nothing changes here; it is `cfar_rank` = 62 in the bundle.
  - *CA would need:*
    - a new kernel epilogue and `decision_version`;
    - upstream pilot-proxy changes: `cuda/f_statistic.{cu,h}`, `fine_decision.py`, `product_contract.py` ("fine_order_statistic_cfar"), `runtime_bundle.py` and tests (21 files mention `cfar_rank`);
    - a re-vendor with new digests;
    - a new exactness design.
- **Selector and tests** (RFIsher):
  - `residual_scores/bundle.py`: per-rank boundaries, the method check, the contiguous `rho` array;
  - `preparation.py`: `candidate_rho_values`, `rho_from_cfar_rank`, the every-rank check;
  - `selection_policy.py`, `thresholds.py`, `archive/selection.py`, `operating.py` and the report modules.

  That is 24 source files and 22 test files (162 test lines) using `rho`. A fixed median rank needs only a single-key family (key 63) plus a relaxed every-rank check or a policy option. CA needs all of that plus a new boundary function.
- **Archive fine tables.**
  - *Release:* `selection.csv` (`rho`, `q_rho`, `diagnostic_rho`), `operating_points.csv`, `channels/chNN/operating_points.csv`, the ledger's selection sections.
  - *Thesis:* `tables/current/held_out_summary.tex` and the `figure_src` exporters and record CSVs (13 files).
  - Every diagnostic number changes under either switch; for example, channel 33's least R goes from 691 to 695 (median rank) or 786 (CA).
- **Thesis text.**
  - *ch06* (42 `\rho` lines): the ranked-background rule, T_(rho), q_rho, the per-rank histograms, the frontier over (rho, eta), the one-sided contamination bound and the invalid-frame rule.
  - *Elsewhere:* ch07 to ch09 (3, 3, 8 lines), the ch04 CFAR remark, ch05 (one line), Appendix B and the Appendix C fine plates.
  - *Under CA*, the contamination bound and the bit-exact claim go. *Under a fixed median rank* both stay, and ch06 already uses the median rank as the primary rule of its synthetic validation ("644 primary median-rank detection" cells).

## Scope and caveats

- The bench treats D's padded bins as independent and puts the pilot on the anchor bin. Real pilots sit off-grid, which leaks into even-anchor bulks too; part 1 shows this.
- The ideal law is a reference, not eta_Pfa: there is no verified transmitter-off null.
- The CA tested is the mean of T. The ratio-of-summed-powers variant was not tested.
- Residuals use the release's shelf-or-floor convention and chain gain, under the unmeasured unity transfer; prices are diagnostics.
- Channel 35's classes use the calibration anchor 107. The era-wide anchor is 111, which is the drift.

## Files and regeneration

**Interpreters.** Analysis ran on `output/fisher-fixes-2026-09-08/analysis-venv`; plots on `~/.venvs/dissertation`. `figstyle/style.py` is a byte copy of `dissertation/figure_src/style.py` (sha256 7ad4b752...). Every run used `-B`, `PYTHONDONTWRITEBYTECODE=1` and one thread.

**Order:**

1. `bulk_behaviour.py last`, then `bulk_behaviour.py 14 ... 36`
2. `make_selection_ca.py`
3. `ca_frontier.py {orig,os,ca} CH`
4. `bench.py`
5. `compare_frontiers.py`
6. `replay63.py CH`
7. `drift35.py`
8. `plot_cfar.py all`

Logs are in `logs/`.

**Main files** (all under `/home/djg/rail/output/cfar-choice-2026-09-29/`):

- Scripts: `common.py`, `bulk_behaviour.py`, `ca_score.py`, `make_selection_ca.py`, `selection_ca.py` and `selection_ca.diff`, `ca_frontier.py`, `compare_frontiers.py`, `replay63.py`, `drift35.py`, `bench.py`, `plot_cfar.py`
- Data: `data/bulk/chNN.{json,npz}`, `data/ideal_bulk.json`, `data/frontier_compare.json`, `data/ca_boundary_check.json`, `data/drift35.json`, `data/bench.json`
- Surfaces: `surfaces/chNN_{orig,os,ca}.{npz,json}` and `surfaces/chNN_os63_replay.json`
- Figures in `figs/`, PDF and PNG: `bulk_mean_over_median`, `ca_frontier_ch{18,19,21,22,31,32,33}`, `ca_frontier_overview`, `drift_ch35`, `bench_ca_os`

## Corrections from the independent check (CHECK.md: PASS; none changes the decision)

1. Section 5, clean-SNR row: the rank search's entry is +0.02 to +0.49 dB at the ranks it picks here (rho 1, 2, 3, 8, 34, 94), not +0.01 to +0.04 dB. This strengthens the case for rho = 63.
2. Section 5 text: the rank search buys 1.006 to 1.68 times (channel 33 is 1.006), not 1.05 to 1.68; the held-out upper end is 1.455, which rounds to 1.46.
3. Section 1: the persistent bins on the proxy-high channels lie on both sides of the anchor (35: +/- 3 to 39; 15: -13 to +21; 24: -58 to +60; 30: -67 to +67; 22: -60 to +58). That fits the strong carrier's own leakage skirt and its drift, not an unmarked pedestal, so census marking would not rescue CA: the skirt moves with the carrier.
4. Section 2: CA is worse for the least R and for m <= 0.5; channel 31 is an exception at m of 0.66 to 0.85, by at most 0.8 percent.
5. Section 4: the losses are measured against the coarse-grid clean OS63 (-8.59 dB), not -8.63 dB.
6. The ideal-law maximum of r depends on the sample (1.0096 in the checker's 3e5 frames); read it as about 1 percent.
