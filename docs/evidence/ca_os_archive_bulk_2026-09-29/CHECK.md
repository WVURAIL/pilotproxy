# Independent check: CA-CFAR against OS-CFAR for the fine decision

29 September 2026. I did not do the analysis. My own code is in `verify/` (logs next to each script). It ran on the analysis venv with `-B`, `PYTHONDONTWRITEBYTECODE=1` and one thread. Nothing was written outside this folder. The three clones are still clean at a281112, 12a1616 and bdfcdce, and `selection_ca.diff` matches a freshly computed diff of `selection_ca.py` against the clone.

## 1. Part 1 recomputed on channels 21, 35 and 15 (`verify/v_bulk.py`)

**My own code for everything except the health gate:** the era cut from the ledger's months, the calibration split from the ledger's boundary time, the bulk mask (even bins, minus anchor +/- 2), T, the ideal-law simulation with a different seed, and every statistic. The release's `Product` supplied only `selected` and the frame and unit times.

**Frame and split counts** match the ledger: 39,617 / 19,813 on channel 21, 29,767 / 14,885 on 35 and 26,703 / 13,355 on 15.

**Every table column matches** on all three channels to the precision the table prints. For example:

| ch | med r | r > 1.05 | w/o near+pers | w/o top 3 | pers bins | pilot-on | > 1.05 on/off |
|---:|---:|---:|---:|---:|---:|---:|---|
| 21 | 1.0040 | 0.149 | 0.140 | 0.052 | 0 | 0.016 | 0.90/0.14 |
| 35 | 4.5970 | 0.985 | 0.015 | 0.984 | 38 | 0.980 | 1.00/0.28 |
| 15 | 1.2567 | 0.971 | 0.141 | 0.309 | 15 | 0.111 | 0.75/1.00 |

**Ideal law** at |B| = 125 (3e5 frames): r has median 1.0003, 99th percentile 1.0046 and 99.9th percentile 1.0060; u has median 1.071. This agrees with the report. The maximum r was 1.0096, against the report's 1.0089, so "none more than 0.9 percent above the median" only describes the report's own sample.

## 2. Part 3 recomputed on channel 35 (same script)

These all match the report:

- the no-carrier gap, 9.62 to 10.81;
- the class counts (9,880 / 10,542 / 8,894 / 451) and their calibration and evaluation shares;
- the class medians of mean/median, Z63 and ZCA;
- the called-clean fractions at eta = 5, eta = 10 and the ideal thresholds (for example, in B 0.0013 / 0.133 and elsewhere 0.0017 / 0.356 at eta = 5);
- the 1st percentiles of ZCA (2.41 and 1.76);
- the AUC, 0.978 for OS and 0.992 for CA.

**Valleys and modes** from my own valley finder:

- Z63: modes 39.5 and 302, valley 126 (report 39 / 298 / 125);
- ZCA: modes 6.6 and 60, valley 29.6 (report 6.4 / 57 / 29).

The called-clean fractions at the valleys differ from the report's by at most 0.002.

## 3. Bench at one setting: 1 bulk bin at +10 dB, even anchor, |B| = 125 (`verify/v_bench.py`)

**Method.** Different from the report's. The reference is simulated (4e5 frames). The D bins enter through scipy's exact noncentral-F CDF, and eta is solved so that Pfa = 1e-3 exactly.

**Thresholds:** CA 1.1001 and OS63 1.1007 (report 1.0999 and 1.1005).

**SNR at Pd = 0.9:**

- clean: CA -8.631 dB and OS63 -8.619 dB, so OS63 needs 0.0116 dB more (report 0.0114 +/- 0.0003);
- with the contaminated bin: CA loses 2.24 dB and OS63 0.011 dB against clean OS63 (report +2.27 and +0.01);
- the 0.03 dB gap on CA comes from the report's reference: its losses are measured against the coarse-grid clean OS63 (-8.59 dB), not the -8.63 dB the heading quotes;
- CA's actual Pfa under contamination is 4e-10 (report: 0).

## 4. The adapted selector reproduces the OS surface (`verify/v_selector.py`, `v_sel33.py`, `v_camask.py`)

**My own driver**, run on channels 19 and 33, called both the unmodified release `select_operating_point` and `selection_ca.select_operating_point(cfar='os')`. Channel 19 has 673,470 points and channel 33 has 1,539,961. On both channels:

- the two point lists are equal, and my own canonical digest agrees (float values hashed through their hex form);
- `write_operating_points` and `operating.choose` output from both paths is byte-identical to the archive's `channels/chNN/operating_points.csv` and `operating_point.csv`;
- the diagnostic, the refusal and the held-out R are equal.

**The CA surfaces carry the right score.** On channels 19, 21 and 33, an independent float Z_CA (ceil(65536 Z_CA)) reproduces the `kept` count at every CA surface point (5,395 / 8,986 / 12,190 points, maximum difference 0). None of the 21 stored surfaces has a feasible point.

## 5. Figures (all 11 PNGs read)

The figures agree with the data and the text. Five cosmetic fixes are optional:

1. In `bulk_mean_over_median` (a), the "5 percent" label runs into the ideal band.
2. `ca_frontier_ch22` and `ca_frontier_ch31` list "floor-only residual" in the legend, but draw no line, because those channels have no floor.
3. In `drift_ch35` (d), the legend covers the ZCA "peak elsewhere" curve, and the valley markers use the same dotted style as "peak in D".
4. The `ca_frontier_ch21` subtitle reaches the right edge.
5. The OS63 markers on channels 22 and 31 are hidden under other markers.

## 6. Does the other option win anywhere? (`verify/v_env.py`, `verify/v_bench_ranks.py`)

**CA against OS63, at each mask budget m** (the least R among points with masked fraction at most m):

- **Channel 31:** CA is below OS63 for m between 0.66 and 0.85, by at most 0.8 percent, at R about 6.6e5 to 7.0e5.
- **Channels 18, 19 and 33:** CA is below OS63 only at m <= 0.002, by less than 0.5 percent.
- **Rank-search envelope:** CA is never below it on any channel.
- **Held out:** CA is worse than OS63 on every channel, by 1.02 to 5.7 times.
- **Clean noise:** CA is better by 0.012 dB.
- **Channel 35:** CA has the higher AUC, but only because it compresses the drifted carriers toward the noise.

None of these wins is material, so the recommendation stands.

**OS63 against the rank search.** The report is right that OS63 is better held out on channel 31 (0.965). It also misses a point in OS63's favour. The report's clean-noise loss of "+0.01 to +0.04 dB" for the rank search covers only ranks 63, 94 and 113. On five of the seven channels the search picks ranks 1 to 8. My check of the clean loss against CA at Pd = 0.9:

| rank | 1 | 2 | 3 | 8 | 34 | 63 | 94 | 113 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss (dB) | 0.49 | 0.23 | 0.15 | 0.06 | 0.017 | 0.012 | 0.017 | 0.037 |

## Corrections for REPORT.md (none changes the decision)

1. **Section 5 table, clean-SNR row.** Change the rank-search entry from "+0.01 to +0.04 dB" to "+0.02 to +0.49 dB at the ranks it picks here (rho 1, 2, 3, 8, 34, 94)". This strengthens the case for OS63.
2. **Section 5 text.** "The rank search buys 1.05 to 1.68 times" contradicts the table and the Decision (1.006 to 1.68; channel 33 is 1.006). The held-out upper end is 1.455, which rounds to 1.46, not 1.45.
3. **Section 1, proxy-high attribution.** "Typically at offsets -26 to -60 from the anchor" is wrong. The persistent bins lie on both sides of the anchor: channel 35 at +/- 3 to 39, channel 15 at -13 to +21, channel 24 at -58 to +60, channel 30 at -67 to +67, channel 22 at -60 to +58. That pattern fits the strong carrier's own leakage skirt and drift. Calling it an "unmarked persistent pedestal" suggests that census marking would fix CA. It would not, because the skirt moves with the carrier.
4. **Section 2 envelopes.** The claim that CA is worse holds for the least R and for m <= 0.5. Add the channel 31 exception at m of 0.66 to 0.85 (at most 0.8 percent).
5. **Section 4 heading.** The losses are measured against the coarse-grid clean OS63 (-8.59 dB), not -8.63 dB.
6. **Ideal-law maximum.** It depends on the sample (1.0096 in my 3e5 frames), so say "about 1 percent" or give the frame count.

CHECK: PASS
