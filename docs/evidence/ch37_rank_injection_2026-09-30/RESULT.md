# Choosing the fixed rank by injection: stopped at the controls

30 September 2026. Implements DESIGN.md. Release of record `results/archive_no_split_2026-09-24_r2`; code from the cfar-choice-2026-09-29 clones, used read-only. Every file is under this folder.

## Verdict

**No rank is chosen.** DESIGN.md section 4 says the two injection modes must agree on channel 37 and on the carrier-off channels. They do not agree, and the gap is large. The objective run (sections 1 to 3 on the 16 objective channels) and its figures were therefore not produced. Per the task, the study stops here and reports the failure.

- **Code-path control (passes).** The ideal i.i.d. control reproduces the rank bench:
  - rho = 41 costs 0.0142 dB (bench 0.0141);
  - rho = 63 costs 0.0116 dB (bench 0.0115);
  - the argmin is 64 / 63 (bench 62 / 64);
  - all six quadrature values of `bench_quad.json` fall within 2 SE.
- **Mode agreement (fails).** On channel 37, which has no television allocation:
  - Mode I (real bulk, ideal designated bins) puts the best rank at 72 to 75 in all 23 channel geometries.
  - Mode R (real bulk, real designated bins, tone injected) puts it at 123 to 125.
  - At rho = 63, mode I loses 0.12 to 0.15 dB and mode R loses 4.5 to 6.2 dB (calibration half, Pfa 1e-3). The combined 2 SE is about 1.9 dB.
  - Only 3 of the 27 control units (23 geometries plus 4 carrier-off channels) agree at rho = 63. A median of 17 percent of ranks agree.
- **Why the modes disagree.** Mode I assumes the designated bins follow the ideal law, and they do not. Take mode I's own Pfa = 1e-3 threshold and apply it to channel 37's real designated bins:
  - the false-alarm rate is 0.36 to 0.48 percent at rho = 41 and 1.5 to 1.8 percent at rho = 63 (calibration half);
  - on the held-out half it is 1.5 to 1.6 percent and 2.9 to 3.3 percent;
  - so the real rate is 3.6 to 33 times the nominal rate.
- **One result holds in every variant: rho = 41 fails the budget on channel 37.** The budget is ch06's accepted straddle loss, "doubling the hypothesis grid bounds the worst-case straddle loss near $0.9$~dB". At rho = 41 on channel 37:
  - mode I loses 3.1 dB on calibration and 4.1 dB held out;
  - mode R loses 5.9 to 7.3 dB and 7.7 to 9.2 dB;
  - even mode I at Pfa 1e-2 loses 3.0 dB held out.
- **Where the "why a quarter?" question now stands.** The injection answer depends on which model of the designated-bin null is right, and the design's own agreement test says that is not yet settled.

Every threshold here is only a device for comparing ranks, not an eta_Pfa for any channel. The verdicts still come from the capture ruling.

## 1. Frames

`rank/extract.py` rebuilt, for every channel's current era, the sorted usable bulk T_(1..|B|), the five designated values, the frame month and the calibration flag. It checked each of these against cfar-choice's per-frame arrays and against the ledger split:

- rows, max_D T and median_B T are asserted equal;
- the near-threshold counts reproduce `design/design_counts.json` exactly, per half, on all 23 channels.

Channel 37 is the control's 40,477 histogram-eligible frames (product sha256 e77d1d41...). They are placed at each of the 23 channel geometries: anchor, designated set and bulk mask.

`blocks.split_blocks` splits channel 37 on its own frame times into:
- 20,240 calibration frames over 66 months;
- 20,237 evaluation frames over 24 months;
- the boundary falls in 2024-08, as DESIGN.md anticipated.

Near-threshold frames per half (w <= w_cut = 1.2962), cal / eval:

| ch | near | ch | near | ch | near |
|---:|---|---:|---|---:|---|
| 14 | 11,619 / 9,728 | 22 | 15 / 146 | 30 | 108 / 0 |
| 15 | 589 / 6,718 | 23 | 17,568 / 16,577 | 31 | 90 / 23 |
| 16 | 13,743 / 11,715 | 24 | 0 / 0 | 32 | 7,924 / 7,526 |
| 17 | 157 / 101 | 25 | 16,226 / 14,676 | 33 | 6,470 / 5,762 |
| 18 | 1,594 / 1,724 | 26 | 7,695 / 5,863 | 34 | 12,866 / 17,594 |
| 19 | 5,009 / 5,097 | 27 | 3,400 / 2,840 | 35 | 421 / 25 |
| 20 | 556 / 565 | 28 | 772 / 914 | 36 | 82 / 11 |
| 21 | 18,006 / 17,062 | 29 | 18,302 / 18,784 | | |

With the lower-quartile screen w_lo, the counts over both halves pooled are in `data/frames/chNN.json` (`near_lo`).

## 2. Controls

### 2a. Ideal i.i.d. control of the mode I code path: passes

Source: `rank/controls.py` and `data/controls_bench_125.json` / `_126.json`.

**Setup.** The bench's layout: |B| = 125 and 126, 20 batches of 20,000 i.i.d. F(4096, 8192) frames, threshold at Pfa 1e-3, and the Pd = 0.9 SNR read from the table. Loss is measured against CA per batch, as in `rank_bench.py`.

**Results** (loss against CA, dB):

| | \|B\| = 125 | bench | \|B\| = 126 | bench |
|---|---|---|---|---|
| rho = 41 | 0.01418 +/- 0.00007 | 0.0141 | 0.01407 | 0.0142 |
| rho = 63 | 0.01157 +/- 0.00007 | 0.0115 | 0.01144 | 0.0114 |
| argmin | 64 | 62 | 63 | 64 |
| ranks within 2 SE of the minimum | 59 to 69 | 58 to 67 | 59 to 68 | |
| quadrature ranks 20, 41, 46, 62, 63, 94 | 6 of 6 within 2 SE | | 6 of 6 | |
| clean ranks outside 2 SE of the bench | 1 of 125 (rank 1, 0.0044 dB) | | 1 of 126 (rank 1, 0.0035 dB) | |

**The +20 dB contaminated columns.** With k = 1, 3, 10, 30 and 54 bins at +20 dB and the clean threshold, every rank the bench solved from its table matches it to within 0.0046 dB.

The letter of the criterion ("within 2 SE at every rank") is not met:
- 48 of 527 contaminated rank checks at |B| = 125 fall outside 2 SE;
- 12 of 532 at |B| = 126.

The misses come in runs of neighbouring ranks within one column. Both benches use common random numbers across ranks and nest the columns (k = 3 is a subset of k = 10), so a column shifts as one realisation. At |B| = 125 the k = 30 column sits at a mean -0.87 SE and the k = 54 column at -0.75 SE.

The same code at |B| = 126 misses at 2.3 percent, with the same maximum deviation. I read this as a pass in substance: the deviations are about 1/200 of the effects below.

### 2b. Mode R against mode I on ideal data: passes, with a known offset

Source: `rank/check_moder_ideal.py` and `data/check_moder_ideal.json`. There were 200,000 ideal frames, and the designated bins were drawn i.i.d., so both modes answer the same question.

- **Shape agrees.** The rho = 63 minus rho = 41 difference is 0.001 dB in mode R and 0.003 dB in mode I.
- **Offset.** Mode R reads 0.015 to 0.054 dB lower at every rank and at both Pfa values.
  - The cause is that the conditional-injection formula of DESIGN.md holds the reference denominator (S1 + S2) fixed for the tone term.
  - A tone is really divided by the same random denominator as the noise, and dropping that makes T' about 4 percent narrower than the noncentral F. This was checked by simulation: the 10 to 90 percent width is 0.0754 against 0.0785.
  - The offset is common to all ranks, so it cancels in L(rho). It is also about 1/100 of the disagreement in 2c. A version that is exact given S0, S1 and S2 is possible, because `fine_power_u64` holds all three.

### 2c. Mode I against mode R on real pilot-free candidates: fails

Sources: `rank/inject.py`, `rank/moder.py` and `rank/summarize_controls.py`, written to `data/controls_summary.json`.

**Setup.**
- Units: channel 37 at the 23 geometries (all frames), plus the near-threshold frames of 19, 26, 27 and 32.
- The two modes use paired calendar-month bootstraps with the same draws.
- Replicates: 200, not the release's 1,000. On this machine the 54 jobs ran at about 1/13 of single-job speed, so 1,000 replicates would have taken more than 10 hours. For a pass/fail test at 2 SE, 200 replicates are ample.
- L(rho) is each unit's loss against its own best rank.
- A rank agrees when |L_I - L_R| <= 2 sqrt(SE_I^2 + SE_R^2).

**Channel 37, over the 23 geometries** (ranges of L in dB; the best rank is where L = 0):

| half, Pfa | best rank I | best rank R | L(41) I | L(41) R | L(63) I | L(63) R | ranks agreeing |
|---|---|---|---|---|---|---|---|
| cal, 1e-3 | 72 to 75 | 123 to 125 | 3.05 to 3.15 | 5.87 to 7.34 | 0.12 to 0.15 | 4.48 to 6.16 | 14 to 29 % |
| eval, 1e-3 | 73 to 76 | 125 | 4.00 to 4.08 | 7.70 to 9.17 | 0.66 to 0.81 | 6.57 to 7.90 | 19 to 21 % |
| cal, 1e-2 | 70 to 73 | 117 to 121 | 0.24 to 0.25 | 0.94 to 1.20 | 0.02 to 0.03 | 0.54 to 0.81 | 16 to 35 % |
| eval, 1e-2 | 71 to 72 | 120 to 124 | 3.00 to 3.08 | 5.28 to 6.07 | 0.11 to 0.14 | 3.81 to 4.71 | 53 to 60 % |

On geometry 21 (cal, 1e-3), the SE at rho = 63 is 0.05 dB in mode I and 0.93 dB in mode R.

**Carrier-off channels** (cal, 1e-3; the eval rows are in the JSON):

| ch | best rank I / R | L(41) I / R | L(63) I / R | ranks agreeing |
|---:|---|---|---|---|
| 19 | 72 / 85 | 3.78 / 1.16 | 0.29 / 0.01 | 47 % |
| 26 | 73 / 73 | 2.55 / 0.52 | 0.23 / 0.02 | 52 % |
| 27 | 68 / 63 | 4.20 / 1.66 | 0.08 / 0.00 | 78 % |
| 32 | 73 / 66 | 2.18 / 1.13 | 0.36 / 0.01 | 55 % |

The best ranks are close here, but the loss curves are not. Mode R's absolute Pd = 0.9 SNR on these frames is near -4.8 dB, against -8.5 dB in mode I.

The carrier-off designated bins are not a null. Mode I's Pfa = 1e-3 threshold at rho = 63 is exceeded by 17 to 24 percent of the near-threshold frames on 19, 26 and 32, and by 67 to 79 percent on 27. Their real Z(63) runs up to the screen's own cut: the 0.999 point on channel 19 is 1.295, against w_cut = 1.2962. So the carrier-off channels cannot serve as the pilot-free cross-check that DESIGN.md assigned them. Only channel 37 can.

### 2d. The mechanism

The real bulks carry a frame-level dispersion that the ideal law lacks:
- the median per-frame robust spread of the bulk is 0.029 on channel 37, 0.032 on 21 and 0.033 on 19, against the ideal 0.027;
- the upper 1 percent of frames sit at 0.087 to 0.10, 3.2 to 3.8 times the ideal spread.

On channel 37 the designated bins widen with the frame. Mode I cannot represent this, because it draws the designated bins from the ideal law whatever the frame's spread. The two modes therefore see different false alarms:

- **Mode I's false alarms** come only from frames whose low-order bulk values are depressed. So low ranks are expensive, and the loss falls steeply until about rho = 70.
- **Mode R's false alarms** come from wide frames, where max_D T is large. Only a high rank rises with the frame's spread and normalises it. So mode R keeps improving up to rho = 125.

This is the "frame-level common effect that mode I cannot represent" named in DESIGN.md section 4. Here it is a common spread, not a common scale.

Mode R's preference for the top ranks cannot be carried over to the archive channels either. A rank near |B| is exactly the one the bench breaks with a single upward-contaminated bulk bin (`rank_bench.json`, k = 1 column). The objective channels 15, 20, 28, 33 and 34 carry such bins.

## 3. Channel 37's own procedure (section 4, run in both modes)

J is the median over the 23 geometries, and W is the worst geometry. The tie set is the ranks with J(rho) - J(rho*) <= 2 paired SE. The budget check compares W_eval(rho*) with 0.9 dB.

| mode, Pfa | rho* (cal) | tie set | held-out argmin | rho* in held-out tie set | W_eval(rho*) | budget | rho = 41: J_cal / W_eval | rho = 63: J_cal / W_eval |
|---|---|---|---|---|---|---|---|---|
| I, 1e-3 | 75 | 68 to 76 | 74 | yes | 0.004 dB | pass | 3.11 / 4.08 dB | 0.13 / 0.81 dB |
| I, 1e-2 | 72 | 68 to 75 | 72 | yes | 0.001 dB | pass | 0.24 / 3.08 dB | 0.02 / 0.14 dB |
| R, 1e-3 | 125 | 124 to 125 | 125 | yes | 0.000 dB | pass | 6.87 / 9.17 dB | 5.47 / 7.90 dB |
| R, 1e-2 | 118 | 116 to 121 | 122 | yes | 0.26 dB | pass | 1.09 / 6.07 dB | 0.69 / 4.71 dB |

**Absolute loss against clean ideal CA** (A = s* + 8.631 dB) at the channel-37 optimum, median over geometries:
- mode I: 0.04 dB on calibration and 0.10 dB held out;
- mode R: 1.08 dB and 2.04 dB;
- adding the 0.91 dB straddle loss gives 1.99 dB and 2.95 dB for mode R.

Mode R's absolute values carry the common offset of 2b, about 0.02 to 0.05 dB in the favourable direction.

The held-out half is the harder one, as flagged, and it is harder in both modes. Mode I's loss at 63 grows from 0.13 to 0.81 dB, and mode R's from 5.5 to 7.9 dB.

## 4. What was not run, and what would unblock it

**Not run:**
- the objective run on the 16 channels;
- the tie-break and the held-out confirmation on those channels;
- the strong-pilot miss rates;
- the sensitivities (w_lo, the 1,000-frame minimum, Pfa 1e-2, the straddle offset, minimax);
- the two requested figures: (a) loss against rank per channel with the median, the chosen rho and the budget line; (b) held out against calibration.

The code for all of these is in place: `inject.py NAME w|wlo|ideal` writes both Pfa values, both offsets and the miss counts. They are held back because the control that licenses mode I failed.

**What would unblock it** (a design decision, not something to settle here):

1. **Give mode I a frame-dependent designated-bin law.** For example, scale each frame's designated-bin dispersion by that frame's measured bulk spread. Then re-run 2c on channel 37. This is the direct fix for the mechanism in 2d.
2. **Use mode R on channel 37 as the only real null**, with the exact S0/S1/S2 tone model of 2b. Then decide how the top-rank preference is to be traded against the upward-contamination breakdown that the objective channels need guarded. That trade is a margin again, which is what this study was meant to remove.
3. **Whatever route is taken**, rho = 41 does not survive the ch06 budget on the one real null the archive has. In no variant does 41 come within the 0.9 dB straddle loss on both halves. rho = 63 survives only under mode I (0.13 / 0.81 dB at Pfa 1e-3; 0.02 / 0.14 dB at 1e-2).

## Files

**Code (`rank/`):**
- `common.py` and `rank_bench.py`: byte copies from fixed-rank-decisions/rank.
- `extract.py`
- `core.py`: mode I tables and solver.
- `inject.py`
- `moder.py`
- `controls.py`
- `check_moder_ideal.py`
- `summarize_controls.py`
- `plot_controls.py`

**Style:** `figstyle/style.py` is a byte copy of the dissertation style module (sha256 7ad4b752...).

**Data (`data/`):**
- `frames/`: per-frame inputs and their json.
- `tables/`: the Pd tables.
- `inject/`, `moder/`: per-unit outputs.
- `controls_bench_125.json`, `controls_bench_126.json`
- `check_moder_ideal.json`
- `controls_summary.json`
- Byte copies: `rank_bench.json`, `bench_quad.json`, `bulk/`, `campaign_last.json`.

**Figure:** `figs/rank_injection_controls.pdf` and `.png` (6.23 in wide, 448.5 pt):
- (a) the ideal control against the bench;
- (b) channel 37, mode I against mode R, with the 0.9 dB budget line;
- (c) the real false-alarm rate of mode I's threshold against the nominal 1e-3.

**Logs:** `logs/`.
