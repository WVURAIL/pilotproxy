# Independent check: rank by injection

30 September 2026. I built none of the study. I re-ran its checks with my own code in `verify/`, which imports nothing from `rank/`:
- `v_ideal.py`: exact quadrature;
- `v_real.py`: both injection modes, read from the raw `fine_power_u64`;
- `v_counts.py`: the screen's sensitivity.

Outputs are `verify/*.json` and `verify/*.log`. Everything ran with `python -B`, `PYTHONDONTWRITEBYTECODE=1` and one thread. Nothing outside this folder was written.

## Verdict in one paragraph

The study chose no rank, so there is no rho*, no held-out confirmation and no budget check on the 16 objective channels to confirm. What it does claim, I reproduce independently:
- the stop condition (the two injection modes disagree on channel 37);
- the size of the disagreement and its cause;
- the channel 37 numbers;
- the carrier-off result;
- the ideal control.

The stop is legitimate. The disagreement is real, at 4.4 combined SE at rho = 63, and it is not a code artefact. The question "why a quarter?" is therefore still open. The one robust finding points against 41, not for it: on the only real null, rho = 41 loses more than ch06's accepted straddle loss ("doubling the hypothesis grid bounds the worst-case straddle loss near $0.9$~dB") on at least one half in all four variants. The quote is verbatim from `ch06_implementation.tex` line 32.

## 1. Ideal control (exact quadrature, `verify/v_ideal.json`)

**Method.**
- **Rank detector:** the exact order-statistic density on a 90,001-point grid.
- **CA:** a 2e6-frame Monte Carlo reference mean. Its offset from `bench_quad.json` is 7e-5 dB at |B| = 125 and 4e-5 dB at |B| = 126, and every loss I quote carries that offset.
- **Contaminated columns:** these are exact. A +20 dB bin always takes a top rank, so rank rho <= n - k is the rho-th order statistic of the n - k clean bins, tested at the clean threshold.

**Results.**
- **Rank detector SNR.** My Pd = 0.9 SNR equals `bench_quad.json` at every quadrature rank (20, 41, 46, 62, 63, 94) to better than 1e-6 dB. All of the loss differences are the CA offset.
- **Clean curve.**

  | | \|B\| = 125 | \|B\| = 126 |
  |---|---|---|
  | L(41) | 0.0141 dB | 0.0141 dB |
  | L(63) | 0.0115 dB | 0.0114 dB |
  | exact argmin | 64 | 64 |
  | ranks within 2e-4 dB of the minimum | 58 to 70 | 58 to 70 |

  The bench's "62 / 64" argmin is Monte Carlo placement on a bottom that is flat to 1e-4 dB. The study's 64 / 63 is equally consistent with it.
- **Contaminated columns.** The study's misses (48 of 527 checks at |B| = 125) come from the bench's own Monte Carlo, not the new code:
  - the exact values fall outside the bench's 2 SE at |B| = 125 in 14 of the 55 checks I made (at most 0.0042 dB, mostly at rank 1);
  - at |B| = 126 they fall outside in 0 of 55;
  - the study's control gets rank 1 right (0.4970 dB against my exact 0.4966) where the bench gives 0.4925.

  "Pass in substance" is therefore confirmed, not just asserted.

**Screen constants.**
- **s999:** exactly -7.1604 dB (design -7.161).
- **w_cut:** exactly 1.29557. The design used 1.29623, a 200,000-frame Monte Carlo quantile, 6.6e-4 high. No objective channel changes.
- **Ideal null's 0.999 point of w:** equals eta(63) = 1.10067. For |B| = 125, w is Z(63).

## 2. Injection re-run on two channels (`verify/v_real_*.json`)

**Data checks.**
- T = 2 S0 / (S1 + S2) is formed from the raw products: the channel 37 control 491.npz, and 767.npz for channel 19 at the release rows.
- The geometry was rebuilt from the ch06 rule: D is the anchor +/- 2, and the bulk is the even bins minus D and the census. It matches the study's `bulk_bins` and `designated`.
- Sorted bulk, D values, the calibration split and the near-threshold counts all match the study's arrays.

**Modes as I implemented them.**
- **Mode I:** eta by root-solve of 1 - mean F(eta t)^5.
- **Mode R:** eta is the empirical (1 - Pfa) quantile of the real Z. The tone is averaged exactly over u through the Beta CDF, not by Monte Carlo.

**Channel 37, geometry 21 (125 bins) and geometry 28 (126 bins).** L in dB against the unit's own best rank. For geometry 28 I allowed rank 126, which the study's candidate set (1 to 125) excludes, so its mode R best reads 126.

| half, Pfa | best I | best R | L(41) I | L(41) R | L(63) I | L(63) R | real FA of eta_I, 41 / 63 | study range (23 geometries) |
|---|---|---|---|---|---|---|---|---|
| cal, 1e-3 | 74 / 74 | 124 / 126 | 3.09 / 3.12 | 6.21 / 7.35 | 0.12 / 0.14 | 4.67 / 5.97 | 0.46 / 1.59 %, 0.42 / 1.62 % | I 3.05 to 3.15, 0.12 to 0.15; R 5.87 to 7.34, 4.48 to 6.16; FA 0.36 to 0.48, 1.5 to 1.8 % |
| eval, 1e-3 | 75 / 75 | 125 / 126 | 4.00 / 4.08 | 8.88 / 8.48 | 0.70 / 0.76 | 7.83 / 7.50 | 1.55 / 3.13 %, 1.56 / 3.07 % | I 4.00 to 4.08, 0.66 to 0.81; R 7.70 to 9.17, 6.57 to 7.90 |
| cal, 1e-2 | 72 / 73 | 120 / 121 | 0.24 / 0.25 | 1.17 / 1.12 | 0.02 / 0.03 | 0.71 / 0.75 | | I 0.24 to 0.25; R 0.94 to 1.20, 0.54 to 0.81 |
| eval, 1e-2 | 71 / 71 | 122 / 121 | 3.00 / 3.07 | 5.51 / 5.47 | 0.12 / 0.14 | 3.95 / 3.93 | | I 3.00 to 3.08, 0.11 to 0.14; R 5.28 to 6.07, 3.81 to 4.71 |

**Significance** (geometry 21, calibration half, Pfa 1e-3; calendar-month block bootstrap, B = 40, paired against rank 125):
- SE[s(63) - s(125)] is 0.041 dB in mode I and 1.03 dB in mode R. The mode gap at 63 is 4.55 dB, which is 4.4 combined SE.
- At 41, the SEs are 0.39 dB and 1.19 dB. The gap there is 3.12 dB, or 2.5 SE.
- The disagreement is well outside 2 SE, as the study reports (its 2 SE is about 1.9 dB, with B = 200).

**Mechanism** (confirmed directly, not inferred).
- **No line in D.** At both geometries, each D position's mean T lies within 0.9993 to 1.0003 of the bulk mean.
- **Frame-level spread.** The per-frame robust spread of the bulk is 0.029 at the median, against the ideal 0.0271. Its 99th percentile is 0.074 on the calibration half and 0.32 on the held-out half.
- **D widens with the frame.**
  - In the widest 1 percent of frames, D's rms deviation about the bulk median is 0.16 (cal) and 0.38 (eval), against 0.029 and 0.033 in the other frames.
  - D rms / bulk spread = 0.96 to 1.03 in both groups.
- **Consequence.** The null is a per-frame location-scale family shared by D and B, which mode I, with its ideal D, cannot represent. The held-out half's tail is 4 times wider, which is why every held-out loss is larger.
- **Comparing with the study's figures.** The study's "3.2 to 3.8 times the ideal" for the upper 1 percent is a different estimator, pooled over both halves. My per-half figures bracket it.

**Channel 19, carrier-off, near-threshold frames.**

| half, Pfa 1e-3 | best I / R | L(41) I / R | L(63) I / R | real FA of eta_I(63) | absolute s90 at best, I / R |
|---|---|---|---|---|---|
| cal | 72 / 60 | 3.78 / 1.16 | 0.29 / 0.01 | 23.8 % | -8.54 / -4.86 dB |
| eval | 69 / 64 | 2.04 / 0.56 | 0.03 / 0.02 | 17.3 % | -8.53 / -4.87 dB |

- This matches the study: 72 / 85, 3.78 / 1.16 and 0.29 / 0.01, FA 17 to 24 percent, s90 near -4.8 dB against -8.5 dB.
- Mode R's best rank is not stable (60 against 85) because its curve stays within 0.025 dB of its minimum from 60 to 90 (85 reads 0.003 dB in my run).
- **Why channel 19 is not a null.** It is not only the screen. D carries a residual line at the anchor: the mean T at anchor and anchor + 1 is 1.026 and 1.021 times the bulk mean (cal), and 1.017 and 1.014 (eval). D's rms is 1.3 to 1.45 times the bulk spread in ordinary frames.
- The study's conclusion that only channel 37 can serve as the mode R null stands.

## 3. Frame selection

- **Counts reproduced.** From cfar-choice's original per-frame arrays, the near-threshold counts reproduce `design_counts.json` on all 23 channels at the as-run cut. The 16-channel objective set follows from the 500-per-half rule.
- **Not tuned post hoc.** The cut was fixed from the ideal law before any injection ran. It does not depend on real data or on the outcome.
- **But not free of convention either.** "Detection saturates at Pd = 0.999" is a chosen level, one step removed from the 3/4. Built the same way at Pd = 0.99, the cut is 1.2447, and the objective set loses 15, 20 and 28:
  - these are the same three that the planned 1,000-frame sensitivity removes;
  - they are three of the five channels the design says should decide the rank.

  At Pd = 0.9999 (cut 1.3386) the set is unchanged. The dependence is therefore covered by an existing sensitivity. It should still be stated as a dependence on 0.999, not as "no fraction is chosen".
- **One wording error.** The task and design say that "below the cut, a frame is pilot-free or its pilot is weak enough to be missed". That is not what the cut does. A pilot at s999 is detected 99.9 percent of the time and passes the screen 99.9 percent of the time. Near-threshold frames include pilots up to the saturation level.
  - This is harmless for mode I, which discards D.
  - It made the design's plan to use screened carrier-off frames as mode R nulls circular, since it screens on the very statistic whose tail it calibrates.
- **Selection-bias argument.** It is correct, and on channel 37 it does not apply (no screen).

## 4. Figure (`figs/rank_injection_controls.pdf` / `.png`)

- **Specification checks:**
  - pdfinfo width 448.5 pt (<= 457.2);
  - PNG 150 dpi;
  - one y-axis per panel;
  - no em dashes in the PDF text;
  - the style module is a byte copy of the dissertation's (sha256 7ad4b752...).
- **Content matches the numbers.**
  - (a): the code path sits on the bench.
  - (b): mode I reaches 0 near 72 to 75, and mode R's median is about 5.5 dB at 63.
  - (c): channel 37 rises from about 2e-3 at rank 1 to 1.6e-2 at 63, and the carrier-off channels sit at 0.2 to 0.8.
- **Cosmetic.** The annotation block in (b) sits over the thin mode R curves near rho 95 to 108. It is still legible, and no text is clipped.
- **Not produced.** The objective figures (a) and (b) do not exist, because the objective was not run.

## 5. What remains before a rank can be chosen

- **Route 1 (the study's first route) is supported by the diagnostic above.** On channel 37, D's rms tracks the frame's bulk spread at a ratio of 1.0. This holds both in the widest 1 percent of frames and in the rest. A mode I that draws D from the ideal law scaled to each frame's measured bulk spread is therefore the model the data support. The channel 37 agreement check should be re-run with it, and then the objective run.
- **Why this matters for the choice.** The mechanism means the order-statistic ratio normalises the level but not the width. Any rank chosen by a loss calibrated at Pfa = 1e-3 is driven by roughly the widest 0.1 percent of frames. That is 20 frames per half on channel 37, and they are concentrated after 2024. The write-up of the eventual choice should say so.

CHECK: BLOCKED: no fixed rank was chosen. The gating control (mode I against mode R on channel 37) genuinely fails. I reproduced it with independent code: best rank 74 to 75 against 124 to 126, and at rho = 63 a loss of 0.12 dB against 4.7 to 6.0 dB, which is 4.4 combined SE. The cause is a frame-level spread shared by D and B that mode I cannot represent. So the 16-channel objective run, rho*, the held-out confirmation, the budget check, the sensitivities and figures (a) and (b) do not exist to confirm. The carrier-off channels are not a usable null, because channel 19 carries a residual line at the anchor. The screen's "no fraction is chosen" rests on the Pd = 0.999 convention, which moves 15, 20 and 28 out of the objective at Pd = 0.99. The one verified result is that rho = 41 exceeds the 0.9 dB ch06 straddle loss on channel 37 in every variant, so the quarter is not yet justified by injection.
