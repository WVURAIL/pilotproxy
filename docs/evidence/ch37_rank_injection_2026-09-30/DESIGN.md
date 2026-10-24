# Choosing the fixed rank by injection on real background bins

30 September 2026. Design only; nothing here has been run beyond the frame counts in section 1 (`design/design_counts.py`, `design/design_counts.json`, `logs/design_counts.log`). Release of record: `results/archive_no_split_2026-09-24_r2`. Code: the cfar-choice-2026-09-29 clones (RFIsher a281112, pilot-proxy bdfcdce, RadioFisher 12a1616), used read-only.

## Why this study

`fixed-rank-decisions-2026-09-29/rank` fixed rho = 41 with the rule (rho + m*) / (|B| + 1) <= 3/4 on every channel. The 3/4 is a convention: RANK.md says so, and its section 3 shows the pick moving from 9 to 62 as the ceiling runs from 0.5 to 1. The question "why a quarter?" has no answer inside that rule.

This design replaces the margin with a measured quantity: the detection loss each rank incurs on the channel's own background, obtained by injecting a pilot into the designated set of real frames. A rank is chosen because it loses least on real bulks, and the only budget it is held to is one the thesis already accepts. No margin fraction enters.

**Definitions (ch06).**
- D = {(f_a + k) mod L_F : |k| <= 2}, the padded main lobe around the measured anchor.
- B is the usable bulk: 125 bins (even anchor) or 126 (odd anchor).
- T[f] = 2 S0[f] / (S1[f] + S2[f]), the deployed statistic (`fine_power_ratio`).
- Z(rho) = max_D T / T_(rho), with T_(rho) the rho-th smallest bulk value.
- Reject when Z(rho) > eta.

**Budget.** ch06's remark on the untapered design: "doubling the hypothesis grid bounds the worst-case straddle loss near $0.9$~dB". That is the loss of a tone half a padded bin off grid: sinc^2(1/4) = 0.811, or -0.91 dB. The thesis accepts it as the price of not tapering.

## 1. Which frames

### 1a. The screen

Only the bulk of a real frame is used, so the screen asks one question: was a pilot near threshold, or absent, when this bulk was recorded? The statistic is

    w = max_D T / median_B T.

The cut comes from the ideal law, not from a chosen fraction. It is a frame's D content at the point where detection saturates:

- **Ideal law:** i.i.d. F(4096, 8192) bins, |B| = 125, D gains (0, g1, 1, g1, 0) with g1 = 0.4053, as in `rank_bench.py`.
- **Pd = 0.9 at Pfa = 1e-3:** clean CA reaches it at -8.631 dB per bin (the bench's value, reproduced to 1e-4 dB).
- **Pd = 0.999:** the clean median-rank detector reaches it at s999 = -7.161 dB.
- **The cut:** w_cut = 1.2962, the upper 0.999 point of w with a pilot at s999. For comparison, the ideal null's 0.999 point is 1.1009 and its median is 1.031.

**Frame classes:**
- **near-threshold:** w <= w_cut. The frame is pilot-free, or its pilot is weak enough that the detector could miss it.
- **mid:** w_cut < w <= 10.
- **on:** w > 10 (cfar-choice's pilot-on definition).

A mid or on frame's pilot lies above the level where detection saturates, so its bulk matters only if the bulk contamination scales with the pilot. That is tested in 1c.

**Selection bias.** The screen's denominator is the median of the bulk being scored. A frame whose median is inflated reads a lower w and is kept more readily, so the selected bulks lean toward contamination. That penalises the high ranks more, so it is conservative against them.

Sensitivity check: reselect with w_lo = max_D T / m_lo, where m_lo = P25 + 0.6745 (P25 - P10) / 0.6071 is RANK.md's lower-quartile location (robust to three quarters of B contaminated). Report any move in rho*.

### 1b. Counts

All 23 channels were counted (current era, the release's calendar split). Source: cfar-choice's per-frame bulk arrays.

- **cal / eval:** calibration and evaluation blocks.
- **m0:** the median of median_B T over near-threshold frames.
- **r>1.05:** the share of frames whose bulk mean exceeds its median by more than 5 percent (cfar-choice's inflation measure).
- **carrier-off:** the date from the transmitter record. On these channels the current era is the post-transition era.

| ch | state | \|B\| | frames (cal / eval) | near (cal / eval) | mid | on | near share | m0 | r>1.05, near | r>1.05, on | carrier-off |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 14 | low | 125 | 17,092 / 17,091 | 11,619 / 9,728 | 8,911 | 3,925 | 0.624 | 1.006 | 0.122 | 0.919 | |
| 15 | high | 126 | 13,355 / 13,348 | 589 / 6,718 | 16,426 | 2,970 | 0.274 | 1.008 | 0.997 | 0.752 | |
| 16 | low | 126 | 18,703 / 18,703 | 13,743 / 11,715 | 10,473 | 1,475 | 0.681 | 1.003 | 0.058 | 0.868 | |
| 17 | high | 125 | 12,992 / 12,985 | 157 / 101 | 2,714 | 23,005 | 0.010 | 1.007 | 0.407 | 1.000 | |
| 18 | low | 125 | 10,107 / 10,105 | 1,594 / 1,724 | 13,961 | 2,933 | 0.164 | 1.013 | 0.183 | 0.824 | |
| 19 | low | 125 | 6,453 / 6,447 | 5,009 / 5,097 | 2,538 | 256 | 0.783 | 1.009 | 0.043 | 0.426 | 2024-11 |
| 20 | low | 126 | 12,018 / 11,979 | 556 / 565 | 14,753 | 8,123 | 0.047 | 0.995 | 0.854 | 0.647 | |
| 21 | low | 125 | 19,813 / 19,804 | 18,006 / 17,062 | 3,904 | 645 | 0.885 | 1.006 | 0.080 | 0.901 | |
| 22 | high | 125 | 19,491 / 19,481 | 15 / 146 | 52 | 38,759 | 0.004 | 1.014 | 0.516 | 1.000 | |
| 23 | low | 125 | 19,361 / 19,355 | 17,568 / 16,577 | 3,947 | 624 | 0.882 | 0.999 | 0.193 | 0.793 | |
| 24 | high | 125 | 5,030 / 5,028 | 0 / 0 | 0 | 10,058 | 0 | - | - | 1.000 | |
| 25 | low | 126 | 18,518 / 18,517 | 16,226 / 14,676 | 5,332 | 801 | 0.834 | 1.002 | 0.124 | 0.874 | |
| 26 | low | 125 | 8,991 / 8,989 | 7,695 / 5,863 | 3,949 | 473 | 0.754 | 1.012 | 0.046 | 0.767 | 2023-04 |
| 27 | low | 125 | 13,728 / 13,725 | 3,400 / 2,840 | 18,476 | 2,737 | 0.227 | 0.996 | 0.103 | 0.737 | 2020 |
| 28 | low | 126 | 16,960 / 16,957 | 772 / 914 | 17,213 | 15,018 | 0.050 | 1.013 | 0.903 | 0.293 | |
| 29 | low | 125 | 20,218 / 20,211 | 18,302 / 18,784 | 2,865 | 478 | 0.917 | 1.002 | 0.038 | 0.910 | |
| 30 | high | 126 | 4,592 / 4,574 | 108 / 0 | 206 | 8,852 | 0.012 | 0.959 | 0.870 | 1.000 | |
| 31 | high | 125 | 10,413 / 10,402 | 90 / 23 | 198 | 20,504 | 0.005 | 1.008 | 0.000 | 1.000 | |
| 32 | low | 126 | 10,513 / 10,511 | 7,924 / 7,526 | 5,181 | 393 | 0.735 | 1.000 | 0.041 | 0.959 | 2020 |
| 33 | low | 125 | 13,799 / 13,796 | 6,470 / 5,762 | 14,212 | 1,151 | 0.443 | 1.022 | 0.457 | 0.858 | |
| 34 | low | 125 | 19,475 / 19,474 | 12,866 / 17,594 | 7,480 | 1,009 | 0.782 | 1.013 | 0.316 | 0.559 | |
| 35 | high | 126 | 14,885 / 14,882 | 421 / 25 | 151 | 29,170 | 0.015 | 0.996 | 0.043 | 1.000 | |
| 36 | ambig. | 125 | 19,614 / 19,603 | 82 / 11 | 2,319 | 36,805 | 0.002 | 0.996 | 0.280 | 0.486 | |
| 37 | control | 125 or 126 | 40,477 (2018-12 to 2026-08, calendar median 2024-08) | all (no allocation) | | | | | | | none |

**Objective set.** A channel enters the rank objective when both halves hold at least 500 near-threshold frames. The count is set by precision, not by tuning:

- The threshold in mode I (section 2) is semi-analytic. It does not need 1/Pfa frames, only enough references to average over.
- The bench's batch spread (per-rank SE 1e-4 dB over 20 batches of 20,000) scales to about 0.003 dB of loss SE at 500 frames. The bootstrap in section 3 measures the actual SE, and a channel whose SE exceeds 0.02 dB at any candidate rank is reported but left out.

Sixteen channels enter: 14, 15, 16, 18, 19, 20, 21, 23, 25, 26, 27, 28, 29, 32, 33 and 34. At 1,000 per half, 15, 20 and 28 would drop out; this is run as a sensitivity. The other seven (17, 22, 24, 30, 31, 35, 36) have 0 to 421 near-threshold frames in a half. They are reported as diagnostics, pooled over both halves where counts allow, and tested by the strong-pilot check of 1c.

**Where the pilot-independent contamination is.** The near-threshold bulks on 15, 20, 28, 33 and 34 are inflated in 32 to 99.7 percent of frames:
- odd-anchor persistent bins on 20 and 28;
- the pedestal on 15;
- the two-sided broadening on 33.

This contamination does not go away when the pilot is weak, so these are the channels that should decide the rank. On the quiet channels (16, 19, 21, 26, 29, 32) the near-threshold share inflated is 4 to 8 percent.

### 1c. Carrier-skirt contamination is a separate class

On the proxy-high channels, near-threshold frames are rare (0 to 1.5 percent, except 15 at 27 percent). There the bulk is dominated by contamination that tracks the carrier.

**Scaling measure.** On on-frames, per channel,

    eps = median over frames of (median_B T - m0) / (max_D T - m0).

It runs from 1.1e-4 to 7.4e-3 across all channels: 1.7e-3 on 35, 2.5e-3 on 31, 3.5e-3 on 30, 7.4e-3 on 17.

**Effect near threshold.** A pilot near threshold has a centre-bin excess of 10^(-0.863) = 0.137. The skirt scaled to that pilot lifts the median by at most 7.4e-3 x 0.137 = 1.0e-3. The ideal median's own standard deviation is 1.2533 x 0.0191 / sqrt(125) = 2.1e-3. When the pilot is strong, a skirt 21 to 40 dB below it lowers Z by the same small factor, so Z stays far above eta: on 35 the median Z(63) of carrier-in-D frames is 314.

**Where the pedestal is not skirt.** The fade frames show which proxy-high contamination is not skirt:
- 31 and 35 are clean when the pilot fades (0 and 4 percent inflated);
- 15, 22 and 30 are not (100, 52 and 87 percent). Their pedestal persists without the carrier.

15 therefore enters the objective. 22 and 30 lack the frames to enter, and their near-threshold bulks are reported as a pooled diagnostic.

**What the study recomputes** per rank:
- eps(rho) with T_(rho) in place of the median;
- the **strong-pilot miss rate**: the share of mid and on frames whose real Z(rho) falls below the channel's calibrated eta(rho).

The claim that skirt frames do not constrain the rank is accepted if the miss rate is zero within its binomial bound at every rank in the tie set of section 3, on every channel. If it is not, the offending ranks are removed from the candidate set, and the report says so.

## 2. Injection

### Two modes

**Mode I (primary, all selected frames): real bulk, ideal D.**
- The frame's real sorted bulk supplies T_(rho) for every rho = 1..|B|.
- The five D bins follow the kernel's tone model on the padded main lobe, as ch06's synthetic validation and the rank bench use:
  - T_f ~ noncentral F(4096, 8192, 4096 s g_f), independent across D and independent of the bulk;
  - the tone sits on the anchor, with gains (0, g1, 1, g1, 0), g1 = D_128(1/2)^2 = 0.4053.
- The real bulk is untouched.
- Because D is independent of the reference, both error rates are exact averages over the real references, with no Monte Carlo in D:

      Pfa_c(eta; rho) = 1 - mean_frames prod_D F(eta T_(rho); 4096, 8192)
      Pd_c(s; eta, rho) = 1 - mean_frames prod_D F_nc(eta T_(rho); 4096, 8192, 4096 s g_f)

- **Why real D can be discarded here.** A near-threshold frame's own D may hold a weak real pilot. Discarding it keeps the null a null.
- **Scale.** A channel-constant scale of D (m0 runs from 0.959 to 1.022) cancels exactly against the threshold calibrated on the same frames.
- **What mode I misses.** A frame-to-frame scale common to D and B. Mode R measures that where real pilot-free D exists.

**Mode R (cross-check, frames with genuinely pilot-free D): real bulk, real D, conditional tone injection.**

Frames:
- channel 37 at each of the 23 channel geometries (anchor, D and bulk mask copied from the channel, as `decision/references.py` builds them);
- the near-threshold frames of the carrier-off channels 19, 26, 27 and 32.

Given the real D-bin value T_f, a coherent tone of per-bin SNR s is added as

    T_f' = T_f + c s g_f + 2 u sqrt(c s g_f T_f),

- c = m0, the channel's clean level;
- u = 2 V - 1, with V ~ Beta((k - 1)/2, (k - 1)/2) and k = 4096.

u is the cosine between the fixed tone vector and a uniformly oriented noise vector on the k-sphere. This is exact for isotropic Gaussian noise in the bin conditional on its observed power. The reference profiles are left unchanged, as in the bench. Their leakage from a tone on the anchor is at most 4.5 percent in power (ch04), and is out of scope for both modes.

In mode R the null threshold is the empirical upper Pfa point of the real Z(rho) on the same frames, and Pd is the injected share above it. Q16 rounding is ignored (display floats). The kernel decision differs from the float one only within 2^-16 of eta.

### Thresholds, Pd and loss

- **Thresholds.** eta_c(rho) at Pfa = 1e-3 per channel and rank, from the channel's selected frames without injection. The same is repeated at Pfa = 1e-2, which is statistically firmer, especially for mode R's empirical quantiles: 5,000 to 37,000 frames give 5 to 37 exceedances at 1e-3.
- **Pd at a fixed SNR.** Pd_c(rho) at s0 = -8.631 dB, the SNR where the clean ideal law gives Pd = 0.9 (clean CA, the bench's ruler).
- **Loss.** s_c(rho) is the SNR at which channel c's rank rho reaches Pd = 0.9, solved from the tabulated product of CDFs as in `rank_bench.py`. The x grid is extended to the largest eta T_(rho) on the archive, with direct brentq where the table cannot reach. Then:
  - per-channel loss against its best rank: L_c(rho) = s_c(rho) - min over rho' of s_c(rho');
  - absolute loss against clean ideal CA: A_c(rho) = s_c(rho) + 8.631 dB.
- **Offsets.** Primary: the tone on the anchor (delta = 0), the bench's model. Secondary: delta = 1/2 padded bin, the straddle worst case with gains sinc^2 at natural-bin offsets -1.25, -0.75, -0.25, 0.25 and 0.75 (0.032, 0.090, 0.811, 0.811, 0.090). It is reported to show that the rank choice does not interact with straddle.

## 3. Selection

**Split.** Each channel's calibration and evaluation blocks are the release's calendar split (`blocks.split_blocks`, with counts asserted against the ledger as `common.channel_setup` does). Channel 37 is split by the same function on its own frame times (its frame-time median is 2024-08).

**Objective, on the calibration half only:**

    J(rho)   = median over the objective channels of L_c(rho)
    W(rho)   = max over the objective channels of L_c(rho)
    rho*     = argmin J(rho)

- **Uncertainty.** Calendar-month block bootstrap within each half, with the release's replicates and seed (`analysis-plan.json` bootstrap). Resampling is common across ranks, so rank differences carry paired SEs.
- **Ties.** The tie set is every rank with J(rho) - J(rho*) <= 2 SE of that difference. Inside it, the pick is the rank with the least W. If ranks are still tied, the pick is the one with the least clean-bench cost (`rank_bench.json`), because it is the only remaining information.
- **Candidates.** rho = 1..125, absolute rank as the bundle's `cfar_rank` stores it. The same rho is used on 125- and 126-bin channels.

**Confirmation, on the held-out half.** Recompute L_c on the evaluation block with eta recalibrated there, and report:
- J_eval(rho*) and W_eval(rho*);
- the held-out argmin and its tie set;
- whether rho* lies in that tie set;
- per-channel L_c(rho*), calibration and held out.

Reference points on the same curves: rho = 41 and rho = 63.

**Budget check.**
- W_eval(rho*), the worst channel's held-out loss at the chosen rank, is compared with ch06's accepted straddle loss, near 0.9 dB. The rank passes if it is below.
- Also reported: the worst channel's absolute loss A_c(rho*) against clean ideal CA, and A_c(rho*) + 0.91 dB, the combined worst case with a tone half a padded bin off grid.

The rank then needs no margin of its own. It costs less, on the worst real bulk the archive holds, than a design convenience the thesis already accepts.

**Sensitivities, reported with the pick:**
- screen w_lo in place of w;
- objective-set minimum of 1,000 per half;
- Pfa = 1e-2;
- delta = 1/2;
- minimax (argmin W) in place of the median.

## 4. Controls

**Ideal i.i.d. control.** The mode I code path is run with each real bulk replaced by i.i.d. F(4096, 8192) draws:
- at each objective channel's own frame count and |B|;
- and at 400,000 frames, |B| = 125 and 126.

It must reproduce `rank_bench.json`'s clean curve within 2 SE at every rank, including:
- rho = 41 at 0.0141 / 0.0142 dB;
- rho = 63 at 0.0115 / 0.0114 dB;
- the argmin at 62 / 64, with its flat bottom over 58 to 67.

The quadrature values in `fixed-rank-decisions-2026-09-29/check/bench_quad.json` serve as the exact check. Two further requirements:
- With k = 1, 3, 10, 30 and 54 bulk bins replaced at +20 dB, it must reproduce the bench's contaminated columns within 2 SE.
- At the real frame counts, the control's own J curve measures how much of the real curve's shape is sampling. A real rho* inside the control's tie set is not distinguished from the clean optimum.

**Channel 37, a second real null.**
- 40,477 histogram-eligible frames, all current era (2018-12 to 2026-08).
- Placed at each of the 23 channel geometries, run through modes I and R.
- It gets its own rho*, J, W, held-out confirmation and budget check.
- Mode I and mode R must agree on L(rho) within their combined bootstrap SE. A disagreement is the frame-level common-scale effect that mode I cannot represent, and it is reported per geometry.
- **Drift caveat.** Channel 37's null widens after 2024: the share of frames above the ideal 1e-3 point at rho = 41 runs from 1.1 to 2.3 percent over 2019 to 2023, then 3.4, 3.9 and 7.0 percent in 2024, 2025 and 2026 (CHECK.md). The held-out half, which starts near 2024-08 if the split falls at the frame-time median, is the harder one, and the confirmation is read with that in mind.

**Carrier-off channels (19, 26, 27, 32).** Mode R on their near-threshold frames against mode I on the same frames: the same agreement requirement as channel 37.

**Scope.** Channel 37 and the carrier-off epochs are real nulls for the bulk, not verified transmitter-off nulls for any channel's current era. Every eta here is a rank-comparison device, not an eta_Pfa. The verdicts still come from the capture ruling.

## 5. Implementation plan

Everything below is written under `/home/djg/rail/output/rank-by-injection-2026-09-30`. Code runs with `python -B`, `PYTHONDONTWRITEBYTECODE=1` and one thread (the `fisher-fixes-2026-09-08/analysis-venv` interpreter, plots in `~/.venvs/dissertation`).

**Byte copies:**
- `rank/common.py` (release reader, era, split, anchor, bulk; clones path unchanged, read-only);
- `rank/rank_bench.py` (table and solver);
- cfar-choice's `data/bulk/*.npz` (frame rows, dmax, median, calibration flag);
- `figstyle/style.py` from `dissertation/figure_src/style.py` (sha256 7ad4b752...).

**Steps:**
1. `extract.py CH`: per channel from `fine_power_u64`, the sorted bulk T (float32, frames x |B|), the five D values, the frame time, the block and the w / w_lo classes. Rows and counts are asserted against the cfar-choice arrays and the table in 1b. Channel 37 reuses `decision/references.py`'s geometry loop on the verified product (sha256 e77d1d41...).
2. `inject.py`: modes I and R, thresholds at 1e-3 and 1e-2, Pd at s0, the s_c(rho) solve, and the bootstrap.
3. `select.py`: J, W, tie sets, rho*, held-out confirmation, budget check and sensitivities, written to `data/selection.json`.
4. `controls.py`: the ideal control and its match to `rank_bench.json` and `bench_quad.json`; the channel 37 and carrier-off agreement.
5. `plot.py`: the figures below.

**Figures** (Appendix C plate style, at most 6.35 in wide, pdfinfo width <= 457.2 pt, PNG at 150 dpi plus PDF, no dual y-axes, no clipped text):
- `rank_injection` on a shared rho axis:
  - (a) L_c(rho) per objective channel on the calibration half, with the median J and worst W;
  - (b) the same, held out;
  - (c) the ideal control against the bench curve.
  - Markers at rho* and at 41 and 63; the 0.9 dB straddle budget as a reference line in (a) and (b).
- `rank_injection_controls`: channel 37 by geometry, mode I against mode R, and the strong-pilot miss rate per rank for the proxy-high channels.
