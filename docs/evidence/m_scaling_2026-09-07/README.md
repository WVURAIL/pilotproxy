# Spatial scaling in M — 2026-09-07

Measured null width and deflection of the coarse and fine designated-set
statistics against the input stream count M, for panel (a) of the four-panel
synthetic-verification figure Chapter 6 asks for (`fig:detection:mscaling`,
`sec:detection:synthetic`), together with panel (c), the fine-axis gain
against the aligned-tone benchmark 5 log10(L) = 10.54 dB measured on a bin
centre *and* off it.

Panels (b) and (d) of that stub are not here: they are `tab:detection:loss`
and `fig_detection_crossings`, rendered by RFIsher's
`rfisher_results/archive/report/detection.py` from the `estimator_transfer`
and `sdr_ota` releases. This directory supplies the two measurements that
module names as missing — the spatial axis and the second statistic — and
nothing else.

This is a synthetic bench. No LimeSDR, no radio, no capture: the Monte Carlo
operates at the integer row-sum level, the exact field both deployed
reductions consume, at 128 windows per stream and the deployed statistics
(`tools/measure_fine_gain.py`, whose batched reduction is checked against
`pilot_proxy.fine_reduction.fine_reduce` by `--stage verify` and by
`tests/core/test_measure_fine_gain.py`). Interpreter:
`/home/djg/rail/delete_me/venvs/ppci/bin/python` (numpy 2.4.2); CPU only.

## Reproduce

    P=/home/djg/rail/delete_me/venvs/ppci/bin/python
    D=docs/evidence/m_scaling_2026-09-07/data

    # gates
    $P tools/measure_fine_gain.py --stage verify --trials 4 --out /tmp/x   # 2048 streams
    $P tools/measure_m_scaling.py --stage verify-rep --trials 3 --streams 2048

    # panel (a): null population, 4 shards x 1000 trials per M (12 x 1000 at M=2048)
    for M in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
      B=$(( 16384 / M )); [ $B -gt 256 ] && B=256; [ $B -lt 8 ] && B=8
      S="101 102 103 104"; [ $M -eq 2048 ] && S="101 102 103 104 105 106 107 108 109 110 111 112"
      for s in $S; do
        $P tools/measure_fine_gain.py --stage h0 --streams $M --trials 1000 \
           --seed $s --batch $B --out $D/null_m$M
      done
      # deflection probes, two levels (the linearity check)
      for sd in 0 -10; do
        $P tools/measure_fine_gain.py --stage sweep --snr-db $sd --streams $M \
           --trials 500 --seed 201 --batch $B --out $D/defl_m$M
      done
      # replicated capture: one stream tiled across all M inputs
      for s in 201 202; do
        $P tools/measure_m_scaling.py --stage null-rep --streams $M --trials 500 \
           --seed $s --batch $B --out $D/rep_null_m$M
      done
      $P tools/measure_m_scaling.py --stage defl-rep --snr-db 0 --streams $M \
         --trials 500 --seed 301 --batch $B --out $D/rep_defl_m$M
    done

    # panel (c): Pd sweep at M=2048, bin centre and worst padded-grid straddle
    for sd in -40 -37 -34 -31 -28 -25 -22 -19 -16 -13 -10; do
      $P tools/measure_fine_gain.py --stage sweep --snr-db $sd --streams 2048 \
         --trials 1200 --seed 7 --batch 8 --out $D/null_m2048
    done
    for sd in -34 -31 -28 -25 -22 -19 -16; do
      $P tools/measure_fine_gain.py --stage sweep --snr-db $sd --half-bin \
         --streams 2048 --trials 1200 --seed 7 --batch 8 --out $D/null_m2048
    done
    # 1 dB refinement across both crossings (the 3 dB grid cannot resolve 0.3 dB)
    for sd in -35 -34 -33 -32 -31 -30 -29 -26 -25 -24 -23 -22 -21 -20; do
      $P tools/measure_fine_gain.py --stage sweep --snr-db $sd --streams 2048 \
         --trials 1200 --seed 8 --batch 8 --out $D/null_m2048
    done
    for sd in -35 -34 -33 -32 -31 -30 -29; do
      $P tools/measure_fine_gain.py --stage sweep --snr-db $sd --half-bin \
         --streams 2048 --trials 1200 --seed 8 --batch 8 --out $D/null_m2048
    done

    $P tools/measure_m_scaling.py --stage report --root $D \
       --report-json docs/evidence/m_scaling_2026-09-07/m_scaling_report.json

Wall time, 12 jobs in parallel on 32 cores: 761 s for the sweep above plus
500 s for the 1 dB refinement; about 86 minutes of single-core CPU in total.

## Populations

155 shards, 132,800 trials, all at 128 windows per stream.

| population | per M | total |
|---|---|---|
| null, independent streams, M = 1…1024 | 4 shards x 1000 | 44,000 |
| null, independent streams, M = 2048 | 12 shards x 1000 | 12,000 |
| deflection probes, 0 and −10 dB per row sum | 2 x 500 | 12,000 |
| null, replicated capture | 2 x 500 | 12,000 |
| deflection probe, replicated capture, 0 dB | 500 | 6,000 |
| Pd sweep at M = 2048, bin centre, 21 SNR points | — | 30,000 |
| Pd sweep at M = 2048, half padded-bin offset, 12 points | — | 16,800 |

Widths on each null population: **raw** = sample standard deviation;
**robust core** = normal-consistent inter-quartile scale 0.7413 (q75 − q25);
**left scale** = median − q15.87, the one-sided bulk scale the archive
calibration uses, carried in the CSV. Deflection is the H1 mean excess over
the H0 mean divided by the H0 width, reported per unit per-row-sum SNR.

## What it shows

**The coarse statistic follows both predictions, over eleven octaves.** The
log-log slope of the robust core width is −0.500 (raw −0.502) across
M = 1…2048, and the implied beta in 1/sqrt(beta M) is flat across the sweep — mean
85.97, sd 3.68, range 79.1 to 92.3 — against the value the statistic's own
degrees of freedom predict, beta = L/1.5 = 85.33; every M is within 4% of
the predicted width. Deflection tracks
sqrt(M) to within ±5% over the same range (0.968 to 1.046 of the law
anchored at M = 1). Nothing about the coarse axis is surprising.

**The fine statistic does not follow 1/sqrt(beta M) with any single beta.**
The implied beta rises monotonically from 0.091 at M = 1 to 1.202 at
M = 2048, a factor of 13, and is still drifting +10% over the last three
octaves. The full-range core slope is −0.638, not −0.5; restricted to
M >= 64 it is −0.534. At the production geometry the fine null is 25%
*narrower* than the naive single-bin prediction (core width 0.0202 against
1/sqrt(M/1.5) = 0.0271, beta = 1.20 against 0.667), and at M = 1 it is
2.7x *wider*. The law is a large-M asymptote for this statistic, and quoting
one beta for it across the sweep is wrong in both directions.

**The raw width of the fine statistic is not a usable number below M ≈ 4.**
The designated-set ratio divides by a chi-square with 4M degrees of freedom,
so its second moment does not exist for M = 1 and is marginal for M = 2:
four independent 1000-trial shards at M = 1 give raw sample widths of 7.75,
9.13, 10.98 and 13.44 — a 55% spread — while the robust core of the same
shards spreads by 15%. Above M = 4 raw and core agree to a few percent.
This is why the panel carries both, and why the core is the one to read.

**Fine deflection grows faster than sqrt(M).** Relative to the law anchored
at M = 64 it is 1.12x at M = 2048 and 0.65x at M = 1. Two measured effects,
not one: the excess per unit SNR falls from 271 at M = 1 to its asymptote of
exactly L = 128 by M ≈ 128 (128.09 measured at M = 2048 against L = 128
predicted), while the core null width falls faster than 1/sqrt(M).

**The replicated capture is exactly flat, and it is exactly flat for a
reason.** Both statistics are ratios of sums over streams, so tiling one
capture across M inputs multiplies numerator and denominator by the same M:
the replicated statistic is *identically* the M = 1 statistic.
`--stage verify-rep` checks it on real trials rather than asserting it —
coarse max|d| = 0 exactly, fine relative 1.1e-14 at M = 2048. The measured
line therefore sits at the M = 1 value at every M: at M = 2048 that is 47x
the independent coarse core width and 174x the independent fine core width.
Drawn as a correlation stress case, which is what the chapter asks for.

**Panel (c): the aligned-tone benchmark is not attained, on the bin centre
or off it.** Empirical H0 thresholds at fixed Pfa on the 12,000-trial null
at M = 2048; crossings at Pd = 0.5 interpolated linearly between adjacent
sampled SNR points (1 dB spacing across both crossings), no extrapolation;
95% intervals from 400 trial-bootstrap replicates that resample whole trials
inside each SNR point and carry the coarse and fine statistic of a resampled
trial together, holding the thresholds at their full-sample values.

| Pfa | offset | coarse SNR@Pd=0.5 | fine SNR@Pd=0.5 | gain | 95% CI |
|---|---|---|---|---|---|
| 1e-2 | bin centre | −22.73 dB | −32.13 dB | 9.40 dB | [9.22, 9.54] |
| 1e-2 | half padded bin | −22.73 dB | −31.85 dB | 9.12 dB | [8.96, 9.26] |
| 1e-3 | bin centre | −21.45 dB | −31.15 dB | 9.69 dB | [9.60, 9.79] |
| 1e-3 | half padded bin | −21.45 dB | −30.80 dB | 9.34 dB | [9.26, 9.43] |

The benchmark 5 log10(L) = 10.54 dB lies above every one of those four
intervals: 0.85 to 1.14 dB above the bin-centre gain and 1.20 to 1.42 dB
above the off-centre gain. The straddle costs 0.28 dB [0.11, 0.44] at
Pfa = 1e-2 and 0.35 dB [0.26, 0.43] at Pfa = 1e-3.

The benchmark's own content is confirmed exactly, which is the point of
measuring both: the coherent concentration the fine axis buys is L in power,
measured 128.09 against L = 128 at M = 2048. What falls short is the
*detection* gain, because the fine statistic pays for that concentration with
a null of 2M degrees of freedom against the coarse statistic's 2ML, and with
a maximum over the five-bin designated set. 5 log10(L) prices the coherent
gain of a tone on a bin centre; it is not the detection gain, and it is not
a universal bound on either offset.

The off-centre case is the worst straddle *on the grid the detector actually
searches*: the window-axis transform is zero-padded x2, so the designated set
lives on a half-native-bin grid and the worst offset is half a padded bin
(injected fine bin 62.5, 5.96 Hz from the anchor), not half a native bin.

Cross-check: `docs/evidence/fine_gain_mc_2026-08-19` measured the
bin-centre gain as 9.32 dB (Pfa 1e-2) and 9.77 dB (Pfa 1e-3) on a 3 dB grid.
This directory's independent seeds and 1 dB grid give 9.40 and 9.69 dB, so
the retained evidence reproduces to about 0.1 dB — but not like for like:
that measurement was read with `measure_fine_gain.py --stage report`, whose
centred curve pools the half-bin shards written beside it (see Files below),
and the same reader on this data returns 9.25 and 9.51 dB. Both runs agree
that the measured gain sits near 9.3–9.7 dB and below the benchmark; neither
is precise to better than a couple of tenths of a decibel.

## What it does not show

* **Nothing here is an over-the-air measurement.** The LimeSDR is not
  involved and is not reachable from this machine. The label "LimeSDR
  detection campaign" on the chapter's stub covers the synthetic bench too;
  this is the synthetic half.
* **No arithmetic ladder.** These trials are float reductions of integer row
  sums. The int4 input quantizer, the quantized weight banks, the frozen
  fixed-point transform and the exact Q16 decision are not exercised, so no
  number here bounds representation loss — that is `tab:detection:loss`, and
  the deflection at M = 2048 here is the ideal-arithmetic value, not the
  packed path's.
* **No weight profile.** The Monte Carlo starts at the row sums, downstream
  of the weights, so the 23 per-channel quantized banks the campaign design
  asks for are not in this study at all.
* **The noise model is i.i.d. per stream, per window and per term**, and the
  pilot is a coherent tone with a uniform random phase per stream, not an
  8-VSB pilot line with residual modulation and no PFB in front of it. The
  independent-stream curve is therefore the fully uncorrelated ideal and the
  replicated curve the fully correlated extreme; the array's actual
  inter-feed correlation is measured by neither, and the replicated case is a
  stress case, not a bound.
* **Two offsets only.** Bin centre and the worst padded-grid straddle. The
  ±1 kHz and ±1.4 kHz fine-span offsets the campaign design asks for carry no
  measurement here and are not interpolated.
* **Deflection is a first-moment statistic, not Pd.** Only panel (c) is a
  detection measurement. Deflection is reported per unit SNR from two probe
  levels, 0 and −10 dB per row sum. The two agree to within 1% for both
  statistics at M >= 128 and disagree by up to 8% (coarse) and 6% (fine) at
  M <= 8 — where 500 trials at the weak probe put the measured excess at or
  below one null core width, so the disagreement is the weak probe's own
  Monte Carlo error and not a demonstrated nonlinearity. The sweep runs one
  probe pair per M, not a Pd curve per M.
* **No frontier.** `fig:detection:frontier` needs duty-cycled, labelled
  frames with a per-frame injection truth, a rank and a multiplier. None of
  those exist in these shards.

## Files

* `m_scaling_report.json` — every number above, keyed by population and M:
  widths (raw, robust core, left scale), per-shard widths, implied beta,
  log-log fits, deflections at both probes, the panel (c) curves, thresholds,
  crossings, gains and bootstrap intervals.
* `m_scaling_widths.csv` — the width and deflection table, one row per
  (population, M, statistic, probe).
* `m_scaling.png` — the three measured panels: null width against M,
  deflection against M, and the Pd curves behind the gain.
* `data/` — 155 shards. `null_m<M>/` null populations, and at M = 2048 the
  panel (c) Pd sweeps as well; `defl_m<M>/` deflection probes;
  `rep_null_m<M>/`, `rep_defl_m<M>/` the replicated-capture stress case.
  Read the gain numbers with `measure_m_scaling.py --stage report`, not with
  `measure_fine_gain.py --stage report`: that tool's centred glob,
  `h1_*dB_s*.npz`, also matches its own half-bin shards `h1_half_*dB_s*.npz`,
  so when both offsets share one directory its "centred" curve silently pools
  the straddled trials. On this data it returns 9.25 and 9.51 dB where the
  unpolluted centred gains are 9.40 and 9.69 dB. Its thresholds and its
  coarse crossings are unaffected (they reproduce exactly: 1.005554 /
  1.080238 at Pfa 1e-2, coarse −22.73 and −21.45 dB), because the coarse
  statistic is a total-power statistic and does not see the offset. The same
  glob is in the reproduce block of `docs/evidence/fine_gain_mc_2026-08-19`,
  whose half-bin sweeps were written into the same directory as its centred
  ones, so its quoted 9.32 / 9.77 dB carry the same pooling and read low.
  Nothing in this directory is corrected for that; it is reported, not
  patched, and `measure_fine_gain.py` is unmodified.
* `tools/measure_m_scaling.py` — the replicated-capture generator and the
  report. New in this run, and not committed.
