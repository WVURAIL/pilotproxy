# Mask-versus-residual frontier against injected truth --- 2026-09-07

This is the synthetic-bench result asked for by chapter 6
(`sec:detection:synthetic`, `fig:detection:frontier`): the frontier of
`eq:detection:frontier` drawn at the production geometry against a
contamination whose value is known frame by frame, and --- the point of the
exercise --- what the residual-score histogram's cumulative sums *claim*
survives in the kept frames set beside what was actually *injected* into them.

Every residual quoted anywhere in the archive analysis is a prefix sum over the
histogram of `eq:detection:histogram` whose per-frame entries are estimates. On
the archive the truth is unknown, so that estimator has never been checked
against one. This is the only place in the work where it can be.

No radio is involved. The frames are synthetic 2048-stream frames built from
the same GNU Radio 8-VSB waveform, the same reference-PFB normalization, the
same weight profile, the same designated set and the same bulk mask as the
crossing and representation-loss results.

## Verdict

**The bookkeeping is exact and the estimator never under-books, but it
over-books by a median factor of 23 and by up to 6.3e3, and once the mask is
working the residual it reports is simply the floor.**

* Over 47,544 evaluated frontier points that retain some injected
  contamination, the claim is below the truth **0** times. The smallest
  claim/truth ratio anywhere is **1.19**.
* The median claim/truth ratio is **23.2**; the largest is **6.28e3**.
* The prefix-sum walk reproduces the direct kept-frame mean exactly: the
  largest absolute disagreement over 60 rank-and-configuration pairs is
  **9.0e-17**.
* **86%** of all 104,815 evaluated points report a residual within 10% of the
  channel floor, and the smallest residual reported anywhere on any frontier is
  **1.0084x** the floor. At the loud shelf the mask removes *every* injected
  frame --- the surviving injection is exactly zero --- and the claim is still
  2.60e-5, the floor.

The last point is the one that bears on chapter 9, and the operating-point
table makes it concrete. In **every** one of the twelve configurations there is
a multiplier at which the mask removes *every* injected frame, so the true
surviving contamination is exactly zero --- at -44 dB and above it costs
exactly the duty cycle (masked 0.100, 0.250, 0.500 at duties 0.10, 0.25, 0.50),
and even at the sub-noise -55 dB shelf it costs 0.84 to 0.92 of the frames. At
that point the claim is 2.60e-5, the floor. Held to an allowance of half the
unmasked injected contamination, the selector accepts only the three loud
configurations and **refuses the other nine** --- refuses, that is, nine
populations it has in fact cleaned completely, because the residual it books
for a perfectly clean kept set is the floor and the floor exceeds the
allowance.

`r_sys` is bounded below by the floor by construction, so a channel whose
science tolerance lies below its floor cannot be given a feasible operating
point no matter how completely the mask cleans the data. The v5 refusal on
every channel is therefore, at least in part, a statement about the floor
convention rather than about surviving contamination. This bench cannot say how
much of the archive's refusal is floor and how much is data --- its floor is
its own --- but it establishes the mechanism and measures its size.

## What produced it

    # 7,000 frames at K=128, L=128, M=2048, on the GPU
    PYTHONPATH=src python3 tools/mask_residual_frontier.py --stage generate --gpu \
        --input-iq generated/atsc/atsc_8vsb_complex64_settled.cfile \
        --waveform-audit generated/atsc/atsc_waveform_audit_settled.json \
        --output-dir docs/evidence/mask_residual_frontier_2026-09-07 \
        --shelf-snr-db -10.0 -44.0 -50.0 -55.0 --duty-cycle 0.10 0.25 0.50 \
        --rho 12 31 62 93 112 \
        --frames-on 1000 --frames-off 2000 --frames-floor 1000

    # the device generator against the host reference, on real frames
    ... --stage audit --gpu --audit-frames 3 ...

    # the frontier, the claim-against-truth panel, the plates and the tables
    ... --stage report --bootstrap-replicates 1000 ...
    ... --stage figure ...

Wall time: 384 s (6 min 24 s) of frame generation, 28 s for the device audit,
53 s for the report, 5 s for the figure. This *is* the full production
configuration --- K=128, L=128, M=2048 streams, the fine designated-set
statistic, the exact Q16 decision. Nothing was reduced.

Environment: `/home/djg/rail/delete_me/venvs/ppci/bin/python` (3.12.3, NumPy
2.4.2, CuPy 14.0.1) on an RTX 5000 Ada laptop GPU.
`study_config.json` carries the immutable identity
`19c90edb30867907b28a1829c089daeffe6b380b24ed1edbc3635ddb47ec0879`.

## The masking rule is the archive's, not a bench imitation

`src/pilot_proxy/testbench/mask_frontier.py` transcribes the deployed selector.
`selector_cross_check.py` runs that transcription and the deployed RFIsher code
side by side on the same frames; `selector_cross_check.txt` is its output:

* the exact per-rank Q16 boundary agrees with
  `rfisher.residual_scores.bundle.required_multipliers_for_frame` on
  **300 frames x 124 ranks with zero mismatches**;
* the empirical candidate staircase agrees with
  `rfisher.preparation.candidate_multiplier_q16_values`;
* the keep rule agrees with `rfisher_results.archive.selection.kept_at`;
* the frontier walk agrees with `rfisher.thresholds.optimize_threshold` on all
  271 evaluable points, to the bit, in both masked fraction and `r_sys`.

The per-frame residual is the archive's `shelf-or-floor linear` convention:
`max(10^(shelf/10), floor)` for a frame with a finite shelf estimate, the floor
otherwise, and never below the floor. The shelf estimate comes from the exact
`uint64` coarse marginals and the exact integer weight norms
(`target_norm_sq = 6338`, `reference_norm_sum_sq = 12650`) through the
geometry's declared transfer, offset `-21.636` dB. The floor is measured the
archive way, from a **disjoint** off population: the 90th percentile of its
finite shelf estimates.

## Populations

One waveform, physical channel 14, offset 0 fine bins, anchor bin 255, bulk
size 124 bins, designated half-width 2, guard 1 bin.

| population | frames | injected shelf | purpose |
| --- | --- | --- | --- |
| `off` | 2000 | none | the null half of every duty cycle |
| `floor` | 1000 | none | the channel floor, disjoint from `off` |
| `on_m10p0` | 1000 | -10 dB (loud) | plates and the transfer closure |
| `on_m44p0` | 1000 | -44 dB | above the crossing |
| `on_m50p0` | 1000 | -50 dB | near the crossing |
| `on_m55p0` | 1000 | -55 dB (sub-noise) | below the detector's reach |

A duty cycle is composed from these banks in order, so a wider duty cycle
contains a narrower one's on frames and the curves are comparable. Every
configuration holds 2000 frames. Input clipping is 0.54% on the null and
1.24% at the loud shelf.

The floor comes out at **-45.92 dB** (2.5615e-5 linear), the 90th percentile of
the 508 of 1000 off frames that resolve a positive excess. Half the null
frames resolving no shelf at all is what a symmetric null should do.

## The estimator's own accuracy, separated from the masking

`transfer_closure` is the mean linear shelf the injected frames report divided
by the linear shelf the bench injected, over the *unmasked* on frames with no
floor. It is a property of the pilot-excess-to-shelf transfer, not of the mask,
and separating it is what lets the frontier's gap be read as masking.

| shelf | transfer closure | claim/truth, no mask | claim/truth over the frontier |
| --- | --- | --- | --- |
| -10 dB | +0.75 dB (1.19x) | 1.19 | 1.19 -- 1.66 |
| -44 dB | +1.05 -- +1.15 dB | 1.94 -- 7.18 | 1.94 -- 1.18e3 |
| -50 dB | +1.63 -- +1.76 dB | 5.38 -- 26.3 | 5.38 -- 4.70e3 |
| -55 dB | +4.06 -- +4.47 dB | 16.5 -- 82.4 | 16.5 -- 6.28e3 |

Two things are visible here and both are real.

The transfer closure degrades as the shelf sinks: +0.75 dB at -10 dB, +4.5 dB
at -55 dB. That is the estimator, not the mask. The per-frame excess is a noisy
positive quantity, and as the true excess falls below the null width its mean
is set by the noise rather than by the signal. A residual booked from it is
therefore high, and increasingly high, exactly in the regime where a residual
matters.

Masking makes the over-booking *worse*, not better, and that is the intended
behaviour read the other way round. The mask removes the frames that carry the
injection, so the truth falls quickly while the claim stops falling at the
floor. Over the frontier the ratio therefore runs away --- at -44 dB from 1.94
unmasked to 1.18e3 at the tightest evaluable multiplier. The residual axis
below the floor carries no information.

Removing the floor from the convention does not remove the bound. Booking the
plain kept-frame mean of `10^(shelf/10)` with unresolved frames at zero still
never under-books (0 of 47,544 points), and still cannot report below
**4.90e-6** (-53.1 dB), the noise-driven positive excess of the kept null
frames. The floor convention raises that bound by a further 7.2 dB; it does not
create it.

## The rank is not a second degree of freedom here

Chapter 6 offers `rho` as an optional second degree of freedom. At this
geometry it is not one. At a matched masked fraction of 0.60 the surviving
injected contamination is identical across `rho` in {12, 31, 62, 93, 112} at
every shelf of -50 dB and above --- exactly zero, the mask having removed every
injected frame --- and at -55 dB the five ranks differ only by the one or two
frames that noise moves across the boundary (a 6% spread at duty 0.50, 40% at
duty 0.10 on a truth of 2e-8). The curves in panel (a) coincide because they
are the same curve. `rho` moves the numerical value of `eta` and not the trade.

## The plates

`fig:detection:frontier` panel (c), at the median rank and duty 0.50, at the
candidate whose masked fraction matches the duty cycle.

At the loud shelf the pilot line stands 30 dB above the fine floor in the
all-frame mean and is gone from the kept-frame mean; the mask removed 1000 of
2000 frames and every injected one. At the sub-noise shelf the coherent line is
still visible in the *fine* spectrum --- 0.15 dB, because L=128 windows of
coherent gain sharpen a line the averaged band floor could never show --- and
the mask reduces it to 0.025 dB without removing it. That is the regime the
chapter says the spectrum cannot carry and the injected truth must: at -55 dB
the mask keeps 1000 frames whose true surviving residual is 4.6e-7 and books
2.6e-5 for them.

## Files

* `study_config.json` --- the immutable study identity and its input hashes.
* `frames/*.npz` --- 7,000 frames: the exact `uint64` fine powers `[3, 256]`,
  the exact coarse marginals, the per-rank Q16 boundaries for all 124 ranks (0
  is the always-masked sentinel), the normalized excess, the shelf estimate,
  the clip fraction and the frame seed. Everything downstream is exactly
  reproducible from these without a GPU.
* `frontier_points.csv` --- the full unsmoothed record, 104,815 points:
  configuration, shelf, duty, `rho`, `eta` and `eta_q16`, kept count, masked
  fraction, `r_sys`, `r_sys` without the floor, the injected truth on the same
  kept set, both bootstrap intervals and the claim-against-truth ratio.
* `frontier_report.json` --- the floor, the populations, the prefix-sum audit,
  the verdict, the per-configuration summaries and the plate spectra.
* `operating_points.csv` --- the minimum-mask envelope of
  `eq:detection:frontier` at two allowances (half and a tenth of the unmasked
  injected contamination), with whether the *truth* is inside the allowance the
  *claim* was admitted under. 30 of 120 rows select a point; the other 90
  refuse, all of them on populations the mask can clean completely. Where a
  point is selected the truth is always inside the allowance --- the refusals
  are the false ones, not the acceptances.
* `device_audit.json` --- the device frame generator against the host
  reference on six real 2048-stream frames: packing, fine powers and coarse
  marginals all bit-identical.
* `selector_cross_check.py`, `selector_cross_check.txt` --- the masking rule
  against the deployed RFIsher code.
* `mask_residual_frontier.png`, `.pdf` --- the three-panel figure.
* `generate.log` --- the generation transcript with per-frame timings.

## Bootstrap

Trial bootstrap, 1000 replicates, frames resampled with replacement at a
candidate grid held fixed at the full-sample staircase, 16th and 84th
percentiles. Median relative interval width is 0.7% on `r_sys` and 14.5% on the
surviving injected truth. No display smoothing anywhere; the empirical steps
are retained.

## What this does not establish

The floor is this bench's floor, measured on this bench's null. The archive's
floors are its own and the numerical conservatism will differ. The frames carry
no feed covariance, no sky signal and no gain structure, and the duty cycle is
an independent Bernoulli label rather than a real transmitter schedule. The
transfer closure quoted here is a by-product, not a replacement for the
estimator-transfer result of `sec:estimator:validation`.
