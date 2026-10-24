# Design decisions

This file records choices that affect the interpretation of CHIME DTV
products. We keep the choice, its reason, and its boundary together so the
same question does not have to be reconstructed during review.

## The CHIME default is norm-corrected positive excess

The CHIME real-data path uses the following comparison:

```text
valid = p_ref_sum != 0
mask  = valid && (p_target * reference_norm_sum_sq > target_norm_sq * p_ref_sum)
```

This is the exact integer form of `F > null_power_ratio`, where

```text
null_power_ratio = 2 * target_norm_sq / reference_norm_sum_sq
```

is the flat-floor `H0` zero point implied by the packed weights. Int4
quantization leaves the target and reference norms unequal. Independent
recomputation from the shipped ATSC 14--36 manifest gives a `null_power_ratio` range of
0.9853298815 to 1.0111111111 rather than exactly 1.

We compare each channel with its own `null_power_ratio` so that the rule is defined by the
shipped weights and can be reproduced from integer powers, with no fitted
operational threshold. Each product records its rule in `mask_rule`; products
made under the plain `F > 1` rule say so there and are read under that rule.

## `K = 128` is the CHIME baseline; `K = 64` is the CHORD baseline

The compiled detector family and detector-core profile support `K = 64` and
`K = 128`. CHIME retains `K = 128` because it matches the CHIME receiver
profile, shipped CHIME manifest, and regression data. The full and pathfinder
CHORD profiles select `K = 64`; their 8,192-sample frames still produce the
frozen 128 windows per stream.

`K = 256` is not implemented or tested and would require a separate fixed-point
bound, weight bank, contract, and validation set.

`K = 512` is outside the current candidate set.

## Guard and reference terms have separate names

The public terms are:

```text
skipped_guard_bins
reference_offset_bins
```

They obey:

```text
reference_offset_bins = skipped_guard_bins + 1
```

A user-authored detector-core configuration specifies
`skipped_guard_bins`. Generated metadata may also record
`reference_offset_bins` so that the selected geometry can be audited.

## Reference placement is adaptive and recorded

The resolver retains the requested reference offset when the receiver
geometry permits it. If a reference reaches a circular FFT edge, it wraps. If
a reference would collide with the forbidden coarse-channel DC tone, it moves
away. A target/DC collision is invalid because moving the target would change
the signal being tested.

In the shipped ATSC 14--36 bank, DTV 14 is the only adaptive case: its lower reference wraps
across the coarse-channel edge. No shipped channel collides with, or shifts
around, the forbidden DC tone.

Run products retain the reference-placement summary. Validators and later
analyses should use the recorded placement rather than reconstructing it from
the nominal offset.

## The runtime bundle is the Kotekan interface

The runtime exporter writes:

```text
detector_contract.json
pilot_profiles.json
weights.bin
weights.manifest.json
sha256sums.txt
```

The deployment model is:

```text
same detector software on every node
same weight bundle on every node
first-frame integer CHIME channel ID selects active profile
non-pilot channel disables detector
pilot channel selects one weight-bank pointer
```

Kotekan's `cudaPilotProxyDetector` stage (Kotekan pull request 1677, in its
`develop` branch) consumes this bundle; this repository holds no Kotekan code.
A valid bundle is not evidence of deployment; commissioning follows
[OPERATOR_HANDOFF.md](OPERATOR_HANDOFF.md).

## The deployed decision is the fine designated-set CFAR in the kernel

The archive products were made with the fine decision inactive. A band applies
it only through a compiled operating policy
(`pilot-proxy compile-operating-policy`, `docs/CHANNEL_OPERATING_POLICY.md`),
which carries the threshold words.

The fine front end replaces the logic after the dot product: instead of
squaring each window's dot product and summing incoherently over all rows, the
windows are integrated coherently (the padded window-axis FFT, per feed) and
only the feed sum remains incoherent. For W = 128 windows this recovers up to
10 log10(sqrt(W)) ~ 10.5 dB of deflection sensitivity over the coarse reduction
(less scalloping, bounded at 3.92 dB at the fine-span edge), and the calibrated
any-bin CFAR turns the refined spectrum into a decision with a pre-registered
false-alarm rate.

Measured ROCs are the reason for the fine path. Coarse-only operating points
fail on the archive: the coarse null bulk and the weak-signal bulk overlap
almost completely, so every coarse operating point is one of two failures. The
null_power_ratio point holds Pd ~ 1.0 everywhere but spends 48.5% of
verified-quiet time by construction, and null-calibrated fixed-Pfa points
collapse on weak channels (null = ch35 off-state, n = 66; signal = 2024+
on-epochs of the first three scanned channels):

| Pfa (null quantile) | theta (F/null_power_ratio) | ch36 Pd | ch35 Pd | ch34 Pd |
|---|---|---|---|---|
| 0.10 (P90) | 1.114 | 0.904 | 0.996 | 0.099 |
| 0.05 (P95) | 1.124 | 0.872 | 0.996 | 0.089 |
| 0.015 (null max) | 1.146 | 0.805 | 0.996 | 0.071 |

The "null" here is a transmitter-off epoch inferred from the archive (ch35's
off state), not a verified signal-free population; a false-alarm limit for
deployment is set only on a band's own verified, null-like population.

The same frames carry a fine designated-set Pd ~ 0.99 on all three channels at
a measured Pfa of 0.091, with the crude +/-2-bin set and an uncalibrated CFAR
multiplier. The fine axis is the only one that holds both ends at once: more
complete than any calibrated coarse point, and cheaper than null_power_ratio.

The mask therefore lives in the kernel, as one solid kernel.
`PilotProxy_Compute_FusedFineMask_U64` takes packed samples, packed weights and
the per-channel bundle constants, and emits one mask bit per aligned frame
(1 = reject; a zero reference forces 0). Each block handles one stream: it
computes the stream's exact int32 row sums in shared memory (never materialized
to global memory), runs the frozen fxfft256 in place with cooperative
butterflies (disjoint writes per stage, so the bits do not depend on
scheduling), and accumulates the exact fine powers of the feed sum. A
last-block epilogue (the completion counter folded into the mask element;
threadfence, then atomic) forms the designated-set decision over the finalized
sums. Row sums are exposed only through an optional debug tap.
`PilotProxy_Compute_FusedFine_U64` is the same launch without the decision, and
`PilotProxy_Compute_FinePowers_U64` is the staged fine-power entry. The epilogue
is deterministic because it reads order-independent exact integer sums. The
governing rule is bit-identity with the frozen reference and golden vectors.

The decision arithmetic is frozen as **fine decision v1** (reference
`src/pilot_proxy/fine_decision.py`, gated by
`tests/core/test_fine_decision.py` against a `fractions.Fraction`
oracle): `F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` held as exact rationals;
null bulk = the bundle's 256-bit mask (built from the pre-registered
`independent_bin_mask`: independent bins, guard- and census-excluded)
intersected with live denominators; CFAR estimate = the bundle-specified
*rank* of the ascending bulk F2 values (order-statistic CFAR; rank
selection by counting, so the selected value is unique under ties and
any representative yields the same decision); fire iff some designated
bin (anchor +/- width, modulo 256 --- the survey's window convention)
exceeds `multiplier_q16 / 2^16` times the rank value. Comparisons are
exact 128/192-bit integer cross products (fuzz-gated, 2M trials vs
Python big ints). Degenerate denominators never fire; a bulk not deeper
than the rank forces mask 0.

The CFAR is rank-based rather than a location+scale estimate, by the same
reasoning that chose the fixed-point FFT: float medians and interpolated
quantiles cannot be made bit-exact across numpy/nvcc/compiler versions, while
the rank statistic is exact, needs no contamination fallback mode (order
statistics reject the measured 0.29% bulk tail by construction), and can be
rescored offline from the survey products' exact `fine_power_u64` terms ---
target, lower-reference, and upper-reference uint64 per fine bin --- because
`F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` is reproduced from them exactly. The
archive fixes the rank at the median of the intended bulk (`rho = 63`,
zero-based `cfar_rank = 62`; 60 and 53 on channels 20 and 34) and takes each
channel's threshold-rule word as its multiplier (see "The recorded threshold is
data, not architecture" below). The deployment compiler carries those words,
and the permanent mask of the channels the rule does not deploy, into the
runtime policy ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)). The
operating point (anchor, width, bulk mask, rank, multiplier) is runtime-bundle
data with provenance (the `fine_calibration` block, exported as
`pending_campaign` or `calibrated` and validated in both states by the bundle
validator); the arithmetic is code. In the archive product contract,
`fine_candidate_decision.active` is false and `reject_mask` is the coarse
norm-corrected positive-excess decision; changing that needs a
schema/decision-contract revision at a survey epoch boundary.

Verification: a threaded CPU emulation of the extracted device code (one
pthread per CUDA thread, a real barrier for `__syncthreads`; mask, fine and tap
outputs bit-identical to the Python reference chain across noise, injected
pilot, wrap-anchor, degenerate-bulk, and tap-unbound scenarios; stable across
repeated concurrent runs), and the on-GPU gates
`tests/kernel/test_fused_fine_gpu.py` and `tests/kernel/test_fused_mask_gpu.py`
(reference bit-equality per frame, injection firing, device-side degenerate
rules, the production no-tap form, 20-repeat last-block determinism, and the
epilogue-cost rate report).

Measured on an A100 (core 2.3.0 build): the fused production form takes
0.769 ms per 2048-stream frame (x55 margin against the 41.9 ms cadence) and the
mask form 0.895 ms (x47; the decision epilogue adds 0.127 ms), about 2% of the
frame cadence.

The gating requirement is a deterministic verification FFT. The project's
validation standard is bit-exact agreement between the CUDA path and the
Python reference; cuFFT and numpy cannot bit-match each other (different
algorithms, operation order, FMA), so the FFT must be owned by the project and
implemented identically on both sides.

The fixed-point transform is frozen as **fxfft256 v1** --- reference
`src/pilot_proxy/fxfft.py`, C port template `cuda/fxfft256_ref.c`, golden
vectors `tests/data/fxfft256_golden_v1.npz` (56 vectors including the
survey-measured pilot offsets and int4-dot row-sum arithmetic), bit-compare
gate `tests/core/test_fxfft256.py`. Spec in brief: 256-point radix-2 DIT, Q15
twiddle table frozen as source literals (nearest-integer, no rounding ties;
+-32768 held in int32 and exact under the rounding rule), butterfly rounding
floor((v + 2^14) / 2^15), no stage scaling, inputs |re|,|im| <= 2^20 provably
overflow-free in int32 (production-bound row sums reach 2^14; contract-limit
stress measures 28 of 31 bits). Measured against the analyzer's complex128
pipeline: spectrum error <= 6.3e-4 of the spectrum peak across all golden
families, and the end-to-end fine statistic moves by <= 5.9e-4 relative
(-0.00026 dB at a +1287 Hz detection peak) --- five orders below the 10.5 dB
coherent gain (`tools/fxfft_report.py`). After the rounded FFT every downstream
quantity (|X|^2, feed sums, the F2 numerators and denominators) is exact
uint64, so the fine decision is deterministic integer arithmetic end to end. A
deterministic float32 FFT (identical butterfly order, FMA disabled) would
verify equally but ports worse. Library FFTs remain cross-check oracles in
tests; the CUDA implementation must reproduce the golden vectors bit-for-bit.

Calibration inputs are all survey-supplied. Per-channel measured line anchors
and designated-set widths need on-epochs only, which every detectable channel
supplies. Per-channel CFAR threshold multipliers need verified nulls: the
measured null is ~19x wider than the iid model, so the design-point P_FA is not
achieved without empirical correction. Conservative margins apply until enough
verified off-epochs accumulate; the null-universality test with the pooled-null
program (classify epochs on/off by fine-line persistence and census; test
universality of the null across channels with off-epochs; pool if universal)
is the path to deep, shared multipliers. The bundle records anchors, sets,
multipliers, and their calibration provenance: epoch lists, quantiles, and
source product hashes.

## The recorded threshold is data, not architecture

The survey keeps the norm-corrected positive excess (F > null_power_ratio) as the
recorded flag: it is exact in integer arithmetic, requires no calibration
input, and keeps the multi-year archive comparable. It is a recording
convention, not a tuned operating point --- scored against the ch35
off-state null it reaches Pd ~ 1.0 only at Pfa ~ 0.485 (Youden
J = Pd - Pfa ~ 0.51) on every channel; it buys completeness with half of
clean time.

Nothing in the architecture pins that choice. The operating point is
carried as per-channel rational data in the runtime bundle
(`positive_excess_half_threshold_num`/`_den` in `pilot_profiles.json`);
the CUDA comparison is threshold-agnostic, and `coarse_power_ratio` plus the fine
spectrum are persisted per frame, so every recorded epoch can be re-scored
offline under a different rule. Retuning a deployment is a bundle
regeneration and revalidation, not a kernel or schema change. Two
selection policies are defensible:

- Fixed false alarm (CFAR): measure a transmitter-off null, set the
  threshold at its (1 - P_FA) quantile, and let Pd follow. This is the
  policy the offline commissioning calibration supports, and the right one
  when clean integration time is the scarce resource. It needs a verified
  null, which the archive does not have.
- ROC-derived (Youden's J): sweep the threshold and maximize
  J = Pd - Pfa. Equal-cost by construction --- on a weak channel it will
  buy detections with false alarms one-for-one (see ch34 below) --- so it
  is the neutral default when no cost model exists for missed
  contamination versus discarded clean time. Measured on this survey for
  reference; not adopted for deployment, because the masking costs are
  not equal.

The archive uses neither policy to set its thresholds. The fine rank is
fixed at the median of the intended bulk, and each channel's word comes
from the threshold rule (`src/pilot_proxy/characterization/threshold_rule.py`):
the knee of its current-era equal-event score distribution
(`src/pilot_proxy/characterization/knee.py`) when the upper end of the kept
DTV power meets RFIsher's tolerance table and the calendar time to one
retained year stays within 20 years; otherwise the loosest word below the
knee that does; otherwise the channel is not deployed and goes to the
permanent mask. The knee is an operating heuristic read from the same data,
and no false-alarm rate is established for any word. This is also why the
ROC measurement reported Youden's J and never selected a threshold with it:
J prices a missed contamination and a discarded clean frame equally, and
they are not equal.

ROC on the 2020-2026 survey products (null = ch35 off-state,
n = 66 frames, proxied across channels; Pfa granularity ~1.5%):

| channel | coarse J_max (theta) | Pd / Pfa at J_max | fine designated set, measured line +/-2 bins |
|---|---|---|---|
| ch36 (506) | 0.837 (1.126) | 0.867 / 0.030 | Pd = 1.000 (no verified null) |
| ch35 (521) | 0.996 (1.157) | 0.996 / 0.000 | Pd = 0.996, Pfa = 0.091, J = 0.905 |
| ch34 (537) | 0.752 (1.030) | 0.979 / 0.227 | Pd = 0.991 (no verified null) |

Three readings. The coarse J-optimal thresholds land at or just above the
null bulk edge when separation is good --- J rediscovers the CFAR
intuition --- and dive into the bulk only when it is not. The fine path at
its fixed CFAR point dominates the coarse J-optimum exactly where the
coarse axis struggles: the coherent gain expressed as ROC separation
rather than dB. And the designated set must anchor at the measured pilot
line, not the nominal bin --- every detected transmitter sits at a stable
nonzero offset, and nominal-bin sets collapse. The table was recomputed from
the released per-pilot products and is frozen in `docs/evidence/youden_j_2026-09-07`.

## The source stamp is memoized per process

`detector_version` embeds a hash of every `*.py` under `src/pilot_proxy`.
The imported code cannot change within a process, so the hash is computed
once per process (first observation) and reused, rather than re-read from
disk at each channel's `begin()`. A source-only stamp difference is
forgivable for cross-pilot combine and stacking, which compare only the
geometry tokens; resume is stricter --- it compares the full
`detector_version` and aborts on any source-tree or package-version
change, so a new build needs a clean `--output-dir`. The stamp-to-commit
mapping is recoverable by recomputing the hash at each candidate
checkout.

## The stored fine surface is the exact integer terms

Products record the fine axis as `fine_power_u64`, shape
`(N, 3, fine_num_bins)` uint64 --- the target, lower-reference, and
upper-reference terms per fine bin --- together with the fine-decision
contract scalars (`fine_status`, `fine_num_bins`, `fine_pad_factor`,
`fine_guard_fine_bins`, `fine_p_fa`, `fine_designated_bins`,
`fine_census_excluded_bins`, `decision_contract_json`). Everything derived
from those terms --- the float ratio, the CFAR location, scale, threshold and
mode, the null-bulk exceedance fraction, and the ragged exceedance list --- is
recomputed in post-processing and is not written. Products that violate the
public contract are unsupported and must be regenerated from authoritative
inputs.

## The frame powers are the sums of the fine powers

Each frame power is `P_r = sum_f S_r[f]`, the all-bin sum of the fixed-point
fine spectrum the fine statistic is built from, so the coarse ratio and the
fine statistic are two readings of one spectrum: one measurement path to
validate and no identity to enforce between paths. A row-domain sum
`sum_m |z[n, m]|^2` would agree only approximately, because the rounded
`fxfft256` does not satisfy Parseval exactly (median relative `2.75e-6` on the
archive frames, up to about 2 per cent on near-empty fill frames). The host
applies the coarse rational-half rule in exact integers, because
`P_0 * half_den` does not fit `uint64` at the fine-sum scale.

Product schema v6 and detector contract revision 2
(`coarse_power_definition = sum_of_fxfft256_fine_powers`) state this
definition. The schema v5 products of record store row-domain powers, but they
also store `fine_power_u64` for every frame, so readers form the v6 frame
powers from it and ignore the stored row-domain powers. Re-deriving the
threshold-rule decisions of record that way changes no word, outcome or reason
(`r_hi` moves by at most `1.6e-5` relative on channels 14-36 and `6.1e-5` on
the channel 37 control). Across the 770,474 archive frames the sign of the
excess, and so `reject_mask`, changes on 50 frames (18 of the 40,606 channel 37
frames), every one within `7e-6` of the boundary.

## Deferred work

The following items are outside the minimum detector result and remain
separate tasks:

- tone catalog and intermodulation classification;
- LimeSDR loopback;
- a `K = 256` implementation;
- production integration of additional threshold modes;
- a common-mode power veto; and
- threshold fitting from CANFAR-measured means.

Offline studies can test alternative thresholds after the scan. That does not
change the shipped real-data mask or add a production mode, so the papers
distinguish the recorded detector output from post-hoc sensitivity studies.
