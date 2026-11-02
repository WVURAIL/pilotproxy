# Design decisions

The purpose of this file is to record the choices that affect the
interpretation of CHIME DTV products. Each choice is kept with its reason and
its limits, so that a reviewer finds the reasoning in one place.

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
quantization leaves the target and reference norms unequal. By independent
recomputation from the shipped manifest for ATSC channels 14 to 36, we find
that `null_power_ratio` ranges from 0.9853298815 to 1.0111111111 instead of
being exactly 1.

We compare each channel with its own `null_power_ratio` so that the rule is defined by the
shipped weights and can be reproduced from integer powers, with no fitted
operational threshold. Each product records its rule in `mask_rule`, and
products made under the plain `F > 1` rule are read under that rule.

## `K = 128` for CHIME and `K = 64` for CHORD

The compiled detector family and detector-core profile support `K = 64` and
`K = 128`. CHIME keeps `K = 128` because it matches the CHIME receiver profile,
the shipped CHIME manifest, and the regression data. The full and Pathfinder
CHORD profiles select `K = 64`, and their 8192-sample frames still produce the
frozen 128 windows per stream.

`K = 256` is neither implemented nor tested. It would require its own
fixed-point bound, weight bank, contract, and validation set.

`K = 512` is not a candidate.

## Guard and reference terms have distinct names

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

The resolver keeps the requested reference offset when the receiver geometry
permits it. If a reference reaches a circular FFT edge, it wraps. If a
reference would collide with the forbidden coarse-channel DC tone, it moves
away. A target and DC collision is invalid, because moving the target would
change the signal being tested.

In the shipped bank for ATSC channels 14 to 36, DTV 14 is the only adaptive
case, and its lower reference wraps across the coarse-channel edge. No shipped
channel collides with, or shifts around, the forbidden DC tone.

Run products store the reference-placement summary. Validators and later
analyses should read the recorded placement instead of reconstructing it from
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

The `cudaPilotProxyDetector` stage of Kotekan (pull request 1677, in its
`develop` branch) consumes this bundle, and this repository holds no Kotekan
code. A valid bundle shows only that the bundle is well formed, and
commissioning follows [OPERATOR_HANDOFF.md](OPERATOR_HANDOFF.md).

## The deployed decision is the fine designated-set CFAR in the kernel

The archive products were made with the fine decision inactive. A band applies
it only through a compiled operating policy
(`pilot-proxy compile-operating-policy`, `docs/CHANNEL_OPERATING_POLICY.md`),
which carries the threshold words.

The fine front end replaces the logic after the dot product. Instead of
squaring the dot product of each window and summing incoherently over all rows,
the windows are integrated coherently by the padded window-axis FFT of each
feed, and only the feed sum remains incoherent. For W = 128 windows, this
recovers up to 10 log10(sqrt(W)) ~ 10.5 dB of deflection sensitivity over the
coarse reduction, less a scalloping loss bounded at 3.92 dB at the fine-span
edge. The calibrated any-bin CFAR then turns the refined spectrum into a
decision with a pre-registered false-alarm rate.

The fine path was chosen on the measured receiver operating characteristic
(ROC) curves. On the archive, the coarse null bulk and the weak-signal bulk
overlap almost completely, so each coarse operating point either discards about
half of the clean time or misses most weak-channel detections. The
null_power_ratio point holds Pd ~ 1.0 on every channel but discards 48.5% of the
verified-quiet time by construction, and the null-calibrated fixed-Pfa points
lose most of their detections on weak channels. In the table below, the null is
the ch35 off-state (n = 66), and the signal is the on-epochs from 2024 on of the
first three scanned channels.

| Pfa (null quantile) | theta (F/null_power_ratio) | ch36 Pd | ch35 Pd | ch34 Pd |
|---|---|---|---|---|
| 0.10 (P90) | 1.114 | 0.904 | 0.996 | 0.099 |
| 0.05 (P95) | 1.124 | 0.872 | 0.996 | 0.089 |
| 0.015 (null max) | 1.146 | 0.805 | 0.996 | 0.071 |

Here, the null is a transmitter-off epoch inferred from the archive (the ch35
off state), and it is not a verified signal-free population. A false-alarm
limit for deployment is set only on the verified, null-like population of the
band itself.

On the same frames, the fine designated set reaches Pd ~ 0.99 on all three
channels at a measured Pfa of 0.091, with the crude +/-2-bin set and an
uncalibrated CFAR multiplier. We find that the fine axis is the only one that is
both more complete than any calibrated coarse point and cheaper in clean time
than null_power_ratio.

Thus, the mask is formed in the kernel, as one solid kernel.
`PilotProxy_Compute_FusedFineMask_U64` takes the packed samples, the packed
weights, and the per-channel bundle constants, and it emits one mask bit per
aligned frame (1 = reject, and a zero reference forces 0). One block is
launched per stream. The exact int32 row sums of the stream are computed in
shared memory and are never written to global memory. The frozen fxfft256 is
then run in place with cooperative butterflies, whose writes in each stage are
disjoint so that the bits do not depend on scheduling, and the exact fine
powers of the feed sum are accumulated. A last-block epilogue forms the
designated-set decision over the finalized sums, with the completion counter
folded into the mask element (threadfence, then atomic). The row sums are
exposed only through an optional debug tap.
`PilotProxy_Compute_FusedFine_U64` is the same launch without the decision, and
`PilotProxy_Compute_FinePowers_U64` is the staged fine-power entry. The epilogue
is deterministic because it reads order-independent exact integer sums. The
governing rule is bit identity with the frozen reference and the golden
vectors.

The decision arithmetic is frozen as fine decision v1. Its reference is
`src/pilot_proxy/fine_decision.py`, gated by `tests/core/test_fine_decision.py`
against a `fractions.Fraction` oracle. The decision is defined as follows:

- `F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` is held as an exact rational.
- The null bulk is the 256-bit mask of the bundle, built from the
  pre-registered `independent_bin_mask` (independent bins with the guard and
  census bins excluded), intersected with the live denominators.
- The CFAR estimate is the bundle-specified rank of the ascending bulk F2
  values (an order-statistic CFAR). The rank is selected by counting, so the
  selected value is unique under ties, and any representative yields the same
  decision.
- The mask fires if and only if some designated bin (anchor +/- width, modulo
  256, the window convention of the survey) exceeds `multiplier_q16 / 2^16`
  times the rank value.

Comparisons are exact 128-bit and 192-bit integer cross products, fuzz-gated
over 2 million trials against Python big integers. Degenerate denominators
never fire, and a bulk no deeper than the rank forces mask 0.

The CFAR uses a rank instead of a location and scale estimate, for the same
reason that the FFT is fixed-point. Float medians and interpolated quantiles
cannot be made bit-exact across numpy, nvcc, and compiler versions. The rank
statistic, however, is exact and needs no contamination fallback mode, since
order statistics reject the measured 0.29% bulk tail by construction. It can
also be rescored offline from the exact `fine_power_u64` terms of the survey
products (the target, lower-reference, and upper-reference uint64 per fine
bin), because `F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])` is reproduced from them
exactly.

The archive fixes the rank at the median of the intended bulk (`rho = 63`,
zero-based `cfar_rank = 62`, and 60 and 53 on channels 20 and 34) and takes
the threshold-rule word of each channel as its multiplier (see "The recorded
threshold is bundle data" below). The deployment compiler carries those words,
together with the permanent mask of the channels the rule does not deploy,
into the runtime policy ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)).
The operating point (anchor, width, bulk mask, rank, and multiplier) is
runtime-bundle data with provenance, held in the `fine_calibration` block. That
block is exported as `pending_campaign` or `calibrated`, and the bundle
validator checks it in both states. The arithmetic is code. In the archive
product contract, `fine_candidate_decision.active` is false, and `reject_mask`
is the coarse norm-corrected positive-excess decision. Changing that requires a
revision of the schema and decision contract at a survey epoch boundary.

The kernel is verified in two ways. First, a threaded CPU emulation of the
extracted device code is run, with one pthread per CUDA thread and a real
barrier for `__syncthreads`. Its mask, fine, and tap outputs are bit-identical
to the Python reference chain for noise, an injected pilot, a wrapped anchor, a
degenerate bulk, and an unbound tap, and they are stable over repeated
concurrent runs. Second, the on-GPU gates `tests/kernel/test_fused_fine_gpu.py`
and `tests/kernel/test_fused_mask_gpu.py` test reference bit-equality per
frame, injection firing, the device-side degenerate rules, the production form
with no tap, and last-block determinism over 20 repeats, and they report the
cost of the epilogue.

On an A100 with the core 2.3.0 build, the fused production form takes 0.769 ms
per 2048-stream frame, a margin of 55 times against the 41.9 ms cadence. The
mask form takes 0.895 ms, a margin of 47 times or about 2% of the frame
cadence, and the decision epilogue adds 0.127 ms of this.

## The source stamp is memoized per process

`detector_version` embeds a hash of every `*.py` under `src/pilot_proxy`. The
imported code cannot change during a process, so the hash is computed once per
process, at first observation, and reused instead of being reread from disk at
the `begin()` of each channel. A stamp that differs only in the source is
accepted for cross-pilot combine and stacking, which compare only the geometry
tokens. Resume is stricter. It compares the full `detector_version` and aborts
on any change of the source tree or package version, so a new build needs a
clean `--output-dir`. The commit behind a stamp can be recovered by recomputing
the hash at each candidate checkout.

## The stored fine surface is the exact integer terms

Products record the fine axis as `fine_power_u64`, a uint64 array of shape
`(N, 3, fine_num_bins)` with the target, lower-reference, and upper-reference
terms per fine bin, together with the fine-decision contract scalars
(`fine_status`, `fine_num_bins`, `fine_pad_factor`, `fine_guard_fine_bins`,
`fine_p_fa`, `fine_designated_bins`, `fine_census_excluded_bins`,
`decision_contract_json`). Every quantity derived from those terms is
recomputed in post-processing and is never written. These are the float ratio,
the CFAR location, scale, threshold, and mode, the null-bulk exceedance
fraction, and the ragged exceedance list. Products that violate the public
contract are unsupported and must be regenerated from authoritative inputs.

## The frame powers are the sums of the fine powers

Each frame power is `P_r = sum_f S_r[f]`, the all-bin sum of the fixed-point
fine spectrum that the fine statistic is built from. The coarse ratio and the
fine statistic are therefore two readings of one spectrum. This means that only
one measurement path needs validation, and no identity between paths needs to
be enforced. A row-domain sum `sum_m |z[n, m]|^2` would agree only
approximately, because the rounded `fxfft256` does not satisfy Parseval exactly
(a median relative difference of `2.75e-6` on the archive frames, and up to
about 2 percent on near-empty fill frames). The host applies the coarse
rational-half rule in exact integers, because `P_0 * half_den` does not fit
`uint64` at the fine-sum scale.

Product schema v6 and detector contract revision 2
(`coarse_power_definition = sum_of_fxfft256_fine_powers`) state this
definition. The schema v5 products of record store row-domain powers, but they
also store `fine_power_u64` for every frame, so readers form the v6 frame
powers from it and ignore the stored row-domain powers. We find that
re-deriving the threshold-rule decisions of record in this way changes no word,
outcome, or reason. `r_hi` moves by at most `1.6e-5` relative on channels 14 to
36 and `6.1e-5` on the channel 37 control. Over the 770474 archive frames, the
sign of the excess, and with it `reject_mask`, changes on 50 frames (18 of the
40606 channel 37 frames), each no more than `7e-6` from the boundary.

## Deferred work

The following items are not part of the minimum detector result and remain
open tasks:

- tone catalog and intermodulation classification;
- LimeSDR loopback;
- a `K = 256` implementation;
- production integration of additional threshold modes;
- a common-mode power veto; and
- threshold fitting from CANFAR-measured means.

Offline studies can test alternative thresholds after the scan. Such a study
leaves the shipped real-data mask unchanged and adds no production mode, so
the papers distinguish between the recorded detector output and post-hoc
sensitivity studies.
