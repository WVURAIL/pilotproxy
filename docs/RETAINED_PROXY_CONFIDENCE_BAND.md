# Retained proxy confidence-band diagnostic

`pilot_proxy.characterization.confidence_band.simultaneous_upper_band` computes
a one-sided studentized event-pairs-bootstrap upper band jointly over supplied
Q16 threshold words and named frame windows. It is a separate analysis: it does
not replace `threshold_rule`'s existing `x + 2 se` guard or change any operating
decision.

## What is bounded

The target is the conditional **signed proxy mean**, with the original frame
weights retained. For a threshold and window, define the contribution of
resampling cluster `b` as

```
A_b = sum(original_frame_weight * signed_excess * keep)
B_b = sum(original_frame_weight * keep)
mu_hat = shelf * sum(A_b) / sum(B_b)
```

The population estimand is the corresponding ratio of expected cluster
contributions. Default clusters are observing events. When the input weights
give every observing event equal total mass, the target retains that weighting;
it does not give every retained event equal mass. Events with no kept frames
contribute zero to both sums and remain members of the resampling population.
The score and excess may depend on the same frame. This is part of the
conditional proxy estimand, not a requirement that the two be independent.

A statistical band for this mean is not a calibration of true injected
retained power or the physical sky shelf. Those interpretations additionally
depend on the pilot-to-shelf transfer, saturation, leakage, propagation and
instrument state. Neither bootstrap resampling nor a larger critical value
supplies an unknown physical calibration allowance.

## Joint resampling and upper-tail direction

Each replicate draws as many clusters as the original sample, with replacement
and equal selection probability. The same multiplicities `m_b` are used in
every window, including overlapping full-era and last-year windows. Original
frame weights are never re-equalized within a window or temporal block.

For a replicate, the mean and standard error are

```
mu_star = shelf * sum(m_b A_b) / sum(m_b B_b)
se_star = shelf * sqrt(sum(m_b (A_b - (mu_star / shelf) B_b)^2))
                 / sum(m_b B_b)
```

The multiplicity in the squared residual sum is `m_b`, not `m_b^2`: repeated
copies are repeated sampling units. The original standard error uses every
`m_b = 1`, matching the existing event-cluster convention without an additional
`G/(G-1)` finite-sample variance factor.

For each replicate, the upper-band pivot is

```
max over eligible words and windows: (mu_hat - mu_star) / se_star
```

Its requested quantile uses NumPy's `higher` order-statistic convention. The
signed band is `mu_hat + critical * se_hat`. Only the final reported `upper`
field applies `max(0, signed_upper)`; the mean, error, pivot and `signed_upper`
retain their signs. This direction matters for asymmetric bootstrap laws.

## Support and failures

Eligibility is fixed once using the original data. The defaults require:

- At least 30 kept resampling clusters.
- Effective retained support `(sum B_b)^2 / sum(B_b^2)` of at least 20.
- Positive retained mass and a finite, positive standard error.

These are declared engineering support floors, not coverage theorems. They
count clusters, not frames. With `cluster_labels`, the API's `kept_events` and
`min_kept_events` refer to resampling clusters; `kept_clusters` is an explicit
output alias. Unsupported points receive infinite upper limits.

The eligible family is not reselected in bootstrap replicates. If any member
has zero retained mass, undefined or zero error, an overflowing pivot, or
cancellation-dominated variance in a replicate, that replicate's joint maximum
is positive infinity. Failed replicates are retained in the quantile. If its
critical value is infinite, the entire eligible family is unavailable. Failure
counts and the number of completed replicates are returned explicitly.

## Grid selection and dependence

The supplied words remain fixed during resampling. For a predeclared grid, the
family is exactly the eligible words and windows supplied in that call. A
threshold chosen from that grid is part of the same joint analysis; this does
not justify extrapolation to omitted words or additional channels.

When the caller instead supplies all distinct observed score words, the result
is an **empirical-staircase simultaneous-bootstrap diagnostic**. The grid was
not predeclared. Any coverage validation must reproduce that grid-building and
selection procedure on each independent dataset; fixed-grid validation alone
does not qualify it. Era definitions, anchors, profiles, weighting rules and
window construction also remain frozen during resampling. If their selection
is part of a desired repeated-sample claim, that selection needs its own
matching validation.

Event-pairs resampling assumes independent, sufficiently exchangeable events.
It does not automatically cover common receiver changes, seasonal drift or
cross-event correlation. Optional integer `cluster_labels` may group complete
original events into calendar-day or longer temporal blocks, without changing
their original frame weights. An original event cannot be split across two
clusters. Changing the grouping is a dependence-sensitivity analysis; it does
not by itself establish independence of the larger blocks.

Before describing a result as a nominal-confidence bound, check repeated-sample
coverage against an appropriate known-target model, including the actual
support, grid and threshold-selection procedure. Independent datasets are the
coverage trials. Dependent points along a single threshold staircase are not
coverage trials. Existing frozen injection products can also test agreement
with known injected retained power, within their recorded conditions, without
turning that test into a universal physical bound.

## API and implementation

```python
result = simultaneous_upper_band(
    frames,                         # threshold_rule.ChannelFrames
    words,                          # positive, sorted, unique np.uint64 words
    windows={"full": full, "last": last},  # boolean frame masks
    cluster_labels=None,             # default: frames.event
    confidence=0.95,
    bootstrap_replicates=999,
    seed=20261016,
    min_kept_events=30,
    min_effective_events=20,
)
```

Each window returns the signed mean, standard error, eligibility/status,
signed and nonnegative upper limits, kept frame/cluster counts, effective
support and original retained weight. Global output includes the critical
value, every joint pivot, settings, assumptions and failure diagnostics. The
arrays and nonfinite values are intentional; reporting tools must preserve
their meanings when serializing rather than silently converting a refusal to
a successful bound.

The implementation aggregates sparse event/entry-word contributions and forms
increments of each event's squared prefixes. Multiplicity-weighted cumulative
sums then recompute each replicate's moments. It allocates no dense
event-by-threshold matrix or replicate-by-event-by-threshold tensor. A common
weight normalization and per-window location centering improve arithmetic
without changing the estimator. `batch_size` controls only the storage of
multinomial count draws and does not change the seeded answer.

Tests compare all means, errors and bootstrap pivots with an independent
event-copy oracle; they cover asymmetric upper-tail direction, overlapping
windows, unequal cluster mass, zero-kept events, failed replicates, support
refusals, clipping, numerical cancellation, malformed input and reproducibility.

## Reproducible command-line runs

The two drivers insert this checkout's `src` directory before importing the
implementation. Run them from the canonical `pilotproxy` checkout, with NumPy
and SciPy available. The examples below use the existing local interpreter
and evidence paths; change paths for another installation. Keep the complete
output directories, input receipts and source revision together.

Record the runtime before a run. A seed and source hash alone do not identify
the numerical environment. The validator records NumPy in its summary, but
neither driver automatically records the full Python/SciPy/platform receipt.
Use a new receipt for every runtime used, including an interrupted run resumed
on another platform; retain all receipts rather than overwriting the original.

```bash
cd /home/djg/rail/pilotproxy
proxy_python=/home/djg/rail/venvs/archive-local/bin/python
proxy_replay=/home/djg/rail/output/proxy-confidence-replay
mkdir -p "$proxy_replay"
"$proxy_python" - "$proxy_replay/runtime.json" <<'PY'
import contextlib
import io
import json
import platform
import sys
from pathlib import Path
import numpy as np
import scipy

configuration = io.StringIO()
with contextlib.redirect_stdout(configuration):
    np.show_config()
receipt = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "numpy_configuration": configuration.getvalue(),
}
Path(sys.argv[1]).write_text(json.dumps(receipt, indent=2) + "\n")
PY
```

### Archive report and frozen-input replay

Create the archive diagnostic from the existing score products:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --scores /home/djg/rail/output/threshold-scores-refit-2026-10-14/scores \
  --published-decisions /home/djg/rail/dtv-paper2-bao-bound/data/decisions.json \
  --output "$proxy_replay/archive" \
  --replicates 1999 --seed 2026101601 --modes event day week
```

The default channel list is 14 through 37; `--channels 14 35 37`, for example,
requests a smaller diagnostic in a separate output directory. The two joint
windows are the full frozen era and frames within `LAST_YEAR_DAYS` of its last
frame. Confidence 0.95 and support floors 30/20 are fixed in this driver.
Day and week modes use one-day and seven-day UTC bins anchored at the Unix
epoch. Each whole event belongs to the bin of its earliest frame. Mode order
also enters the bootstrap seed, so retain the recorded order when reproducing
a run. The modes are separate sensitivity analyses, not a joint confidence
family across groupings or channels.

Each `scores/chN.npz` requires its `chN.json` metadata and the original product
referenced by that metadata for the live loader. The report records score-file
and metadata hashes and freezes the exact analysis arrays in
`inputs/chN.npz`, with a companion JSON checksum/provenance receipt. The live
loader checks original archive size; it does **not** verify the original
archive SHA256. That limitation is explicit in the receipt.

An offline replay uses those frozen arrays and checks their stored SHA256
before loading them:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --frozen-inputs "$proxy_replay/archive/inputs" \
  --published-decisions /home/djg/rail/dtv-paper2-bao-bound/data/decisions.json \
  --output "$proxy_replay/archive-offline" \
  --replicates 1999 --seed 2026101601 --modes event day week
```

`--scores` and `--frozen-inputs` are mutually exclusive. Offline replay needs
the frozen NPZ/JSON pairs, this implementation and its policy data; it does
not need the original score or archive files. The optional
`--published-decisions` file checks reproduction of the accepted DTV words
(channel 37 is the radio astronomy control). Preserve it to reproduce that
check. A moved input path can change receipt identity without changing the
arrays or numerical answer.

Outputs include `protocol.json`, `inputs/`, a JSON report and compressed NPZ
arrays for every channel/mode, and `summary.json`. The NPZ files preserve the
full grid, eligibility, upper limits and bootstrap pivots. JSON nonfinite
numbers become `null`; inspect status/failure fields and the NPZ arrays to
distinguish unavailable limits from finite results. Reports hash the core,
driver, existing rule, policy inputs, frozen arrays and result arrays, and
record resolved policy constants. Reusing an output directory verifies the
protocol and each existing result's identities/checksum; mismatches refuse
reuse. Use a new directory after changing code, protocol or inputs.

The original guard is recomputed and remains the operative result. The
`proxy_band_substitution_diagnostic` is only a sensitivity calculation through
the unchanged rule API. Its `se` input encodes half the nonnegative difference
between the band and the clipped proxy mean, so the API's `2 se` term supplies
the band margin. This encoded value is **not a standard error**; if the band
falls below that mean, the substitution conservatively uses the mean. The
report does not write accepted detector words or modify the guard.

### Independent-dataset validation

Run the declared seven-scenario protocol, including existing campaign products:

```bash
"$proxy_python" tools/validate_proxy_confidence_band.py \
  --output "$proxy_replay/validation" \
  --replicates 400 --bootstrap-replicates 399 --seed 20261016 \
  --campaign /home/djg/rail/results/fine_detector_validation_2026-09-09/campaign
```

Here `--replicates` counts independent datasets per scenario; it is distinct
from `--bootstrap-replicates` within each dataset. The fixed protocol uses
320 events per analytic dataset and 160 rows per empirical dataset, full and
final-half windows, confidence 0.95 and support floors 30/20. It reconstructs
the observed threshold staircase, support eligibility and diagnostic knee/
threshold selection in every dataset. The selection budget is fixed from the
declared population target before dataset draws; this tests a specified
selection procedure, not the entire archive policy or profile-fitting process.

The five analytic scenarios are `null_gaussian`, `independent_gaussian`,
`coupled_unequal`, `correlated_wrong_unit` and `correlated_blocks`. They cover
known conditional proxy targets, unequal event sizes and score/proxy coupling.
The last two share a Gaussian disturbance across four events and compare
incorrect event-level resampling with the corresponding block grouping. The
wrong-unit case deliberately violates the independent-cluster assumption.

The empirical frontier population comes from the checked-in
`docs/evidence/mask_residual_frontier_2026-09-07/study_config.json` and
`frames/off.npz` / `frames/on_m55p0.npz`. With `--campaign`, the driver also
reads the campaign plan, its referenced cache plan and 16-trial evaluation
shards for channel/case pairs (14, 0), (14, 4) and (35, 0), at 2,048 streams.
It verifies cache/shard checksums and shard identities, reduces 512 trial IDs
per case, and uses the available levels nearest requested coordinates
-55, -50 and -44 dB. The empirical campaign population uses channel 14,
case 0, at the middle selected level. These populations retain paired score
and proxy rows; independently resampling them creates known-target statistical
datasets, not new physical waveform observations.

`campaign_diagnostic.json` compares proxy means with the **requested model
data-shelf coordinate**. That coordinate is not measured total digital input
INR, and the difference is not an isolated estimate of selection bias. Reused
trial IDs pair cells and levels and do not add independent trials. Omitting
`--campaign` runs six scenarios and changes the family size and protocol.
This driver has no `--frozen-inputs` option: direct replay needs the referenced
fixtures, campaign/cache products when requested, and a Git checkout for the
revision receipt. An external portable replay must preserve the exact joint
populations and separately record any I/O substitutions and their hashes.

The driver writes `protocol.json` before outcomes, atomic per-scenario
checkpoints every ten datasets, per-scenario results, optional campaign
diagnostics and `summary.json`. Checkpoints resume only when their protocol
hash matches. Preserve input checksums and verify them before resuming; a
matching protocol alone does not establish unchanged empirical inputs. The
summary records source/fixture hashes, the checkout commit and NumPy version.

For each scenario, `qualified95` requires both zero unavailable datasets and
a one-sided Clopper-Pearson upper limit on the joint miss probability of at
most 0.05. The tail probability is `0.05 / number_of_scenarios`, adjusting the
qualification statements across the declared scenario family. Raw observed
coverage alone does not pass this gate. Knee/selected-point misses and
unavailable counts are reported separately; refusals must not be presented as
successful finite bounds. Any qualification is conditional on that declared
population. It does not certify physical shelf power, arbitrary archive
dependence, profile selection, or a simultaneous statement across channels.
These instructions specify the analysis and its gate; they make no claim that
the completed validation passes it. The existing guard remains unchanged.
