# Retained proxy confidence-band diagnostic

The purpose of
`pilot_proxy.characterization.confidence_band.simultaneous_upper_band` is to
compute a one-sided upper band on the signed proxy mean jointly over the
supplied Q16 threshold words and named frame windows. The band comes from a
studentized event-pairs bootstrap. As a diagnostic, it leaves the existing `x + 2 se` guard of
`threshold_rule` and every operating decision unchanged.

## What is bounded

The bounded quantity is the conditional signed proxy mean, with the original
frame weights kept. For a threshold and a window, the contribution of resampling
cluster `b` is defined as

```
A_b = sum(original_frame_weight * signed_excess * keep)
B_b = sum(original_frame_weight * keep)
mu_hat = shelf * sum(A_b) / sum(B_b)
```

The population estimand is the corresponding ratio of the expected cluster
contributions. By default, the clusters are the observing events. When the input
weights give every observing event equal total mass, the target keeps that
weighting, so the retained events do not each receive equal mass. Events with no
kept frames contribute zero to both sums and stay in the resampling population.
The score and the excess may both depend on the same frame. This dependence is
part of the conditional proxy estimand, and the two need not be independent.

A statistical band for this mean does not calibrate the true injected retained
power or the physical sky shelf. Those quantities also depend on the
pilot-to-shelf transfer, saturation, leakage, propagation, and the instrument
state. An unknown physical calibration allowance is supplied by neither
bootstrap resampling nor a larger critical value.

## Joint resampling and upper-tail direction

Each replicate draws as many clusters as the original sample has, with
replacement and equal selection probability. In every window, the same
multiplicities `m_b` are used, even where the full-era and last-year windows
overlap. The original frame weights are never re-equalized in a window or
temporal block.

For a replicate, the mean and standard error are

```
mu_star = shelf * sum(m_b A_b) / sum(m_b B_b)
se_star = shelf * sqrt(sum(m_b (A_b - (mu_star / shelf) B_b)^2))
                 / sum(m_b B_b)
```

The squared residual sum is weighted by `m_b`, since each repeated copy is its
own sampling unit. A weight of `m_b^2` would treat the copies as a single unit.
For the original standard error, `m_b = 1` for every cluster, which matches the
existing event-cluster convention, with no added `G/(G-1)` finite-sample
variance factor.

For each replicate, the upper-band pivot is

```
max over eligible words and windows: (mu_hat - mu_star) / se_star
```

For the requested quantile of the pivot, NumPy's `higher` order-statistic
convention is used. The signed band is `mu_hat + critical * se_hat`. Only the
final reported `upper` field applies `max(0, signed_upper)`, and the mean,
error, pivot, and `signed_upper` keep their signs. The sign of the pivot,
`mu_hat - mu_star`, selects the upper tail, and the band depends on this
direction when the bootstrap law is asymmetric.

## Support and failures

Eligibility is fixed once from the original data. By default, a point is
eligible when it has:

- at least 30 kept resampling clusters;
- an effective retained support `(sum B_b)^2 / sum(B_b^2)` of at least 20; and
- positive retained mass and a finite, positive standard error.

These are declared engineering floors on the support, and no coverage theorem
rests on them. They count clusters, never frames. With `cluster_labels`, the API
fields `kept_events` and `min_kept_events` refer to resampling clusters, and
`kept_clusters` is given as an explicit output alias. Unsupported points receive
infinite upper limits.

In the bootstrap replicates, the eligible family is not reselected. If any
member has zero retained mass, an undefined or zero error, an overflowing pivot,
or a variance dominated by cancellation in a replicate, the joint maximum of
that replicate is positive infinity. Failed replicates are kept in the quantile.
If the critical value is infinite, the entire eligible family is unavailable.
Both the failure counts and the number of completed replicates are returned with
the result.

## Grid selection and dependence

During resampling, the supplied words stay fixed. For a predeclared grid, the
family is exactly the eligible words and windows supplied in that call. A
threshold chosen from that grid is covered by the same joint analysis, but words
and channels that were not supplied are not covered.

When the caller supplies all distinct observed score words, the grid was not
predeclared, and the result is known as an empirical-staircase
simultaneous-bootstrap diagnostic. A coverage validation of this diagnostic must
repeat the grid construction and selection on each independent dataset, since a
fixed-grid validation does not qualify it. Era definitions, anchors, profiles,
weighting rules, and window construction also stay frozen during resampling. If
a repeated-sample claim is to cover their selection, that selection needs its
own matching validation.

Event-pairs resampling assumes independent and sufficiently exchangeable events.
Common receiver changes, seasonal drift, and correlation between events are not
covered by this assumption. With the optional integer `cluster_labels`, complete
original events are grouped into calendar-day or longer temporal blocks, and the
original frame weights are kept. An original event cannot be split over two
clusters. A change of grouping is a sensitivity analysis for dependence, and it
does not by itself show that the larger blocks are independent.

A result is described as a bound at nominal confidence only after its
repeated-sample coverage is checked against a suitable model with a known
target, using the actual support, grid, and threshold-selection procedure. Each
coverage trial is an independent dataset, and dependent points along one
threshold staircase do not count as trials. Existing frozen injection products
can also test agreement with a known injected retained power under their
recorded conditions, and such a test holds only for those conditions.

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

Each window returns the signed mean, the standard error, the eligibility and
status, the signed and nonnegative upper limits, the kept frame and cluster
counts, the effective support, and the original retained weight. Across all
windows, the output holds the critical value, every joint pivot, the settings,
the assumptions, and the failure diagnostics. Both the arrays and the nonfinite
values carry meaning, and a reporting tool must preserve that meaning when it
serializes them, so that a refusal is never reported as a successful bound.

In the implementation, the sparse contributions for each event and entry word
are aggregated, and the increments of each event's squared prefixes are formed.
From cumulative sums weighted by the multiplicities, the moments of each
replicate are then recomputed. No dense event-by-threshold matrix or
replicate-by-event-by-threshold tensor is allocated. A common weight
normalization and a location centering in each window improve the arithmetic and
leave the estimator unchanged. `batch_size` sets only the storage of the
multinomial count draws and does not change the seeded answer.

Tests compare all means, errors, and bootstrap pivots with an independent oracle
that copies events. They cover the upper-tail direction for asymmetric laws,
overlapping windows, unequal cluster mass, events with no kept frames, failed
replicates, support refusals, clipping, numerical cancellation, malformed input,
and reproducibility.

## Reproducible command-line runs

The two drivers insert the `src` directory of this checkout before they import
the implementation, so they are run from the canonical `pilotproxy` checkout
with NumPy and SciPy available. In the commands below, `/path/to/` marks the
interpreter, the inputs, and a new output directory. Complete output
directories, input receipts, and the source revision are kept together.

Before a run, the runtime is recorded, because a seed and a source hash do not
determine the numerical environment. The validator records the NumPy version in
its summary, but neither driver records the full receipt of Python, SciPy, and
platform. Each runtime gets its own receipt, and so does an interrupted run
resumed on another platform. All receipts are kept, and the original is never
overwritten. A receipt is written with:

```bash
cd /path/to/pilotproxy
proxy_python=/path/to/venv/bin/python
proxy_replay=/path/to/new-output
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

The archive diagnostic is made from the existing score products:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --scores /path/to/threshold-scores/scores \
  --published-decisions /path/to/dtv-mnras/data/decisions.json \
  --output "$proxy_replay/archive" \
  --replicates 1999 --seed 2026101601 --modes event day week
```

By default, the channel list is 14 through 37, and a smaller diagnostic, such as
`--channels 14 35 37`, is written to its own output directory. The two joint
windows are the full frozen era and the frames no more than `LAST_YEAR_DAYS`
days before its last frame. In this driver, the confidence of 0.95 and the
support floors of 30 and 20 are fixed. The day and week modes use one-day and
seven-day UTC bins anchored at the Unix epoch, and each whole event belongs to
the bin of its earliest frame. Since the mode order also enters the bootstrap
seed, a reproduction keeps the recorded order. Each mode is its own sensitivity
analysis, and the modes do not form a joint confidence family over groupings or
channels.

For the live loader, each `scores/chN.npz` requires its `chN.json` metadata and
the original product named in that metadata. The report records the hashes of
the score file and the metadata and freezes the exact analysis arrays in
`inputs/chN.npz`, beside a JSON receipt of checksums and provenance. The live
loader checks the size of the original archive but not its SHA256, and the
receipt states this limit.

An offline replay uses those frozen arrays and checks their stored SHA256
before loading them:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --frozen-inputs "$proxy_replay/archive/inputs" \
  --published-decisions /path/to/dtv-mnras/data/decisions.json \
  --output "$proxy_replay/archive-offline" \
  --replicates 1999 --seed 2026101601 --modes event day week
```

`--scores` and `--frozen-inputs` are mutually exclusive. An offline replay needs
the frozen NPZ and JSON pairs, this implementation, and its policy data, and it
runs without the original score or archive files. The optional
`--published-decisions` file checks that the accepted DTV words are reproduced
(channel 37 is the radio astronomy control), so the file is kept for that
check. Moving an input path can change the receipt identity while the arrays and
the numerical answer stay the same.

The outputs contain `protocol.json`, `inputs/`, a JSON report and compressed NPZ
arrays for every channel and mode, and `summary.json`. In the NPZ files, the
full grid, the eligibility, the upper limits, and the bootstrap pivots are
preserved. Nonfinite numbers become `null` in JSON, so an unavailable limit is
told apart from a finite result by the status and failure fields and by the NPZ
arrays. Each report hashes the core, the driver, the existing rule, the policy
inputs, the frozen arrays, and the result arrays, and it records the resolved
policy constants. When an output directory is reused, the protocol and the
identities and checksum of each existing result are verified, and any mismatch
refuses the reuse. A change of code, protocol, or inputs needs a new directory.

The original guard is recomputed and remains the operative result. The
`proxy_band_substitution_diagnostic` is a sensitivity calculation through the
unchanged rule API. Its `se` input holds half the nonnegative difference between
the band and the clipped proxy mean, so the `2 se` term of the API supplies the
band margin. This encoded value is therefore not a standard error. If the band
falls below the mean, the substitution uses the mean, which is the conservative
choice. The report writes no accepted detector words and leaves the guard
unchanged.

### Independent-dataset validation

The declared protocol of seven scenarios, with the existing campaign products,
is run with:

```bash
"$proxy_python" tools/validate_proxy_confidence_band.py \
  --output "$proxy_replay/validation" \
  --replicates 400 --bootstrap-replicates 399 --seed 20261016 \
  --campaign /path/to/fine_detector_validation_2026-09-09/campaign
```

Here, `--replicates` counts the independent datasets per scenario, and
`--bootstrap-replicates` counts the bootstrap replicates in each dataset. Under
the fixed protocol, each analytic dataset has 320 events and each empirical
dataset has 160 rows, with full and final-half windows, a confidence of 0.95,
and support floors of 30 and 20. In every dataset, the observed threshold
staircase, the support eligibility, and the diagnostic selection of knee and
threshold are reconstructed. The selection budget is fixed from the declared
population target before the datasets are drawn. Thus, the protocol tests one
specified selection procedure, and the full archive policy and the profile
fitting are not tested.

The five analytic scenarios are `null_gaussian`, `independent_gaussian`,
`coupled_unequal`, `correlated_wrong_unit`, and `correlated_blocks`. They cover
known conditional proxy targets, unequal event sizes, and coupling between score
and proxy. In the last two, four events share a Gaussian disturbance, and
incorrect resampling by event is compared with the corresponding block grouping.
The wrong-unit case violates the independent-cluster assumption on purpose.

The empirical frontier population is taken from the checked-in
`docs/evidence/mask_residual_frontier_2026-09-07/study_config.json` and
`frames/off.npz` / `frames/on_m55p0.npz`. With `--campaign`, the driver also
reads the campaign plan, the cache plan it names, and the 16-trial evaluation
shards for the channel and case pairs (14, 0), (14, 4), and (35, 0), at 2048
streams. After the checksums of the cache and shards and the shard identities
are verified, 512 trial IDs are reduced per case, and the available levels
nearest the requested coordinates of -55, -50, and -44 dB are used. For the
empirical campaign population, channel 14, case 0, is used at the middle
selected level. These populations keep the score and proxy rows paired.
Resampling them independently produces statistical datasets with a known target,
and no new physical waveform observations are made.

`campaign_diagnostic.json` compares the proxy means with the requested model
coordinate of the data shelf. This coordinate is a model input and differs from
the measured total digital input INR, and the difference between the proxy mean
and this coordinate is not an isolated estimate of selection bias. Reused trial IDs pair cells and levels and
add no independent trials. Without `--campaign`, six scenarios are run, and the
family size and protocol change. This driver has no `--frozen-inputs` option. A
direct replay needs the referenced fixtures, the campaign and cache products
when they are requested, and a Git checkout for the revision receipt. A portable
replay elsewhere must preserve the exact joint populations and record each I/O
substitution and its hash.

The driver writes `protocol.json` before any outcome, then atomic checkpoints
for each scenario every ten datasets, the results of each scenario, the
optional campaign diagnostics, and `summary.json`. A checkpoint resumes only
when its protocol hash matches. A matching protocol does not show that the
empirical inputs are unchanged, so the input checksums are kept and verified
before a resume. The summary records the source and fixture hashes, the
checkout commit, and the NumPy version.

For each scenario, `qualified95` requires zero unavailable datasets and a
one-sided Clopper-Pearson upper limit of at most 0.05 on the joint miss
probability. The tail probability is `0.05 / number_of_scenarios`, which adjusts
the qualification statements for the declared family of scenarios. Raw observed
coverage alone does not pass this gate. Misses at the knee and at the selected
points, and the unavailable counts, are reported on their own, and a refusal
must never be presented as a successful finite bound. A qualification holds only
for the declared population. Physical shelf power, arbitrary archive dependence,
profile selection, and a simultaneous statement over channels are not certified
by it. This section specifies the analysis and its gate, and the outcome of the
completed validation is given in the next section.


## Separately calibrated model margin

`tools/calibrate_proxy_confidence_band.py` adds a conservative option that holds
only under its stated models. The event and block estimator, the original event
weights, the joint windows, the support rules, the bootstrap algorithm, and the
production `x + 2 se` guard are all unchanged. In the original validation of
seven models, which is kept, the joint misses in 400 datasets were 18, 23, 20,
94, 18, 25, and 21, in the scenario order above with the two empirical models
last. All bands were finite, but none of the models passed the stricter
family-adjusted qualification gate. For the models that match the assumptions,
the observed miss rates lie between 4.5% and 6.25%, and these rates do not by
themselves show a numerical defect. With 94 misses, the wrong-unit control had a
far larger miss rate of 23.5%.

Independent datasets are used for calibration and for validation. For each
calibration dataset, `c` denotes its original bootstrap critical value, and `mu`
denotes the known conditional proxy mean of the population. Its scalar score is

```
R = max(0, max over eligible words/windows ((mu - mu_hat) / se_hat))
    / max(c, 1)
```

The score covers the complete observed-staircase procedure, the overlapping
windows, the support selection, and the randomness of the finite bootstrap. A
multiplier `M` then gives the upper band

```
mu_hat + M * max(c, 1) * se_hat
```

With the floor of one on the critical value, the widening stays monotone even
when a bootstrap critical value is negative. `widen_upper_band` preserves the
raw means, standard errors, and bootstrap pivots, clips only the final
nonnegative branch, rejects multipliers below one, and refuses a nonfinite
widened critical value. An original band that is unavailable stays unavailable.

For each of the six models that match the assumptions, 400 independent
calibration scores determine a one-sided nonparametric tolerance limit with a
declared content of 0.975. The smallest order rank `k` that satisfies

```
P[Binomial(400, 0.975) >= k] <= 0.05 / 6
```

is 398 (one-based), with an exact tail of 0.002547621772282387. Rank 397 gives
0.009683002762933034 and does not meet the rule. For each model, the multiplier
is its 398th ascending score, floored at one, and the common multiplier is the
maximum over the six models. No failed or unavailable calibration dataset is
dropped, and any such dataset refuses a finite calibration.

When the complete-dataset scores of each specified model are independent and
identically distributed, this order-statistic construction gives at least 95%
simultaneous confidence that the resulting multiplier covers at least 97.5% of
future datasets from each of the six models. The Bonferroni correction controls
the six statements, so the six calibrations need not be independent of each
other. Ties make the order-statistic guarantee conservative. Such a guarantee
concerns the score distribution under the declared simulation laws, with their
sample sizes and procedure. It makes no distribution-free claim about arbitrary
archive data or physical DTV power. Tolerance content and confidence are
distinguished in the [NIST tolerance-limit
reference](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm),
and the order-statistic method is due to [Wilks
(1941)](https://doi.org/10.1214/aoms/1177731788).

For calibration, the seed namespace is fixed at 20261016031. After the
multiplier is frozen, 400 fresh datasets per scenario are drawn with seed
20261016032, with distinct hashed purposes for the dataset and bootstrap draws.
Under the unchanged validation gate, all six matched models must have no
unavailable datasets and a one-sided exact binomial upper miss limit of at most
.05 at `alpha = .05/7`. The seventh scenario, incorrect resampling by event
under four-event dependence, is kept out of the calibration and reported as a
negative control. When its lower miss limit is above .05, the assumption failure
is confirmed. If the conservative margin happens to hide that failure, no
retuning follows. No seed, model population, or multiplier is replaced after the
held-out outcomes are known.

### Calibration and held-out replay

The compact frozen population package holds `empirical_populations.npz` and
`population_receipt.json`, and the driver verifies the SHA256 of the NPZ before
writing any output. Its score and proxy rows are the exact finite populations
extracted by the original validator, and their source receipts are kept on
their own. With this package, the large waveform products are not re-read,
and no new physical trials are created.

```bash
cd /path/to/pilotproxy
export PYTHONPATH="$PWD/src"
proxy_python=/path/to/venv/bin/python
proxy_band_run=/path/to/new-output/confidence
proxy_populations=/path/to/proxy-confidence-validation-compact

"$proxy_python" tools/calibrate_proxy_confidence_band.py \
  --phase calibrate --output "$proxy_band_run" \
  --populations "$proxy_populations" \
  --base-revision 0adde4b775fd8428b790b64e3494205072da51c3

"$proxy_python" tools/calibrate_proxy_confidence_band.py \
  --phase validate --output "$proxy_band_run" \
  --populations "$proxy_populations" \
  --base-revision 0adde4b775fd8428b790b64e3494205072da51c3
```

By default, 400 datasets are drawn per phase and model, with 399 bootstrap
replicates. Before any outcome, the protocol, source hashes, population
identity, both seed namespaces, support rules, and success criterion are
written. The runtime receipts record Python, NumPy, SciPy, and the platform, and
every record names its runtime, replicate index, and dataset and bootstrap
seeds. For each record, the runtime identity is the hash of the compact, sorted
JSON form of the runtime object, and the evidence manifest also hashes the bytes
of the receipt file. Checkpoints are written every ten datasets and resume only
with the same protocol and the same ordered seed identities. In the validation
phase, the hashes of the calibration file are rechecked and all six order
statistics are recomputed from the complete saved records, so a printed
multiplier is never trusted.

The outputs are `protocol.json`, the runtime receipts, six `calibrate-*.json`
checkpoints, `calibration.json`, seven `validate-*.json` checkpoints,
`summary.json`, and `heldout_scenarios.csv`. All records and their hashes are
preserved, optionally in a lossless compressed archive. Since the base revision
names the checkout before the change, and the frozen source hashes name the
implementation that actually ran, the final source commit is recorded on its
own. A different source, protocol, or input needs a new output directory. Exact
numerical replay also depends on the recorded numerical runtime. A different
platform needs a new output directory as well, because the inherited JSON writer
uses native newline bytes, and a protocol rewritten on another platform can
change its byte hash and refuse the resume. For the completed experiment, one
recorded native Windows runtime was used throughout.

The archive reporter accepts the verified artifact explicitly:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --frozen-inputs /path/to/previous/archive/inputs \
  --output "$proxy_band_run/archive-diagnostic" \
  --channels 37 --modes event --replicates 399 \
  --model-calibration "$proxy_band_run/calibration.json"
```

Before it processes archive data, the reporter recomputes the calibration
artifact from its saved records, and it refuses a different source or bootstrap
replicate count. Applied to an archive, the multiplier gives an unqualified
sensitivity diagnostic, because the archive has not been shown to meet the
model's independence, population, and selection assumptions. Neither the
accepted words nor the production sampling guard is changed, and no physical
transfer-calibration allowance is supplied.


### Held-out result

The frozen multiplier is 1.5651996017565912, set by the calibration tolerance
limit of the dependence model with correct blocks. All 2400 calibration datasets
and 2800 held-out datasets produced finite available bands. All six models that
match the assumptions passed the unchanged qualification gate:

| Held-out model | Misses / 400 | Family-adjusted upper miss limit |
| --- | ---: | ---: |
| Null Gaussian | 0 | 1.228% |
| Independent Gaussian | 1 | 1.743% |
| Coupled score/proxy, unequal event sizes | 0 | 1.228% |
| Correlated events, correct blocks | 2 | 2.189% |
| Frozen empirical frontier | 0 | 1.228% |
| Frozen empirical campaign | 2 | 2.189% |

For the wrong-unit negative control, 19 of 400 datasets missed (4.75%), with
lower and upper limits of 2.524% and 8.012%. It failed qualification, but with
the widened margin its lower limit is below 5%, so the run does not show that
its miss probability exceeds 5%. Thus, this run supports the six declared model
results, but after widening the negative control no longer detects the nominal
undercoverage. Before widening, the control had 94 misses in 400. No procedure,
seed, or multiplier was changed in response.

These results are joint over the eligible staircase and the windows. Diagnostic
selected points exist in only some of the datasets, and their own counts are
kept in the summary. They must not be described as 400 successful operating
selections. The complete artifact is kept with the private evidence:
`summary.json`, `heldout_scenarios.csv`, `complete_records.zip`, the raw
records, the runtime and test receipts, and `MANIFEST.json`. The protocol SHA256
is `2b83b2901c4d1ebc1869f07b987dede01ab85a6d0eb2893af6dc7cd06b7f0e24`, and the
SHA256 of the calibrated artifact is
`3fe6f4625dd0d45f01cc1c62922eedba376d1fcb3ba7a722429d0c053759a319`. On Windows,
the recorded runtime was Python 3.12.14, NumPy 2.5.4, and SciPy 1.18.1. A resume
with no remaining work verified that all protocol, calibration, validation, and
summary bytes were unchanged. An offline archive replay of channel 37 checked
the opt-in reporting path. It is an archive diagnostic only and qualifies no
model.
