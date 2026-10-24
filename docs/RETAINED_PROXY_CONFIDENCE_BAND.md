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
and SciPy available. Replace the `/path/to/` placeholders with your interpreter,
inputs and a new output directory. Keep the complete
output directories, input receipts and source revision together.

Record the runtime before a run. A seed and source hash alone do not identify
the numerical environment. The validator records NumPy in its summary, but
neither driver automatically records the full Python/SciPy/platform receipt.
Use a new receipt for every runtime used, including an interrupted run resumed
on another platform; retain all receipts rather than overwriting the original.

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

Create the archive diagnostic from the existing score products:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --scores /path/to/threshold-scores/scores \
  --published-decisions /path/to/dtv-mnras/data/decisions.json \
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
  --published-decisions /path/to/dtv-mnras/data/decisions.json \
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
  --campaign /path/to/fine_detector_validation_2026-09-09/campaign
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


## Separately calibrated model margin

`tools/calibrate_proxy_confidence_band.py` adds a conservative, explicitly
model-conditional option. It leaves the event/block estimator, original event
weights, joint windows, support rules and bootstrap algorithm unchanged. It
also leaves the production `x + 2 se` guard unchanged. The original seven-model
validation is retained: 18, 23, 20, 94, 18, 25 and 21 joint misses among 400
datasets, respectively, in the scenario order above with the two empirical
models last. All were finite, but none passed the stricter family-adjusted
qualification gate. The assumption-matched observed miss rates of 4.5–6.25%
do not, by themselves, demonstrate a numerical defect. The wrong-unit control
had a substantially larger 23.5% miss rate.

The new procedure uses **independent calibration and validation datasets**.
For each calibration dataset, let `c` be its original bootstrap critical value
and let `mu` be the known population conditional proxy mean. Its scalar score is

```
R = max(0, max over eligible words/windows ((mu - mu_hat) / se_hat))
    / max(c, 1)
```

The score includes the complete observed-staircase procedure, overlapping
windows, support selection and finite-bootstrap randomness. A multiplier `M`
then gives the upper band

```
mu_hat + M * max(c, 1) * se_hat
```

The positive critical floor makes widening monotone even when a bootstrap
critical value is negative. `widen_upper_band` preserves raw means, standard
errors and bootstrap pivots, clips only the final nonnegative branch, rejects
multipliers below one and refuses a nonfinite widened critical value. An
unavailable original band remains unavailable.

For each of the six assumption-matched models, 400 independent calibration
scores determine a one-sided nonparametric tolerance limit. The declared
content is 0.975. The smallest order rank `k` satisfying

```
P[Binomial(400, 0.975) >= k] <= 0.05 / 6
```

is **398** (one-based): its exact tail is 0.002547621772282387; rank 397 gives
0.009683002762933034 and does not meet the rule. The model's multiplier is the
398th ascending score, floored at one. The common multiplier is the maximum
across the six models. No failed or unavailable calibration dataset is dropped:
any such dataset refuses finite calibration.

For independent, identically distributed complete-dataset scores from each
specified model, this order-statistic construction gives at least 95%
simultaneous calibration confidence that the resulting multiplier covers at
least 97.5% of future datasets from **each of those six models**. Bonferroni
controls the six statements; independence across the six calibrations is not
needed. Ties make the order-statistic guarantee conservative. This is a
statement about the score distribution under the declared simulation laws,
including their sample sizes and procedure. It is not a distribution-free
claim about arbitrary archive data or physical DTV power. The distinction
between tolerance content and confidence is described in the
[NIST tolerance-limit reference](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/tolelimi.htm);
the order-statistic approach originates in
[Wilks (1941)](https://doi.org/10.1214/aoms/1177731788).

The calibration namespace is fixed at seed 20261016031. After the multiplier is
frozen, 400 fresh datasets per scenario use seed 20261016032, with distinct
hashed dataset/bootstrap purposes. The validation gate is unchanged: no
unavailable datasets and the one-sided exact binomial upper miss limit at
`alpha = .05/7` must be at most .05 for **all six matched models**. The seventh
scenario, incorrect event resampling under four-event dependence, is excluded
from calibration and reported as a negative control. Its lower miss limit
above .05 establishes assumption failure when observed; it does not trigger
retuning if the conservative margin happens to mask that failure. No seeds,
model populations or multiplier are replaced after held-out outcomes.

### Calibration and held-out replay

The compact frozen population package contains `empirical_populations.npz` and
`population_receipt.json`; the driver verifies the NPZ SHA256 before writing
output. Its score/proxy rows are the exact finite populations extracted by the
original validator, with their source receipts retained separately. This avoids
re-reading large waveform products and does not create new physical trials.

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

Defaults fix 400 datasets per phase/model and 399 bootstrap replicates. The
protocol, source hashes, population identity, both seed namespaces, support
rules and success criterion are written before any outcomes. Runtime receipts
record Python, NumPy, SciPy and platform; every record identifies its runtime,
replicate index and dataset/bootstrap seeds. The per-record runtime identity hashes
the compact, sorted JSON representation of the runtime object; the evidence
manifest separately hashes the receipt file bytes. Checkpoints every ten datasets
resume only with the same protocol and ordered seed identities. Validation
rechecks calibration-file hashes and recomputes all six order statistics from
complete saved records, rather than trusting a printed multiplier.

Outputs are `protocol.json`, runtime receipts, six `calibrate-*.json`
checkpoints, `calibration.json`, seven `validate-*.json` checkpoints,
`summary.json` and `heldout_scenarios.csv`. Preserve all records, optionally in
a lossless compressed archive, plus their hashes. Record the eventual source
commit separately; the base revision identifies the pre-change checkout, while
the frozen source hashes identify the actual implementation used. A different
source, protocol, or input needs a new output directory. Exact numerical replay
also depends on the recorded numerical runtime. Use a new output directory for
a different platform: the inherited JSON writer uses native newline bytes, so
cross-platform protocol rewriting can change its byte hash and refuse resume.
The completed experiment uses one recorded native Windows runtime throughout.

The archive reporter accepts the verified artifact explicitly:

```bash
"$proxy_python" tools/report_proxy_confidence_band.py \
  --frozen-inputs /path/to/previous/archive/inputs \
  --output "$proxy_band_run/archive-diagnostic" \
  --channels 37 --modes event --replicates 399 \
  --model-calibration "$proxy_band_run/calibration.json"
```

It refuses a different source or bootstrap replicate count and recomputes the
calibration artifact from its saved records before processing archive data.
Applying this multiplier to an archive remains an unqualified **sensitivity
diagnostic**: matching the model's independence, population and selection
assumptions has not been established for that archive. It changes neither
accepted words nor the production sampling guard and supplies no physical
transfer-calibration allowance.


### Held-out result

The frozen multiplier is **1.5651996017565912**, selected by the correctly
blocked dependence model's calibration tolerance limit. All 2,400 calibration
datasets and 2,800 held-out datasets produced finite available bands. All six
assumption-matched models passed the unchanged qualification gate:

| Held-out model | Misses / 400 | Family-adjusted upper miss limit |
| --- | ---: | ---: |
| Null Gaussian | 0 | 1.228% |
| Independent Gaussian | 1 | 1.743% |
| Coupled score/proxy, unequal event sizes | 0 | 1.228% |
| Correlated events, correct blocks | 2 | 2.189% |
| Frozen empirical frontier | 0 | 1.228% |
| Frozen empirical campaign | 2 | 2.189% |

The wrong-unit negative control had 19/400 misses (4.75%), with lower and
upper limits 2.524% and 8.012%. It **failed qualification**, but the widened
margin did not establish that its miss probability exceeds 5%. Thus this run
supports the six declared model results while its negative control is not a
successful detector of nominal undercoverage after widening. No procedure,
seed or multiplier was changed in response. Before widening, the control had
94/400 misses.

These are joint eligible-staircase/window results. Diagnostic selected points
exist in only a subset of datasets; their separate counts remain in the
summary and must not be described as 400 successful operating selections.
The complete artifact is kept with the private evidence: `summary.json`, `heldout_scenarios.csv`, `complete_records.zip`, the
raw records, runtime/test receipts and `MANIFEST.json`. The protocol SHA256 is
`2b83b2901c4d1ebc1869f07b987dede01ab85a6d0eb2893af6dc7cd06b7f0e24`;
the calibrated artifact SHA256 is
`3fe6f4625dd0d45f01cc1c62922eedba376d1fcb3ba7a722429d0c053759a319`.
The recorded runtime was Python 3.12.14, NumPy 2.5.4 and SciPy 1.18.1 on Windows.
A no-work resume verified all protocol, calibration, validation and summary
bytes unchanged. An offline channel-37 archive replay checked the opt-in
reporting path; it remains an archive diagnostic, not a model qualification.
