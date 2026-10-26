# From baseband to a conditional threshold

This page gives the workflow for a new run. Frozen studies and their commands
are kept in `docs/evidence/` and in the paper repositories, and a new run does
not take its defaults from them. [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) maps
each scientific step to its code and tests.

## 1. Reduce baseband

The CUDA detector projects each input stream onto the target weight and two
reference weights. Next, the 128 windows are transformed, and the squared
amplitudes are summed over the inputs. One per-pilot product with exact fine
powers is written for each pilot. In schema v6, the frame powers are the sums of
those fine powers, and v5 archives are read through the same sums (see
[PRODUCT_SCHEMA.md](PRODUCT_SCHEMA.md)).

Each run follows either the [local processing](LOCAL_PROCESSING.md) guide or the
[CANFAR](CANFAR_RUNBOOK.md) runbook. Receiver profiles supply the packing,
Nyquist zone, input order, and block size. The [CHIME frequency
plan](../projects/chime_atsc/frequency_plan.yaml) lists the science channels and
the channel-37 control.

## 2. Export frames and determine eras

`characterize histogram-frames` preserves frame coordinates, health selection
and event membership. `characterize fine-histogram-frames` adds exact Q16 keep
boundaries. Its defaults reproduce the baseline marker geometry. A change to
`--designated-half-width` or `--marker-guard-bins` requires regenerated scores
and a new calibration, since a cache made with one geometry is not valid for
another.

`characterize era-boundaries --line-identity` determines the valid current era.
The paper uses the line-identity era list in
[projects/chime_atsc/eras/line_identity_2026-10-17](../projects/chime_atsc/eras/line_identity_2026-10-17).
Other lists in `projects/chime_atsc/eras/` are provenance records.
[POWER_ERAS.md](POWER_ERAS.md) describes a second tool, which proposes power
eras.

## 3. Read the score cache and knee

A cache contains each frame's exact keep word, support, event and weight.
`characterization.threshold_rule.load_frames` authenticates the stored v1 knee
and recomputes the v2 knee. Rule v2 maximizes the absolute difference between
the equal-event CDF and its chord in log word, so it can choose either bend. As
an operating heuristic, the knee sets no false-alarm rate and is not an optimum
in observing time.

The retained power is the average of the signed excesses, with the nonnegative
floor applied to that mean only and never to individual frames. Its uncertainty
is clustered by source event, and unsupported frames stay in the retention
denominator.

## 4. Assign a conditional threshold

From the repository root, with the package installed:

```sh
python tools/threshold_rule.py --scores <score-cache> --out <new-output> \
    --policy-budget <policy-budget.json> \
    --exclude "33=pilot off the detector's frequency"
```

The policy budget file holds a `budget` mapping from channel to nominal-strength
cap. In the paper, the budget adds channel 19's containment allowance to the
finite-amplitude checks, and a reproduction of the paper uses its complete
`data/policy_budget.json`. A new input population needs its own justified
budget. `--finite-budget` is an alias of `--policy-budget`.

The tool tests `a = (gamma x_hi)^2 N G` against RFIsher's combined-distance
response, at the knee first and then below it. Both the full era and its recent
year must pass. `gamma` is a stack-to-proxy amplitude ratio, and
[TERMINOLOGY.md](TERMINOLOGY.md) explains how it differs from a coherent
fraction. `G` is a model-family budget for an unmeasured temporal cross-moment.
Neither the short captures nor channel 37 determine the cross-day DTV residual.

By default, the calendar cost in years is `1 / kappa`, with a 20-year ceiling
and no added duty penalty. The frozen decision files were made with an
operating-duty factor and are reproduced with:

```sh
--calendar-factor 6.574444324363269
```

With this option, both the selection and the plot coordinates are scaled. The
JSON keys `deployed*`, `coherent_fraction`, and `finite_budget` keep their
spelling for existing consumers, and their meanings are given in
[TERMINOLOGY.md](TERMINOLOGY.md).

## 5. Compile a runtime policy

The [policy compiler](CHANNEL_OPERATING_POLICY.md) binds the decisions to the
receiver profile, the frequency IDs, the era, and the calibrated bundle.
Unassigned channels become permanent masks, and an assigned channel needs a
valid profile at its exact word. Channel 37 is kept as a control and is not one
of the DTV science channels.

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json \
    --output-dir <policy-review> --require-complete
```

Compiling the bundle neither activates the detector nor certifies its science
assumptions.

## 6. Integrate and commission

The Kotekan stage `cudaPilotProxyDetector` is in Kotekan's `develop` branch
([kotekan PR 1677](https://github.com/kotekan/kotekan/pull/1677)). It selects
the detector window at run time (64 for CHORD and 128 for CHIME), and the CHIME
include is detect-only. Commissioning follows the [operator
handoff](OPERATOR_HANDOFF.md). First, the consumer revision is inspected, and
the first run is then made with masking disabled. The live run is a later
qualification step.
