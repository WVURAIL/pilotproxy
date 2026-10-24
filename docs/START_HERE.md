# From baseband to a conditional threshold

This is the workflow for a new run. Frozen studies and their commands live in
`docs/evidence/` and the paper repositories; they are not defaults for new runs.
[PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) maps each scientific step to its code and tests.

## 1. Reduce baseband

The CUDA detector projects each input stream onto the target and two reference
weights, transforms the 128 windows, and sums squared amplitudes across inputs.
It writes one per-pilot product with exact fine powers. The v6 frame powers are
sums of those fine powers; v5 archives are read through the same sums.
See [PRODUCT_SCHEMA.md](PRODUCT_SCHEMA.md).

Choose [local processing](LOCAL_PROCESSING.md) or [CANFAR](CANFAR_RUNBOOK.md).
Receiver profiles supply packing, Nyquist zone, input order and block size.
The [CHIME frequency plan](../projects/chime_atsc/frequency_plan.yaml) identifies
the science channels and the channel-37 control.

## 2. Export frames and establish eras

`characterize histogram-frames` preserves frame coordinates, health selection
and event membership. `characterize fine-histogram-frames` adds exact Q16 keep
boundaries. Its defaults reproduce the baseline marker geometry; changing
`--designated-half-width` or `--marker-guard-bins` requires regenerated scores
and calibration. A cache made with one geometry is not valid for another.

`characterize era-boundaries --line-identity` determines the valid current era.
The paper uses the line-identity era list in
[projects/chime_atsc/eras/line_identity_2026-10-17](../projects/chime_atsc/eras/line_identity_2026-10-17).
The other lists in `projects/chime_atsc/eras/` are provenance records.
[POWER_ERAS.md](POWER_ERAS.md) describes the separate power-era proposal tool.

## 3. Read the score cache and knee

A cache contains each frame's exact keep word, support, event and weight.
`characterization.threshold_rule.load_frames` authenticates the stored v1 knee
and recomputes the v2 knee. Rule v2 maximizes the **absolute** difference
between the equal-event CDF and its log-word chord. It can choose either bend;
it is not a false-alarm threshold or an observing-time optimum.

The retained power averages signed excesses before applying a nonnegative floor
to the mean. It does not floor individual frames. Its uncertainty is clustered
by source event. The unsupported frames remain in the retention denominator.

## 4. Assign a conditional threshold

From the repository root, with the package installed:

```sh
python tools/threshold_rule.py --scores <score-cache> --out <new-output> \
    --policy-budget <policy-budget.json> \
    --exclude "33=pilot off the detector's frequency"
```

The policy budget file has a `budget` mapping from channel to nominal-strength
cap. The paper combines finite-amplitude checks with channel 19's containment
allowance; use its complete `data/policy_budget.json`. A new input population
needs its own justified budget. `--finite-budget` is an alias of
`--policy-budget`.

The tool tests `a = (gamma x_hi)^2 N G` against RFIsher's combined-distance
response, at the knee first and then below it. Both the full era and its recent
year must pass. `gamma` is a stack-to-proxy amplitude ratio, not a coherent
fraction. `G` is a model-family budget for an unmeasured temporal cross-moment.
Neither the short captures nor channel 37 determine the cross-day DTV residual.

By default, calendar years are `1 / kappa`, with a 20-year ceiling and no added
duty penalty. To reproduce the frozen decision files made with an operating-duty
factor, pass:

```sh
--calendar-factor 6.574444324363269
```

This option scales both selection and plot coordinates. The `deployed*`,
`coherent_fraction`, and `finite_budget` JSON keys keep their spelling for
existing consumers; their meaning is in [TERMINOLOGY.md](TERMINOLOGY.md).

## 5. Compile a runtime policy

The [policy compiler](CHANNEL_OPERATING_POLICY.md) binds decisions to the
receiver profile, frequency IDs, era and calibrated bundle. Unassigned channels
become permanent masks; assigned channels need a valid profile at the exact
word. Channel 37 is a control, not a DTV science channel.

```sh
pilot-proxy compile-operating-policy --plan policy_plan.json \
    --output-dir <policy-review> --require-complete
```

Bundle compilation does not activate the detector or certify its science assumptions.

## 6. Integrate and commission

The Kotekan stage `cudaPilotProxyDetector` is in Kotekan's `develop` branch
([kotekan PR 1677](https://github.com/kotekan/kotekan/pull/1677)). It selects the
detector window at run time (CHORD 64, CHIME 128); the CHIME include is
detect-only. Follow the [operator handoff](OPERATOR_HANDOFF.md), inspect the
consumer revision, and start with masking disabled. The live run is a separate
qualification step.
