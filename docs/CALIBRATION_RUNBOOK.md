# Calibration runbook

This page is for whoever runs the DTV pilot mask on the CHORD Pathfinder after December 2026. It lists the
manual calibration as it is done today, from recording to the checks after deployment, with the commands. The
background for each step is in [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md), the node-level checklist in
[PATHFINDER_RECALIBRATION.md](PATHFINDER_RECALIBRATION.md), and release in
[OPERATOR_HANDOFF.md](OPERATOR_HANDOFF.md). The automatic recalibration that would replace this loop is a
design only: [AUTOMATIC_RECALIBRATION.md](AUTOMATIC_RECALIBRATION.md).

The commands assume the setup of PATHFINDER_RECALIBRATION.md: PilotProxy in `~/pp-venv` on cx50, a checkout in
`~/pilotproxy` for `tools/`, and a login on the choco host. Kotekan is `develop` (or `chord`) with
kotekan#1924 and #1927, so bundles are fine decision v3, `dtv_runtime_bundle_dir` and `dtv_mask_control` are
render variables, and both DTV stages switch the mask at the same frame.

Pathfinder receiver IDs of the bound channels: 2531 = ch18, 2715 = ch24, 2869 = ch29, 3115 = ch37 (the
control, which has no pilot).

## 1. Record a detect-only run in calibration mode, at least one sidereal day

Ask the operators for calibration mode (`dtv_calibration_mode: true` in choco's `vars.yaml`, which recv1 also
reads), fixed digital gains and no other DTV stage on the X-engines. Check the nodes, then switch the mask off
and the recording on from the choco host:

```bash
~/pp-venv/bin/python tools/pathfinder_preflight.py --nodes cx47 cx42 --recv recv1
at=$(( ($(date +%s) + 20) * 1000000000 ))
choco set cx dtv_mask_enabled "{\"enabled\": false, \"valid_at_time_ns\": $at}"
choco set cx dtv_write_enabled '{"enabled": true}'
```

Record at least one whole sidereal day (23.93 h); two or more is better, because the test in step 3 needs data
the calibration did not see. On Pathfinder Run 1 (21 to 23 November 2026) a 6 h calibration failed the control
check because it missed the Taurus A transit, which raises the control's false alarms to about 30% for about
30 minutes once per sidereal day. Calibrations of 12 h, one sidereal day and 47.9 h passed. Note every restart
and its reason; a change to gains, inputs or F-engines splits the recording.

At the end, `choco set cx dtv_write_enabled '{"enabled": false}'` and ask for calibration mode off. Combine
the files (about 56 000 a day) before reading them:

```bash
~/pp-venv/bin/python tools/combine_pathfinder_dtv_run.py --acq "/mnt/cs00/data/dtv/acq_*" --out <combined dir>
```

`tools/calibration_length.py` (PATHFINDER_RECALIBRATION.md step 3) finds the shortest whole number of sidereal
days whose words still pass, and gives the `calibration_share` for step 2.

## 2. Calibrate

`process-run` splits the recording in time, runs `calibrate-bundle` on the first part and `test-bundle` on the
rest. It writes `calib/bundle/`, `calib/bundle.report.json`, `calib/test.json`, the spectrograms and
`calib/summary.md`:

```bash
~/pp-venv/bin/pilot-proxy process-run --bundle-dir <bundle of the run> \
    --run "<combined dir>/acq_*/dtv_fine_powers*.h5" --epoch "<dates> Pathfinder" \
    --calibration-share <share> --night 49.3208 -119.6236 \
    --threshold power_limit --power-limit power_limit.json --output-dir calib
```

The same steps one at a time are `plan-run` (writes `plans/calibrate.json` and `plans/test.json`),
`pilot-proxy calibrate-bundle --plan plans/calibrate.json --output-dir bundle` and `test-bundle` (step 3).
The bundle of the run must bind the control, or `plan-run` stops.

Each channel gets two words.

**Default word.** The generic threshold, for a consumer with no word of its own. Today it is the 1% rule: the
99% quantile of the control's keep words (plan key `null_quantile`, default 0.99; `--null-quantile` on
`plan-run`/`process-run`), over the control's night frames when `--night` is given (plan key
`"default_word": {"night": {...}}`), scaled to each channel by `mu0_c g_c / (mu0_37 g_37)`. The report gives
it as `word_source` `control word x scale`.

*Pending a parallel PR (not merged when this page was written):* the default rule moves to a per-channel false
discovery rate of 1%. That PR adds the plan key that chooses between the two rules, with the FDR rule as the
new default. Check the CHANGELOG and update this paragraph with the key name once it merges.

**BAO word.** The word that keeps the most frames while the kept DTV shelf power meets the checked power limit
of the channel (`--threshold power_limit`, plan key `"threshold": "power_limit"`). `power_limit.json` holds the
`power_limit` object of RUNNING_THE_DETECTOR.md: the RFIsher channel table of the Pathfinder (`table`), the
limits in mK² (`limits`), `curvature`, `margin` and `pilot_passband`. Leave `coupling` out: each channel then
takes `f = 1`, the worst case, and the word is the one that passes the bias test alone. The report gives
`power_limit_word_q16` with `power_limit_word_source` `power limit`, or the reject word 1
(`reject, power limit: <reason>`) where no word passes; the reject word masks the channel whole.

The measured-coupling BAO test (coupling records, noise and remainder tests, RFIsher budgets) is not in this
tool. It exists only in the dissertation scripts (WVURAIL/dissertation, private to the group), so a deployed
bundle carries the worst-case word.

Read `calib/summary.md`, `calib/bundle.report.json` (each channel's `word_source`, `pilot_share`, the BAO word or
its reject reason) and the spectrograms before going on.

## 3. Test and qualify

`process-run` has already tested the bundle on the held-out part (`calib/test.json`). On its own:

```bash
~/pp-venv/bin/pilot-proxy test-bundle --bundle-dir calib/bundle --plan calib/plans/test.json --output-json calib/test.json
```

Each check prints PASS or FAIL, and the exit status is 1 on any failure. The masked-fraction checks (every
channel at the word Kotekan applies, the BAO word where set, and the control at its word) pass when
`|bias| <= std`: the held-out mean masked fraction less the calibration's, against the hour-to-hour standard
deviation of the calibration. Passing means the words mask new data as they masked the calibration data. It
does not show that a word catches weak pilots; a person decides that from the report and the spectrograms.

Qualify only a passing report:

```bash
~/pp-venv/bin/pilot-proxy qualify-bundle --bundle-dir calib/bundle --test-report calib/test.json \
    --reviewer "<your name>" --output-dir calib/bundle-qualified
sha256sum calib/bundle-qualified/sha256sums.txt | cut -c1-8
```

The command refuses a failed report, one with no passing control check, or one of another bundle. It writes
`mask_qualification` (report and reviewer) into each calibrated channel and gives the bundle a new name. Kotekan
enforces this: since kotekan#1924, `DtvRfiMask` will not start with the mask on, and refuses a
`/dtv_mask_enabled` request that turns it on, unless every calibrated channel carries `mask_qualification`
(`mask_unqualified_reason` in `lib/stages/DtvRfiMask.cpp`). A detect-only bundle can still run with the mask off.
The control needs no qualification; it is masked only with `mask_control`.

## 4. Deploy

1. Add the qualified bundle to Kotekan by a pull request, as
   `config/chord/dtv_bundles/<purpose>-<8 hex digits from above>/`, with a line in that directory's
   `README.md`. To make it the default, change the default of `dtv_runtime_bundle_dir` in
   `config/chord/dtv.j2` to its installed path. To use it for one run only, render with
   `dtv_runtime_bundle_dir: /usr/local/etc/kotekan/chord/dtv_bundles/<name>`.
2. After `make install` on the nodes, check the digests in place:
   `cd /usr/local/etc/kotekan/chord/dtv_bundles/<name> && sha256sum -c sha256sums.txt`.
3. Restart Kotekan. The bundle is read once at startup. Check the log for one `bound ATSC physical channel`
   line per pilot and no `detector DISABLED`.
4. Both switches start off at every start. Switch them from the choco host (a POST straight to a node can be
   undone by choco):

   ```bash
   at=$(( ($(date +%s) + 20) * 1000000000 ))
   choco set cx dtv_mask_enabled "{\"enabled\": true, \"valid_at_time_ns\": $at}"
   choco set cx dtv_write_enabled '{"enabled": true}'
   ```

   With kotekan#1927, `DtvRfiMask` and `N2Accumulate` switch at the same frame, the first at or after
   `valid_at_time_ns`; a 0 or a time already passed applies from the next whole 5 s. `DtvRfiMask` logs
   `DTV mask on`.

Render settings (Kotekan variables, set in choco's `vars.yaml` or at render time):

| Variable | Standard operation | Meaning |
|---|---|---|
| `dtv_enabled` | true | false leaves the detector, the DTV mask and the recording out |
| `dtv_runtime_bundle_dir` | the qualified bundle | installed bundle directory |
| `dtv_calibration_mode` | false, true for a recording or the checks in step 5 | also sends and writes powers, fine support and fine powers |
| `dtv_mask_control` | false | true masks the control like a pilot. Keep it false: the control must not be masked, since its unmasked false alarms are the null check |

`dtv_write_enabled` and `dtv_mask_enabled` are REST switches, not render variables, on the Pathfinder.

## 5. Checks after deployment

Run with the mask on and calibration mode on for at least a day (the checks need the fine powers).

**Published pair against recorded fine powers.** For every block, the `[n_peak, d_rho]` row in `dtv_stat`
must equal `pilot_proxy.fine_decision.fine_stat_v3` of that block's `dtv_fine_powers`. Any difference means
the stage and the reference disagree, and the mask must go off until it is understood. On Run 2 this was done
by `check_stat_identity.py` in the run's analysis folder in the WVURAIL library; it is not in this
repository.

**Masked share against excised share, per visibility bin.**

```bash
~/pp-venv/bin/python tools/compare_pathfinder_dtv_mask.py --dtv-dir <acq> --vis-dir <visibility acq> \
    --bundle-dir <deployed bundle> --output-dir compare
```

With the mask on, the share of each visibility bin that `DtvRfiMask` masked (from `dtv_stat` at the
channel's word) against the correlator's `frac_rfi` has a slope near 1, and no bin is excised less than the
detector masked (`excised_below_detector` 0). The slope sits below 1 by the share of kept frames the
correlator excises on its own, about 0.3%. With the mask off the slope is near 0.

**Control masked share at night.**

```bash
~/pp-venv/bin/python tools/check_pathfinder_dtv_run.py --dtv-dir <acq> --bundle-dir <deployed bundle> \
    --vis-dir <visibility acq> --output-dir check
```

`check/masked_by_hour.csv` gives the masked percent of each channel by UTC hour; the control is decided from
its `dtv_stat` at its own word, so its false alarms show even though it is not masked. At night (Sun below
the horizon at DRAO) it should sit at its design, 1% under the current default rule, apart from the Taurus A
hour. A night share well away from the design means the null has moved: recalibrate.

## 6. When to recalibrate (manual, today)

- **Era change.** A transmitter switched on or off, or a new station: the old words describe a different
  shelf. Recalibrate on a recording from the new era.
- **ATSC 3.0 conversion.** Check the station records (FCC licensing and the stations' announcements) before
  each calibration. A converted channel keeps its DTV shelf but loses its pilot, so the detector never fires
  and the mask fails open. Mask such a channel statically with the permanent list
  (`permanent_mask_freq_ids` of the detector and `DtvRfiMask`). The Pathfinder templates do not pass that list
  yet, so this needs a Kotekan config change in `config/chord/dtv.j2`.
- **Pilot outside its target band.** A pilot moved out of the ±1.5 kHz target band around its nominal frequency fails open in the
  same way. Treat it as a converted channel until the bundle is rebuilt.
- **Drift of the control's masked share.** Night share away from its design (step 5), or a persistent
  summed-input warning after the bad-input mask changed.
- Also after any change to gains, quantization, inputs, frequency map or Kotekan DTV stage
  (RUNNING_THE_DETECTOR.md, The loop).
