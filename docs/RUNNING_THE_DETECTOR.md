# Running the DTV Detector

This guide is for the person who runs the digital television (DTV) detector on the CHORD Pathfinder, CHORD, or the CHIME upgrade after the author has left. It explains what the detector needs, why each number has to be measured on the telescope, and the commands that produce those numbers. The stage reference is `docs/sphinx/user/modules/pilotproxy.rst` in Kotekan, and masking the visibilities is covered by `docs/OPERATOR_HANDOFF.md` in PilotProxy.


## What the detector needs

The detector tests for a small excess of power at a television pilot over nearby fine frequency bins. For each block of 41.9 ms, the ratio at the pilot is tested against the same ratio over a set of reference bins, and the block is rejected when the excess passes a threshold. The inputs to this test are stored in a directory known as the runtime bundle.

| Bundle content | What sets it | How it is obtained |
|---|---|---|
| Weights | Receiver geometry and the pilot frequency | Computed, no data needed |
| Anchor bin | The true pilot frequency of each transmitter | Measured from a capture |
| Reference bins and rank | Lines near the pilot, which must be left out | Default, plus any measured exclusions |
| Threshold word | The frames of each channel, above the noise of the telescope | Knee of the kept fraction, with a floor from a channel with no pilot |

The word is the part that does not transfer between telescopes, and each channel has its own. For each frame, we find the smallest word that keeps it. The kept fraction against the logarithm of the word rises over the frames with no pilot or a weak pilot and then bends, and the word at this bend, known as the knee, is the word of largest height above the chord of the curve. Every capture has equal weight, so a long capture does not set the curve. The knee is a generic starting point, with no link to a science requirement. A channel with a strong pilot in nearly every frame has no clean frames to bend away from, so its knee lands in the spread of the pilot itself and masks little. Calibration gives such a channel the floor instead, so it is masked whenever the pilot is seen, which is nearly always. A frame mask recovers little of such a channel, and it is a candidate to flag whole.

The floor keeps a word above the noise. It is the 99% quantile of the scores of a channel with no pilot, and it depends on the number of inputs summed and on the real array. White noise gives its lowest value, which `"num_inputs"` simulates (2.15 as a ratio for 32 inputs, 1.48 for 128, and 1.10 for 2048). A real control can sit well above this, so a simulated floor is only a first guess and a floor from a short capture is a lower limit. Each telescope needs a control record of its own.

No channel is qualified for masking yet. The stage refuses words marked `archive_review_only`, and every bundle made by the commands below is a detect-only commissioning bundle. In detect-only mode the detector records its decisions and the visibilities do not change.


## The loop

The same four steps are used for a first run, a new telescope, and a recalibration.

1. Record data from the telescope.
2. Calibrate, which builds a new bundle.
3. Test the bundle on data it was not made from.
4. Deploy the bundle and run.

The word stays fixed during a run. A threshold that followed the live data would rise with a stronger pilot and stop masking it, and the kept data would have no fixed definition. The live data instead feeds the next bundle, and each new bundle is tested before it is deployed.

The loop is repeated after a change to the gains or quantization, the input population, or the frequency map of a node. It is also repeated when a pilot moves out of its marker bins or a new line appears in the reference bins, and both appear in `dtv_powers` before the mask changes.


## Step 1: Record

Calibration reads each channel to be bound and a control channel with no pilot. Channel 37 is the control for CHIME and the Pathfinder. The best source is the detector itself. With `dtv_calibration_mode` and `dtv_write_enabled` on, a detect-only run writes the fine powers of every block to `dtv_fine_powers*.h5`, and each file of about 43 s counts as one capture (`dtv_fine_powers` in the plan). The channels and the control must be bound in the bundle of that run, and the voltages are not needed. Reading these files needs `h5py` and `hdf5plugin`. A run of a few days covers the changes over a day, and it gives far more than the 30 captures a knee needs.

A telescope with no data starts from its parameters alone. A plan with empty capture lists, `"control": null` and `"num_inputs"` gives a first bundle at the simulated floor, and this bundle is enough to record the data for the next one:

```json
{
  "receiver_profile": "configs/receiver_profiles/chord_pathfinder_dtv_fengine.json",
  "detector_core_profile": "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json",
  "capture_format": "dtv_fine_powers",
  "epoch": "first bundle",
  "channels": {"24": [], "37": []},
  "control": null,
  "num_inputs": 128
}
```

Baseband voltages work as well, as CHIME baseband HDF5 (`chime_baseband`) or Pathfinder baseband HDF5 (`chord_baseband`). The CHIME upgrade has no voltage capture path of its own yet, and `config/chime/dtv.j2` does not write fine powers yet, so its first captures come from the CHIME/FRB baseband system.

Merged Pathfinder files are chunked over all 6144 frequencies, so a read of one channel costs about 97 s per block on cx50. The channels are therefore extracted once, to a small file on `/scratch`:

```bash
python3 tools/extract_baseband_channels.py --input baseband_subset.00000000.h5 --output day1.h5 --coarse-freq 2531 2715 2869 3115 --blocks 20
```

Calibration and testing use different epochs, at different times of day. The null quantile sets the amount of control data. A floor at 99% needs at least 50 control frames, and more frames give a steadier floor. A knee needs at least 30 captures of each channel, spread over the conditions the telescope sees, because a curve from a few captures describes only those captures.


## Step 2: Calibrate

`calibrate-bundle` exports the weights, measures each anchor, finds the floor from the control channel, finds the knee of each channel, and writes a sealed bundle. It reads a plan:

```json
{
  "receiver_profile": "configs/receiver_profiles/chime_dtv_fengine.json",
  "detector_core_profile": "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json",
  "capture_format": "chime_baseband",
  "epoch": "2026-11 CHIME upgrade commissioning",
  "null_quantile": 0.99,
  "channels": {"35": ["capture_521.h5"], "29": ["capture_614.h5"]},
  "control": {"channel": 37, "captures": ["capture_491.h5"]},
  "exclude_bins": {"29": []}
}
```

```bash
pilot-proxy calibrate-bundle --plan calibrate.json --output-dir bundle
```

Paths to profiles are relative to the plan, and capture lists take glob patterns such as `"/scratch/dtv/run1/dtv_fine_powers*.h5"`. For a telescope with no captures yet, `"control": null` and `"num_inputs": 32` give a simulated first floor. The command writes `bundle/` and `bundle.report.json`, and the report must be read. A channel marked `nominal` showed no pilot in the capture, so its anchor is the expected bin and a capture with the pilot on should replace it. Lines in the reference bins are added to `exclude_bins`.

The report gives `word_source` for each channel. The word is the knee when the pilot was seen in at least 30 captures, in at most 90% of the frames, and the knee lies above the floor, and in any other case it is the floor. `pilot_share` estimates the share of frames with a pilot, the share masked at the floor less the clean frames the floor masks, and `pilot_in_nearly_every_frame` marks a share above 90%. A first bundle from a few captures therefore holds the floor on every channel, which is enough for a detect-only run. Its `knee_interval_q16` gives the 16th and 84th percentiles of the knee over resampled captures, and a wide interval means more captures are needed.

Three plan keys replace a measured value. `"word_q16"` sets the floor and the word of every channel, `"words": {"29": 165551}` sets the word of one channel, and `"anchors": {"29": 202}` sets an anchor. A word belongs to its geometry, so a word moved from another record must come with its anchor, search and reference bins. The report gives the strongest bin of each channel, which shows faint lines, so each anchor is read against the report before the bundle is tested. The strongest bin of a faint line can move between captures, so an anchor needs several epochs, and a single short capture is only trusted for a strong pilot.


## Step 3: Test

`test-bundle` checks a bundle on captures it was not made from. The plan names the channels and the control:

```json
{
  "capture_format": "chime_baseband",
  "channels": {"35": ["other_521.h5"], "29": ["other_614.h5"]},
  "control": ["other_491.h5"]
}
```

```bash
pilot-proxy test-bundle --bundle-dir bundle --plan test.json --output-json bundle.test.json
```

Each check prints PASS or FAIL, and the exit status is 1 on any failure.

| Check | Why it exists |
|---|---|
| Structure and digests | A damaged bundle stops the instance at startup |
| Every bound channel has values | A pending channel stops the instance at its first frame |
| Test data differ from calibration data | A word tested on its own data proves nothing |
| Rank support on each channel | A frame with too few reference bins is not tested |
| Masked fraction on each channel | The word masks other captures as it masked the calibration captures |
| False alarms on the control | The floor gives the rate it claims |

Kotekan's `tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh` tests the code, and `test-bundle` tests the numbers. Both are needed after a change to the detector core.


## Step 4: Deploy and run

Kotekan reads the bundle from a local path once, at startup, and checks every digest. There is no service to query, and none is needed. A bundle changes only at a restart, and a failed lookup at startup would stop the correlator along with the detector.

A passing bundle is added through a Kotekan pull request to the bundle directory of the telescope, `config/chime/dtv_bundles/` or `config/chord/dtv_bundles/`. Each bundle is an immutable directory named `<purpose>-<first 8 digits of the sha256 of sha256sums.txt>`, and a new calibration gets a new directory. The deployment step copies it to the node:

```bash
/opt/kotekan/dtv/<purpose>-<digest>/
```

```bash
sha256sum -c sha256sums.txt
```

### CHORD Pathfinder

The detector always runs in `config/chord/pathfinder.j2`. These variables are set when it and `config/chord/recv.j2` are rendered:

```yaml
dtv_runtime_bundle_dir: /opt/kotekan/dtv/<bundle>
dtv_apply_mask: false
dtv_write_enabled: false
dtv_calibration_mode: true
```

Without `dtv_calibration_mode` only the mask is sent. With it, the powers, fine support and fine powers are sent too, and both templates must be rendered with the same value. The mask and recording switches start from the values above, and either is changed over REST until the next restart:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"enabled": true, "valid_at_time_ns": 0}' <node>:12048/dtv_mask_enabled
```

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"enabled": true}' <node>:12048/dtv_write_enabled
```

A POST sent before the stages start has no effect. A `valid_at_time_ns` of 0 applies the mask from the next frame.

The recv node writes the products under `/mnt/cs00/data/dtv/acq_<start time>/`, with a new directory for each recv start, and one file for each half and window of about 43 s. The `coarse_freq` and `fpga_seq_num` attributes identify the half and the window. A block missing from a window is a row of 0xFF bytes, and the reader skips it.

After a run, `plan-run` splits the files in time, with the first two thirds for calibration and the rest for the test. On a 48 h run, `--calibration-share 0.5` calibrates on the first day and tests on the second, so each covers a whole day:

```bash
pilot-proxy plan-run --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --epoch "2026-11 Pathfinder shakedown" --calibration-share 0.5 --output-dir run1/plans
```

```bash
pilot-proxy calibrate-bundle --plan run1/plans/calibrate.json --output-dir run1/bundle
```

```bash
pilot-proxy test-bundle --bundle-dir run1/bundle --plan run1/plans/test.json --output-json run1/bundle.test.json
```

Calibration holds the fine powers of one channel at a time, about 13 GB for a day. On a 90 min synthetic run of four channels, calibration took 18 s, the test 10 s and the spectrograms 15 s, and the time grows in proportion to the run.

CHORD has no archive for the science rule of the threshold, so these runs set its words. A 48 h run calibrates on the first day and tests on the second, and a short run with the new bundle then confirms the masks on the nodes. `test-bundle` checks only that the second day is masked like the first, so it passes a poor word. Each channel's `word_source` and `pilot_share` in the report, and its spectrogram, are read before the bundle is used.

The bundle of the run must bind the control, or `plan-run` stops, because the run then gives no floor. A pilot that fades during the run can fail the masked fraction check on the last third, so a failure on a weak channel is read against its spectrogram before the word is changed.

### CHIME upgrade

The template `config/include/dtv_chime_upgrade.j2` must be ported before the first run. It was written for the single-domain `chime_upgrade_n2.j2`, and `develop` now has `config/chime/chime_upgrade_2numa.j2`, where each buffer carries a `_n<numa>` suffix. The port places the detector stage, its three buffers, and its writers in the `for numa in numa_domains` loop, with one detector on each GPU.

The node must carry the `freq_id` values of the bundle, and the voltage metadata must carry the 16 `coarse_freq` values of each domain. The template has no REST switch, so the detector runs from startup.

### Checking a run

The first check is the log. Each pilot must have a `bound ATSC physical channel` line, matching the frequency map of the node. A `detector DISABLED` line means the node has no bundle pilot, and its zero masks mean the blocks were not tested.

| Quantity | Limit |
|---|---|
| Detector time per block, from `/gpu_profile` | Far below 41.9 ms |
| Voltage ring, from `/buffers` | No backlog |
| N2 products | No new frame drops |
| Disk use on CHIME | About 38 GB per hour |

`tools/analyze_dtv_run.py` reads the raw CHIME products and checks equal frame counts, gaps in `fpga_seq_num`, rank support, and the masked fraction of each channel. The masked fractions should agree with the `test-bundle` result. To compare the stage with the reference bit for bit, a baseband dump taken during the run is passed through `pilot-proxy replay-fine-capture`.

A run with `dtv_calibration_mode` also gives a spectrogram of each bound channel with and without the mask, in 10 s chunks:

```bash
pilot-proxy run-spectrogram --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --channel 18 24 29 37 --output-dir spectrograms
```

The decision is recomputed from the fine powers and the bundle word, so the control shows what its word would remove. A chunk with no kept frames is grey. The axis is recorded time, so a gap in the recording does not appear.

The same command writes `masked_summary.csv`, with the masked percent and the longest masked stretch of each channel. The percent on the control is its false-alarm rate at its word.

For a threshold from the run, `run-scores` writes the score cache of the threshold rule: each frame's smallest keeping word, its frame powers and its 10 s event. It reads the files once, and `--jobs` scores them in parallel:

```bash
pilot-proxy run-scores --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --channel 18 24 29 --jobs 32 --output-dir scores
```

```bash
python tools/threshold_rule.py --scores scores --power-limits limits.json --out thresholds
```

The rule keeps the most data while the BAO bias from the residual shelf power stays under one standard deviation in the RFIsher table, with a margin of two standard errors. `limits.json` gives each channel's checked power limit in mK². The result is in-sample, so it is checked on a later part of the run or on another run.

On CHIME, the detector is removed by a restart without `dtv_enabled`, and the rendered config then has no detector stages. A change of bundle, word, or channel list always needs a restart.


## Common failures

| Message or symptom | Cause | Action |
|---|---|---|
| `sha256 mismatch` | Bundle file changed or was copied badly | Copy it again from the bundle directory |
| Stops at first frame on an uncalibrated channel | Bound profile is `pending_campaign` | Run `calibrate-bundle` with that channel, or drop it |
| `missing bundle receiver namespace` | Bundle was made for the other telescope | Use the matching receiver profile |
| `detector DISABLED` | No bundle pilot on this node | Check the frequency map |
| N2 falls behind | Detector holds the voltage ring | Stop the detector and report its time per block |
| Control false alarms fail | Floor is too low, or the control has a line | Check `dtv_powers`, then recalibrate with more epochs |
| Masked fraction test fails | The test captures differ from the calibration captures | Read the report, then recalibrate with captures that cover both |


## Bad feeds

The detector sums every input in the voltage ring, and it does not read the bad feed mask.


## Applying the mask

Masking needs thresholds from a calibration campaign on the telescope, chosen against the science tolerance, then the shadow acceptance in `docs/OPERATOR_HANDOFF.md`, and only then `dtv_apply_mask: true`.
