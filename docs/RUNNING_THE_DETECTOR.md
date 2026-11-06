# Running the DTV Detector

This guide is for the person who runs the digital television (DTV) detector on the CHORD Pathfinder, CHORD, or the CHIME upgrade after the author has left. It explains what the detector needs, why each number has to be measured on the telescope, and the commands that produce those numbers. The stage reference is `docs/sphinx/user/modules/pilotproxy.rst` in Kotekan, and masking the visibilities is covered by `docs/OPERATOR_HANDOFF.md` in PilotProxy.


## What the detector needs

The detector tests for a small excess of power at a television pilot over nearby fine frequency bins. For each block of 41.9 ms, the ratio at the pilot is tested against the same ratio over a set of reference bins, and the block is rejected when the excess passes a threshold. The inputs to this test are stored in a directory known as the runtime bundle.

| Bundle content | What sets it | How it is obtained |
|---|---|---|
| Weights | Receiver geometry and the pilot frequency | Computed, no data needed |
| Anchor bin | The true pilot frequency of each transmitter | Measured from a capture |
| Reference bins and rank | Lines near the pilot, which must be left out | Default, plus any measured exclusions |
| Threshold word | The noise behaviour of the telescope | Measured from a channel with no pilot |

The word is the part that does not transfer between telescopes. It depends on the number of inputs summed and on the behaviour of the real array. White noise gives a floor, which `"num_inputs"` simulates (1.84 as a ratio for 32 inputs, 1.36 for 128, and 1.08 for 2048). A real control can sit well above this floor, so a simulated word is only a first guess and a word from a short capture is a lower limit. Each telescope needs a control record of its own.

No channel is qualified for masking yet. The stage refuses words marked `archive_review_only`, and every bundle made by the commands below is a detect-only commissioning bundle. In detect-only mode the detector records its decisions and the visibilities do not change.


## The loop

The same four steps are used for a first run, a new telescope, and a recalibration.

1. Capture voltages on the telescope.
2. Calibrate, which builds a new bundle.
3. Test the bundle on other captures.
4. Deploy the bundle and run.

The loop is repeated after a change to the gains or quantization, the input population, or the frequency map of a node. It is also repeated when a pilot moves out of its marker bins or a new line appears in the reference bins, and both appear in `dtv_powers` before the mask changes.


## Step 1: Capture

Calibration needs baseband voltages of each channel to be bound and of a control channel with no pilot, on the block grid of the telescope. Channel 37 is the control for CHIME and the Pathfinder. Two formats are read today: CHIME baseband HDF5 (`chime_baseband`) and Pathfinder baseband HDF5 (`chord_baseband`). The CHIME upgrade has no voltage capture path of its own yet, so its first captures come from the CHIME/FRB baseband system.

Merged Pathfinder files are chunked over all 6144 frequencies, so a read of one channel costs about 97 s per block on cx50. The channels are therefore extracted once, to a small file on `/scratch`:

```bash
python3 tools/extract_baseband_channels.py --input baseband_subset.00000000.h5 --output day1.h5 --coarse-freq 2531 2715 2869 3115 --blocks 20
```

Calibration and testing use different epochs, at different times of day. The null quantile sets the amount of control data. A word at 99% needs at least 1000 scores, and each control frame gives about 50.


## Step 2: Calibrate

`calibrate-bundle` exports the weights, measures each anchor, sets the word from the control channel, and writes a sealed bundle. It reads a plan:

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

Paths to profiles are relative to the plan. For a telescope with no captures yet, `"control": null` and `"num_inputs": 32` give a simulated first word. The command writes `bundle/` and `bundle.report.json`, and the report must be read. A channel marked `nominal` showed no pilot in the capture, so its anchor is the expected bin and a capture with the pilot on should replace it. Lines in the reference bins are added to `exclude_bins`.

Two plan keys replace a measured value. `"word_q16"` sets the word, and `"anchors": {"29": 202}` sets an anchor. The report gives the strongest bin of each channel, which shows faint lines, so each anchor is read against the report before the bundle is tested. The strongest bin of a faint line can move between captures, so an anchor needs several epochs, and a single short capture is only trusted for a strong pilot.


## Step 3: Test

`test-bundle` checks a bundle on captures it was not made from. The plan names a strong channel, a quiet channel, and the control:

```json
{
  "capture_format": "chime_baseband",
  "strong": {"35": ["other_521.h5"]},
  "quiet": {"29": ["other_614.h5"]},
  "control": ["other_491.h5"],
  "min_strong_masked": 0.9,
  "max_quiet_masked": 0.2
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
| Strong channel is masked | The detector sees a real pilot |
| Quiet channel is rarely masked | The word and anchor do not fire on a clear channel |
| False alarms on the control | The word gives the rate it claims |

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

To enable the detector, these variables are set when `config/chord/pathfinder.j2` is rendered:

```yaml
dtv_enabled: true
dtv_apply_mask: false
dtv_runtime_bundle_dir: /opt/kotekan/dtv/<bundle>
dtv_output_dir: /scratch/dtv/<run>
```

Detection and recording start off, and both are turned on over REST:

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"enabled": true, "valid_at_time_ns": 0}' <node>:12048/dtv_detector_enabled
```

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"enabled": true}' <node>:12048/dtv_write_enabled
```

A POST sent before the stages start has no effect, so the log line `PilotProxy detector on` must be seen. A restart returns both switches to off. The same POST with `"enabled": false` stops either one.

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

`tools/analyze_dtv_run.py` reads the raw CHIME products and checks equal frame counts, gaps in `fpga_seq_num`, rank support, and the masked fraction of the strong and quiet channels. The masked fractions should agree with the `test-bundle` result. To compare the stage with the reference bit for bit, a baseband dump taken during the run is passed through `pilot-proxy replay-fine-capture`.

To remove the detector, Kotekan is restarted without `dtv_enabled`, and the rendered config then has no detector stages. A change of bundle, word, or channel list always needs a restart.


## Common failures

| Message or symptom | Cause | Action |
|---|---|---|
| `sha256 mismatch` | Bundle file changed or was copied badly | Copy it again from the bundle directory |
| Stops at first frame on an uncalibrated channel | Bound profile is `pending_campaign` | Run `calibrate-bundle` with that channel, or drop it |
| `missing bundle receiver namespace` | Bundle was made for the other telescope | Use the matching receiver profile |
| `detector DISABLED` | No bundle pilot on this node | Check the frequency map |
| N2 falls behind | Detector holds the voltage ring | Stop the detector and report its time per block |
| Control false alarms fail | Word is too low, or the control has a line | Check `dtv_powers`, then recalibrate with more epochs |
| Strong channel test fails | The pilot is weak on this array, with its ratio near the word | Read the report, then choose a stronger channel |
| Quiet channel test fails | Word is too low, or the channel has a faint line | Read the report, then raise the word or choose another quiet channel |


## Bad feeds

The detector sums every input in the voltage ring, and it does not read the bad feed mask.


## Applying the mask

Masking needs thresholds from a calibration campaign on the telescope, chosen against the science tolerance, then the shadow acceptance in `docs/OPERATOR_HANDOFF.md`, and only then `dtv_apply_mask: true`.
