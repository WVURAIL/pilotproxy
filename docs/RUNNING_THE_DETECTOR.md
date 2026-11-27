# Running the DTV Detector

This guide is for the person who runs the digital television (DTV) detector on the CHORD Pathfinder, CHORD, or the CHIME upgrade after the author has left. It explains what the detector needs, why each number has to be measured on the telescope, and the commands that produce those numbers. The stage reference is `docs/sphinx/user/modules/pilotproxy.rst` in Kotekan, and masking the visibilities is covered by `docs/OPERATOR_HANDOFF.md` in PilotProxy.


## What the detector needs

The detector tests for a small excess of power at a television pilot over nearby fine frequency bins. For each block of 41.9 ms, the ratio at the pilot is tested against the same ratio over a set of reference bins, and the block is rejected when the excess passes a threshold. The inputs to this test are stored in a directory known as the runtime bundle.

| Bundle content | What sets it | How it is obtained |
|---|---|---|
| Weights | Receiver geometry and the pilot frequency | Computed, no data needed |
| BAO word | The residual power the BAO measurement tolerates on each channel | Power-limit test on recorded frames, with the RFIsher table of the telescope |
| Default word | The noise of the telescope, for a backend with no word of its own | The control's 99% word, scaled to each channel |

Under fine decision v3 the bundle carries no search geometry: the detector takes the largest target over all 256 fine bins against the median reference of the 128 even bins, and publishes the pair `[n_peak, d_rho]`. Each consumer applies its own word to the pair. The words are the part that does not transfer between telescopes, and each channel has its own. For each frame, we find the smallest word that keeps it, its keep word.

The BAO word is the point of the calibration: the word that keeps the most frames while the kept DTV power meets the checked power limit of the channel (see Choosing the threshold). The correlator path applies it where calibration sets one. The default word is the generic threshold beside it, for a backend with no word of its own. Under v3 it is set by a false discovery rate of 1% (Benjamini and Hochberg, `pi0 = 1`): the null of each channel is the control channel, which has no pilot, scaled to the channel's null level by `lambda_c = mu0_c g_c / (mu0_37 g_37)` (`mu0` is the weight-norm ratio of the stored weights, `g` an optional passband factor from the plan). For a score `eta`, `E_c(eta)` is the share of the channel's frames above `eta` and `E_0(eta)` the share of control frames above `eta / lambda_c`: the empirical share while at least 20 control frames lie above, and beyond that a generalized Pareto tail fitted to the control frames above its 98th percentile. The default word is the Q16 word of the smallest channel score with `E_0 / E_c <= 0.01`. A channel where no score qualifies, a channel with no captures, and the control itself mask nothing at the default: their word is `2^64 - 1`, the largest word. Both shares are counted over the night frames when the plan names a night site (below), without capture weights. This is the rule of the dissertation (Sec. 2.3); on the archive score cache it gives the dissertation's words exactly (`tests/core/test_default_word_fdr_archive.py`). The tail fit needs at least 20 control frames above the 98th percentile, so at least 1000 control frames with a decision. The earlier rule, the 99% quantile of the control's keep words scaled by `lambda_c`, is kept as `"default_word": {"rule": "control_quantile"}` (below); the Pathfinder runs of November 2026 used it. It depends on the number of inputs summed and on the real array. White noise gives its lowest value, which `"num_inputs"` simulates when no control is recorded, so a simulated word is only a first guess and a word from a short control capture is a lower limit. Each telescope needs a control record of its own.

Bundles of fine decisions v1 and v2, which a plan names explicitly, still carry an anchor bin, a search half-width, reference bins and a rank, and their default word is the knee of the kept fraction with a floor from the control. Their rules are kept in the report fields named below.

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

Calibration reads each channel to be bound and a control channel with no pilot. Channel 37 is the control for CHIME and the Pathfinder. The best source is the detector itself. With `dtv_calibration_mode` and the recording on (`/dtv_write_enabled` on the Pathfinder, `dtv_write_enabled` on CHIME), a detect-only run writes the fine powers of every block to `dtv_fine_powers*.h5`, and each file of about 43 s counts as one capture (`dtv_fine_powers` in the plan). `tools/combine_pathfinder_dtv_run.py` stacks an acquisition's files into one per half, and the run commands read a window of a combined file as they read a per-window file. They read a file of several frames with per-frame metadata (kotekan#1925) by its frames. The channels and the control must be bound in the bundle of that run, and the voltages are not needed. Reading these files needs `h5py` and `hdf5plugin`. A run of a few days covers the changes over a day, and it gives far more than the 30 captures a knee needs.

A telescope with no data starts from its parameters alone. A plan with empty capture lists, `"control": null` and `"num_inputs"` gives a first bundle at the simulated control word, and this bundle is enough to record the data for the next one:

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

Baseband voltages work as well, as CHIME baseband HDF5 (`chime_baseband`) or Pathfinder baseband HDF5 (`chord_baseband`). The CHIME upgrade has no voltage capture path of its own yet. Its first captures can come from the CHIME/FRB baseband system, and since kotekan#1880 `config/chime/dtv.j2` writes fine powers in calibration mode as well.

Merged Pathfinder files are chunked over all their frequencies (7424 since kotekan#1884 widened the band to 150-1600 MHz, 6144 before), so a read of one channel cost about 97 s per block on cx50 at 6144. The Pathfinder receiver profile still gives the earlier science band (300-1500 MHz, channels 1536-7679) in its descriptive `science_band` field. It is left as it is, since the shipped bundles bind the profile by hash and the replay and operating-policy tools refuse a bundle whose profile hash differs. Every DTV pilot and the control lie in both bands. The channels are therefore extracted once, to a small file on `/scratch`:

```bash
python3 tools/extract_baseband_channels.py --input baseband_subset.00000000.h5 --output day1.h5 --coarse-freq 2531 2715 2869 3115 --blocks 20
```

Calibration and testing use different epochs, at different times of day. The null quantile sets the amount of control data. A control word at 99% needs at least 100 control frames, and more frames give a steadier word. The BAO word needs at least 30 kept captures of each channel (`min_captures`), spread over the conditions the telescope sees, because a test on a few captures describes only those captures.


## Step 2: Calibrate

`calibrate-bundle` exports the weights, finds the control word, sets the default word and, with the power-limit threshold, the BAO word of each channel, and writes a sealed bundle. It reads a plan:

```json
{
  "receiver_profile": "configs/receiver_profiles/chime_dtv_fengine.json",
  "detector_core_profile": "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json",
  "capture_format": "chime_baseband",
  "epoch": "2026-11 CHIME upgrade commissioning",
  "null_quantile": 0.99,
  "channels": {"35": ["capture_521.h5"], "29": ["capture_614.h5"], "37": ["capture_491.h5"]},
  "control": {"channel": 37, "captures": ["capture_491.h5"]},
  "exclude_bins": {"29": []}
}
```

```bash
pilot-proxy calibrate-bundle --plan calibrate.json --output-dir bundle
```

The control channel is also listed under `channels`, so the bundle carries it. Paths to profiles are relative to the plan, and capture lists take glob patterns such as `"/scratch/dtv/run1/**/*dtv_fine_powers*.h5"`. `**` also finds the files in the `acq_*` folders that Kotekan's HDF5 writer makes for each run, and the leading `*` takes the host name it puts in front of each file by default. For a telescope with no captures yet, `"control": null` and `"num_inputs": 32` give a simulated first word. The command writes `bundle/` and `bundle.report.json`, and the report must be read. A channel marked `nominal` showed no pilot in the capture, and a capture with the pilot on should replace it before its BAO word is trusted. Under v3 the anchor search only decides whether the pilot was seen, and a `nominal` channel gets no BAO word and carries the reject word (`pilot not seen in the calibration captures`). It counts the frames whose keep word passes the channel's default word, and a v3 keep word takes every bin, so a line anywhere in the channel counts as well.

The report gives `word_source` for each channel's default word: under v3 `fdr`, `fdr: masks nothing`, `fdr, no captures: masks nothing` or `fdr, the control is not masked by default`, with the rule `control_quantile` `control word x scale`, or `plan`. Under `fdr` each channel with captures has an `fdr` record (`word`, `channel_share`, `control_share` and whether it is `empirical` or from the `tail`, the estimate `fdr`, the `frames` used and whether they are the night frames), and the report gives `fdr_level` and the tail fit of the control (`fdr_null`). The pilot-seen gate below still counts the frames above the scaled control word, under either rule. The BAO word of a v3 channel is kept apart, in `power_limit_word_q16` with `power_limit_word_source` `power limit`, or the reject word 1 with `reject, power limit: <reason>` for a channel with no passing word. It also gives `control_quantile_q16`, `reference_level`, and each channel's `mu0`, `passband_factor` and `scale`. The default word needs no captures of the channel, so a first bundle from a short control capture already holds it on every channel, which is enough for a detect-only run. `pilot_share` estimates the share of frames with a pilot. For a v1 or v2 bundle, the word is the knee when the pilot was seen in at least 30 captures, in at most 90% of the frames, and the knee lies above the floor, and otherwise the floor; `knee_word_q16` and `knee_interval_q16` report it.

Plan keys replace a measured value. `"word_q16"` sets the word of every channel, `"words": {"29": 165551}` sets the word of one channel, `"default_word": {"passband_factor": {"29": 1.004}}` sets the passband factor of the default word, `"default_word": {"rule": "control_quantile"}` takes the earlier rule (`--default-word-rule` of `plan-run`, `process-run` and `tools/calibration_length.py`), and `"default_word": {"fdr_level": 0.01}` sets the false discovery rate of the rule `fdr`. `"default_word": {"night": {"latitude_deg": 49.3208, "longitude_deg_east": -119.6236}}` takes the default word over the control (and channel) frames recorded with the Sun below the horizon at that site (DRAO here), as CHIME's 21 cm analyses use the data, since in the day the Sun raises the control's scores. `plan-run` and `process-run` write it with `--night 49.3208 -119.6236`, and `test-bundle` then tests the night frames of the control. A word belongs to its test and its weights, so a word moved from another record must come with its decision version and weights. `"anchors": {"29": 202}` sets an anchor: for a v1 or v2 bundle it places the search, and under v3 it passes the pilot-seen gate of the BAO test. For a v1 or v2 bundle, a word moved from another record must also come with its anchor, search and reference bins.


### Choosing the threshold

A plan takes one of two thresholds. `"threshold": "knee"`, the default, gives each channel the default word above (the knee for v1 and v2). `"threshold": "power_limit"` gives each channel the word that keeps the most frames while the upper residual power meets a checked limit:

```
f * power_scale * (x + 2 s)^2 <= limit,    x = max(C_s e(u) - o - delta(u), 0) / r_c
```

With the RFIsher table and limits of the BAO forecast, this is the BAO acceptance test, and its word is the BAO threshold. The kept mean excess `e(u)` and its error `s` come from the same fine powers as the default word. The null `o` and the selection shift `delta(u)` are measured on the control; under v3 the control frames kept at a channel's word `u` are those with keep word at most `floor(u / lambda_c)`, with `lambda_c` the scale of the default word, so this plan needs a recorded control. Only control frames that the decision decides enter (a frame masked at every word is left out), and a word at which the control keeps no frame has no measured shift and cannot pass. The excess is the frame-level ratio `e = F / mu_0 - 1`, `F = 2 P_0 / (P_1 + P_2)`, the normalized pilot excess of the archive products, with each channel and the control at the null level `mu_0` of its own weights. The report gives `estimate` and `mu0`. The plan names the rest:

```json
{
  "threshold": "power_limit",
  "power_limit": {
    "table": "channel_table.json",
    "limits": "limits.json",
    "coupling": {"29": 0.2},
    "curvature": {"29": [1.0e-6, 1.0e-7]},
    "margin": 1.0e-6,
    "pilot_passband": {"29": 0.8464, "37": 0.9802}
  }
}
```

`table` is the RFIsher channel table of this telescope, which supplies `power_scale`, and `limits` gives each channel's checked power limit in mK². These two are required, and a table of another telescope gives a wrong word. `coupling` is the fraction `f` of the kept shelf power that reaches the map, and a channel with none takes 1, the worst case. `curvature` gives each calibrated channel the passband offset of the ratio at its pilot less that at the control position, with its standard error, and `margin` is added to the upper estimate. Both are in units of the shelf estimate, and both are required: write 0 for a term measured as none. Leaving them out can read a channel low, toward a pass: the offset difference is negative on some channels, and where an estimate clips at zero only the error terms remain. For CHIME they are the dissertation's offset difference `(d_c, d_se)`, measured on the control frame spectra under each channel's weights, and its route term (`estimate_offset.json`, fields `curvature`, `curvature_se` and `route_disagreement`). `pilot_passband` gives every channel of the plan, the control included, its pilot passband factor `r_c`: the pilot is a tone and passes the channelizer with another gain than the shelf, so the corrected estimate reads `r_c` times the shelf, and the test divides the estimate and its margin by `r_c`. It is required as well: `r_c` is below 1 on most channels, so leaving it out reads them low. For CHIME it is the field `r_c` of the dissertation's `pilot_passband.json`. A word must keep at least 30 captures (`min_captures`).

Under v3, a channel with a passing word carries it as `power_limit_multiplier_q16` beside its default word (`power_limit_word_source` `power limit`). A channel with no passing word, or with no captures, carries the reject word 1 there instead. DtvRfiMask and the relay apply the power-limit word where a bundle has one, and the smallest word keeps only a frame with no target power, so they mask the channel whole, as the operating policy does. That needs no permanent list, which the Pathfinder templates do not pass. Under v1 and v2 the passing word is the channel's word, with `word_source` `power limit`. `bundle.decisions.json` records each outcome in the form that `compile-operating-policy` reads ([CHANNEL_OPERATING_POLICY.md](CHANNEL_OPERATING_POLICY.md)), and the report keeps the default word of each channel beside its test. The control is tested with no corrections, as a check on the run.

The test bounds the bias of the residual at the given `f`. It does not test a residual that enters as noise. Its word is in-sample, so it is tested on later data before it masks. This threshold holds the control in memory beside each channel, about 13 GB more for a day.

`plan-run` and `process-run` take `--threshold power_limit --power-limit inputs.json`, where the file holds the `power_limit` object above.


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
| Every frame decided on each channel | A v3 frame whose median reference is zero, or a v1/v2 frame with too few reference bins, has no decision |
| Masked fraction on each channel | The word the stage applies (the BAO word where set) masks other captures as it masked the calibration captures |
| False alarms on the control | The control word masks other control frames as it masked the calibration frames. A test plan without a control fails |

Both masked-fraction checks group the frames by the UTC hour of their time and take each hour's fraction over the frames with a decision. A check passes when `|bias| <= std`: the bias is the test's mean masked fraction less the calibration's, both weighted by frames, and `std` is the standard deviation of the calibration's hourly fractions at that word (`masked_by_hour` in the report, `applied_masked_by_hour` at a BAO word). This is the rule of the dissertation's BAO test. Frames within an hour share the sky and the gains, so the hour-to-hour spread, not the count of frames, sets what the test data may differ by. A calibration of one hour has no spread and fails, and a report without these fractions asks for a recalibration. Captures that record no frame time are binned by capture, and the report says so.

Kotekan's `tests/ci-scripts/pilotproxy/run_pilotproxy_tests.sh` tests the code, and `test-bundle` tests the numbers. Both are needed after a change to the detector core.

No check shows that a word catches weak pilots, since the captures do not say which blocks hold one. A person reads the report, each channel's `word_source` and `pilot_share`, and the spectrograms, and only then qualifies the bundle for masking:

```bash
pilot-proxy qualify-bundle --bundle-dir bundle --test-report bundle.test.json \
  --reviewer "Name" --output-dir bundle-qualified
```

The command refuses a report that failed, has no passing control check, or is of another bundle. The new bundle names the report and the reviewer in each calibrated channel's `mask_qualification`, and has new digests and a new name.

After `process-run`, the test report is `run1/test.json`. Kotekan names a bundle directory `<purpose>-<first 8 hex digits of the SHA-256 of sha256sums.txt>`, and `sha256sum bundle-qualified/sha256sums.txt | cut -c1-8` gives the digits for the qualified bundle.


## Step 4: Deploy and run

Kotekan reads the bundle from a local path once, at startup, and checks every digest. There is no service to query, and none is needed. A bundle changes only at a restart, and a failed lookup at startup would stop the correlator along with the detector.

A passing bundle is added through a Kotekan pull request to the bundle directory of the telescope, `config/chime/dtv_bundles/` or `config/chord/dtv_bundles/`. Each bundle is an immutable directory named `<purpose>-<first 8 digits of the sha256 of sha256sums.txt>`, and a new calibration gets a new directory. `make install` copies the whole config tree, bundles included, to `<prefix>/etc/kotekan/`, so with the default prefix a bundle is on the node at:

```bash
/usr/local/etc/kotekan/chord/dtv_bundles/<purpose>-<digest>/
```

The digests can be checked there before a restart:

```bash
sha256sum -c sha256sums.txt
```

With `CMAKE_INSTALL_PREFIX=/usr`, the configs go to `/etc/kotekan/` instead, and the bundle path must be changed to match: `dtv_runtime_bundle_dir` on CHIME, and on the Pathfinder the `set` line in `config/chord/dtv.j2` (with kotekan#1924, `dtv_runtime_bundle_dir`).

### CHORD Pathfinder

[PATHFINDER_RECALIBRATION.md](PATHFINDER_RECALIBRATION.md) is the checklist for the steps below, with the commands.

`config/chord/pathfinder.j2` runs the detector unless it is rendered with `dtv_enabled: false`, which leaves the detector, the DTV mask and the recording out, and n2k then reads the SK mask. On `develop` (kotekan#1919) the bundle path is fixed by the `set` line in `config/chord/dtv.j2`, the installed `pathfinder-commissioning-e2e90219`, and a new bundle is named in that line by a Kotekan pull request. With kotekan#1924, `dtv_runtime_bundle_dir` is a render variable of `pathfinder.j2` with that path as its default. `config/chord/telescope.j2` reads two variables for both `pathfinder.j2` and `config/chord/recv.j2`, so render both with the same values:

```yaml
dtv_enabled: true
dtv_calibration_mode: true
```

Without `dtv_calibration_mode` only the mask is sent, and with kotekan#1924 `dtv_stat` as well (port 11041), in every mode. With it, the powers, fine support and fine powers are sent too. On `develop` the control is decided and recorded like a pilot, and never masked. With kotekan#1924, rendering `pathfinder.j2` with `dtv_mask_control: true` masks it like a pilot, in the detector and in `DtvRfiMask`, for any bundle version.

There is no `dtv_apply_mask` or `dtv_write_enabled` variable on the Pathfinder. The mask and recording switches start off at every start, and either is changed over REST until the next restart:

```bash
at=$(( ($(date +%s) + 20) * 1000000000 ))
curl -X POST -H 'Content-Type: application/json' -d "{\"enabled\": true, \"valid_at_time_ns\": $at}" <node>:12048/dtv_mask_enabled
```

```bash
curl -X POST -H 'Content-Type: application/json' -d '{"enabled": true}' <node>:12048/dtv_write_enabled
```

A POST sent before the stages start has no effect. Post a `valid_at_time_ns` a few seconds ahead, as above, so every node switches at the same frame; `tools/pathfinder_dtv_switch.py` does this for a list of nodes (`--lead`, 20 s by default). A mask change applies from the first frame starting at or after that time, in `DtvRfiMask` and in the `frac_dtv_mask_enabled` that `N2Accumulate` records. With kotekan#1927 both round a posted 0, or a time already passed, up to the next whole 5 s. An earlier build applies it at each stage's next frame, and `N2Accumulate`, downstream, then records the switch up to about a second before the data change.

On the live Pathfinder, the site control system choco stores both switches and pushes its values back to the nodes, so a POST sent straight to a node can be undone. There, switch them from the choco host, for every X-engine (`cx`) or one node (`cx/cx47`):

```bash
at=$(( ($(date +%s) + 20) * 1000000000 ))
choco set cx dtv_mask_enabled "{\"enabled\": true, \"valid_at_time_ns\": $at}"
choco set cx dtv_write_enabled '{"enabled": true}'
```

`tools/pathfinder_mask_schedule.sh` switches the mask through choco in this way, on a schedule.

The recv node writes the products under `/mnt/cs00/data/dtv/acq_<start time>/`, with a new directory for each recv start, and one file for each half and window of about 43 s. The `coarse_freq` and `fpga_seq_num` attributes identify the half and the window, and `fpga_seq_time_nsec` gives the window's time. The tools order the files by that time, so a recording split by restarts, even one that resets `fpga_seq_num`, keeps its order. A block missing from a window is a row of 0xFF bytes, and the reader skips it.

After a run, one command does every step below and writes a summary of each channel to `summary.md`, whose first line names the bundle's decision version: the scale of its default word (v3) or its anchor (v1 and v2), the word and its source, the pilot share, the masked percent on each part under the new bundle, the longest masked stretch, and the test result. Under v3 the word, its eta and its source are those of the power-limit word where calibration wrote one, and the scale is still that of the default word. It also writes the plans, the bundle and its report, the test record, and the spectrograms of each part:

```bash
pilot-proxy process-run --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --epoch "Pathfinder commissioning" --calibration-share 0.25 --output-dir run1
```

New bundles are written for fine decision v3, which Kotekan runs since kotekan#1924: it publishes `dtv_stat` and applies the words in `DtvRfiMask`. Add `--decision-version fine_decision_v2` to `process-run` or `plan-run` for a stage built without kotekan#1924. A Kotekan stage that runs only v1 refuses v2 and v3 bundles, so on nodes whose build lacks kotekan#1914 (in `develop` since 2026-11-20), add `--decision-version fine_decision_v1`.

The steps can also be run one at a time. `plan-run` splits the files in time, with the first two thirds for calibration and the rest for the test. `--calibration-share` moves the split. On a recording of several days, a share that ends the calibration on a whole day keeps a full day in each part, as 0.25 does on four days:

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

These runs set the deployed words on every telescope; the archive values characterize the method and are not deployed. A detect-only recording of several days calibrates on its first part and tests on the days after. The recording also sets how long the calibration must be. Run `process-run` with `--calibration-share` at increasing first parts, and take the shortest whose words still pass on the rest. The first part is at least a day, since conditions change within one: on the Pathfinder, one hour gave a channel-29 v2 knee of eta 3.90 from its first half and 2.22 from its second. A run of at least a day with the new bundle then keeps the mask on throughout and confirms the masks on the nodes. Visibilities with and without the mask wait for a second, unmasked correlation, so that both come from the same frames. The bundle is qualified as in Step 3 before the mask is switched on.

The bundle of the run must bind the control, or `plan-run` stops, because the run then gives no control word. A pilot that fades during the run can fail the masked fraction check on the last third, so a failure on a weak channel is read against its spectrogram before the word is changed.

### CHIME upgrade

`config/chime/chime_upgrade_2numa.j2` includes `config/chime/dtv.j2` in each NUMA domain when `dtv_enabled` is set, with one detector on each GPU (kotekan#1859). It is detect only, and no visibility or weight consumer reads its outputs. These variables are set when it is rendered:

```yaml
dtv_enabled: true
dtv_runtime_bundle_dir: /usr/local/etc/kotekan/chime/dtv_bundles/<bundle>
dtv_write_enabled: true
dtv_output_dir: /scratch/dtv
```

The bundle profiles must use `chime_freq_id`, and the voltage metadata must carry the 16 `coarse_freq` values of each domain. There is no REST switch, so the detector runs and writes from startup. The detector drops the inputs flagged by `bufferBadInputs` and writes only the mask (and with kotekan#1924 `dtv_stat` beside it) unless `dtv_calibration_mode` is set, which adds the powers, fine support and fine powers as on CHORD. The first bundle is `chime-first-56005bb7` (kotekan#1880).

### Checking a run

The first check is the log. Each pilot must have a `bound ATSC physical channel` line, matching the frequency map of the node. A `detector DISABLED` line means the node has no bundle pilot, and its zero masks mean the blocks were not tested.

| Quantity | Limit |
|---|---|
| Detector time per block, from `/gpu_profile` | Far below 41.9 ms |
| Voltage ring, from `/buffers` | No backlog |
| N2 products | No new frame drops |
| Disk use on CHIME | About 38 GB per hour |

`tools/analyze_dtv_run.py` reads the raw CHIME products and checks equal frame counts, gaps in `fpga_seq_num`, rank support (v1 and v2), and the masked fraction of each channel. With `--bundle-dir`, it decides the v3 channels from `dtv_stat` at each channel's word. The masked fractions should agree with the `test-bundle` result. To compare the stage with the reference bit for bit, a baseband dump taken during the run is passed through `pilot-proxy replay-fine-capture`. That command reads only CHORD `rawFileWrite` captures in the layout of Kotekan `c81def55`, so it cannot yet replay a CHIME run or a run of a current build ([capture replay guide](CAPTURE_FINE_REPLAY.md#supported-capture-format)).

A run with `dtv_calibration_mode` also gives a spectrogram of each bound channel with and without the mask, in 10 s chunks:

```bash
pilot-proxy run-spectrogram --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --channel 18 24 29 37 --output-dir spectrograms
```

The decision is recomputed from the fine powers and the word the stage applies, so the control shows what its word would remove. Under v3 that word is the power-limit word where calibration wrote one, and the default word otherwise. A chunk with no kept frames is grey. The axis is recorded time, so a gap in the recording does not appear.

The same command writes `masked_summary.csv`, with the masked percent and the longest masked stretch of each channel. The percent on the control is its false-alarm rate at its word.

The `power_limit` threshold (the plan key `threshold`, or `--threshold power_limit` of `plan-run` and `process-run`) sets the word of the BAO acceptance test in the bundle (Step 2). To study the sweep of a channel by itself, `run-scores` writes the score cache of the threshold rule: each frame's smallest keeping word, its frame powers and its 10 s event. It reads the files once, and `--jobs` scores them in parallel:

```bash
pilot-proxy run-scores --bundle-dir bundle --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --channel 18 24 29 --jobs 32 --output-dir scores
```

```bash
python tools/threshold_rule.py --scores scores --table channel_table.json --channels 18,24,29 --power-limits limits.json --out thresholds
```

This rule keeps the most data while the BAO bias from the residual shelf power stays under one standard deviation in the RFIsher table, with a margin of two standard errors. It applies no control corrections. `channel_table.json` is the RFIsher channel table of this telescope: without `--table` the tool takes the packaged CHIME inter-cylinder table, which gives a wrong word on any other telescope. `--channels` names the channels of the score cache, since the tool otherwise reads every DTV channel of the table. `limits.json` gives each channel's checked power limit in mK². The result is in-sample, so it is checked on a later part of the run or on another run.

On CHIME, the detector is removed by a restart without `dtv_enabled`, and the rendered config then has no detector stages. A change of bundle, word, or channel list always needs a restart.


## Common failures

| Message or symptom | Cause | Action |
|---|---|---|
| `sha256 mismatch` | Bundle file changed or was copied badly | Copy it again from the bundle directory |
| Stops at first frame on an uncalibrated channel | Bound profile is `pending_campaign` | Run `calibrate-bundle` with that channel, or drop it |
| `missing bundle receiver namespace` | Bundle was made for the other telescope | Use the matching receiver profile |
| `detector DISABLED` | No bundle pilot on this node | Check the frequency map |
| N2 falls behind | Detector holds the voltage ring | Stop the detector and report its time per block |
| Control false alarms fail | Control word is too low, or the control has a line | Check `dtv_powers`, then recalibrate with more epochs |
| Masked fraction test fails | The test captures differ from the calibration captures | Read the report, then recalibrate with captures that cover both |


## Bad feeds

On CHORD, the detector packs every input flagged by `bufferBadInputs` as zero, as if it carried no signal, and `bufferBadInputs` also flags the RFI antennas (`bf_mask_name` in `config/chord/dtv.j2`). On CHIME it does the same, since kotekan#1880. A change in the flagged inputs changes the noise of the sum. When the plan gives `num_inputs`, each profile records it as `summed_inputs`, and with kotekan#1906 the detector logs a warning while the count of good inputs differs from it by more than 10%. Recalibrate if the warning persists.


## Applying the mask

Masking needs a calibration on the telescope, a passing `test-bundle` report with the control, `qualify-bundle`, then the shadow acceptance in `docs/OPERATOR_HANDOFF.md`, and only then switching the mask on. With kotekan#1906, Kotekan's `DtvRfiMask` refuses to start with the mask on unless every calibrated channel of the bundle is qualified, and a POST that switches the mask on with such a bundle stops Kotekan.
