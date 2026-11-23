# Recalibrating the Pathfinder DTV detector

Recalibrate after a change to the telescope or its inputs (new or moved inputs, new gains, a new receiver
configuration), when a pilot leaves its fine bins, when a check fails, or on a new Kotekan stage. This page
is the checklist; [RUNNING_THE_DETECTOR.md](RUNNING_THE_DETECTOR.md) explains each step, and
[OPERATOR_HANDOFF.md](OPERATOR_HANDOFF.md) covers release.

## What you need

- Logins on cx50 and the choco host (the choco admins group). The site home is shared by both.
- PilotProxy on cx50, in a virtual environment in your home, and a checkout for `tools/` (the commands
  below run from it):

  ```bash
  /opt/kotekan_env/bin/python -m venv ~/pp-venv
  ~/pp-venv/bin/pip install "pilot-proxy[calibration] @ https://github.com/WVURAIL/pilotproxy/archive/main.tar.gz" h5py hdf5plugin matplotlib
  git clone https://github.com/WVURAIL/pilotproxy.git ~/pilotproxy && cd ~/pilotproxy
  ```

- The X-engines on a Kotekan build from `chord` (develop plus the CHORD changes) or `develop`, with no other
  DTV stage: a development branch can change the voltages before the detector reads them. Agree the time
  with the operators, and ask them to keep the digital gains fixed for the recording.
- `dtv_calibration_mode` on in the X-engine configs. recv1 takes `dtv_calibration_mode: true` from choco's
  `vars.yaml`.

## 1. Check the nodes

```bash
~/pp-venv/bin/python tools/pathfinder_preflight.py --nodes cx47 cx42 --recv recv1
```

It reads each node's version and configuration and changes nothing. It fails on a branch other than
`chord` or `develop` (`--branches` sets the list) or on a DTV subtraction.

## 2. Record several days, mask off

From the choco host, which keeps the switches and pushes them to the nodes (a POST sent straight to a node
can be undone):

```bash
at=$(( ($(date +%s) + 20) * 1000000000 ))
choco set cx dtv_mask_enabled "{\"enabled\": false, \"valid_at_time_ns\": $at}"
choco set cx dtv_write_enabled '{"enabled": true}'
```

A mask change carries a `valid_at_time_ns` a few seconds ahead, so every node switches at the same frame
(`tools/pathfinder_mask_schedule.sh` and `tools/pathfinder_dtv_switch.py` do the same). With kotekan#1927 a
posted 0, or a time already passed, applies from the next whole 5 s. An earlier build applies it at each
stage's next frame, and `frac_dtv_mask_enabled` then records the switch up to about a second early.

recv1 writes under `/mnt/cs00/data/dtv/acq_<start>/`, about 27 GB a day (1.14 GB in a calibration-mode
hour, 2026-11-20). Record for longer than the calibration and test could need. Interruptions are fine:
each Kotekan restart opens a new `acq_<start>/`, and the tools order the files by the time of their first
frame. Keep the build and the configuration the same across restarts, with calibration mode and recording
on, and note when and why each interruption happened. A change to the gains, the inputs or the F-engines
splits the recording for calibration. At the end, set `dtv_write_enabled` back to false and ask for
calibration mode to be switched off again.

recv writes one file per half, product and 43 s window, about 56 000 a day, and each takes about 25 ms to
open over NFS. Combine each closed acquisition into one file per half and product:

```bash
~/pp-venv/bin/python tools/combine_pathfinder_dtv_run.py --acq "/mnt/cs00/data/dtv/acq_*" --out <combined dir>
```

Every window is read back against its source, and `<combined dir>/acq_<start>/combine.json` lists the files
and their checksums. An acquisition with a file changed in the last 10 minutes is skipped as still open, and
one already combined is skipped unless recv has added files to it since, as after a quiet gap; it is then
combined again. Once recv files are deleted, new files of that acquisition are refused (exit 1), as combining
again would drop the deleted windows. On 2026-11-21 a 2.9 h acquisition went from 6712 files to 27 in 167 s. The
tools below take `"<combined dir>/acq_*/dtv_fine_powers*.h5"` in place of the recv paths, with the same
result; read cold, its calibration took 37 s against 45 s from the recv files. Delete the recv files only
once every acquisition is combined and verified.

With kotekan#1925 (`record_per_frame_metadata`), recv writes the frames of every half to files of several
frames, with each frame's metadata in `frame_metadata`. The tools read those files as they are, and the
combine tool leaves them out.

## 3. Find how long the calibration must be

```bash
V=/mnt/cs00/data/kotekan_vis_files/subset
~/pp-venv/bin/python tools/pathfinder_input_counts.py --vis "$V/acq_<run>*/vis_*.h5" "$V/acq_<run>*/.partial/vis_*.h5" \
    --output inputs.npz
~/pp-venv/bin/python tools/calibration_length.py --bundle-dir <bundle of the run> \
    --run "/mnt/cs00/data/dtv/acq_*/dtv_fine_powers*.h5" --hours 23.93 47.87 71.8 --input-counts inputs.npz \
    --night 49.3208 -119.6236 --output-dir length
```

The study takes the same `--night` and `--decision-version` as process-run in step 4, so it tests the words
that step deploys. With `--night`, a length or block with no control capture at night is skipped.

The bad-input mask changes as the operators' script flags feeds: on 2026-11-20 it moved between 111 and
113 good inputs of 128 about every 14 minutes. `pathfinder_input_counts.py` reads it from the visibility
files of the run's acquisitions and prints each change. The detector sums only the good inputs, so
`--input-counts` adds each channel's good inputs to every row.

Each length calibrates on the first part and tests on the rest. Lengths are clock hours from the first
capture, so a gap leaves its hours empty; the tool prints each gap. Calibrate on whole sidereal days, 23.93 h
each. Taurus A transits once per sidereal day and raises the control's false alarms to about 30% for about
30 minutes. `test-bundle` compares each masked fraction with the hour-to-hour spread of the calibration, and a
calibration with no transit in it understates that spread: on Run 1 the 6 h calibration failed the control
check (bias/std 1.69), and 12, 18 and 24 h passed. Take the shortest number of sidereal days whose words still
pass, and its `calibration_share` in `length/length.json`. Conditions also change within a day: on 2026-11-20
one hour gave channel 29's v2 knee as eta 3.90 from its first half and 2.22 from its second. `--block-hours 4` shows how each word follows the time of day, and with `--input-counts` how
it follows the good inputs.

## 4. Calibrate, test and qualify

```bash
~/pp-venv/bin/pilot-proxy process-run --bundle-dir <bundle of the run> \
    --run "/mnt/cs00/data/dtv/acq_<start>/dtv_fine_powers*.h5" --epoch "<date> Pathfinder" \
    --calibration-share <calibration_share of the chosen length> --night 49.3208 -119.6236 --output-dir run1
```

`--night` sets the control word over the night frames at DRAO, as the dissertation does. Taurus A transits near
local midnight in November and still raises the night false alarms, so the calibration part covers whole
sidereal days (step 3).

New bundles are fine decision v3, which needs kotekan#1924. Add `--decision-version fine_decision_v2` for a
stage built without it, and `--decision-version fine_decision_v1` for one built without kotekan#1914. Read
`run1/summary.md` and the spectrograms, then qualify the bundle with the test report `run1/test.json`
(Step 3 of the run guide).

## 5. Deploy

On `develop` (kotekan#1919) the X-engines read the bundle at the path fixed by the `set` line in
`config/chord/dtv.j2`, the installed `pathfinder-commissioning-e2e90219`. A new bundle goes into Kotekan
through a pull request that adds it under
`config/chord/dtv_bundles/<purpose>-<first 8 hex of sha256(sha256sums.txt)>` and changes that line to its
installed path. The `chord` branch can carry the pull request before it merges into `develop`, and the nodes
take the bundle at the next deploy. With kotekan#1924, `dtv_runtime_bundle_dir` is a render variable with
that path as its default, so a run can name another installed bundle without a template change. With
kotekan#1906, the stage refuses to switch the mask on with a bundle that has not been qualified.

## 6. Characterize with the mask on

With calibration mode on, switch the mask on from the choco host and keep it on for at least a day. On
`develop` the control is never masked. With kotekan#1924, render the X-engine configs with
`dtv_mask_control: true` so that the control is masked like a pilot, in the detector and in `DtvRfiMask`,
for any bundle version:

```bash
at=$(( ($(date +%s) + 20) * 1000000000 ))
choco set cx dtv_mask_enabled "{\"enabled\": true, \"valid_at_time_ns\": $at}"
choco set cx dtv_write_enabled '{"enabled": true}'
```

The detector decides on every frame whatever the switch, so the spectra of all frames and of kept frames
come from the same frames. Afterwards, `tools/check_pathfinder_dtv_run.py --dtv-dir <acq> --bundle-dir
<bundle> --output-dir check` gives those spectrograms with the integrated spectrum, and the masked percent
by UTC hour. At the end, set `dtv_write_enabled` back to false and ask for calibration mode to be switched
off again.

Visibilities with and without the mask wait for a second correlation of the pilot frequencies without the
DTV mask, so that both come from the same frames. Switching the mask in blocks
(`tools/pathfinder_mask_schedule.sh`) gives the two from different blocks, so use it only to check the mask
path; `tools/compare_pathfinder_dtv_mask.py --dtv-dir <acq> --vis-dir <visibility acq>` reads such a run. A
visibility file still being written sits in the acquisition's `.partial/`; once its recv has stopped it is
readable there.

## Not possible yet

The shadow acceptance of the handoff replays sampled decisions from voltages saved during the run. CHORD has
no baseband dump yet, so that replay waits for one; the recorded fine powers still let every decision be
rescored offline.

## Rehearse first

```bash
~/pp-venv/bin/python tools/make_pathfinder_mock_run.py mock --windows 252 --burst-blocks 100 800 --gzip
```

writes three hours of synthetic recv files, on which steps 3 and 4 run as on real data. Add
`--gap 100 40 --reset-seq` to leave out half an hour with a Kotekan and F-engine restart in it.
