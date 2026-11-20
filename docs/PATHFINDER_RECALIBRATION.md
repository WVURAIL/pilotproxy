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

- The X-engines on a Kotekan build from `develop`, with no other DTV stage: a development branch can change
  the voltages before the detector reads them. Agree the time with the operators.
- `dtv_calibration_mode` on in the X-engine configs. recv1 takes `dtv_calibration_mode: true` from choco's
  `vars.yaml`.

## 1. Check the nodes

```bash
~/pp-venv/bin/python tools/pathfinder_preflight.py --nodes cx47 cx42 --recv recv1
```

It reads each node's version and configuration and changes nothing. It fails on a branch other than
`develop` (`--branches` adds others) or on a DTV subtraction.

## 2. Record several days, mask off

From the choco host, which keeps the switches and pushes them to the nodes (a POST sent straight to a node
can be undone):

```bash
choco set cx dtv_mask_enabled '{"enabled": false, "valid_at_time_ns": 0}'
choco set cx dtv_write_enabled '{"enabled": true}'
```

recv1 writes under `/mnt/cs00/data/dtv/acq_<start>/`, about 27 GB a day (1.14 GB in a calibration-mode
hour, 2026-11-20). Record for longer than the calibration and test could need. At the end, set
`dtv_write_enabled` back to false and ask for calibration mode to be switched off again.

## 3. Find how long the calibration must be

```bash
~/pp-venv/bin/python tools/calibration_length.py --bundle-dir <bundle of the run> \
    --run "/mnt/cs00/data/dtv/acq_<start>/dtv_fine_powers*.h5" --hours 3 6 12 24 48 --output-dir length
```

Each length calibrates on the first part and tests on the rest. Take the shortest length whose words still
pass. Conditions change within a day, so expect at least one: on 2026-11-20 one hour gave channel 29's knee
as eta 3.90 from its first half and 2.22 from its second. `--block-hours 4` shows how each word follows the
time of day.

## 4. Calibrate, test and qualify

```bash
~/pp-venv/bin/pilot-proxy process-run --bundle-dir <bundle of the run> \
    --run "/mnt/cs00/data/dtv/acq_<start>/dtv_fine_powers*.h5" --epoch "<date> Pathfinder" \
    --calibration-share <length / recording> --output-dir run1
```

Add `--decision-version fine_decision_v1` while the stage on the nodes lacks kotekan#1914. Read
`run1/summary.md` and the spectrograms, then qualify the bundle with the test report `run1/test.json`
(Step 3 of the run guide).

## 5. Deploy

For commissioning and debugging, the X-engines can read the bundle from the site home: set
`dtv_runtime_bundle_dir` to it when the configurations are rendered. A production bundle goes into Kotekan
through a pull request under `config/chord/dtv_bundles/<purpose>-<first 8 hex of sha256(sha256sums.txt)>`.
With kotekan#1906, the stage refuses to switch the mask on with a bundle that has not been qualified.

## 6. Characterize with the mask on and off

With calibration mode on and `dtv_mask_control: true` (kotekan#1915, so the control is masked like a
pilot), alternate the mask from the choco host, in tmux. One day of 10-minute blocks:

```bash
BLOCK=600 BLOCKS=144 bash tools/pathfinder_mask_schedule.sh 2>&1 | tee -a mask_schedule.log
```

Afterwards:

- `tools/check_pathfinder_dtv_run.py --dtv-dir <acq> --bundle-dir <bundle> --output-dir check`: spectrograms
  of all frames and kept frames with the integrated spectrum, and the masked percent by UTC hour;
- `tools/compare_pathfinder_dtv_mask.py --dtv-dir <acq> --vis-dir <visibility acq> --output-dir compare`: the
  visibilities with and without the mask, with the control as the null.

A visibility file still being written sits in the acquisition's `.partial/`; once its recv has stopped it is
readable there.

## Not possible yet

The shadow acceptance of the handoff replays sampled decisions from voltages saved during the run. CHORD has
no baseband dump yet, so that replay waits for one; the recorded fine powers still let every decision be
rescored offline.

## Rehearse first

```bash
~/pp-venv/bin/python tools/make_pathfinder_mock_run.py mock --windows 252 --burst-blocks 100 800 --gzip
```

writes three hours of synthetic recv files, on which steps 3 and 4 run as on real data.
