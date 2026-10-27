# CANFAR bounded-run guide

This guide gives the remote CANFAR GPU-session workflow for CHIME DTV pilot
runs, the alternative to the local workstation path in
[`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md). The parameter register, gates,
and run record are in
[`RERUN_PARAMETER_REGISTER.md`](RERUN_PARAMETER_REGISTER.md),
[`VALIDATION_GATES.md`](VALIDATION_GATES.md), and the run ledger, and this
guide changes none of them. The schema v5 archive products of record were made
with detector core 2.3.0. A build of the current sources is core 4 and writes
schema v6 (see [`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md)), so a run made with it
is a new cohort (checklist item 7).

The approved run is archive estimation and sufficient-statistic
reprocessing. The coarse positive-excess flag is kept as a bootstrap
diagnostic. The fine rank and eta decision is inactive, and no calibrated
detection policy is applied.

For bounded remote archive-scale work, the entry point is
`pilot-proxy chime-scan`.

For the measured local workstation profile, use
[`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md). The build instructions below
target whatever GPU architecture the remote session provides and cannot
override the approved local settings.

`pilot-proxy chime-run` reads an already-staged HDF5 directory in one process
and is used for local calibration and regression comparisons. CADC and CANFAR
work starts with `chime-scan`.

---

## Baseline detector contract

The executable detector contract is:

```text
detector_window_samples = 128
skipped_guard_bins = 1
reference_offset_bins = 2
mask mode = positive_excess
```

That mask mode records the coarse bootstrap flag and defines an after-mask
diagnostic spectrum. The calibrated detection policy comes later. Setting
`fine_products=on` stores the exact fine powers and leaves fine detection
disabled.

The detector core supports compile-time `K = 64` and `K = 128` builds. The
CHIME receiver profile and shipped CHIME weight bank select `K = 128`, and `K`
is never tuned at run time. Another value would require its own supported
kernel geometry, weight bank, profile, and validation campaign.

## Launch a GPU session

First, on the workstation, clone this repository and install the `canfar`
client (`pip install canfar`). The launch script below runs locally, and the
repository is cloned in the session later.

The `pilot-proxy-detector` path requires a CUDA GPU. The helper script launches
a CANFAR GPU notebook session through the `canfar` client, or reuses a running
or pending session with the same name. The client also needs a Harbor CLI secret
to pull the session image. Obtain that secret from
`https://images.canfar.net` under your profile, then either export it or let
`setup_env.sh` prompt for and store it.

```bash
read -r -p "Registry user: " CANFAR_REGISTRY_USER
read -r -s -p "Registry secret: " CANFAR_REGISTRY_SECRET
printf '\n'
export CANFAR_REGISTRY_USER CANFAR_REGISTRY_SECRET

python scripts/launch_gpu_session.py
```

Use `python scripts/launch_gpu_session.py --status` to inspect it. Use
`python scripts/launch_gpu_session.py --destroy` only when the remote session
should be torn down.

The default session name is `cupy-gpu`. Open the printed URL and complete the
environment setup in that session's terminal.

---

## Environment setup

Clone this repository before running the setup script. The script recreates the
target virtual environment from scratch.

```bash
git clone https://github.com/WVURAIL/pilotproxy.git ~/pilotproxy
cd ~/pilotproxy

VENV_DIR=~/pilotproxy-venv PILOT_PROXY_DIR=~/pilotproxy bash scripts/setup_env.sh

source ~/pilotproxy-venv/bin/activate
```

Do not point `VENV_DIR` at an environment that must be preserved. The script
uses `python -m venv --clear`, installs this repository in editable mode,
checks the bundled archive components, and builds the CUDA library when a GPU
and `nvcc` are available. It refuses protected, checkout-overlapping, and unowned non-empty
targets and keeps its ownership record beside the environment so an interrupted
rebuild can be retried. If this is the first guarded rerun of a genuine virtual
environment created by an older checkout, add
`PILOT_PROXY_ADOPT_LEGACY_VENV=1` to the command once. This opt-in must never
be used for an arbitrary directory.

In the expected CANFAR notebook environment, the home directory is on
persistent `/arc` storage. Activate the virtual environment in every new
session before running `pip` or `pilot-proxy`. A bare install against the
session image's Python may fail because that Python environment is read-only.
Rerun `setup_env.sh` only when the environment should be rebuilt.

If the environment should not be cleared, use the manual installation path:

```bash
python3.12 -m venv --system-site-packages ~/pilotproxy-venv
source ~/pilotproxy-venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r "$HOME/pilotproxy/requirements/archive.txt"
python -m pip install -e "$HOME/pilotproxy[archive,test]"
```

Then confirm that the archive commands load:

```bash
pilot-proxy chime-survey --help
pilot-proxy chime-inventory --help
pilot-proxy chime-scan --help
```

The production analyzer also needs a working CUDA/CuPy runtime and a built or
staged `libpilotproxy.so`.

---

## Remote session validation example

This is the on-node sequence for the alternate remote workflow. Run it after
environment setup and stop at the first failure. It does not replace the SM89
gates required for the approved local run.

```bash
# S2.0 - architecture of the GPU this session actually got (no architecture is
#        requestable through launch_gpu_session.py); record it in the ledger
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '. ')
echo "session arch sm_${SM}"

# S2.1 - on-node library built from the current sources and reporting core 4
make -C cuda clean && make -C cuda SM="$SM"
python - <<'PY'
from pilot_proxy.kernel import PilotProxyKernel
v = PilotProxyKernel().version
assert v.major == 4, f"stale kernel library: {v.as_string()}"
print("kernel core", v.as_string())
PY

# S2.2 - CUDA regression + exact row-sum parity (integers, no tolerance)
make -C cuda test_cuda SM="$SM"

# S2.3 - GPU pytest gates: numpy-reference equality, bit-exact coarse marginal
#        identity, and the pre-registered ULP gate on the cupy fine FFT.
#        these fail on a stale library and are never skipped
python -m pytest tests/kernel -q

# full suite, with the GPU and archive tests run instead of skipped
python -m pytest tests -q
```

Frame parity was settled by the 23-channel census over the integrated spectra
of the archive, so no odd-channel profile override is needed. As an optional
check, use any archived baseband file with an odd `freq_id`:

```bash
H5_FILE=/absolute/path/to/odd_channel_baseband.h5
python tools/framing_audit.py "$H5_FILE"
```

The bounded smoke test below is the remote real-file gate. Its pass criteria
on the per-pilot schema and its interrupt and resume check are listed at the
end of that section.

---

## Required inputs

For a local scan, provide CHIME HDF5 baseband files whose names end in the
selected `freq_id`, such as `baseband_<event>_844.h5`. For other layouts, pass
`--source-freq-id-regex '<regex-with-one-capturing-group>'`. The value is
stored as `source_freq_id_regex` for the bundled local source, and an explicit
`--set 'source_freq_id_regex=...'` takes precedence over the flag.

For a CADC/CANFAR scan, provide:

- a valid CADC proxy certificate;
- an `inventory.jsonl` produced by `pilot-proxy chime-survey`;
- the CHIME `freq_id` values to scan, or an inventory that they can be inferred
  from.

The detector resolves these runtime artifacts unless explicit alternatives are
passed:

- `configs/receiver_profiles/chime_dtv_fengine.json` - must be the
  frame-verified `chime_dtv_fengine` profile
  (`baseband_frame.channel_center_normalized = 0.0`);
- `weights/chime_dtv_weights_k128.bin` - manifest
  `receiver_profile_hash` starts `c06744b1b4362d21`;
- `cuda/libpilotproxy.so`, or its staged cache copy;
- a working CuPy/CUDA runtime.

Check the receiver profile and layout before staging archive data:

```bash
pilot-proxy check-profile \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json

pilot-proxy check-layout \
  --receiver-profile configs/receiver_profiles/chime_dtv_fengine.json \
  --stream-map configs/stream_maps/chime_feed_pol_example.json
```

---

## Selection convention

`pilot-proxy chime-scan --select` uses the CHIME `freq_id` coarse-channel
identifier. It does not use the ATSC physical-channel number.

For ATSC physical channels 14 through 36, the corresponding pilot `freq_id`
set is:

```text
506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844
```

For a one-channel smoke test, choose a `freq_id` present in the inventory or
local directory. `844` is the expected coarse channel for the ATSC 14 pilot.
Do not substitute `396-412`, which are not the `freq_id` values of this pilot
selection for DTV channels 14 to 36.

---

## CADC inventory

Renew the CADC proxy certificate:

```bash
read -r -p "CADC username: " CADC_USERNAME
cadc-get-cert -u "$CADC_USERNAME"
```

Begin with a bounded inventory:

```bash
pilot-proxy chime-survey \
  --freq-ids 506,521,537,552,568,583,598,614,629,644,660,675,690,706,721,736,752,767,783,798,813,829,844 \
  --name chime-pilots \
  --max-events 5
```

The survey resolves its output location through PilotProxy's canonical
inventory root and prints the path it wrote, for example:

```text
~/datatrawl-inventories/chime-pilots/inventory.jsonl
```

Inspect the inventory without downloading baseband data:

```bash
pilot-proxy chime-inventory \
  --inventory ~/datatrawl-inventories/chime-pilots/inventory.jsonl
```

With `--inventory-name`, `chime-scan` resolves the inventory through the same
bundled resolver, so it works from any directory. For an inventory stored
elsewhere, pass `--source-root <survey-root>` or `--inventory <path>`
explicitly. Increase `--max-events` only after the bounded scan succeeds.

---

## Bounded detector smoke test

First verify the GPU runtime:

```bash
nvidia-smi
python - <<'PY'
import cupy as cp
print("CuPy", cp.__version__)
print("GPU count", cp.cuda.runtime.getDeviceCount())
PY
```

Then process one file and one full analysis chunk for `freq_id 844`:

```bash
pilot-proxy chime-scan \
  --output-dir "$HOME/pilotproxy_runs/detector_smoke_844" \
  --source cadc-datatrail \
  --inventory-name chime-pilots \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

The file cap intentionally leaves part of the inventory unprocessed, so this
bounded smoke test explicitly acknowledges that with `--allow-partial`.
`scan_scope.json` remains the authoritative record of the omitted units.

Validate the combined products and generate the diagnostic plots:

```bash
pilot-proxy validate-products \
  --run-dir "$HOME/pilotproxy_runs/detector_smoke_844" \
  --output-json "$HOME/pilotproxy_runs/detector_smoke_844/product_validation.json"

pilot-proxy chime-plot \
  --run-dir "$HOME/pilotproxy_runs/detector_smoke_844" \
  --clean-figures
```

After these commands, the run directory should contain:

```text
run_config.json
stats.json
input_manifest.json
product_validation.json
chime_detector_outputs.npz
chime_spectrogram_cache.npz
chime_integrated_spectra.npz
chime_reductions_10s.npz
tables/mask_summary_by_pilot.csv
figures/*.png
```

An event-keyed combine also writes `chime_frame_identity.npz`. Current products
must contain event-keyed frame identity, and combine refuses inputs without it.

Pass criteria for the remote smoke test on the per-pilot schema:

```bash
python - <<'PY'
import numpy as np, glob
import os
p = sorted(glob.glob(os.path.expanduser(
        "~/pilotproxy_runs/detector_smoke_844/**/_per_pilot/844.npz"),
        recursive=True))
z = np.load(p[-1], allow_pickle=False)
sv = str(np.asarray(z["schema_version"]).reshape(()).item())
ev = str(np.asarray(z["source_event_key_schema_version"]).reshape(()).item())
dv = str(np.asarray(z["detector_version"]).reshape(()).item())
fs = str(np.asarray(z["fine_status"]).reshape(()).item())
assert sv == "pilotproxy_per_pilot_product_v6", sv
assert ev == "pilotproxy_namespaced_source_event_key_v1", ev
assert "pilot-proxy/" in dv and "kernel=4." in dv, dv
assert fs == "enabled", fs
assert z["fine_power_u64"].shape[1:] == (3, int(z["fine_num_bins"])), z["fine_power_u64"].shape
assert z["fine_power_u64"].dtype == np.uint64, z["fine_power_u64"].dtype
print("schema", sv)
print("event identity", ev)
print("fine bins", int(z["fine_num_bins"]),
      "| exact fine terms", z["fine_power_u64"].shape)
PY
```

The product stores only the exact fine terms. The null-bulk exceedance
fraction, like every other derived fine quantity, is recomputed in
post-processing from `fine_power_u64`. It is an in-sample threshold diagnostic
and gives no independent measure of the false-alarm rate. Read it
together with the stored fine terms.

For the resume half of the gate, run the same `chime-scan` command in a new
output directory with `--max-files 2` and no chunk cap. The smoke product above
was built with `max_chunks_per_file=1`, and a capped product refuses completion
under a different cap by design. Interrupt the run after the first per-pilot
checkpoint lands (watch for `_per_pilot/<freq_id>.npz` in the output
directory), run the identical command again, and confirm that the frame count
continues without duplication and that `validate-products` passes on the
result. If the bounded run finishes before it can be interrupted, relaunch the
identical command with a higher `--max-files` on the same directory. The
relaunch restores the checkpoint and continues through the same resume path.
Keep `--allow-partial` on these bounded resume commands, and remove both the
file cap and that flag for the production run.

---

## Optional remote bootstrap zero-point diagnostic

This diagnostic is not a gate for the approved local run. For a remote study
with its own approval, test the detector on a channel that should approximate
the no-pilot hypothesis, H0. Choose a DTV pilot frequency that lies in the
selected CHIME coarse channel but whose physical channel has no station listed
in the 500-mile census. The control is selected from the census, with no
propagation prediction. Do not choose an arbitrary coarse channel with no
nominal ATSC pilot in band, since the analyzer marks that case invalid and
forms no local-reference power ratio.

Two constraints bound that selection for this archive. First, the census epoch
must match the data epoch. The shipped census reflects its 2026 retrieval date,
while archive events span earlier years, and a channel that is quiet in the
census can have been occupied at observation time. The interior analysis of
channel 27 (`freq_id` 644) records 38 percent positive-excess bootstrap flags
in 2020 Q3 against quiet endpoints in 2025 and 2026. Second, the shipped census
lists at least one station on every physical channel from 14 to 36, so this
pilot set has no strictly station-free control. Use the most isolated channels
as approximate controls, and take the operating zero point from an empirical
calibration instead of a census assumption.

Run the bounded scan for the census-control channel and for one quiet channel
with a known pilot. Then:

1. Read `null_power_ratio` from `chime_detector_outputs.npz`. The
   `_per_pilot/<freq_id>.npz` product does not store it, so derive it there
   from the stored norms as `2*target_norm_sq/reference_norm_sum_sq` (see
   `null_power_ratio_from_weight_norms` in
   `src/pilot_proxy/detector_contract.py`). The `chime-run` batch path also records
   `null_power_ratio_by_channel` in `stats.json`, but the combined `chime-scan`
   statistics do not copy that array.
2. Over frames with `valid = 1`, compare the mean `coarse_power_ratio` with
   `null_power_ratio`, and inspect the valid-frame mask fraction. Under the
   tested white-noise model, the norm-corrected zero-excess boundary gives a
   fraction near one half. On real data, this fraction is a diagnostic
   expectation and never a pass condition by itself.
3. If the control result is strongly displaced, check the weight bank,
   `mask_rule`, channel selection, and structured interference before expanding
   the scan.

`tests/core/test_mask_zero_point.py` performs the corresponding white-noise
regression with the shipped weights. The on-sky check tests the additional
instrument and archive path that the synthetic regression cannot cover.

## Remote long-run checklist

Use this checklist only for a remote run with its own approval. It records the
failure modes of any remote GPU session. Its on-node binary, session cadence,
and checkpoint settings belong to the remote path and are never local
production settings.

1. Run every remote gate in full. The remote build, GPU, smoke, and interrupt
   and resume checks are run with no shortcuts, since a multi-day run will
   cross session restarts, and resume is what makes a session death cost
   minutes instead of days.
2. Preserve the binary at launch. Detector products pin `kernel_sha256`, and
   kernel builds are not byte-reproducible, since identical sources, flags, and
   toolkit produce different bytes on each invocation (nvcc embeds
   per-invocation artifacts). Thus, the preserved copy must be used, and a
   rebuild can never replace it:

   ```bash
   kernel_sha=$(sha256sum cuda/libpilotproxy.so | awk '{print $1}')
   KERNEL_LIB="$PWD/cuda/libpilotproxy-sm${SM}-${kernel_sha:0:16}.so"
   cp --no-clobber --preserve=mode,timestamps cuda/libpilotproxy.so "$KERNEL_LIB"
   cmp -s cuda/libpilotproxy.so "$KERNEL_LIB" || exit 1
   export KERNEL_LIB
   sha256sum cuda/libpilotproxy.so "$KERNEL_LIB"
   ```

   Every initial launch and resume passes `--lib-path "$KERNEL_LIB"`.
3. Check inventory completeness before the run. Confirm that the survey
   inventory covers every selected `freq_id` (all 23 for the DTV band), and
   explain any channel whose unit selection looks anomalous. Sparse channels
   and late-starting spans are acceptable only when the archive holds no more
   data, and this must be confirmed before the run. The source survey has zero
   terminally incomplete events. Three events were still pending after two
   attempts, and an authenticated metadata check found all 23 selected files
   absent for each of them, with no errors or sub-floor objects. The
   supplemental resolution is part of the frozen bundle. Its sparse channels
   are `freq_id` 598 (1543 units, ending 2023-09-13) and 690 (1767 units,
   ending 2026-04-16). Record these as archive coverage, since no processing
   loss is involved.
4. Plan around the session lifetime. Know why the previous scan session ended
   (expiry or crash), and budget the relaunch cadence around the session
   lifetime. Checkpointing (`--checkpoint-every 50`) with unit-level resume
   bounds the loss, provided items 1 and 2 hold.
5. Stop at the first product. After about the first 50 units of the first
   channel, stop and check the product before the remaining channels run for
   days. The measured pilot line must be at its known offset (`freq_id` 506:
   KSKN at +739 Hz), the coarse excess must agree with the prior cohort, and
   the product must pass validation. These stop criteria are checked in the
   first hour, while a failure still costs little.
6. Accept each channel while the run continues. Validate and audit each
   product as it completes. `tools/audit_per_pilot.py` re-derives every
   internally checkable quantity from first principles and is safe on live
   checkpoints. Set `RUN_DIR` to the run path, and check the heartbeat with
   `find "$RUN_DIR" -type f -mmin -60 -print -quit` at least daily.
7. Record the cohort. The products of each run pin one binary and one source
   commit. Record the hash, commit, image and CUDA version, and channel list in
   the run ledger at launch, and never mix cohorts in one output directory.
8. Check `cupy` on the node as well as the `.so`. The detector pre-flight
   fails closed with "cupy/CUDA is not importable" before any unit is
   staged, so a node with a freshly built `cuda/libpilotproxy.so` can still
   be unusable. Check it before launch:

   ```bash
   python -c "import cupy; print(cupy.__version__)"
   ```

9. Read the quarantine ledger before accepting the run. A unit whose bytes
   cannot yield one complete transform is rejected at probe time and
   quarantined, so the run continues past it instead of aborting. Quarantine,
   however, is persistent, and a later run skips that unit without examining
   it again. The scan prints each such unit as `QUARANTINE <name>: ...`, and
   the scope gate reports `quarantined=<n>`. Confirm that every entry is a
   short acquisition (the file opens cleanly and holds its full declared
   extent) and that none is a staging problem. The completed inventory has 170377
   units. Prior products show 4692 units with no complete frame, and the
   existing quarantine contains three corrupt units. One corrupt unit is also
   below the one-frame size estimate of the inventory. A frozen inventory that
   excludes all 4695 unusable units therefore has 165682 units in 8983
   events. Preserve the full inventory with an exclusion ledger beside it. The
   approved archive run uses the frozen inventory on either path. Use of the
   full inventory would require a new approval for `--allow-partial` and an
   exact final-quarantine audit.
10. Expect the terminal combine over all channels to be empty. The stack keeps
   only the `(event, frame)` identities common to every selected pilot, and a
   triggered event lights only a few surrounding channels of the 23. Over 115
   measured events on the 23 DTV channels, no event was present in all of
   them, and this outcome is expected. The approved terminal deliverable is
   all 23 per-pilot v5 products. Build any channel subset afterward as a
   derived product, and never restrict the archive scan for it.

---

## Approved full archive run

The approved archive run has two qualified paths, and the operator picks
exactly one for the whole run. The local path is authoritative as written. Its
production command is in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md), and
nothing in this file overrides the local parameter register, the SM89 gates,
the controls of four workers and eight slots, the 250-unit checkpoint, or the
output paths. The remote path is qualified on the GPU architecture of its own
session, with its own on-node kernel build and digest, its own gate evidence,
and its own topology, and none of these are local settings. Both paths share
the frozen source revision, package-source digest, weight bank, and frozen
inventory. Products from the two paths pin different kernel digests, so they
must never be mixed in one cohort or output directory (checklist item 7).

The authoritative results are all 23 per-pilot v5 products. A merged
common-frame stack is a derived projection and may be empty when archive
coverage is ragged.

---

## Local staged-data equivalent

For HDF5 data already on disk, run the same detector analyzer through the local
source:

```bash
export LOCAL_H5=/path/to/chime_hdf5

pilot-proxy chime-scan \
  --input-dir "$LOCAL_H5" \
  --output-dir "$HOME/pilotproxy_runs/local_detector_smoke_844" \
  --source local \
  --analyzer pilot-proxy-detector \
  --select 844 \
  --max-files 1 \
  --max-chunks-per-file 1 \
  --allow-partial
```

If the filenames do not end in `_<freq_id>.h5`, add
`--source-freq-id-regex '<regex-with-one-capturing-group>'` to the complete
`chime-scan` command.

The flag is stored as `source_freq_id_regex` for the bundled local source, and
an explicit `--set 'source_freq_id_regex=...'` takes precedence over the flag.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi: command not found` | CPU-only host or unavailable driver utility | Move the detector run to a GPU node |
| `nvcc: command not found` | CUDA compiler toolkit is not on `PATH` | Load the CUDA toolkit/module or set `NVCC`/`PATH` |
| `pilot-proxy-detector needs cupy/CUDA` | Production detector started in a CPU-only environment | Use a GPU node |
| `no files matched` | `--select` does not match the inventory or local filename `freq_id` | Run `pilot-proxy chime-inventory` or inspect the filenames |
| first file's center implies a different `freq_id` | Inventory or filename label disagrees with HDF5 metadata | For local data, pass the current parser with `--set source_freq_id_regex=...`; for archive data, rebuild the inventory |
| combine finds no common events | The selected channels contain different event sets | Run `pilot-proxy chime-combine --work-dir "$RUN_DIR/_per_pilot" --report`, choose a stated subset, and recombine with `--drop` |
| all frames are invalid | The selected coarse channel does not contain the nominal pilot, or the reference denominator is zero | Check `freq_id`, HDF5 frequency metadata, and the detector weights |

---

## Restart policy

Use a new output directory when the existing products are suspect. Resume only
when the partial run is known to have the same channel selection, frame cap,
weights, detector geometry, and provenance. The analyzer rejects several
incompatible resume cases, but that validation does not classify a scientifically
bad run.

Detector checkpoints bind the PilotProxy source tree and pin the kernel by hash
(`kernel_sha256=` in `detector_version`), so a code change or a kernel rebuild
during a run makes the existing partial products refuse to resume. Preserve the
current library under a digest-derived filename with the launch procedure
above, and resume with `--lib-path <preserved copy>`. After a source change,
start a fresh output directory. Move to a new source tree or kernel only at a
run boundary.

Keep a failed run until its failure has been classified. Do not commit generated
products while diagnosing the run.

---

## Archive policy

For each accepted run, archive:

- the per-pilot products and their audit record;
- `product_validation.json` when terminal combine succeeds;
- the receiver profile and any stream map used;
- the weight bank manifest;
- the source commit hash or source archive;
- the exact CADC inventory used.

Do not commit CANFAR products, local HDF5 files, generated figures, or CUDA build
artifacts to the source repository.

## Inventory and resume compatibility

Completed `inventory.jsonl` files remain readable, and this covers inventories
under the default `~/datatrawl-inventories` path. Do not resume an in-progress survey state
written by another checkout, and use a fresh `pilot-proxy chime-survey --name`
or `--out` instead. Detector checkpoints fail closed after source changes, so
use a fresh scan output directory after updating the checkout. See
[INTEGRATION.md](../INTEGRATION.md#inventory-and-resume-compatibility).

## Sharded runs

The archive scan of record ran as channel-partitioned shards, one per GPU
session, with per-shard gates, tripwires, and an external supervisor that
resumes a shard across archive outages. Its scripts, with the digests they
assert, are in [`scripts/canfar/`](../scripts/canfar/README.md), and the run
ledger records what each of them found.
