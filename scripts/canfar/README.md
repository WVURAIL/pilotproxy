# CANFAR shard tooling

These scripts ran the CHIME pilot-tone archive scan on CANFAR as channel-partitioned
shards, and they are kept as the record of that run. The paths under
`/arc/home/$CANFAR_USER/`, the session hostnames, the channel partition and every
digest they assert are specific to that run and are read alongside its run ledger.

`CANFAR_USER` is the CANFAR account name. It is not committed, so it must be
exported before running, and every shell script here stops if it is unset.

The session image must be `images.canfar.net/skaha/astroml-cuda:latest`. The
plain `astroml` image has a GPU and driver but no CUDA toolkit, so every gate
passes and the scan then dies on its first file in cupy's JIT. Such a
session is refused by the bootstrap at its image check, before any other setup.

## Scripts in run order

| step | script | role |
|---|---|---|
| 1 | `canfar_probe_bootstrap.sh` | Per-node setup: refuse a toolkit-less image, clone at the frozen revision, restore the runtime offline from the freeze bundle's wheelhouse, reuse or build the node's kernel, measure fetch throughput. Once per session. |
| 2 | `canfar_smoke_844.sh` | Cross-arch qualification: rerun the local rehearsal (channel 844, first 8 files) and compare the product unit by unit against the local sm89 reference. Integer fields must be bit-identical. |
| 3 | `canfar_shard.sh` | Shard controller: `update`, `gate`, `launch`, `resume`, `run`, `tripwire`, `status`, `stop`. Every launch and resume re-runs the full gate chain, which has an md5-verified byte fetch over the route the scan will use and a cupy JIT compile. |
| 4 | `canfar_supervise.sh` | Keeps a shard running across archive outages and transient fetch failures, launching it if it has never run and resuming it otherwise. Distinguishes a refused gate (nothing ran) from a scan that ran and exited. |
| 5 | `canfar_merge_channel.sh` | Hands a channel finished by a helper shard to the shard that owns it, so the owner skips it instead of re-fetching. Checks run-wide identity and that the product is the channel it claims. |
| | `canfar_compare_duplicates.py` | Compares the channels two shards both built, field by field and NaN-aware. The byte-identical reproduction result in the ledger comes from this script. |
| 6 | `canfar_closeout.sh` | Assembles the 23 canonical products, split across two non-overlapping shard directories, into one run directory. |

The qualified sm90 kernel, the `pp_switch` kit (inventory, launcher scripts) and
the runtime freeze tar are kept in the campaign's `kit/` directory beside the
products and the run ledger, on the WVU RAIL OneDrive under
`Datasets/pilot-tone-pipeline/products/chime_pilots_rebuild_20260829/kit/`.

## After the closeout

`canfar_closeout.sh assemble` assembles and verifies the run directory and does
nothing else. The `_per_pilot/` products of the assembled run directory are the
input of the threshold pipeline (see `docs/CHIME_RUN_WORKFLOW.md`).

## Operating notes

* A bash async job (`nohup cmd &`) inherits SIGINT ignored, so scans are
  started with `setsid -f` and stopped with INT then TERM.
* `/minoc/capabilities` stays at HTTP 200 through outages. Only a real byte
  fetch or `/raven/availability` shows that the archive is serving, and even
  `/raven/availability` can report healthy during an interruption.
* A pinned replica must be throughput-probed before use. UVic measured
  0.18-1.7 MiB/s, far below the 15-187 MiB/s of raven when both were healthy.
* Removing a staging directory under a live scan permanently quarantines the
  units in flight, so staging is removed only after `stop` confirms exit.
* One dead download worker ends the scan, and not every transient is retried
  in `fetch()`. `canfar_supervise.sh` restarts the scan for this reason.
* Launching a session rewrites the shared 30-day certificate with a 7-day
  one, so the certificate is re-minted after every launch.
* `/arc` in a running session serves a cached copy, and a freshly uploaded
  script is not immediately visible there. The current version can be run at
  once with `git show origin/main:<path> > /tmp/<name>`.

The general procedure is in `docs/CANFAR_RUNBOOK.md`.
