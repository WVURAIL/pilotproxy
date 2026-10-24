# CANFAR shard tooling

These scripts ran the CHIME pilot-tone archive scan on CANFAR as channel-partitioned
shards. They are kept as the record of that run rather than as a general deployment
kit: paths under `/arc/home/$CANFAR_USER/`, the session hostnames, the channel
partition, and every digest they assert are specific to that run and are read
alongside its run ledger.

`CANFAR_USER` is the CANFAR account name. It is not committed: export it
before running, and every shell script here stops if it is unset.

**Session image must be `images.canfar.net/skaha/astroml-cuda:latest`.** The
plain `astroml` image has a GPU and driver but no CUDA toolkit: every gate
passes and the scan then dies on its first file inside cupy's JIT. The
bootstrap refuses such a session up front.

## The flow

| step | script | role |
|---|---|---|
| 1 | `canfar_probe_bootstrap.sh` | Per-node setup: refuse a toolkit-less image, clone at the frozen revision, restore the runtime offline from the freeze bundle's wheelhouse, reuse or build the node's kernel, measure fetch throughput. Once per session. |
| 2 | `canfar_smoke_844.sh` | Cross-arch qualification: rerun the local rehearsal (channel 844, first 8 files) and compare the product unit-by-unit against the local sm89 reference. Integer fields must be bit-identical. |
| 3 | `canfar_shard.sh` | Shard controller: `update`, `gate`, `launch`, `resume`, `run`, `tripwire`, `status`, `stop`. Every launch and resume re-runs the full gate chain, ending in an md5-verified byte fetch over the route the scan will use, and a cupy JIT compile. |
| 4 | `canfar_supervise.sh` | Keeps a shard running across archive outages and transient fetch failures, launching it if it has never run and resuming it otherwise. Distinguishes a refused gate (nothing ran) from a scan that ran and exited. |
| 5 | `canfar_merge_channel.sh` | Hands a channel finished by a helper shard to the shard that owns it, so the owner skips it instead of re-fetching. Checks run-wide identity and that the product is the channel it claims. |
| — | `canfar_compare_duplicates.py` | Field-level, NaN-aware comparison of the channels two shards both built; the basis of the byte-identical reproduction result in the ledger. |
| 6 | `canfar_closeout.sh` | Assembles the 23 canonical products, split across two non-overlapping shard directories, into one run directory. |

The qualified sm90 kernel, the `pp_switch` kit (inventory, launcher scripts) and
the runtime freeze tar are kept in the campaign's `kit/` directory beside the
products and the run ledger, on the WVU RAIL OneDrive under
`Datasets/pilot-tone-pipeline/products/chime_pilots_rebuild_20260829/kit/`.

## After the closeout

`canfar_closeout.sh assemble` only assembles. The assembled run directory's
`_per_pilot/` products are the input of the threshold pipeline (see
`docs/CHIME_RUN_WORKFLOW.md`).

## Operating notes

* a bash async job (`nohup cmd &`) inherits SIGINT ignored, so scans are
  started with `setsid -f` and stopped with INT then TERM;
* `/minoc/capabilities` stays HTTP 200 through outages -- only a real byte
  fetch, or `/raven/availability`, proves the archive is serving, and even
  `/raven/availability` can report healthy during an interruption;
* a pinned replica must be throughput-probed before use: UVic measured
  0.18-1.7 MiB/s against raven's 15-187 when both were healthy;
* removing a staging directory under a live scan permanently quarantines the
  units in flight -- never remove staging until `stop` confirms exit;
* one dead download worker ends the scan, and not every transient is retried
  inside `fetch()`, which is why the supervisor exists;
* launching a session rewrites the shared 30-day certificate with a 7-day
  one, so re-mint after every launch;
* `/arc` in a running session serves a cached copy: a freshly uploaded script
  is not immediately visible there. To run the current version now, use
  `git show origin/main:<path> > /tmp/<name>`.

See `docs/CANFAR_RUNBOOK.md` for the general procedure.
