# Validation gates for the archive run

These are the mandatory gates for the approved archive estimation and
sufficient-statistic reprocessing run, on either qualified path: the local WSL
host or a CANFAR GPU session. The coarse positive-excess flag is
retained as a bootstrap diagnostic; the fine rank/eta decision is inactive and
no calibrated detection policy is applied.

The run of record passed these gates with its preserved detector core 2.3.0
library (digests in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md)) and wrote
schema-v5 products. The current code binds detector core 4 and writes schema v6
(see [`PRODUCT_SCHEMA.md`](PRODUCT_SCHEMA.md)); the blocks below check that
build.

Gate evidence must match one clean source revision, its package-source digest,
and the exact detector library that will run: the preserved SM89 artifact on
the local path, or the node-built SM90 artifact on the CANFAR path. A source
change after a gate invalidates that gate, and so does rebuilding the library.
The scientific and product pins are in
[`RERUN_PARAMETER_REGISTER.md`](RERUN_PARAMETER_REGISTER.md), the local
production command is in [`LOCAL_PROCESSING.md`](LOCAL_PROCESSING.md), the
CANFAR production commands are in [`CANFAR_RUNBOOK.md`](CANFAR_RUNBOOK.md), and
the actual values belong in the run ledger for the launched path.

## Ground truth already closed

The frame correction and 23-channel parity census established the CHIME frame
convention and the DC-centered weight bank, which the shipped receiver profile
and weight manifest encode. The supporting gates are:

1. `tools/framing_audit.py` reports `VERDICT: ALIGNED` for event 68399317,
   `freq_id` 844 when run with four chunks.
2. The production-geometry NumPy replica recovers the known signal with a mean
   coarse excess above 1200 percent, while a bank built for the half-band frame
   convention stays near the null.
3. The parity census covers all 23 selected channels. An additional odd-channel
   framing audit is optional.
4. The shipped CHIME bank selects `K = 128`, one skipped guard cell, and a
   two-cell reference offset.

Changing the detector window, weights, references, repack, frame convention, or
fine transform creates a new run contract and requires new approval.

## Final-source CPU gates

Run these from a clean checkout of the exact revision recorded in the ledger:

```bash
test -z "$(git status --porcelain)"
git rev-parse HEAD
PYTHONPATH=src python - <<'PY'
from pilot_proxy.provenance import package_source_sha256
print(package_source_sha256())
PY
PYTHONPATH=src python -m pytest tests -q
make -C cuda test_ref
```

Pass criteria:

- the source tree is clean;
- the complete test tree has zero failures;
- optional tests skip only when their documented external dependency is absent;
- the CPU kernel reference exits zero; and
- the commit and package-source digest are recorded in the ledger.

## Final-source RTX/SM89 gates

The approved host is the local RTX 5000 Ada workstation. Do not replace the
already preserved library during this run. The explicit load below and the
real-file gates exercise that exact artifact. The later build and kernel-test
commands exercise a disposable SM89 build from the same frozen CUDA source.

```bash
: "${KERNEL_LIB:?path of the preserved core 4 SM89 library}"
: "${KERNEL_SHA256:?sha256 of that library}"
: "${KERNEL_MANIFEST:?path of its preservation manifest}"
: "${KERNEL_MANIFEST_SHA256:?sha256 of that manifest}"
printf '%s  %s\n' \
  "$KERNEL_SHA256" "$KERNEL_LIB" \
  "$KERNEL_MANIFEST_SHA256" "$KERNEL_MANIFEST" | sha256sum --check --strict

PYTHONPATH=src python - "$KERNEL_LIB" <<'PY'
import sys
from pilot_proxy.kernel import PilotProxyKernel

kernel = PilotProxyKernel(sys.argv[1])
assert kernel.version.major == 4
assert kernel.supports_fine_powers()
assert kernel.supports_fused_fine()
print("kernel", kernel.version.as_string(), kernel.get_fine_specs())
PY

make -C cuda test_cuda SM=89
PYTHONPATH=src python -m pytest tests/kernel -q
```

Pass criteria:

- the preserved byte digest matches;
- the library reports detector core 4 and the required exact fine capabilities;
- the disposable SM89 CUDA reference and fixed-point parity tests exit zero;
- the kernel test directory has zero failures and zero skips against that
  disposable build; and
- the preserved bytes pass the explicit capability check and real-file gates.

## Final-source H100/SM90 gates

The qualified alternate host is a CANFAR H100 GPU session. The preserved SM89
artifact cannot execute there. Build the library on the node from the same
frozen CUDA source, then pin those bytes by digest for the life of the run.

```bash
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '. ')
make -C cuda clean && make -C cuda SM="$SM"
# nvcc is not byte-reproducible: this build IS the artifact. Preserve it
# outside the checkout, gate that file, and record its digest in the ledger.
KERNEL_SHA=$(sha256sum cuda/libpilotproxy.so | cut -d' ' -f1)
KERNEL_LIB="$HOME/pp_run/kernel/libpilotproxy-sm${SM}-${KERNEL_SHA:0:12}.so"
mkdir -p "$(dirname "$KERNEL_LIB")" && cp cuda/libpilotproxy.so "$KERNEL_LIB"
echo "record in the ledger: $KERNEL_LIB  $KERNEL_SHA"

PYTHONPATH=src python - "$KERNEL_LIB" <<'PYEOF'
import sys
from pilot_proxy.kernel import PilotProxyKernel

kernel = PilotProxyKernel(sys.argv[1])
assert kernel.version.major == 4
assert kernel.supports_fine_powers()
assert kernel.supports_fused_fine()
print("kernel", kernel.version.as_string(), kernel.get_fine_specs())
PYEOF

make -C cuda test_cuda SM=90
PYTHONPATH=src python -m pytest tests/kernel -q
```

Pass criteria:

- the build used the frozen CUDA source and the default detector configuration;
- the library reports detector core 4 and the required exact fine capabilities;
- the SM90 CUDA reference and fixed-point parity tests exit zero;
- the kernel test directory has zero failures and zero skips against that
  build;
- the built digest is recorded in the ledger and rechecked at every resume; and
- rebuilding the library ends this chain: repeat every gate above and record the
  new digest before resuming.

## Real-file and resume gates

Repeat both gates after the source revision is frozen:

1. Process one full detector chunk from the 2048-stream `freq_id` 844 file.
   Require product validation, nonempty receiver-state identity, and peak VRAM
   below 13,900 MiB.
2. Process eight archive units with four download workers, eight staged-file
   slots, and a two-unit rehearsal checkpoint. Interrupt after a durable
   checkpoint, rerun the identical command, and require eight unique unit keys,
   no duplicate frames, valid v5 products, and empty staging.
3. On a path that runs concurrent shards, run the production shard count at
   once, one distinct `freq_id` per shard, each with its own output and staging
   directory. Require every shard's products to validate, each shard to hold
   exactly one distinct physical channel, no unit key or frame duplicated
   across shards, no shard reading or writing another shard's output or
   staging, and the summed peak device memory to stay inside the gate 1
   headroom bound. This gate does not apply to a single-process path.

The exact commands and evidence paths are in `LOCAL_PROCESSING.md`. These gates
must embed the final package-source digest and preserved kernel digest. A capped
rehearsal output is never reused for production.

For every per-pilot v5 product inspected during these gates, require:

- all valid frames remain in the per-frame arrays on both sides of zero excess;
- exact target, lower-reference, upper-reference, and summed-reference powers;
- exact fine powers with shape `[N, 3, 256]` and unsigned 64-bit dtype;
- one nonempty `unit_scope`, raw `unit_git_version_tag`, and
  `unit_input_map_sha256` per unit;
- every input-map identity is a 64-character lowercase SHA-256 digest;
- `fine_status == enabled` and an inactive fine candidate decision;
- the positive-excess bit affects only the recorded bootstrap flag and the
  after-mask diagnostic spectrum; and
- the source, kernel, profile, bank, and inventory identities match the ledger.

The authoritative products are `_per_pilot/<freq_id>.npz`. The combined product
is a derived common-frame projection and is not a substitute for them.

## Production settings

The approved local command uses:

- the frozen 165,682-unit inventory and all 23 selected `freq_id` values;
- four download workers and eight staged-file slots;
- a 250-unit production checkpoint interval;
- the preserved SM89 library named above;
- `fine_products=on`; this retains exact fine powers and does not enable fine
  detection;
- no file cap, no chunk cap, and no partial-run acknowledgement; and
- a fresh output directory outside the checkout.

The approved CANFAR command uses, per shard:

- the same frozen 165,682-unit inventory, one `freq_id` per shard, with the
  union over shards covering all 23 selected values;
- five concurrent shards, the VRAM-bound maximum for the session budget;
- four download workers and sixteen staged-file slots;
- a 50-unit production checkpoint interval;
- the node-built SM90 library pinned by digest above;
- `fine_products=on`; this retains exact fine powers and does not enable fine
  detection;
- staging on `/scratch`, not `/arc`;
- no file cap, no chunk cap, and no partial-run acknowledgement; and
- a fresh output directory and a fresh staging directory per shard, outside the
  checkout.

The full synthetic detection frontier, residual-contamination calibration,
eta selection at the fixed median rank and final detection policy are separate
from this run. They are not prelaunch gates because this run preserves the sufficient statistics needed to
perform them without reacquiring the archive.
