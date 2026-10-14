# Survey identity and recovery contract

New surveys preserve `survey_manifest.json` as a stable configuration guard.
Software versions, the source revision and source-tree digest, public certificate
expiry/fingerprint, start/end time, result inventory hash and row count belong to
separate execution records in `survey_runs/`. Resuming creates a new execution
record and updates `latest_run`; it does not silently rewrite the producer of
an earlier run. Software identity follows the package's first-observation per
process contract. A commit alone does not describe uncommitted source; use the
source digest alongside it. Missing optional provenance is explicitly null.
Certificate fingerprints cover only the public certificate, never private keys.

`inventory.meta.json` identifies its metadata writer, hashes the inventory it
actually describes and points to available execution history. Adding metadata
to an old inventory does not recover its original survey toolchain. Run history
is not a row-by-row reconstruction of the producer of a legacy inventory.

A survey holds the existing exclusive output lock. Before working, it writes
`.survey.views-stale`; after the final SQLite-to-JSON/text render it removes the
marker. The marker survives SIGKILL. Inventory enumeration refuses marked views.
Rerun the same survey command and configuration to refresh views and continue;
do not remove the marker to certify a stale file. An ordinary interruption
records its outcome, refreshes committed views when possible, and prints the
resume instruction. An unfinished execution record has no known finish time.

Archive exclusions have different meanings:

- `no-selected-candidates`: the reader selected no files. It is an explicit
  `empty-selection` disposition and fails strict completeness; it proves nothing
  about archive holdings.
- `below-reader-floor`: objects exist but are too small for the reader. The
  ledger retains their names, sizes and byte floor separately from absent counts.
- `aged-out` / `max-attempts`: the existing empty-event policy accepted an event
  whose selected objects were absent, under its recorded age/retry rule.
- `unverified-restored-collection`: a bare replica needed a collection prefix,
  and no selected object could establish that inferred location. The event stays
  pending regardless of its age or number of retries; details live in
  `pending_events.json`. A later successful pass clears the pending entry.

Bare `data/` paths are restored only when exactly one collection is configured.
Resolved rows record `collection_restored`. Multiple possible collections and
parent traversal are refused. This records a reconstruction, not independent
archive provenance; source-qualified replies remain preferable.

The read-only legacy adapter reconstructs `baseband_<event>_<freq_id>.h5` only
for documented schema-1 CHIME raw-baseband rows with matching event-directory
identity and geometry fields. It accepts the old trailing-directory slash,
labels the compatibility conversion and preserves the old floating `n_frames`
value as a legacy estimate. It does not turn that size-derived estimate into
an observed HDF5 frame count, rewrite old files, or permit unrelated partial rows.

The regression suite includes real SIGINT/SIGKILL and concurrent-output tests.
It uses local archive fakes; passing it does not verify remote CADC availability
or relabel any historical CANFAR product.
