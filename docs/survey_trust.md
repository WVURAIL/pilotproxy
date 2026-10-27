# Survey identity and recovery contract

The purpose of this contract is to record which software wrote each survey and
to let an interrupted survey resume safely. A new survey keeps
`survey_manifest.json` as a stable configuration guard. Each execution writes
its own record in `survey_runs/` with the software versions, the source revision
and source-tree digest, the expiry and fingerprint of the public certificate,
the start and end times, and the hash and row count of the result inventory. A
resume creates a new execution record and updates `latest_run`, and the producer
recorded for an earlier run is never rewritten. Software identity is taken at
its first observation in each process, as in the rest of the package. A commit
does not describe uncommitted source, so the source digest is recorded beside
it. Missing optional provenance is written as null. Certificate fingerprints
cover the public certificate only, and private keys are never fingerprinted.

`inventory.meta.json` names its metadata writer, hashes the inventory it
describes, and points to the available execution history. Adding metadata to an
existing inventory does not recover the survey toolchain that wrote it, and the
run history does not record which run wrote each inventory row.

A survey holds the existing exclusive output lock. The marker
`.survey.views-stale` is written before any work and removed after the final
render from SQLite to the JSON and text views. The marker survives SIGKILL, and
inventory enumeration refuses views that carry it. Rerunning the same survey
command with the same configuration refreshes the views and continues the
survey. The marker is never removed by hand, since the view it marks is still
stale. An ordinary interruption records its outcome, refreshes the committed
views when possible, and prints the resume instruction. An unfinished execution
record has no finish time.

Each archive exclusion code has its own meaning:

- `no-selected-candidates`: the reader selected no files. The event is given an
  explicit `empty-selection` disposition and fails strict completeness. The code
  carries no information about the archive holdings.
- `below-reader-floor`: the objects exist but are too small for the reader. The
  ledger keeps their names, sizes, and byte floor apart from the absent counts.
- `aged-out` / `max-attempts`: the existing empty-event policy accepted an event
  whose selected objects were absent, under its recorded age and retry rule.
- `unverified-restored-collection`: a bare replica needed a collection prefix,
  and no selected object could confirm the inferred location. The event stays
  pending at any age or retry count, with the details in `pending_events.json`.
  A later successful pass clears the pending entry.

Bare `data/` paths are restored only when exactly one collection is configured,
and each resolved row records `collection_restored`. A path that could belong to
more than one collection, or that uses parent traversal, is refused. The
restored path is a reconstruction made by this package, and a reply that names
its source collection remains the better provenance.

The read-only schema-1 adapter reconstructs `baseband_<event>_<freq_id>.h5` only
for documented schema-1 CHIME raw-baseband rows whose event directory and
geometry fields match. It accepts a trailing slash on the directory and labels
the converted rows. The floating `n_frames` value is derived from the file size
and is kept as an estimate, never as an observed HDF5 frame count. The adapter
rewrites no existing file and refuses unrelated partial rows.

The regression suite sends real SIGINT and SIGKILL signals and runs
concurrent-output tests against local archive fakes. A passing suite therefore
says nothing about remote CADC availability, and it changes no label on an
existing CANFAR product.
