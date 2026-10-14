# Scientific provenance snapshots

This tree is an explicit exception to the repository's normal generated-output
policy. It contains small, dated evidence snapshots needed to reproduce or
audit manuscript and dissertation claims. A file's presence here does not make
it an active runtime product or a current software contract.

Each dated directory must state:

- what claim or decision the snapshot supports;
- which files are committed and how they were produced;
- which inputs or larger result bundles remain external; and
- whether the evidence is complete, provisional, or historical.

Do not overwrite a dated snapshot with a rerun. Add a new dated directory and
link the superseding record from both READMEs. Large raw captures, ordinary
scan products, and rebuildable intermediate arrays remain outside git; retain
their hashes and archive locations in the corresponding snapshot instead.

One earlier snapshot was removed on 2026-10-07. It is kept only in the private provenance repository (see `PROVENANCE.md`).
