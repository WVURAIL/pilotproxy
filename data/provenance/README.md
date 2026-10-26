# Scientific provenance snapshots

This tree is an explicit exception to the generated-output policy of the
repository. It holds small, dated evidence snapshots that are needed to
reproduce or audit the claims of the manuscripts and the dissertation. The
files here are records, and none of them is an active runtime product or a
current software contract.

Each dated directory must state:

- the claim or decision the snapshot supports
- the files that are committed and how they were produced
- the inputs or larger result bundles that stay external
- the status of the evidence (complete, provisional, or historical)

A dated snapshot is never overwritten with a rerun. The rerun goes in a new
dated directory, and both READMEs link to the record that supersedes the old
one. Large raw captures, ordinary scan products, and rebuildable intermediate
arrays stay out of git, and their hashes and archive locations are kept in the
snapshot they belong to.

One earlier snapshot was removed on 2026-10-07. It is kept only in the private
provenance repository (see `PROVENANCE.md`).
