# Provenance

This repository, WVURAIL/pilotproxy, starts from one snapshot commit. Its first
(root) commit is the tree of the private development repository at:

| This repository | Private provenance repository |
|---|---|
| first commit (`git rev-list --max-parents=0 HEAD`) | `WVURAIL/pilot-proxy@76b5ecf4937bb89a340ac8dc31a2ebb71ff41c26` (main, 2026-10-14; private) |

The private repository keeps the full development history and is the record of
provenance. The commits cited in the papers, in the dissertation, in kotekan's vendored
copy (`external/pilotproxy/VENDOR.json`), in the provenance stamps of products made
before this snapshot, and in this tree as `<path>@<commit>` (for example in
`tests/data/removed_paths.txt`, `weights/README.md` and the frozen records) are
commits of that repository. They do not exist here. Readers who need them, such as
the dissertation committee, can be given access on request.

## What differs from the snapshot

- Material that was never meant for release (a manuscript draft and its working
  notes, and an archive listing) was removed from the private repository's tree
  before the snapshot and is not in this history. Rows of
  `tests/data/removed_paths.txt` that named it were dropped.
- Real CHIME/FRB event numbers used as examples in tests, docs and one docstring were
  replaced with made-up ones of the same length and order.
- Workstation paths in docs and tests became `~/rail` examples or environment
  variables (`PILOT_PROXY_ARCHIVE_BUNDLE`, `PILOT_PROXY_JULY_INVENTORIES`); the tests
  that need those inputs skip without them.
- The package version restarts at 1.0.0, and products now stamp the repository
  `WVURAIL/pilotproxy`. The detector core and schema versions are unchanged.
- CHIME-derived files are listed in [CHIME_DERIVED.md](CHIME_DERIVED.md), with the
  code-only cut to use until CHIME clears their release.

## Records keep their paths

Frozen records (`docs/evidence/`, `data/provenance/`, `docs/runtime_freezes/` and
`tests/data/threshold_decisions_2026-10-12.json`) keep the absolute paths of the
workstation that wrote them (`/home/djg/rail/...`). They are records, so they were
not edited. The code never reads those paths.
