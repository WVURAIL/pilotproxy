# Provenance

This repository, WVURAIL/pilotproxy, starts from one snapshot commit. Its first
(root) commit is the tree of the private development repository at:

| This repository | Private provenance repository |
|---|---|
| first commit (`git rev-list --max-parents=0 HEAD`) | `WVURAIL/pilot-proxy@76b5ecf4937bb89a340ac8dc31a2ebb71ff41c26` (main, 2026-10-14; private) |

The private repository kept the full development history and is the record of
provenance. It was retired from GitHub on 2026-10-20; its full history, pull request
refs included, is kept as a git bundle in the WVU RAIL lab archive. The commits cited
in the papers, in the dissertation, in the provenance stamps of products made before
this snapshot, and in this tree as `<path>@<commit>` (for example in
`tests/data/removed_paths.txt`, `weights/README.md` and the frozen records) are
commits of that repository. They do not exist here. Readers who need them, such as
the dissertation committee, can be given access on request.

The code-only cut WVURAIL/pilotproxy-code was retired on the same day. Each of its
commits was a cut of a commit here, named in its message, and kotekan now vendors the
detector core from this repository.

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

## Ported after the snapshot

Work continues in this repository. The private repository took one more change after
the snapshot, ported here:

| This repository | Private provenance repository |
|---|---|
| Coherence gain as the lag sum at 33 frames, 40.00 for every band; tolerance table on inter-cylinder baselines | `76b5ecf..d59baa24d7e8c3083569ce17e3de4fa612dba222` (its PR 55) |

After this port the tree corresponds to `WVURAIL/pilot-proxy@d59baa2`, except for the
differences listed above.
