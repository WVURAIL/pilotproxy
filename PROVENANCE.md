# Provenance

This repository, WVURAIL/pilotproxy, starts from one snapshot commit. Its first
(root) commit is the tree of the private development repository at commit
`76b5ecf4937bb89a340ac8dc31a2ebb71ff41c26`, and the next change on main ports that
repository's commit `d59baa24d7e8c3083569ce17e3de4fa612dba222` (coherence gain as the
lag sum at 33 frames, 40.00 for every band; tolerance table on inter-cylinder
baselines). From there, all work is in this repository.

The private development repository holds the full development history and is the
record of provenance. Its history is kept as a git bundle in the WVU RAIL lab
archive. Commits cited in this tree as `<path>@<commit>` (in
`tests/data/removed_paths.txt` and in the frozen records), and the provenance stamps
of products made before the snapshot, refer to that repository unless the commit is
in this one. Readers who need them, such as the dissertation committee, can be given
access on request.

## What differs from the snapshot

- A manuscript draft, its working notes and an archive listing were not meant for
  release and are not in this history.
- Real CHIME/FRB event numbers used as examples in tests, docs and one docstring are
  replaced with made-up ones of the same length and order.
- Workstation paths in docs and tests are environment variables
  (`PILOT_PROXY_ARCHIVE_BUNDLE`, `PILOT_PROXY_JULY_INVENTORIES`) or examples; the
  tests that need those inputs skip without them.
- Package versions in this repository start at 1.0.0, and products stamp the
  repository `WVURAIL/pilotproxy`. Detector core and schema versions continue the
  development repository's.
- CHIME-derived files are listed in [CHIME_DERIVED.md](CHIME_DERIVED.md), with the
  code-only cut to use until CHIME clears their release.

## Records keep their paths

Frozen records (`docs/evidence/`, `data/provenance/`, `docs/runtime_freezes/` and
`tests/data/threshold_decisions_2026-10-12.json`) keep the absolute paths of the
workstation that wrote them. They are records and are not edited. The code never
reads those paths.
