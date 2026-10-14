# Channel 37 rank-injection controls

Frozen record of the 2026-09-30 study, including channel 37 at 23 geometries, four carrier-off controls and the ideal code-path checks. No study was rerun for this package.

The two injection modes disagree. The study stopped at its control gate and chose no rank. Its curves compare rank behavior; they do not calibrate any channel's operational false-alarm threshold. The multi-rank results remain a recorded comparison alongside the fixed-median detector.

Read `RESULT.md` and `CHECK.md` together. `data/controls_summary.json` preserves the per-geometry curves, uncertainties, real-null exceedance rates and the study's selection diagnostics. Per-unit JSON records, extraction/geometry metadata, design and verification results are included. `rank/` and `verify/` retain the original source bytes and paths.

`INPUTS.json` pins the extracted frames, intermediate arrays, lookup tables, source products and dependency commits. Large arrays are not copied. Product hashes from the release ledger are explicitly identified as recorded hashes; all other external hashes were checked during packaging. Restoring those inputs and the recorded code layout is required to repeat the original calculation. These files are evidence, not package entry points.

`MANIFEST.json` records every packaged byte and its source. Run `python -B verify.py` to check this package without executing the study. Its manifest hash is pinned in the adjacent F1 inventory.
