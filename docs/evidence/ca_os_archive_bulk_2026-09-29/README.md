# CA/OS archive-bulk statistics

Frozen record of the 2026-09-29 archive-bulk comparison on all 23 channels. No study was rerun for this package.

`data/bulk/chNN.json` contains the channel statistics, persistent-bin profiles and geometry. `data/ideal_bulk.json` is the recorded ideal reference. The statistic r is mean_B(T)/median_B(T); it describes how the archive bulk differs from the ideal reference and is not an operational false-alarm calibration.

`bulk_behaviour.py` and `common.py` preserve the producer bytes, except that the RFIsher commit ID in `common.py` was re-pinned to the rewritten history on 2026-10-24. `verify/v_bulk.py` and its recorded output preserve the independent check. `REPORT.md` and `CHECK.md` also discuss adjacent bench and frontier comparisons; those broader comparisons are outside this package's archive-bulk scope.

`reader/b2_cfar.json` preserves the later figure table, including its bulk percentiles, without recomputation. The reader's extractor and source-hash list accompany it. Its bench section is a recorded comparison, not the newly rerun F1 bench.

`INPUTS.json` pins the omitted per-frame arrays, source products and dependency commits. Large arrays are not copied. Product hashes from the release ledger are explicitly identified as recorded hashes; the per-frame arrays were hashed during packaging. Restoring the recorded inputs and code layout is required to repeat the original calculation. These files are evidence, not package entry points.

`MANIFEST.json` records every packaged byte and source. Run `python -B verify.py` to check this package without executing the study. Its manifest hash is pinned in the adjacent F1 inventory.
