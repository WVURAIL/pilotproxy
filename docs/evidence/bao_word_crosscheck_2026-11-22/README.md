# BAO-word cross-check against the dissertation — 2026-11-22

`calibrate-bundle`'s power-limit path (`bundle_calibration._limit_channel`, pilotproxy d664158, shelf window 96
fine bins) run on the CHIME archive frames of record and compared with the dissertation's worst-case test
(`data/assembly/corrected_rule.json` and `sweeps65`, dissertation v3-rerun c07c6fb).

PilotProxy gets the inputs a plan would carry: the dissertation's power limits and RFIsher channel table,
`lambda_c` from `default_words.json`, the curvature `(d_c, d_se)` of each channel and the route term 9.6e-7 as
`margin` (`estimate_offset.json`). The channel frames are those of the v3 score cache, with their fine powers
from the archive products, and the control frames are its score-cache frames (40477), as the dissertation's
`scripts/median/selection.py` reads them. A second run of each channel uses `curvature` and `margin` of zero.

| | Channels 14–36 |
|---|---|
| Keep word of every frame, capture weight, sweep words | identical |
| Ratio at every word | agrees to 2.2e-13 (relative) |
| Best word and its ratio | the same on 23 of 23 |
| Passing channels | none in either |

With `curvature` and `margin` of zero, the ratios of channels 16, 19, 23 and 26 fall by 27 to 52% (16: 1.017
against 1.397), because their estimate clips at zero and only the error terms remain. Hence a power-limit plan
must now give both. With every valid control frame of the product (40601) in place of the score cache's, the
ratios move by up to 3% and no outcome changes (`h96_c07c6fb_product.json`).

`h96_c07c6fb_cache.json` holds, for each channel, the comparison and both runs. `crosscheck.py REV HALF
[channels]` reproduces it from the dissertation checkout, the v3 score cache and the archive products; its paths
are those of the machine it ran on. The change that makes the two fields required leaves this path unchanged.
