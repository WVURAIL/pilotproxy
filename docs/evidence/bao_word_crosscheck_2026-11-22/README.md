# BAO-word cross-check against the dissertation — 2026-11-22

`calibrate-bundle`'s power-limit path (`bundle_calibration._limit_channel`, pilotproxy 90618b2, ratio estimate
`e = F / mu_0 - 1`) run on the CHIME archive frames of record and compared with the dissertation's worst-case test
(`data/assembly/corrected_rule.json` and `sweeps65`, dissertation v3-rerun ad38579).

PilotProxy gets the inputs a plan would carry: the dissertation's power limits and RFIsher channel table,
`lambda_c` from `default_words.json`, the offset difference `(d_c, d_se)` of each channel as `curvature` and the
route term 9.62e-7 as `margin` (`estimate_offset.json`). The channel frames are those of the v3 score cache, with
their fine powers from the archive products, and the control frames are its score-cache frames (40477). Each
channel and the control take `mu_0` from their bundle weights. A second run of each channel uses `curvature` and
`margin` of zero.

| | Channels 14–36 |
|---|---|
| `mu_0` of each channel and of the control | equal to the products' (control 1.002832) |
| Excess of every frame | within 6e-14 of the products' `pilot_excess_from_fine_powers` |
| Keep word of every frame, capture weight, sweep words | identical |
| Ratio at every word | agrees to 2.9e-13 (relative) |
| Best word and its ratio | the same on 23 of 23 |
| Passing channels | none in either |

With `curvature` and `margin` of zero, channels 21, 29 and 32 would pass (least ratios 0.31, 0.14 and 0.21,
against 4.2, 7.7 and 4.9), because their offset differences are negative (-2.6e-5, -1.5e-5 and -3.0e-5 in shelf
units). Hence a power-limit plan must give both. Dropping the control frames that the score cache marks undecided
changes nothing.

`ratio_ad38579.json` holds, for each channel, the comparison and both runs. `crosscheck.py REV [channels]`
reproduces it from the dissertation checkout, the v3 score cache and the archive products (it writes
`out/ratio_REV_cache.json`); its paths are those of the machine it ran on.
