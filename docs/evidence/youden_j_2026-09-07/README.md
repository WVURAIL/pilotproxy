# Coarse-versus-fine ROC and Youden J, on the released per-pilot products

`youden_j.txt` is the verbatim output of

    python3 analysis/youden_j.py --products <the released per-pilot directory>

run on the v5 rebuild `chime_pilots_rebuild_20260829` on 2026-09-07. It is the
committed form of the comparison the candidate-selection record
(`docs/DESIGN_DECISIONS.md`, 2026-07) made when it chose the fine designated-set
CFAR as the runtime decision, recomputed on the products actually released so
the dissertation can cite an analysis rather than a recollection.

## Populations

* **null** --- channel 35 (`freq_id` 521) frames in its verified
  transmitter-off era, through 2021-10. The off state comes from the era, not
  from non-detection: 11,199 frames.
* **signal** --- the 2025-and-later on-epoch frames of channels 36, 35 and 34
  (18,296 / 18,049 / 16,193 frames).

## What it shows

The fine statistic dominates the coarse one on every channel, and the margin
widens exactly where the shelf is weak. At a null-quantile false-alarm rate of
0.05 the fine detection probability is 0.997 or better on all three channels
while the coarse one falls from 0.998 on channel 35 to 0.715 on 36 and 0.063 on
34. Youden J, the maximum of `P_d - P_fa` over the threshold, is 0.959--0.987
fine against 0.860--0.985 coarse.

The coarse positive-excess rule --- `Q > 1`, the rule with no threshold ---
holds `P_d` near unity but rejects **44.2%** of verified-quiet time on these
products. The design record states 48.5% for the same point on the superseded
products; the difference is the product rebuild, not a change of rule. Either
figure is the same statement: the point is unusable as a survey policy, which
is why the runtime decision is the fine designated-set CFAR.

The anchors (fine bins 62, 111 and 102) are each channel's own, taken as the
argmax of its mean signal-epoch fine spectrum, and the null frames are scored
in the same window.
