"""The narrowband-marker detector (today the ATSC 8-VSB pilot).

An emitter that carries a narrowband marker at a known offset from its band
edge is detected on the marker: the fine order-statistic CFAR over the marker
bins against a bulk of independent bins. The marker-specific parts live here:
the fine anchor and marker bins (:mod:`.anchors`), the exact rank scores
(:mod:`.scores`) and the per-frame fine score export
(:mod:`.fine_histogram_frames`).
"""
