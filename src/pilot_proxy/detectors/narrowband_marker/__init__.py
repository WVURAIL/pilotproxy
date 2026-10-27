"""The narrowband-marker detector.

An emitter that carries a narrowband marker at a known offset from its band
edge is detected on the marker. The only marker implemented is the ATSC 8-VSB
pilot. Detection uses the fine order-statistic constant false alarm rate
(CFAR) test, which compares the marker bins with a bulk of independent bins.
The marker-specific modules are the fine anchor and marker bins
(:mod:`.anchors`), the exact rank scores (:mod:`.scores`), the rescoring of a
score cache (:mod:`.rescore`), and the per-frame fine score export
(:mod:`.fine_histogram_frames`).
"""
