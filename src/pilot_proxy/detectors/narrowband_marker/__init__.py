"""The narrowband-marker detector.

An emitter that carries a narrowband marker at a known offset from its band
edge is detected on the marker. The only marker implemented is the ATSC 8-VSB
pilot. Detection uses the fine order-statistic constant false alarm rate
(CFAR) test, which compares the marker bins with a bulk of independent bins.
The marker-specific modules are the marker bins and reference bulk
(:mod:`.anchors`), the exact rank scores (:mod:`.scores`) and the per-frame
fine score export (:mod:`.fine_histogram_frames`).
"""
