# coding=utf-8
"""Stream kinds shared by the Pilot Proxy archive components.

A stream kind names the sample format that a reader passes to an analyzer in
memory. It is independent of the receiver and of the on-disk container that
supplied the samples.
"""
from __future__ import annotations

# One uint8 per complex sample: offset-binary int4 real in the high nibble and
# offset-binary int4 imaginary in the low nibble, yielded as baseband frames.
STREAM_PACKED_COMPLEX_INT4_BASEBAND = "packed-complex-int4-baseband-frame"


__all__ = ["STREAM_PACKED_COMPLEX_INT4_BASEBAND"]
