# coding=utf-8
"""Metadata contract for PilotProxy's packed reader/analyzer pair."""
from __future__ import annotations


from pilot_proxy.archive.interfaces import stream_compatibility

from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.archive.stream_kinds import (
    STREAM_PACKED_COMPLEX_INT4_BASEBAND,
)


def test_packed_reader_and_detector_declare_matching_stream_contract() -> None:
    reader_info = ChimeBasebandPackedReader.info
    analyzer_info = PilotProxyDetectorAnalyzer.info

    assert reader_info.stream_kind == STREAM_PACKED_COMPLEX_INT4_BASEBAND
    assert analyzer_info.accepts_stream_kinds == (
        STREAM_PACKED_COMPLEX_INT4_BASEBAND,
    )

    compatibility = stream_compatibility(reader_info, analyzer_info)
    assert compatibility.compatible is True
    assert compatibility.reader_kind == STREAM_PACKED_COMPLEX_INT4_BASEBAND
