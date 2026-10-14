# coding=utf-8
from __future__ import annotations

import inspect


from pilot_proxy.archive.detector import (  # noqa: E402
    PilotProxyDetectorAnalyzer,
    _callable_accepts_keyword,
)


def test_detector_capability_is_discovered_before_invocation() -> None:
    def supported(*, packed, weights, kernel, emit_row_projections=False):
        raise TypeError("an internal detector error must not become a fallback")

    def unsupported(*, packed, weights, kernel):
        return {}

    def extensible(**kwargs):
        return kwargs

    assert _callable_accepts_keyword(supported, "emit_row_projections")
    assert not _callable_accepts_keyword(unsupported, "emit_row_projections")
    assert _callable_accepts_keyword(extensible, "emit_row_projections")


def test_consume_file_does_not_catch_detector_type_errors() -> None:
    source = inspect.getsource(PilotProxyDetectorAnalyzer.consume_file)
    assert "except TypeError" not in source
