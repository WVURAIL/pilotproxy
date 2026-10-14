#!/usr/bin/env python3
"""
The CADC probe boundary -- the code that decides whether a file EXISTS.

Every other survey test replaces `_cadc_size` wholesale, so the three lines
that turn a CADC answer into an archive verdict have never run under test.
That matters more than it looks: `_cadc_size` returning `(None, None)` is the
single decision that marks a file definitively ABSENT, which the survey then
turns into an `empty` event and, once aged, a permanent `aged-out` write-off.
Anything else must land in `errored`, which is retried and can never silently
delete a file from the inventory.

The classification is a substring match on the exception CLASS NAME
(`cadc.py:473`), so it is only as good as the names CADC raises. An expired or
missing certificate must NOT be mistaken for "file absent" -- that is the
failure mode that would silently shrink a published inventory.

Offline: no CADC, no network, no cert.
"""
from __future__ import annotations

import sys
import types

import pytest

from pilot_proxy.archive.sources import cadc as src


# The real families resolve to OSError plus (optionally) HttpException and the
# requests families; subclassing OSError keeps these tests independent of
# whether cadcutils is installed, while preserving the thing under test -- the
# class NAME.
class NotFoundException(OSError):
    pass


class UnauthorizedException(OSError):
    pass


class ForbiddenException(OSError):
    pass


class SslException(OSError):
    pass


class _Info:
    def __init__(self, size):
        self.size = size


class _Client:
    """Minimal stand-in for StorageInventoryClient."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def cadcinfo(self, uri):
        self.calls.append(uri)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _source_with(monkeypatch, result):
    s = src.CadcDatatrailSource()
    client = _Client(result)
    monkeypatch.setattr(s, "_get_client", lambda: client)
    monkeypatch.setattr(src.time, "sleep", lambda *_a, **_k: None)
    return s, client


URI = "cadc:CHIMEFRB/data/chime/baseband/raw/2020/07/15/astro_1/baseband_1_506.h5"


def test_present_file_returns_its_size(monkeypatch):
    # the success path: nothing in the repo exercised `return info.size, None`
    s, client = _source_with(monkeypatch, _Info(91311880))
    assert s._cadc_size(URI) == (91311880, None)
    assert client.calls == [URI]


def test_notfound_is_the_definitive_absent_verdict(monkeypatch):
    # (None, None) is the ONLY way a file becomes "absent"; it must not retry
    s, client = _source_with(monkeypatch, NotFoundException("no such artifact"))
    assert s._cadc_size(URI, retries=3) == (None, None)
    assert len(client.calls) == 1, "an absent verdict must be decided in one call"


@pytest.mark.parametrize("exc", [
    UnauthorizedException("cert expired"),
    ForbiddenException("not permitted"),
    SslException("handshake failed"),
    OSError("connection reset"),
])
def test_credential_and_transport_failures_are_never_absent(monkeypatch, exc):
    # THE cert-expiry safety property. An expired proxy certificate must land
    # in `errored` (retryable, and escalated to `incomplete`), never in the
    # absent verdict that would write the file out of the inventory forever.
    s, _ = _source_with(monkeypatch, exc)
    size, err = s._cadc_size(URI, retries=1, base=0.0)
    assert size is None
    assert err is exc, "must be reported as an error, not as a definitive absence"


def test_the_real_cadc_notfound_name_still_matches(monkeypatch):
    # pins the coupling to the actual library: if CADC ever renames its
    # exception, the substring test above stops protecting anything.
    cadc_exceptions = pytest.importorskip("cadcutils.exceptions")
    assert "NotFound" in cadc_exceptions.NotFoundException.__name__
    for sibling in ("UnauthorizedException", "ForbiddenException"):
        cls = getattr(cadc_exceptions, sibling, None)
        if cls is not None:
            assert "NotFound" not in cls.__name__, (
                f"{sibling} would be misread as a definitive absence")


def test_missing_certificate_builds_an_anonymous_subject(monkeypatch, tmp_path):
    # `_make_client` silently falls back to an anonymous Subject when the cert
    # is absent. That branch fires the moment a certificate expires and is
    # removed, so pin which Subject each case builds.
    built = {}

    class _Subject:
        def __init__(self, certificate=None):
            built["certificate"] = certificate

    fake_net = types.SimpleNamespace(Subject=_Subject)
    monkeypatch.setitem(sys.modules, "cadcutils",
                        types.SimpleNamespace(net=fake_net))
    monkeypatch.setitem(sys.modules, "cadcutils.net", fake_net)
    monkeypatch.setitem(sys.modules, "cadcdata", types.SimpleNamespace(
        StorageInventoryClient=lambda subj: ("client", subj)))

    s = src.CadcDatatrailSource()

    real_cert = tmp_path / "cadcproxy.pem"
    real_cert.write_text("x")
    s._make_client(cert=str(real_cert))
    assert built["certificate"] == str(real_cert)

    built.clear()
    s._make_client(cert=str(tmp_path / "gone.pem"))
    assert built["certificate"] is None, (
        "a missing cert must be visibly anonymous, not silently authenticated")
