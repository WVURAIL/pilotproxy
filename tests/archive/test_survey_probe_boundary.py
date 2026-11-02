#!/usr/bin/env python3
"""
Tests of the CADC probe boundary, the code that determines the presence of a
file.

Every other survey test replaces `_cadc_size` as a whole, so the three lines
that turn a CADC answer into an archive verdict run under test only here. A
return of `(None, None)` from `_cadc_size` is the single decision that marks
a file definitively absent. The survey then turns the file into an `empty`
event and, once it is old enough, a permanent `aged-out` write-off. Any other
answer must land in `errored`, which is retried and can never delete a file
from the inventory with no warning.

The classification is a substring match on the name of the exception class
in `_cadc_size`, so it is only as good as the names CADC raises. An expired or
missing certificate must never be read as an absent file, since that failure
would shrink a published inventory with no warning.

No test reaches CADC or the network, or needs a certificate.
"""
from __future__ import annotations

import sys
import types

import pytest

from pilot_proxy.archive.sources import cadc as src


# The real exception families resolve to OSError and, when installed, to
# HttpException and the requests families. Subclassing OSError keeps these
# tests independent of the cadcutils install and keeps the class name, which is
# the thing under test.
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
    # the success path, `return info.size, None`, which no other test runs
    s, client = _source_with(monkeypatch, _Info(91311880))
    assert s._cadc_size(URI) == (91311880, None)
    assert client.calls == [URI]


def test_notfound_is_the_definitive_absent_verdict(monkeypatch):
    # (None, None) is the only way a file becomes "absent", and it is decided
    # with no retry
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
    # An expired proxy certificate must land in `errored`, which is retried and
    # escalated to `incomplete`. It must never reach the absent verdict, which
    # would write the file out of the inventory forever.
    s, _ = _source_with(monkeypatch, exc)
    size, err = s._cadc_size(URI, retries=1, base=0.0)
    assert size is None
    assert err is exc, "must be reported as an error, not as a definitive absence"


def test_the_real_cadc_notfound_name_still_matches(monkeypatch):
    # This test is tied to the real library. If CADC renames its exception,
    # the substring match no longer protects anything.
    cadc_exceptions = pytest.importorskip("cadcutils.exceptions")
    assert "NotFound" in cadc_exceptions.NotFoundException.__name__
    for sibling in ("UnauthorizedException", "ForbiddenException"):
        cls = getattr(cadc_exceptions, sibling, None)
        if cls is not None:
            assert "NotFound" not in cls.__name__, (
                f"{sibling} would be misread as a definitive absence")


def test_missing_certificate_builds_an_anonymous_subject(monkeypatch, tmp_path):
    # `_make_client` falls back to an anonymous Subject with no warning when the
    # cert is absent. That branch fires as soon as an expired certificate is
    # removed, so the test checks the Subject built in each case.
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
