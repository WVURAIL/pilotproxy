#!/usr/bin/env python3
"""
Offline tests for the Storage Inventory service override -- the escape hatch
that lets a run fetch straight from a replica when the global locator is
degraded. No CADC access: cadcdata's client is replaced by a recorder, so the
tests lock the actual constructor contract (whether resource_id is passed, and
with what) rather than restating the helper's logic.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_storage_service_override.py
"""
from __future__ import annotations

import pytest

from pilot_proxy.archive.sources import cadc as src

# cadcdata is optional (the CADC client is not installed in the smoke jobs).
# The module under test imports it lazily inside _make_client, so importing
# pilot_proxy stays fine; only these tests need the real class to patch.
cadcdata = pytest.importorskip("cadcdata")


_UNSET = object()


class _RecordingClient:
    """Stands in for cadcdata.StorageInventoryClient.

    resource_id defaults to a sentinel so a test can tell "argument omitted"
    (library default applies) from "argument passed", which a real default
    string would hide.
    """

    def __init__(self, subject, resource_id=_UNSET, host=None, insecure=False):
        self.subject = subject
        self.resource_id = resource_id
        self.host = host


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(cadcdata, "StorageInventoryClient", _RecordingClient)
    return _RecordingClient


# ==========================================================================
# storage_service_override(): unset/blank means "use the default locator"
# ==========================================================================
def test_override_absent_when_env_unset(monkeypatch):
    monkeypatch.delenv(src.STORAGE_SERVICE_ENV, raising=False)
    assert src.storage_service_override() is None


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_override_absent_when_env_blank(monkeypatch, blank):
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, blank)
    assert src.storage_service_override() is None


def test_override_value_is_stripped(monkeypatch):
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV,
                       "  ivo://cadc.nrc.ca/uvic/minoc \n")
    assert src.storage_service_override() == "ivo://cadc.nrc.ca/uvic/minoc"


def test_override_accepts_a_bare_url(monkeypatch):
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "https://ws-uv.canfar.net/minoc")
    assert src.storage_service_override() == "https://ws-uv.canfar.net/minoc"


# ==========================================================================
# _make_client(): the override reaches the real constructor, or is omitted
# ==========================================================================
def test_client_omits_resource_id_without_override(monkeypatch, recorder):
    monkeypatch.delenv(src.STORAGE_SERVICE_ENV, raising=False)
    client = src.CadcDatatrailSource()._make_client()
    assert client.resource_id is _UNSET, (
        "with no override the library default locator must apply -- passing "
        "resource_id explicitly would pin behaviour the library owns")


def test_client_receives_resource_id_from_override(monkeypatch, recorder):
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/uvic/minoc")
    client = src.CadcDatatrailSource()._make_client()
    assert client.resource_id == "ivo://cadc.nrc.ca/uvic/minoc"


def test_blank_override_does_not_pin_a_service(monkeypatch, recorder):
    """A blank value must behave as "unset", not as an empty resource_id."""
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "  ")
    client = src.CadcDatatrailSource()._make_client()
    assert client.resource_id is _UNSET


def test_override_is_read_per_client_not_cached_at_import(
        monkeypatch, recorder):
    """Each client construction re-reads the environment.

    Clients are built lazily per thread, so a value captured once at import
    would silently apply the wrong route to threads created later.
    """
    monkeypatch.delenv(src.STORAGE_SERVICE_ENV, raising=False)
    source = src.CadcDatatrailSource()
    assert source._make_client().resource_id is _UNSET
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/sfu/minoc")
    assert source._make_client().resource_id == "ivo://cadc.nrc.ca/sfu/minoc"


def test_client_still_carries_a_subject(monkeypatch, recorder):
    """The override must not displace certificate-based authentication."""
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "ivo://cadc.nrc.ca/uvic/minoc")
    client = src.CadcDatatrailSource()._make_client()
    assert client.subject is not None
