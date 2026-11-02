#!/usr/bin/env python3
"""
Offline tests for the Storage Inventory service override.

The override lets a run fetch straight from a replica when the global locator
is degraded. The cadcdata client is replaced by a recorder, so no test reaches
CADC. The tests check the constructor contract itself, which is the presence
of resource_id and its value, and do not restate the logic of the helper.

Run:  PYTHONPATH=src python -m pytest tests/archive/test_storage_service_override.py
"""
from __future__ import annotations

import pytest

from pilot_proxy.archive.sources import cadc as src

# cadcdata is optional, and the CADC client is not installed in the smoke
# jobs. The module under test imports it lazily in _make_client, so
# pilot_proxy still imports without it. Only these tests need the real class
# to patch.
cadcdata = pytest.importorskip("cadcdata")


_UNSET = object()


class _RecordingClient:
    """Stand-in for cadcdata.StorageInventoryClient that records its arguments.

    resource_id defaults to a sentinel so a test can distinguish between an
    argument that is not passed, where the library default applies, and one
    that is passed. A real default string would make the two cases look
    the same.
    """

    def __init__(self, subject, resource_id=_UNSET, host=None, insecure=False):
        self.subject = subject
        self.resource_id = resource_id
        self.host = host


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(cadcdata, "StorageInventoryClient", _RecordingClient)
    return _RecordingClient


# storage_service_override(): an unset or blank value selects the default locator
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


# _make_client(): the override reaches the real constructor, or is omitted
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
    """A blank value behaves as unset and never passes an empty resource_id."""
    monkeypatch.setenv(src.STORAGE_SERVICE_ENV, "  ")
    client = src.CadcDatatrailSource()._make_client()
    assert client.resource_id is _UNSET


def test_override_is_read_per_client_not_cached_at_import(
        monkeypatch, recorder):
    """The environment is read again at each client construction.

    Clients are built lazily per thread, so a value captured once at import
    would apply the wrong route to threads created later, with no warning.
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
