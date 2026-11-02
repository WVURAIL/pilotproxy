"""Check installed-version metadata before starting a Datatrail survey."""
import importlib.metadata

import pytest

from pilot_proxy.archive import datatrail_client


@pytest.mark.parametrize(
    "raw,parsed,available",
    [("0.10.9", (0, 10, 9), False), ("0.11", (0, 11), True),
     ("0.11.0", (0, 11, 0), True), ("1.0.2+local.7", (1, 0, 2), True)],
)
def test_metadata_version_drives_contract_gate(monkeypatch, raw, parsed, available):
    lookups = []

    def version(distribution):
        lookups.append(distribution)
        return raw

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert datatrail_client._cli_version() == parsed
    ok, detail = datatrail_client.Datatrail.api_available()
    assert ok is available
    assert lookups == ["datatrail-cli", "datatrail-cli"]
    if available:
        assert detail == ""
    else:
        assert "0.10.9 predates" in detail
        assert "datatrail-cli>=0.11" in detail


@pytest.mark.parametrize("error", [importlib.metadata.PackageNotFoundError("datatrail-cli"), OSError("unreadable metadata")])
def test_missing_or_unreadable_metadata_refuses_contract(monkeypatch, error):
    def version(distribution):
        assert distribution == "datatrail-cli"
        raise error

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert datatrail_client._cli_version() is None
    ok, detail = datatrail_client.Datatrail.api_available()
    assert ok is False
    assert "cannot determine the datatrail-cli version" in detail


def test_unknown_version_text_refuses_contract(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "unknown")
    assert datatrail_client._cli_version() is None
    assert datatrail_client.Datatrail.api_available()[0] is False
