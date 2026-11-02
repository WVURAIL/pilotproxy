"""The compatibility listing keeps raw names and its documented empty result."""

import json

import pytest

from pilot_proxy.archive.datatrail_client import Datatrail
from test_datatrail_adapter import _install_fake_cli


@pytest.mark.parametrize("outcome", ["children", "empty", "failure"])
def test_public_children_preserves_names_and_checked_outage_distinction(monkeypatch, outcome):
    names = ["20260904T181800Z", "calibration container", "event123456789"]
    expected = names if outcome == "children" else []
    ok = outcome != "failure"
    calls = []

    def handler(arguments):
        calls.append(arguments)
        assert arguments == ["ls", "scope.capture", "container/2026"]
        if not ok:
            return 1, json.dumps({"error": "Service unavailable"}), ""
        return 0, json.dumps({"datasets": expected}), ""

    _install_fake_cli(monkeypatch, handler)
    client = Datatrail()
    assert client.children("scope.capture", "container/2026") == expected
    assert client.children_checked("scope.capture", "container/2026") == (expected, ok)
    assert len(calls) == 2
