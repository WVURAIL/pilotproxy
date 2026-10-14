"""Tests for the CANFAR GPU session launcher."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "launch_gpu_session", REPO / "scripts" / "launch_gpu_session.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class _FakeSession:
    def __init__(self, fetch_results):
        self.fetch_results = list(fetch_results)
        self.create_calls = []

    def fetch(self, *, kind):
        assert kind == "notebook"
        if len(self.fetch_results) > 1:
            return self.fetch_results.pop(0)
        return self.fetch_results[0]

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return ["session-id"]


def _install_fake_canfar(monkeypatch, session) -> None:
    canfar = types.ModuleType("canfar")
    canfar.__path__ = []
    sessions = types.ModuleType("canfar.sessions")
    sessions.Session = lambda: session
    canfar.sessions = sessions
    monkeypatch.setitem(sys.modules, "canfar", canfar)
    monkeypatch.setitem(sys.modules, "canfar.sessions", sessions)


def _run(monkeypatch, session, *args: str) -> int:
    _install_fake_canfar(monkeypatch, session)
    monkeypatch.setenv("CANFAR_REGISTRY_USER", "test-user")
    monkeypatch.setenv("CANFAR_REGISTRY_SECRET", "test-secret")
    monkeypatch.setattr(sys, "argv", ["launch_gpu_session.py", *args])
    return launcher.main()


@pytest.mark.parametrize("flag", ["--cores", "--ram", "--gpu", "--timeout"])
def test_main_rejects_nonpositive_resource_values(monkeypatch, capsys, flag):
    monkeypatch.setattr(sys, "argv", ["launch_gpu_session.py", flag, "0"])

    with pytest.raises(SystemExit) as exc_info:
        launcher.main()

    assert exc_info.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_main_creates_session_and_returns_when_running(monkeypatch, capsys):
    session = _FakeSession(
        [
            [],
            [
                {
                    "id": "session-id",
                    "name": "test-session",
                    "status": "Running",
                    "connectURL": "https://example.test/session",
                }
            ],
        ]
    )
    monkeypatch.setattr(launcher.time, "monotonic", lambda: 100.0)

    result = _run(
        monkeypatch,
        session,
        "--name",
        "test-session",
        "--image",
        "example.test/image:tag",
        "--cores",
        "2",
        "--ram",
        "16",
        "--gpu",
        "2",
        "--timeout",
        "30",
    )

    assert result == 0
    assert session.create_calls == [
        {
            "name": "test-session",
            "image": "example.test/image:tag",
            "kind": "notebook",
            "cores": 2,
            "ram": 16,
            "gpu": 2,
        }
    ]
    assert "https://example.test/session" in capsys.readouterr().out


def test_main_uses_monotonic_deadline_and_fails_on_timeout(monkeypatch, capsys):
    session = _FakeSession(
        [
            [],
            [{"id": "session-id", "name": "test-session", "status": "Pending"}],
        ]
    )
    clock = 100.0
    sleeps = []

    def monotonic():
        return clock

    def sleep(seconds):
        nonlocal clock
        sleeps.append(seconds)
        clock += seconds

    monkeypatch.setattr(launcher.time, "monotonic", monotonic)
    monkeypatch.setattr(launcher.time, "sleep", sleep)

    result = _run(
        monkeypatch,
        session,
        "--name",
        "test-session",
        "--timeout",
        "5",
    )

    assert result == 1
    assert sleeps == [5.0]
    assert "not Running after 5s" in capsys.readouterr().err
