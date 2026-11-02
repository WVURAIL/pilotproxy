"""Exercise the CADC timeout policy at the Requests session boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import socket
import threading
from types import SimpleNamespace

import pytest

from pilot_proxy.archive.sources import cadc, cadc_transport

requests = pytest.importorskip("requests")


class _RecordingAdapter(requests.adapters.BaseAdapter):
    """Return a local response without opening a socket."""

    def __init__(self):
        self.calls = []

    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        response = requests.Response()
        response.request = request
        response.status_code = 200
        response._content = b"archive metadata"
        return response

    def close(self):
        pass


def _client():
    session = requests.Session()
    adapter = _RecordingAdapter()
    session.mount("https://", adapter)
    client = SimpleNamespace(
        _cadc_client=SimpleNamespace(_get_session=lambda: session),
    )
    request = requests.Request("HEAD", "https://cadc.invalid/object").prepare()
    return client, session, adapter, request


@pytest.mark.parametrize(
    "remaining,expected",
    [(None, (30.0, 120.0)), (200.0, (30.0, 120.0)),
     (45.0, (30.0, 45.0)), (5.0, (5.0, 5.0))],
)
def test_session_policy_caps_each_wait_and_preserves_request_options(
    monkeypatch, remaining, expected,
):
    client, session, adapter, request = _client()
    monkeypatch.setattr(cadc_transport.time, "monotonic", lambda: 1000.0)
    default_socket_timeout = socket.getdefaulttimeout()
    assert cadc_transport.configure_request_timeout(client) is True
    wrapped = session.send
    assert cadc_transport.configure_request_timeout(client) is True
    assert session.send is wrapped

    deadline = None if remaining is None else 1000.0 + remaining
    with cadc_transport.request_deadline(deadline):
        response = session.send(
            request, timeout=900.0, stream=True, verify=False,
            cert="certificate.pem", proxies={},
        )
    assert response.content == b"archive metadata"
    assert len(adapter.calls) == 1
    sent_request, options = adapter.calls[0]
    assert sent_request is request
    assert options == {
        "timeout": expected, "stream": True, "verify": False,
        "cert": "certificate.pem", "proxies": {},
    }
    assert socket.getdefaulttimeout() == default_socket_timeout


@pytest.mark.parametrize("deadline", [1000.0, 999.0])
def test_expired_deadline_refuses_send_and_restores_context(monkeypatch, deadline):
    client, session, adapter, request = _client()
    monkeypatch.setattr(cadc_transport.time, "monotonic", lambda: 1000.0)
    assert cadc_transport.configure_request_timeout(client)
    with pytest.raises(TimeoutError, match="survey outage deadline exceeded"):
        with cadc_transport.request_deadline(deadline):
            session.send(request)
    assert adapter.calls == []
    session.send(request)
    assert adapter.calls[0][1]["timeout"] == (30.0, 120.0)


def test_nested_deadline_restores_outer_budget_after_exception(monkeypatch):
    client, session, adapter, request = _client()
    monkeypatch.setattr(cadc_transport.time, "monotonic", lambda: 1000.0)
    assert cadc_transport.configure_request_timeout(client)
    with cadc_transport.request_deadline(1040.0):
        with pytest.raises(ValueError, match="interrupted probe"):
            with cadc_transport.request_deadline(1007.0):
                session.send(request)
                raise ValueError("interrupted probe")
        session.send(request)
    session.send(request)
    assert [kwargs["timeout"] for _, kwargs in adapter.calls] == [
        (7.0, 7.0), (30.0, 40.0), (30.0, 120.0),
    ]


def test_concurrent_probe_deadlines_remain_thread_local(monkeypatch):
    monkeypatch.setattr(cadc_transport.time, "monotonic", lambda: 1000.0)
    barrier = threading.Barrier(2)

    def probe(deadline):
        client, session, adapter, request = _client()
        assert cadc_transport.configure_request_timeout(client)
        with cadc_transport.request_deadline(deadline):
            barrier.wait(timeout=10)
            session.send(request)
            barrier.wait(timeout=10)
        session.send(request)
        return [kwargs["timeout"] for _, kwargs in adapter.calls]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe, [1011.0, 1200.0]))
    assert results == [
        [(11.0, 11.0), (30.0, 120.0)],
        [(30.0, 120.0), (30.0, 120.0)],
    ]


def test_source_probe_installs_policy_once_and_does_not_retain_deadline(monkeypatch):
    client, session, adapter, request = _client()
    monkeypatch.setattr(cadc_transport.time, "monotonic", lambda: 1000.0)

    def cadcinfo(uri):
        assert uri == "cadc:CHIMEFRB/object"
        session.send(request, timeout=(60.0, 600.0))
        return SimpleNamespace(size=91311880)

    client.cadcinfo = cadcinfo
    source = cadc.CadcDatatrailSource()
    constructions = []

    def make_client():
        constructions.append(client)
        return client

    monkeypatch.setattr(source, "_make_client", make_client)
    assert source._cadc_size("cadc:CHIMEFRB/object", deadline=1003.0) == (91311880, None)
    assert source._cadc_size("cadc:CHIMEFRB/object") == (91311880, None)
    assert constructions == [client]
    assert source._local.socket_timeout_fallback is False
    assert [kwargs["timeout"] for _, kwargs in adapter.calls] == [
        (3.0, 3.0), (30.0, 120.0),
    ]


def test_legacy_client_selects_socket_fallback():
    assert cadc_transport.configure_request_timeout(SimpleNamespace()) is False
