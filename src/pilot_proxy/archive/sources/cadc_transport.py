"""Connect and read timeouts for the optional CADC client stack.

``StorageInventoryClient.cadcget`` has no timeout argument. Its cadcutils
transport uses a Requests session that supplies its own timeout, so the process
default from ``socket.setdefaulttimeout`` has no effect on supported CADC
clients. This module sets the connect and read timeouts on each thread-owned
session. For simple test clients and older transports, the socket default is
set under a lock and restored after each call.
"""
from __future__ import annotations

import http.client
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps

CADC_CONNECT_TIMEOUT_SECONDS = 30.0
CADC_READ_TIMEOUT_SECONDS = 120.0
CADC_REQUEST_TIMEOUT = (
    CADC_CONNECT_TIMEOUT_SECONDS,
    CADC_READ_TIMEOUT_SECONDS,
)

_SOCKET_DEFAULT_LOCK = threading.RLock()
_REQUEST_DEADLINE = ContextVar("pilot_proxy_cadc_request_deadline", default=None)


@lru_cache(maxsize=1)
def expected_errors() -> tuple:
    """Return the archive, network, and local I/O errors a source may retry or report.

    The optional dependencies are imported lazily, so the source still imports
    in a local-only installation. Any other exception is a programming or
    environment-contract failure and must abort the run.
    """
    errors = [OSError]
    try:
        from cadcutils import exceptions as cadc_exceptions
    except ImportError:
        pass
    else:
        errors.append(cadc_exceptions.HttpException)
    try:
        from requests.exceptions import RequestException
    except ImportError:
        pass
    else:
        errors.append(RequestException)
    try:
        from urllib3.exceptions import HTTPError as Urllib3HTTPError
    except ImportError:
        pass
    else:
        # Requests can pass urllib3 errors through unwrapped while a body is
        # streamed. A truncated transfer raises ProtocolError, and an inactive
        # read can raise ReadTimeoutError. Both can be retried as network
        # failures. They derive from urllib3.HTTPError and are not subclasses
        # of Requests' RequestException.
        errors.append(Urllib3HTTPError)
    errors.append(http.client.IncompleteRead)
    return tuple(errors)


@contextmanager
def socket_timeout(seconds: float, *, deadline=None):
    """Set and restore the process socket timeout under a lock, capped at the deadline."""
    with _SOCKET_DEFAULT_LOCK:
        timeout = float(seconds)
        if deadline is not None:
            # The remaining time is computed after the process-wide lock is
            # acquired, since another legacy request may have used up the
            # budget while this caller waited for the lock.
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("survey outage deadline exceeded")
            timeout = min(timeout, remaining)
        previous = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            yield
        finally:
            socket.setdefaulttimeout(previous)


@contextmanager
def request_deadline(deadline):
    """Cap a supported Requests call at an optional monotonic deadline.

    Context variables are local to each probe thread, so concurrent archive
    requests can carry different survey deadlines, and the process network
    defaults are not changed.
    """
    token = _REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REQUEST_DEADLINE.reset(token)


def configure_request_timeout(client) -> bool:
    """Apply the timeout policy to one thread-owned cadcutils session.

    Returns False when the client does not expose the supported transport
    session. Callers then fall back to :func:`socket_timeout`. The read timeout
    is a limit on inactivity, so a healthy large transfer may run longer than it
    while bytes continue to arrive.
    """
    transport = getattr(client, "_cadc_client", None)
    get_session = getattr(transport, "_get_session", None)
    if not callable(get_session):
        return False
    try:
        session = get_session()
        original_send = session.send
    except (AttributeError, TypeError):
        return False
    if getattr(original_send, "_pilot_proxy_bounded_timeout", False):
        return True

    @wraps(original_send)
    def bounded_send(request, **kwargs):
        timeout = CADC_REQUEST_TIMEOUT
        deadline = _REQUEST_DEADLINE.get()
        if deadline is not None:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("survey outage deadline exceeded")
            timeout = (
                min(CADC_CONNECT_TIMEOUT_SECONDS, remaining),
                min(CADC_READ_TIMEOUT_SECONDS, remaining),
            )
        kwargs["timeout"] = timeout
        return original_send(request, **kwargs)

    bounded_send._pilot_proxy_bounded_timeout = True
    try:
        session.send = bounded_send
    except (AttributeError, TypeError):
        return False
    return True


def describe_exception(exc: BaseException) -> str:
    """Return an exception's name and message without raising another error.

    Some library exceptions return a non-string from ``__str__``. For example,
    cadcutils' HttpException can wrap an HTTPError that way. Normal formatting
    would then raise TypeError in the retry handler. The wrapped
    exception's message is preferred, and repr is used when no message is
    available.
    """
    name = type(exc).__name__
    for candidate in (exc, getattr(exc, "orig_exception", None)):
        if candidate is None:
            continue
        try:
            text = str(candidate)
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(text, str) and text:
            return f"{name}: {text}"
    try:
        text = repr(exc)
    except Exception:                                      # noqa: BLE001
        return name
    return f"{name}: {text}" if isinstance(text, str) and text else name
