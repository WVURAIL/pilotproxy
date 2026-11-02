"""Report expected on-disk read failures as UnreadableUnitError."""
from __future__ import annotations

from contextlib import contextmanager

from pilot_proxy.archive.interfaces import UnreadableUnitError


@contextmanager
def unreadable_file():
    """Convert HDF5, open and schema failures to UnreadableUnitError.

    OSError, KeyError and ValueError are converted. Other exceptions are
    programming errors and pass through unchanged.
    """
    try:
        yield
    except UnreadableUnitError:
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise UnreadableUnitError(str(exc)) from exc
