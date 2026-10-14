"""Compatibility import path for the instrument loader.

The loader lives in :mod:`pilot_proxy.config.instrument` and the hardware files
in ``pilot_proxy/instruments/``. This module keeps the survey engine's import
path working.
"""
from __future__ import annotations

from pilot_proxy.config.instrument import (
    DEFAULT_NFFT,
    Instrument,
    Readiness,
    instrument_readiness,
    list_instrument_names,
    load_instrument,
    nyquist_sign,
)

__all__ = [
    "DEFAULT_NFFT",
    "Instrument",
    "Readiness",
    "instrument_readiness",
    "list_instrument_names",
    "load_instrument",
    "nyquist_sign",
]
