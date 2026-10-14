# coding=utf-8
"""Reader-contract guards for the Pilot Proxy archive analyzers.

The archive survey writes inventory metadata with the telescope's canonical
reader. For CHIME that canonical reader is ``chime-baseband`` (complex64), but
the detector analyzer needs the PilotProxy-specific ``chime-baseband-packed``
reader so it can losslessly repack native uint8 samples.
These tests make wrong reader pairings fail with a clear error instead of silently producing
nonsense.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.archive import detector as detector_mod
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer


NFFT = 16_384
K = 128
N_FEEDS = 2
CH14_CENTER_HZ = 470.3125e6
CH14_FREQ_ID = 844


def _instrument():
    return SimpleNamespace(
        nfft=NFFT,
        sense=-1,
        fs_hz=390_625.0,
        f0_mhz=800.0,
    )


def _stub_kernel():
    specs = SimpleNamespace(
        K=K,
        N=3,
        bits=4,
        reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": K,
            "num_weight_terms": 3,
            "sample_bits_per_component": 4,
            "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(specs=specs, version=SimpleNamespace(as_string=lambda: "test"))


def _unused_detector_fn(*, packed, weights, kernel):
    raise AssertionError("reader dtype guard should run before detector_fn")


def test_detector_fft_backend_falls_back_when_cupy_runtime_unusable(monkeypatch):
    class _BadRuntime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("CUDA runtime is not usable")

    class _BadCuda:
        runtime = _BadRuntime()

    class _BadCupy:
        cuda = _BadCuda()
        float32 = np.float32

    monkeypatch.setattr(detector_mod.accel, "import_cupy", lambda: _BadCupy)

    assert detector_mod._detector_fft_backend() is np


def _first_meta():
    return {
        "f_center_hz": CH14_CENTER_HZ,
        "nfft": NFFT,
        "num_input_streams": N_FEEDS,
        "unit_key": f"baseband_evt_{CH14_FREQ_ID}.h5",
        "unit_name": f"baseband_evt_{CH14_FREQ_ID}.h5",
        "scope": "local",
    }


def test_detector_rejects_complex_reader_chunks_with_actionable_hint():
    analyzer = PilotProxyDetectorAnalyzer()
    ctx = RunContext(
        instrument=_instrument(),
        selection=[CH14_FREQ_ID],
        options={
            "detector_fn": _unused_detector_fn,
            "kernel": _stub_kernel(),
            "weights_by_channel": {
                14: np.ones((3, K), dtype=np.int8),
            },
        },
    )
    analyzer.begin(ctx, _first_meta())

    wrong_reader_chunk = np.zeros((NFFT, N_FEEDS), dtype=np.complex64)
    with pytest.raises(ValueError, match="chime-baseband-packed"):
        analyzer.consume_file([wrong_reader_chunk], _first_meta())


def test_detector_rejects_input_stream_count_change():
    analyzer = PilotProxyDetectorAnalyzer()
    ctx = RunContext(
        instrument=_instrument(), selection=[CH14_FREQ_ID], options={
            "detector_fn": _unused_detector_fn,
            "kernel": _stub_kernel(),
            "weights_by_channel": {14: np.ones((3, K), dtype=np.int8)},
        },
    )
    analyzer.begin(ctx, _first_meta())
    wrong_feed_count = np.zeros((NFFT, N_FEEDS + 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="input-stream count"):
        analyzer.consume_file([wrong_feed_count], _first_meta())
