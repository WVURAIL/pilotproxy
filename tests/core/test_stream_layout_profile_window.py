# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace

from pilot_proxy.integration import DetectorFrameLayout
from pilot_proxy.integration.stream_layout import layout_from_receiver_profile


def test_layout_uses_receiver_profile_detector_window() -> None:
    profile = SimpleNamespace(
        frame_size_samples=8192,
        detector_window_samples=64,
        num_input_streams=128,
    )

    layout = layout_from_receiver_profile(profile)

    assert isinstance(layout, DetectorFrameLayout)
    assert layout.frame_size_samples == 8192
    assert layout.detector_window_samples == 64
    assert layout.windows_per_stream == 128
    assert layout.num_input_streams == 128
    assert layout.detector_rows_per_frame == 128 * 128


def test_layout_does_not_replace_profile_window_with_chime_default() -> None:
    assert layout_from_receiver_profile(SimpleNamespace(
        frame_size_samples=8192, detector_window_samples=64, num_input_streams=1
    )).detector_window_samples != 128
