"""Streaming continuity, gap isolation, arithmetic and bounded-state checks."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.integration.sdr_upgrade_adapter import (
    DigitalAdapterConfig, FRAME_SAMPLES, INPUT_RATE_HZ, K, L, adapt_record,
)
from pilot_proxy.integration.sdr_upgrade_streaming import (
    INPUT_BLOCK_SAMPLES, MAX_HISTORY_SAMPLES, StreamingDigitalAdapter,
)


def stream(x, cfg, sizes, origin=0):
    adapter = StreamingDigitalAdapter(cfg, first_input_index=origin)
    frames = []
    offset = index = 0
    while offset < len(x):
        size = sizes[index % len(sizes)]
        frames.extend(adapter.push(x[offset:offset + size], input_start_index=origin + offset))
        offset += min(size, len(x) - offset)
        index += 1
        state = adapter.state
        assert state["pending_input_samples"] < INPUT_BLOCK_SAMPLES
        assert state["mixed_history_samples"] <= MAX_HISTORY_SAMPLES
        assert state["pending_frame_samples"] < FRAME_SAMPLES
    final = adapter.finish()
    frames.extend(final["frames"])
    return frames, final["receipt"], adapter


def compare_batch(frames, batch):
    y = np.stack([r["resampled_frame"] for r in frames])
    np.testing.assert_allclose(y, batch["resampled_frames"], rtol=0, atol=2e-12)
    for key, plural in [("packed_frame", "packed_frames"), ("projections_i32", "projections_i32"),
                        ("term_power_sums", "term_power_sums"), ("coarse_ratio", "coarse_ratio")]:
        np.testing.assert_array_equal(np.stack([r[key] for r in frames]), batch[plural])
    np.testing.assert_array_equal([r["frame_start_seconds_from_segment_start"] for r in frames],
                                  batch["frame_start_seconds"])


@pytest.mark.parametrize("sizes", [[1], [127, 128, 129, 8191, 8192, 8193], [83_890], [200_000]])
def test_arbitrary_chunk_partitions_match_batch_and_fixed_internal_blocks(sizes):
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, -23_456.125, -4, 3.0)
    rng = np.random.default_rng(119_009)
    x = rng.normal(size=175_001) + 1j * rng.normal(size=175_001)
    frames, receipt, adapter = stream(x, cfg, sizes, origin=5_000_007)
    batch = adapt_record(x, cfg)
    compare_batch(frames, batch)
    canonical, _, _ = stream(x, cfg, [len(x)])
    for actual, expected in zip(frames, canonical):
        np.testing.assert_array_equal(actual["resampled_frame"], expected["resampled_frame"])
        j = actual["first_untrimmed_output_index"]
        assert actual["sample_center_time_numerator"] == 5_000_007 * 25 + j * 128 - 4096
    assert receipt["discarded_incomplete_frame_samples"] == batch["metadata"]["unused_full_support_tail_samples"]
    assert receipt["full_support_output_samples"] == batch["metadata"]["timing"]["output_samples_with_full_filter_support"]
    assert adapter.state["closed"]


def test_explicit_gap_discards_partial_frame_and_matches_two_independent_records():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 90_000., 4, 5.)
    t = np.arange(103_123)
    a = np.exp(2j * np.pi * (cfg.mixer_hz + cfg.term_frequencies_hz[0]) * t / INPUT_RATE_HZ + .4j)
    b = np.zeros(113_567, dtype=complex)
    s = StreamingDigitalAdapter(cfg, first_input_index=23)
    first = s.push(a, input_start_index=23)
    boundary = s.gap(701)
    first.extend(boundary["frames"])
    compare_batch(first, adapt_record(a, cfg))
    assert boundary["receipt"]["discarded_incomplete_frame_samples"] > 0
    assert boundary["receipt"]["missing_input_samples_to_next_segment"] == 701
    origin = 23 + len(a) + 701
    assert s.next_input_index == origin
    second = s.push(b, input_start_index=origin)
    second.extend(s.finish()["frames"])
    compare_batch(second, adapt_record(b, cfg))
    assert {r["segment_id"] for r in first} == {0}
    assert {r["segment_id"] for r in second} == {1}
    assert all(r["ratio_status"] == "undefined_both_zero" for r in second)
    assert np.all(second[0]["resampled_frame"] == 0)  # No pre-gap tone leaks in.


def test_reset_does_not_stitch_two_subframe_segments_and_restart_is_explicit():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 5.)
    s = StreamingDigitalAdapter(cfg)
    assert s.push(np.ones(50_000, dtype=complex), input_start_index=0) == []
    boundary = s.reset(first_input_index=1234, reason="unknown transport loss; new source index")
    assert boundary["frames"] == []
    assert boundary["receipt"]["frame_count"] == 0
    assert s.push(np.ones(50_000, dtype=complex), input_start_index=1234) == []
    assert s.finish()["frames"] == []
    with pytest.raises(RuntimeError):
        s.push(np.ones(1, dtype=complex), input_start_index=s.next_input_index)
    with pytest.raises(RuntimeError):
        s.finish()
    assert s.reset(first_input_index=0, reason="new recording") is None
    frames = s.push(np.ones(100_000, dtype=complex), input_start_index=0)
    frames.extend(s.finish()["frames"])
    assert len(frames) == 1
    assert frames[0]["segment_id"] == 2
    assert frames[0]["ratio_status"] == "infinite"


@pytest.mark.parametrize("length", [0, 1, 127, 128, 327, 328, 329, 8191, 8192, 83900])
def test_finish_full_support_count_no_padding_and_tail_accounting(length):
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 1.)
    frames, receipt, _ = stream(np.zeros(length, dtype=complex), cfg, [37])
    expected = max(0, (length - 1) * 25 // 128 + 1 - 64) if length else 0
    assert receipt["full_support_output_samples"] == expected
    assert receipt["frame_count"] == len(frames) == expected // FRAME_SAMPLES
    assert receipt["discarded_incomplete_frame_samples"] == expected % FRAME_SAMPLES


@pytest.mark.parametrize("kind", ["nan", "infinite", "real", "matrix", "overlap", "gap"])
def test_invalid_push_leaves_state_unchanged(kind):
    s = StreamingDigitalAdapter(DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 1.))
    s.push(np.ones(3, dtype=complex), input_start_index=0)
    before = s.state
    inputs = {
        "nan": (np.r_[np.ones(9000, dtype=complex), complex(np.nan, 0)], 3),
        "infinite": (np.array([1j * np.inf]), 3), "real": (np.ones(4), 3),
        "matrix": (np.ones((2, 2), dtype=complex), 3),
        "overlap": (np.ones(4, dtype=complex), 2), "gap": (np.ones(4, dtype=complex), 4),
    }
    x, index = inputs[kind]
    with pytest.raises(ValueError):
        s.push(x, input_start_index=index)
    assert s.state == before


def test_empty_chunks_noop_and_invalid_boundary_does_not_lose_data():
    s = StreamingDigitalAdapter(DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 1.))
    s.push(np.ones(123, dtype=complex), input_start_index=0)
    before = s.state
    assert s.push(np.empty(0, dtype=complex), input_start_index=123) == []
    for call in [lambda: s.gap(0), lambda: s.gap(-1), lambda: s.gap(True),
                 lambda: s.reset(first_input_index=-1, reason="x"),
                 lambda: s.reset(first_input_index=0, reason=""), lambda: s.finish(reason="")]:
        with pytest.raises(ValueError):
            call()
        assert s.state == before


def test_half_step_rounding_caveat_is_exposed_and_chunk_independent():
    # Place one non-saturated component exactly at a batch half-step. Batch
    # and streaming may straddle this tie by an ulp; do not assert universal
    # packed equality for this deliberately adversarial arithmetic boundary.
    rng = np.random.default_rng(99173)
    x = rng.normal(size=100_000) + 1j * rng.normal(size=100_000)
    reference = adapt_record(x, DigitalAdapterConfig(INPUT_RATE_HZ, 23_456.7, 0, 1.))
    scale = .5 / abs(reference["resampled_frames"].real[0, 0, 11])
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 23_456.7, 0, scale)
    a, _, _ = stream(x, cfg, [193, 1, 8192])
    b, _, _ = stream(x, cfg, [len(x)])
    assert a[0]["minimum_unclipped_quantizer_half_step_distance"] < 1e-12
    np.testing.assert_array_equal(a[0]["packed_frame"], b[0]["packed_frame"])


def test_longer_stream_state_is_bounded_and_output_arrays_are_owned_by_caller():
    cfg = DigitalAdapterConfig(INPUT_RATE_HZ, 0., 0, 1.)
    s = StreamingDigitalAdapter(cfg)
    counts = 0
    saved = None
    for block in range(129):
        values = np.full(INPUT_BLOCK_SAMPLES, complex(block % 4, -1))
        out = s.push(values, input_start_index=block * INPUT_BLOCK_SAMPLES)
        values[:] = 900 + 900j  # No retained alias to caller input.
        for frame in out:
            counts += 1
            if saved is None:
                saved = frame["resampled_frame"]
                copy = saved.copy()
        assert s.state["mixed_history_samples"] <= MAX_HISTORY_SAMPLES
    counts += len(s.finish()["frames"])
    assert counts > 10
    np.testing.assert_array_equal(saved, copy)
    assert saved.shape == (L, K)
