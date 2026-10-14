# coding=utf-8
from __future__ import annotations

import math

import numpy as np

from pilot_proxy.testbench.audit_atsc_signal import audit_atsc_iq

AUDIT_TEST_SAMPLE_RATE_HZ = 1024.0
AUDIT_TEST_CHANNEL_WIDTH_HZ = 512.0
AUDIT_TEST_PILOT_OFFSET_HZ = 64.0
AUDIT_TEST_SAMPLES = 8192
AUDIT_TEST_RNG_SEED = 1234
AUDIT_TEST_PILOT_AMPLITUDE = 8.0
AUDIT_TEST_SEARCH_HALF_WIDTH_HZ = 20.0
AUDIT_TEST_PILOT_WINDOW_HALF_WIDTH_HZ = 1.0
AUDIT_TEST_PILOT_EXCLUSION_HZ = 8.0
AUDIT_TEST_EDGE_EXCLUSION_HZ = 16.0
PILOT_FREQUENCY_TOLERANCE_HZ = 0.2
HALF_SCALE = 2.0


def test_audit_atsc_iq_measures_expected_pilot(tmp_path) -> None:
    sample_rate_hz = AUDIT_TEST_SAMPLE_RATE_HZ
    channel_width_hz = AUDIT_TEST_CHANNEL_WIDTH_HZ
    pilot_offset_hz = AUDIT_TEST_PILOT_OFFSET_HZ
    expected_pilot_hz = -channel_width_hz / HALF_SCALE + pilot_offset_hz
    n = AUDIT_TEST_SAMPLES
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(AUDIT_TEST_RNG_SEED)
    shelf = (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)
    pilot = AUDIT_TEST_PILOT_AMPLITUDE * np.exp(2j * np.pi * expected_pilot_hz * t)
    iq = np.asarray(shelf + pilot, dtype=np.complex64)
    path = tmp_path / "synthetic_atsc.cfile"
    iq.tofile(path)

    audit = audit_atsc_iq(
        input_iq=path,
        sample_rate_hz=sample_rate_hz,
        channel_width_hz=channel_width_hz,
        pilot_offset_hz=pilot_offset_hz,
        max_samples=n,
        pilot_search_half_width_hz=AUDIT_TEST_SEARCH_HALF_WIDTH_HZ,
        pilot_window_half_width_hz=AUDIT_TEST_PILOT_WINDOW_HALF_WIDTH_HZ,
        pilot_exclusion_hz=AUDIT_TEST_PILOT_EXCLUSION_HZ,
        edge_exclusion_hz=AUDIT_TEST_EDGE_EXCLUSION_HZ,
    )

    assert (
        abs(audit["measured_pilot_frequency_hz"] - expected_pilot_hz)
        < PILOT_FREQUENCY_TOLERANCE_HZ
    )
    assert math.isfinite(audit["measured_pilot_to_data_power_db"])
    assert math.isclose(
        audit["measured_pilot_below_data_db"],
        -audit["measured_pilot_to_data_power_db"],
    )
    assert audit["occupied_bandwidth_hz"] > 0.0
    assert audit["shelf_flatness_db"] > 0.0
    assert "quality" in audit
    assert audit["quality"]["num_quality_checks"] == 5
    assert 0.0 <= audit["quality_score"] <= 1.0


def test_audit_atsc_iq_quality_passes_with_matching_thresholds(tmp_path) -> None:
    sample_rate_hz = AUDIT_TEST_SAMPLE_RATE_HZ
    channel_width_hz = AUDIT_TEST_CHANNEL_WIDTH_HZ
    pilot_offset_hz = AUDIT_TEST_PILOT_OFFSET_HZ
    expected_pilot_hz = -channel_width_hz / HALF_SCALE + pilot_offset_hz
    n = AUDIT_TEST_SAMPLES
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(AUDIT_TEST_RNG_SEED)
    shelf = (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)
    pilot = AUDIT_TEST_PILOT_AMPLITUDE * np.exp(2j * np.pi * expected_pilot_hz * t)
    iq = np.asarray(shelf + pilot, dtype=np.complex64)
    path = tmp_path / "synthetic_atsc.cfile"
    iq.tofile(path)

    audit = audit_atsc_iq(
        input_iq=path,
        sample_rate_hz=sample_rate_hz,
        channel_width_hz=channel_width_hz,
        pilot_offset_hz=pilot_offset_hz,
        max_samples=n,
        pilot_search_half_width_hz=AUDIT_TEST_SEARCH_HALF_WIDTH_HZ,
        pilot_window_half_width_hz=AUDIT_TEST_PILOT_WINDOW_HALF_WIDTH_HZ,
        pilot_exclusion_hz=AUDIT_TEST_PILOT_EXCLUSION_HZ,
        edge_exclusion_hz=AUDIT_TEST_EDGE_EXCLUSION_HZ,
        expected_pilot_below_data_db=0.0,
        pilot_below_data_tolerance_db=100.0,
        min_occupied_bandwidth_hz=0.0,
        max_occupied_bandwidth_hz=sample_rate_hz,
        max_shelf_flatness_db=100.0,
        max_edge_rolloff_db=100.0,
    )

    assert audit["quality_passed"] is True
    assert audit["quality_score"] == 1.0
