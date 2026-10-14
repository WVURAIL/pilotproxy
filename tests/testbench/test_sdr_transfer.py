# coding=utf-8
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import runpy

import numpy as np
import pytest


DRIVER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "sdr_transfer.py")
)
FREEZER = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "tools"
        / "freeze_estimator_transfer.py"
    )
)
SAMPLE_RATE_HZ = float(DRIVER["GNU_RADIO_ATSC_SYMBOL_RATE_HZ"])
PILOT_BELOW_DATA_DB = 11.918446870168612
CAPTURE_SAMPLES = max(
    2 * int(DRIVER["SPECTRUM_SEGMENT_SAMPLES"]),
    int(
        DRIVER["required_iq_samples"](
            iq_sample_rate_hz=SAMPLE_RATE_HZ,
            adc_sample_rate_hz=DRIVER["REFERENCE_ADC_SAMPLE_RATE_HZ"],
            num_output_samples=128,
        )
    ),
)


def _transmit_args(
    tmp_path: Path,
    *,
    input_path: Path | None = None,
    audit_path: Path | None = None,
    event_limit: int | None = None,
):
    worker = tmp_path / "stream_worker"
    worker.touch()
    worker.chmod(0o755)
    library = tmp_path / "libLimeSuite.so"
    library.touch()
    values = [
        "--input-iq",
        str(input_path or tmp_path / "generated.cfile"),
        "--output-dir",
        str(tmp_path / "run"),
        "--transmit",
        "--rf-authorized",
        "--frequency-hz",
        "473000000",
        "--physical-channel",
        "14",
        "--tx-gain-db",
        "0",
        "--rx-gain-db",
        "0",
        "--tx-rms",
        "0.02",
        "--device-serial",
        "TEST123",
        "--tx-antenna",
        "BAND2",
        "--rx-antenna",
        "LNAW",
        "--antenna-separation-cm",
        "30",
        "--stream-worker",
        str(worker),
        "--limesuite-library",
        str(library),
        "--settle-samples",
        "4096",
        "--capture-samples",
        str(CAPTURE_SAMPLES),
        "--capture-guard-samples",
        "4096",
        "--sync-marker-samples",
        "256",
        "--marker-search-samples",
        "128",
        "--session-guard-samples",
        "4096",
        "--frame-size-samples",
        "128",
    ]
    if audit_path is not None:
        values.extend(["--waveform-audit-json", str(audit_path)])
    if event_limit is not None:
        values.extend(["--event-limit", str(event_limit)])
    return DRIVER["build_parser"]().parse_args(values)


def _write_stream_status(request: dict[str, object]) -> None:
    Path(str(request["stream_status_path"])).write_text(
        json.dumps(
            {
                "schema_version": DRIVER["STREAM_STATUS_SCHEMA"],
                "valid": True,
                "rx_host_rate_hz": request["sample_rate_hz"],
                "tx_host_rate_hz": request["sample_rate_hz"],
                "requested_filter_bandwidth_hz": request[
                    "radio_filter_bandwidth_hz"
                ],
                "rx_lpf_bandwidth_hz": request["radio_filter_bandwidth_hz"],
                "tx_lpf_bandwidth_hz": request["radio_filter_bandwidth_hz"],
                "gfir_enabled": True,
                "tx_off_samples": request["capture_samples"],
                "session_samples": request["session_samples"],
                "tx_off_underrun": 0,
                "tx_off_overrun": 0,
                "tx_off_dropped_packets": 0,
                "rx_underrun": 0,
                "rx_overrun": 0,
                "rx_dropped_packets": 0,
                "tx_underrun": 0,
                "tx_overrun": 0,
                "tx_dropped_packets": 0,
            }
        ),
        encoding="utf-8",
    )


def _write_input(tmp_path: Path) -> tuple[Path, Path]:
    input_path = tmp_path / "generated.cfile"
    rng = np.random.default_rng(3)
    count = 4096 + CAPTURE_SAMPLES
    clean = (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    ).astype(np.complex64)
    clean.tofile(input_path)
    input_path.with_suffix(".cfile.json").write_text(
        json.dumps(
            {
                "schema_version": "fstat_atsc_clean_iq_v1",
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "num_iq_samples": int(clean.size),
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "waveform_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": "pilotproxy_atsc_waveform_audit_v1",
                "input_iq": str(input_path),
                "quality_passed": True,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "measured_pilot_below_data_db": PILOT_BELOW_DATA_DB,
            }
        ),
        encoding="utf-8",
    )
    return input_path, audit_path


def test_schedule_uses_requested_grid_and_trial_counts() -> None:
    schedule = DRIVER["build_schedule"]()
    mixtures = [event for event in schedule if event.kind == "mixture"]
    counts = Counter(event.snr_db for event in mixtures)

    assert sorted(counts) == [float(value) for value in range(-42, 1, 3)]
    assert sum(counts.values()) == 1800
    for snr_db, count in counts.items():
        assert count == (240 if snr_db <= -30.0 else 60)
    assert len(schedule) == 1833

    for pass_index in range(1, 4):
        one_pass = [event for event in schedule if event.pass_index == pass_index]
        assert [event.kind for event in one_pass[:5]] == [
            "tx_off",
            "tx_zero",
            "signal_only",
            "noise_only",
            "drift",
        ]
        assert one_pass[4].note == "pass_start"
        assert one_pass[-1].kind == "drift"
        assert one_pass[-1].note == "pass_end"
        order = [event.snr_db for event in one_pass if event.kind == "mixture"]
        assert order != sorted(order)


def test_schedule_is_pinned_by_seed() -> None:
    first = DRIVER["build_schedule"](seed=77)
    second = DRIVER["build_schedule"](seed=77)
    third = DRIVER["build_schedule"](seed=78)

    assert first == second
    assert first != third


def test_channel_center_matches_generated_pilot_offset() -> None:
    assert DRIVER["expected_center_frequency_hz"](14) == pytest.approx(
        473_000_000.440559,
        abs=1.0e-6,
    )


def test_native_gain_mapping_preserves_device_gain_convention() -> None:
    assert DRIVER["native_lime_gain_db"](-12.0) == 0
    assert DRIVER["native_lime_gain_db"](8.0) == 20
    assert DRIVER["native_lime_gain_db"](38.0) == 50


@pytest.mark.parametrize("snr_db", [-42.0, -30.0, -3.0, 0.0])
def test_mixture_keeps_level_and_snr(snr_db: float) -> None:
    rng = np.random.default_rng(9)
    clean = (
        rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000)
    ).astype(np.complex64)
    mixed, metadata = DRIVER["make_tx_waveform"](
        clean,
        kind="mixture",
        snr_db=snr_db,
        target_rms=0.08,
        sample_rate_hz=SAMPLE_RATE_HZ,
        bandwidth_hz=6.0e6,
        seed=123,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
    )

    assert np.sqrt(np.mean(np.abs(mixed) ** 2)) == pytest.approx(
        0.08,
        abs=2.0e-7,
    )
    assert metadata["actual_data_shelf_snr_db"] == pytest.approx(
        snr_db,
        abs=1.0e-10,
    )


def test_dry_run_writes_full_plan_despite_event_limit(tmp_path: Path) -> None:
    output_dir = tmp_path / "plan"
    args = DRIVER["build_parser"]().parse_args(
        [
            "--input-iq",
            str(tmp_path / "generated.cfile"),
            "--output-dir",
            str(output_dir),
            "--event-limit",
            "12",
        ]
    )
    calls: list[object] = []

    assert DRIVER["run"](
        args,
        capture_runner=lambda request: calls.append(request),
    ) == 0
    assert calls == []
    plan = json.loads((output_dir / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "dry_run"
    assert plan["driver"] == DRIVER["RADIO_DRIVER"]
    assert plan["detector_backend"] == "cuda"
    assert plan["num_input_streams"] == 1
    assert len(plan["events"]) == 1833


def test_result_loader_rejects_conflicting_event_identity(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    planned = {
        "event_index": 1,
        "pass_index": 1,
        "kind": "tx_zero",
    }
    rows = [
        {
            "event": planned,
        },
        {
            "event": {
                "event_index": 1,
                "pass_index": 1,
                "kind": "noise_only",
            }
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="conflicting planned identities"):
        DRIVER["_load_result_rows"](path, planned_events=[planned])


def test_result_loader_accepts_planned_subset(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    planned = [
        {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        {"event_index": 2, "pass_index": 1, "kind": "noise_only"},
    ]
    path.write_text(
        json.dumps({"event": planned[1]}) + "\n",
        encoding="utf-8",
    )

    rows = DRIVER["_load_result_rows"](path, planned_events=planned)

    assert list(rows) == [2]


def test_result_loader_rejects_wrong_single_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    row = {"event": {**planned, "kind": "noise_only"}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the run plan"):
        DRIVER["_load_result_rows"](path, planned_events=[planned])


def test_result_loader_rejects_swapped_index(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    planned = [
        {"event_index": 1, "pass_index": 1, "kind": "tx_zero"},
        {"event_index": 2, "pass_index": 1, "kind": "noise_only"},
    ]
    row = {"event": {**planned[1], "event_index": 1}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the run plan"):
        DRIVER["_load_result_rows"](path, planned_events=planned)


def test_result_loader_rejects_extra_index(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    row = {"event": {**planned, "event_index": 2}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="is not in the run plan"):
        DRIVER["_load_result_rows"](path, planned_events=[planned])


@pytest.mark.parametrize("event_index", [1.0, "1", True])
def test_result_loader_rejects_non_integer_index(
    tmp_path: Path,
    event_index: object,
) -> None:
    path = tmp_path / "events.jsonl"
    planned = {"event_index": 1, "pass_index": 1, "kind": "tx_zero"}
    row = {"event": {**planned, "event_index": event_index}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be integers"):
        DRIVER["_load_result_rows"](path, planned_events=[planned])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tx_antenna", "BAND1", "BAND2"),
        ("rx_antenna", "LNAH", "LNAW"),
    ],
)
def test_tv_antenna_paths_are_required(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    args = _transmit_args(tmp_path)
    DRIVER["validate_transmit_args"](args)
    setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        DRIVER["validate_transmit_args"](args)


def test_waveform_audit_must_match_input(tmp_path: Path) -> None:
    input_path, audit_path = _write_input(tmp_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["input_iq"] = str(tmp_path / "different.cfile")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    args = _transmit_args(
        tmp_path,
        input_path=input_path,
        audit_path=audit_path,
    )

    with pytest.raises(SystemExit, match="does not match --input-iq"):
        DRIVER["resolve_pilot_calibration"](args)


def test_fixed_quantization_scale_must_be_positive(tmp_path: Path) -> None:
    args = DRIVER["build_parser"]().parse_args(
        [
            "--input-iq",
            str(tmp_path / "generated.cfile"),
            "--output-dir",
            str(tmp_path / "plan"),
            "--quantization-scale",
            "0",
        ]
    )

    with pytest.raises(SystemExit, match="positive and finite"):
        DRIVER["run"](args)


def _marker_capture(
    marker: np.ndarray,
    expected: list[int],
    offsets: list[int],
) -> np.ndarray:
    capture = np.zeros(
        expected[-1] + offsets[-1] + marker.size + 512,
        dtype=np.complex64,
    )
    for start, offset in zip(expected, offsets):
        observed = start + offset
        capture[observed : observed + marker.size] = marker
    return capture


def test_sync_markers_align_a_continuous_capture() -> None:
    marker = DRIVER["make_sync_marker"](
        samples=256,
        sample_rate_hz=SAMPLE_RATE_HZ,
        target_rms=0.02,
    )
    expected = [512, 2048, 3584]
    slots = [
        {"event_index": index, "marker_start_sample": start}
        for index, start in enumerate(expected, start=1)
    ]
    capture = _marker_capture(marker, expected, [19, 19, 19])

    aligned = DRIVER["align_session_slots"](
        capture,
        slots=slots,
        marker=marker,
        initial_search_samples=128,
        local_search_samples=128,
    )

    assert [item["observed_marker_sample"] for item in aligned] == [
        start + 19 for start in expected
    ]
    assert [item["stream_offset_samples"] for item in aligned] == [19, 19, 19]
    assert [item["separation_error_samples"] for item in aligned] == [0, 0, 0]
    assert all(item["correlation"] > 0.99 for item in aligned)


def test_sync_markers_reject_a_stream_slip() -> None:
    marker = DRIVER["make_sync_marker"](
        samples=256,
        sample_rate_hz=SAMPLE_RATE_HZ,
        target_rms=0.02,
    )
    expected = [512, 2048]
    slots = [
        {"event_index": index, "marker_start_sample": start}
        for index, start in enumerate(expected, start=1)
    ]
    capture = _marker_capture(marker, expected, [19, 89])

    with pytest.raises(RuntimeError, match="sample slip"):
        DRIVER["align_session_slots"](
            capture,
            slots=slots,
            marker=marker,
            initial_search_samples=128,
            local_search_samples=128,
        )


def test_received_input_calibration_uses_control_power() -> None:
    result = DRIVER["calibrated_received_shelf_snr"](
        tx_metadata={"signal_power": 3.0, "noise_power": 1.0},
        control_calibration={
            "receiver_and_leakage_power": 2.0,
            "signal_composite_power": 10.0,
            "signal_data_shelf_power": 8.0,
            "injected_noise_power": 20.0,
            "received_pilot_below_data_db": PILOT_BELOW_DATA_DB,
        },
    )
    expected_data_power = 0.75 * 8.0
    expected_noise_power = 2.0 + 0.25 * 20.0
    expected_snr = expected_data_power / expected_noise_power

    assert result["receiver_noise_power"] == pytest.approx(2.0)
    assert result["signal_path_power"] == pytest.approx(10.0)
    assert result["noise_path_power"] == pytest.approx(20.0)
    assert result["data_shelf_path_power"] == pytest.approx(8.0)
    assert result["received_composite_power"] == pytest.approx(7.5)
    assert result["received_data_shelf_power"] == pytest.approx(expected_data_power)
    assert result["received_noise_power"] == pytest.approx(expected_noise_power)
    assert result["data_shelf_snr_linear"] == pytest.approx(expected_snr)
    assert result["data_shelf_snr_db"] == pytest.approx(
        10.0 * math.log10(expected_snr)
    )


def test_received_control_spectrum_separates_data_and_pilot() -> None:
    rng = np.random.default_rng(51)
    count = 262_144
    receiver_scale = 0.002
    data_power = 0.01
    receiver = (
        rng.standard_normal((3, count)) + 1j * rng.standard_normal((3, count))
    ) / math.sqrt(2.0)
    receiver *= receiver_scale
    data = (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    ) / math.sqrt(2.0)
    data *= math.sqrt(data_power)
    in_band_data_power = data_power * 6.0e6 / SAMPLE_RATE_HZ
    pilot_ratio = 10.0 ** (-PILOT_BELOW_DATA_DB / 10.0)
    pilot_amplitude = math.sqrt(in_band_data_power * pilot_ratio)
    sample = np.arange(count)
    pilot_hz = -3.0e6 + DRIVER["ATSC_PILOT_OFFSET_HZ"]
    pilot = pilot_amplitude * np.exp(2j * math.pi * pilot_hz * sample / SAMPLE_RATE_HZ)
    injected_noise = (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    ) / math.sqrt(2.0)
    injected_noise *= math.sqrt(data_power)

    result = DRIVER["calibrate_received_controls"](
        tx_zero=receiver[0].astype(np.complex64),
        signal_only=(receiver[1] + data + pilot).astype(np.complex64),
        noise_only=(receiver[2] + injected_noise).astype(np.complex64),
        sample_rate_hz=SAMPLE_RATE_HZ,
        bandwidth_hz=6.0e6,
        expected_pilot_below_data_db=PILOT_BELOW_DATA_DB,
    )

    assert result["signal_data_shelf_power"] > 0.0
    assert result["signal_pilot_excess_power"] > 0.0
    assert result["injected_noise_power"] > 0.0
    assert result["received_pilot_below_data_db"] == pytest.approx(
        PILOT_BELOW_DATA_DB,
        abs=1.0,
    )


def _transfer_row(
    *,
    trial: int,
    input_snr_linear: float,
    target: int,
    lower: int,
    upper: int,
    scale: float = 8.0,
    input_noise_power: float = 1.0,
) -> dict[str, object]:
    return {
        "status": "complete",
        "event": {
            "kind": "mixture",
            "snr_db": -18.0,
            "pass_index": 1,
            "trial_index": trial,
        },
        "received_input_calibration": {
            "data_shelf_snr_linear": input_snr_linear,
            "data_shelf_snr_db": 10.0 * math.log10(input_snr_linear),
            "received_data_shelf_power": input_snr_linear * input_noise_power,
            "received_noise_power": input_noise_power,
        },
        "detector": {
            "quantization_scale": scale,
            "p_target_u64": target,
            "p_ref_lower_u64": lower,
            "p_ref_upper_u64": upper,
            "p_ref_sum_u64": lower + upper,
            "normalized_coarse_power_ratio": 1.0,
            "normalized_pilot_excess": 0.0,
            "coarse_power_ratio": 1.0,
            "target_weight_norm_sq": 2,
            "reference_weight_norm_sum_sq": 4,
        },
    }


def _control_row(
    kind: str,
    *,
    target: int,
    lower: int,
    upper: int,
    scale: float = 8.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "complete",
        "event": {
            "kind": kind,
            "snr_db": None,
            "pass_index": 1,
            "trial_index": None,
        },
        "detector": {
            "quantization_scale": scale,
            "p_target_u64": target,
            "p_ref_lower_u64": lower,
            "p_ref_upper_u64": upper,
            "p_ref_sum_u64": lower + upper,
            "target_weight_norm_sq": 2,
            "reference_weight_norm_sum_sq": 4,
        },
    }
    if kind == "noise_only":
        data_power = 8.0
        row["received_control_calibration"] = {
            "receiver_and_leakage_power": 2.0,
            "signal_composite_power": 9.0,
            "signal_data_shelf_power": data_power,
            "signal_pilot_excess_power": (
                data_power * 10.0 ** (-PILOT_BELOW_DATA_DB / 10.0)
            ),
            "injected_noise_power": 20.0,
            "received_pilot_below_data_db": PILOT_BELOW_DATA_DB,
        }
    return row


def test_transfer_table_pools_raw_power_at_one_scale() -> None:
    rows = [
        _transfer_row(
            trial=0,
            input_snr_linear=0.01,
            target=40,
            lower=10,
            upper=10,
        ),
        _transfer_row(
            trial=1,
            input_snr_linear=0.04,
            target=60,
            lower=15,
            upper=15,
            input_noise_power=3.0,
        ),
    ]

    summaries, trials = DRIVER["build_transfer_tables"](
        rows,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
    )

    assert len(summaries) == 1
    assert len(trials) == 2
    summary = summaries[0]
    assert summary["requested_data_shelf_snr_db"] == pytest.approx(
        10.0 * math.log10((0.01 + 0.04 * 3.0) / 4.0)
    )
    assert summary["pooled_received_data_shelf_power"] == pytest.approx(0.13)
    assert summary["pooled_received_noise_power"] == pytest.approx(4.0)
    assert summary["quantization_scale"] == pytest.approx(8.0)
    assert summary["gpu_pooled_p_target"] == 100
    assert summary["gpu_pooled_p_ref_lower"] == 25
    assert summary["gpu_pooled_p_ref_upper"] == 25
    assert summary["gpu_pooled_normalized_coarse_power_ratio"] == pytest.approx(4.0)
    assert summary["trials"] == 2

    changed = [dict(row) for row in rows]
    changed[1] = {
        **changed[1],
        "detector": {
            **changed[1]["detector"],
            "quantization_scale": 7.0,
        },
    }
    with pytest.raises(RuntimeError, match="one quantization scale"):
        DRIVER["build_transfer_tables"](
            changed,
            pilot_below_data_db=PILOT_BELOW_DATA_DB,
        )


def test_transfer_table_builds_expected_curve_from_controls() -> None:
    mixture = _transfer_row(
        trial=0,
        input_snr_linear=0.5,
        target=40,
        lower=10,
        upper=10,
    )
    mixture["tx"] = {"signal_power": 3.0, "noise_power": 1.0}
    mixture["received_input_calibration"][
        "received_pilot_below_data_db"
    ] = PILOT_BELOW_DATA_DB
    rows = [
        _control_row("tx_zero", target=10, lower=5, upper=5),
        _control_row("signal_only", target=50, lower=10, upper=10),
        _control_row("noise_only", target=20, lower=10, upper=10),
        mixture,
    ]

    summaries, trials = DRIVER["build_transfer_tables"](
        rows,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
    )

    summary = summaries[0]
    assert summary["gpu_control_expected_p_target"] == pytest.approx(42.5)
    assert summary["gpu_control_expected_p_ref_sum"] == pytest.approx(20.0)
    assert summary[
        "gpu_control_expected_normalized_coarse_power_ratio"
    ] == pytest.approx(4.25)
    assert summary["gpu_control_expected_normalized_pilot_excess"] == pytest.approx(
        3.25
    )
    assert summary["received_input_pilot_below_data_db"] == pytest.approx(
        PILOT_BELOW_DATA_DB
    )
    assert summary["detector_output_pilot_below_data_db"] == pytest.approx(
        PILOT_BELOW_DATA_DB
    )
    assert trials[0]["received_input_pilot_below_data_db"] == pytest.approx(
        PILOT_BELOW_DATA_DB
    )


def test_control_expected_curve_preserves_nonpositive_excess() -> None:
    mixture = _transfer_row(
        trial=0,
        input_snr_linear=0.5,
        target=40,
        lower=10,
        upper=10,
    )
    mixture["tx"] = {"signal_power": 1.0, "noise_power": 1.0}
    rows = [
        _control_row("tx_zero", target=5, lower=10, upper=10),
        _control_row("signal_only", target=5, lower=10, upper=10),
        _control_row("noise_only", target=5, lower=10, upper=10),
        mixture,
    ]

    summaries, _ = DRIVER["build_transfer_tables"](
        rows,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
    )

    summary = summaries[0]
    assert summary["gpu_control_expected_normalized_pilot_excess"] == pytest.approx(
        -0.5
    )
    assert math.isnan(summary["gpu_control_expected_pilot_excess_db"])
    assert math.isnan(summary["gpu_control_expected_data_shelf_snr_db"])


def test_control_expected_curve_requires_one_scale() -> None:
    mixture = _transfer_row(
        trial=0,
        input_snr_linear=0.5,
        target=40,
        lower=10,
        upper=10,
    )
    mixture["tx"] = {"signal_power": 1.0, "noise_power": 1.0}
    rows = [
        _control_row("tx_zero", target=10, lower=5, upper=5),
        _control_row(
            "signal_only",
            target=50,
            lower=10,
            upper=10,
            scale=7.0,
        ),
        _control_row("noise_only", target=20, lower=10, upper=10),
        mixture,
    ]

    with pytest.raises(RuntimeError, match="one quantization scale"):
        DRIVER["build_transfer_tables"](
            rows,
            pilot_below_data_db=PILOT_BELOW_DATA_DB,
        )


def test_hidden_worker_request_cannot_bypass_authorization(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": DRIVER["SCHEMA_VERSION"],
                "driver": DRIVER["RADIO_DRIVER"],
                "rf_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unauthorized"):
        DRIVER["main"](["--worker-config", str(request_path)])


def test_mocked_pass_uses_one_aligned_continuous_session(tmp_path: Path) -> None:
    input_path, audit_path = _write_input(tmp_path)
    args = _transmit_args(
        tmp_path,
        input_path=input_path,
        audit_path=audit_path,
        event_limit=5,
    )
    requests: list[dict[str, object]] = []
    offset = 23

    def capture_runner(request: dict[str, object]) -> None:
        DRIVER["validate_worker_request"](request)
        requests.append(dict(request))
        assert [slot["kind"] for slot in request["slots"]] == [
            "tx_zero",
            "signal_only",
            "noise_only",
            "drift",
        ]
        tx = np.fromfile(Path(str(request["tx_iq_path"])), dtype=np.complex64)
        rng = np.random.default_rng(44)
        receiver_noise = (
            rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size)
        ).astype(np.complex64)
        receiver_noise *= np.float32(0.00025 / math.sqrt(2.0))
        capture = receiver_noise
        capture[offset:] += tx[:-offset]
        capture.tofile(Path(str(request["session_capture_path"])))
        tx_off = (
            rng.standard_normal(CAPTURE_SAMPLES)
            + 1j * rng.standard_normal(CAPTURE_SAMPLES)
        ).astype(np.complex64)
        tx_off *= np.float32(0.00025 / math.sqrt(2.0))
        tx_off.tofile(Path(str(request["tx_off_capture_path"])))
        _write_stream_status(request)

    class Detector:
        def __init__(self) -> None:
            self.sizes: list[int] = []
            self.scales: list[float | None] = []

        def measure(self, capture: np.ndarray) -> dict[str, object]:
            self.sizes.append(int(capture.size))
            self.scales.append(args.quantization_scale)
            return {
                "backend": "cuda",
                "num_input_streams": 1,
                "estimated_data_shelf_snr_db": -18.0,
                "p_target_u64": 40,
                "p_ref_lower_u64": 10,
                "p_ref_upper_u64": 10,
                "p_ref_sum_u64": 20,
                "coarse_power_ratio": 2.0,
                "normalized_coarse_power_ratio": 4.0,
                "normalized_pilot_excess": 3.0,
                "quantization_scale": 8.0,
                "target_weight_norm_sq": 2,
                "reference_weight_norm_sum_sq": 4,
            }

    def control_calibration(**_kwargs: object) -> dict[str, float]:
        data_power = 8.0e-5
        return {
            "receiver_and_leakage_power": 2.0e-6,
            "signal_composite_power": 9.0e-5,
            "signal_data_shelf_power": data_power,
            "signal_pilot_excess_power": (
                data_power * 10.0 ** (-PILOT_BELOW_DATA_DB / 10.0)
            ),
            "injected_noise_power": 1.0e-4,
            "received_pilot_below_data_db": PILOT_BELOW_DATA_DB,
            "pilot_ratio_error_db": 0.0,
            "psd_bin_width_hz": SAMPLE_RATE_HZ / 16_384.0,
        }

    detector = Detector()
    run_globals = DRIVER["run"].__globals__
    original_calibration = run_globals["calibrate_received_controls"]
    run_globals["calibrate_received_controls"] = control_calibration
    try:
        assert DRIVER["run"](
            args,
            capture_runner=capture_runner,
            detector=detector,
        ) == 0
    finally:
        run_globals["calibrate_received_controls"] = original_calibration

    assert len(requests) == 1
    assert requests[0]["driver"] == DRIVER["RADIO_DRIVER"]
    plan = json.loads(
        (tmp_path / "run" / "run_plan.json").read_text(encoding="utf-8")
    )
    assert plan["radio"]["device_serial"] == "TEST123"
    assert plan["radio"]["native_tx_gain_db"] == 12
    assert plan["radio"]["native_rx_gain_db"] == 12
    assert len(detector.sizes) == 7
    assert detector.sizes == [CAPTURE_SAMPLES] * 7
    assert detector.scales == [None, None, None, 8.0, 8.0, 8.0, 8.0]
    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"]["kind"] for row in rows] == [
        "tx_off",
        "tx_zero",
        "signal_only",
        "noise_only",
        "drift",
    ]
    assert rows[0]["session_alignment"] is None
    assert all(
        row["session_alignment"]["stream_offset_samples"] == offset
        for row in rows[1:]
    )
    assert rows[-1]["received_input_calibration"] is not None
    assert all(row["radio"]["agc"] is False for row in rows)
    assert all(
        row["quantization_scale_source"] == "pass_controls" for row in rows
    )
    assert rows[0]["detector"] is None
    assert args.quantization_scale == pytest.approx(8.0)
    assert (tmp_path / "run" / "events.csv").is_file()
    assert (tmp_path / "run" / "sessions" / "pass_01.json").is_file()


def test_interrupted_pass_resume_freezes_canonical_event_ledger(
    tmp_path: Path,
) -> None:
    input_path, audit_path = _write_input(tmp_path)
    offset = 23

    def capture_runner(request: dict[str, object]) -> None:
        DRIVER["validate_worker_request"](request)
        tx = np.fromfile(Path(str(request["tx_iq_path"])), dtype=np.complex64)
        rng = np.random.default_rng(44)
        capture = (
            rng.standard_normal(tx.size) + 1j * rng.standard_normal(tx.size)
        ).astype(np.complex64)
        capture *= np.float32(0.00025 / math.sqrt(2.0))
        capture[offset:] += tx[:-offset]
        capture.tofile(Path(str(request["session_capture_path"])))
        tx_off = (
            rng.standard_normal(CAPTURE_SAMPLES)
            + 1j * rng.standard_normal(CAPTURE_SAMPLES)
        ).astype(np.complex64)
        tx_off *= np.float32(0.00025 / math.sqrt(2.0))
        tx_off.tofile(Path(str(request["tx_off_capture_path"])))
        _write_stream_status(request)

    class Detector:
        def __init__(self, fail_on: int | None = None) -> None:
            self.calls = 0
            self.fail_on = fail_on

        def measure(self, _capture: np.ndarray) -> dict[str, object]:
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("planned interruption")
            return {
                "backend": "cuda",
                "num_input_streams": 1,
                "estimated_data_shelf_snr_db": -18.0,
                "p_target_u64": 40,
                "p_ref_lower_u64": 10,
                "p_ref_upper_u64": 10,
                "p_ref_sum_u64": 20,
                "coarse_power_ratio": 2.0,
                "normalized_coarse_power_ratio": 4.0,
                "normalized_pilot_excess": 3.0,
                "quantization_scale": 8.0,
                "target_weight_norm_sq": 2,
                "reference_weight_norm_sum_sq": 4,
            }

    def control_calibration(**_kwargs: object) -> dict[str, float]:
        data_power = 8.0e-5
        return {
            "receiver_and_leakage_power": 2.0e-6,
            "signal_composite_power": 9.0e-5,
            "signal_data_shelf_power": data_power,
            "signal_pilot_excess_power": (
                data_power * 10.0 ** (-PILOT_BELOW_DATA_DB / 10.0)
            ),
            "injected_noise_power": 1.0e-4,
            "received_pilot_below_data_db": PILOT_BELOW_DATA_DB,
            "pilot_ratio_error_db": 0.0,
            "psd_bin_width_hz": SAMPLE_RATE_HZ / 16_384.0,
        }

    frozen_path = tmp_path / "frozen" / "events.jsonl"
    duplicate_line_counts: list[int] = []
    run_globals = DRIVER["run"].__globals__
    original_calibration = run_globals["calibrate_received_controls"]
    original_rewrite = run_globals["_rewrite_result_rows"]

    def freeze_then_rewrite(
        path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        if len(rows) == 5:
            duplicate_line_counts.append(
                len(path.read_text(encoding="utf-8").splitlines())
            )
            FREEZER["copy_canonical_event_ledger"](path, frozen_path)
        original_rewrite(path, rows)

    run_globals["calibrate_received_controls"] = control_calibration
    run_globals["_rewrite_result_rows"] = freeze_then_rewrite
    try:
        first_args = _transmit_args(
            tmp_path,
            input_path=input_path,
            audit_path=audit_path,
            event_limit=5,
        )
        with pytest.raises(RuntimeError, match="planned interruption"):
            DRIVER["run"](
                first_args,
                capture_runner=capture_runner,
                detector=Detector(fail_on=6),
            )

        with (tmp_path / "run" / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write('{"event":')

        resume_args = _transmit_args(
            tmp_path,
            input_path=input_path,
            audit_path=audit_path,
            event_limit=5,
        )
        resume_args.resume = True
        assert DRIVER["run"](
            resume_args,
            capture_runner=capture_runner,
            detector=Detector(),
        ) == 0
    finally:
        run_globals["calibrate_received_controls"] = original_calibration
        run_globals["_rewrite_result_rows"] = original_rewrite

    assert duplicate_line_counts == [8]
    frozen_rows = [
        json.loads(line)
        for line in frozen_path.read_text(encoding="utf-8").splitlines()
    ]
    source_rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(frozen_rows) == 5
    assert len(source_rows) == 5
    assert [row["event"]["event_index"] for row in frozen_rows] == [1, 2, 3, 4, 5]
    assert source_rows == frozen_rows
