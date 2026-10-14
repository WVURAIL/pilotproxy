"""Offline fixtures only: no devices, SDR bindings, streams or radio commands."""
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.testbench import sdr_protocol as p


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": p.digest(path)}


@pytest.fixture
def spec(tmp_path):
    wave = tmp_path/"fixture.cfile"
    np.full(96, .1+.01j, dtype=np.complex64).tofile(wave)
    result = {"schema": p.SCHEMA, "run_id": "FICTIONAL-OFFLINE-TEST",
              "waveform": {"path": str(wave), "sha256": p.digest(wave), "start_sample": 0,
                           "samples": 96, "source_startup_discard_samples": 200000},
              "implementation_sources": [{"path": __file__, "sha256": p.digest(__file__)}],
              "noise_model": "independent_complex_gaussian",
              "radio": {"frequency_hz": 100000000., "sample_rate_hz": 10000000.,
                        "emission_bandwidth_hz": 6000000., "radio_filter_bandwidth_hz": 8000000., "tx_rms": .01,
                        "device_serial": "FICTIONAL", "antenna_separation_cm": 10., "antenna_polarization": "fixture",
                        "clock_source": "fixture", "native_tx_gain_db": 12, "native_rx_gain_db": 12,
                        "tx_gain_setting_db": 0., "rx_gain_setting_db": 0.,
                        "tx_antenna": "fictional TX", "rx_antenna": "fictional RX"},
              "capture": {"settle_samples": 32, "marker_guard_samples": 16, "capture_samples": 64},
              "guards": {"clip_level": .98, "rms_min": .001, "rms_max": .35,
                         "clip_fraction_max": .0001, "marker_correlation_min": .15},
              "slots": [{"slot_index": i, "kind": kind, "seed": i+50, "data_shelf_snr_db": -18. if kind == "mixture" else None}
                        for i, kind in enumerate(["tx_zero", "signal_only", "noise_only", "mixture"])],
              "rf_limits": {"frequency_low_hz": 95000000., "frequency_high_hz": 105000000.,
                            "power_metric": "average_eirp_dbm", "maximum_power_dbm": -20.,
                            "provided_by": "FICTIONAL TEST CONSTRAINTS; not authorization"}}
    audit = {"schema": "pilotproxy-stationary-input-audit-v1", "waveform_sha256": p.digest(wave),
             "start_sample": 0, "samples": 96, "source_startup_discard_samples": 200000,
             "stationarity_passed": True, "method": "fixture only", "thresholds": {"fixture": 1},
             "segments": [{"start_sample": a, "stop_sample": a+32, "passed": True} for a in (0, 32, 64)]}
    result["stationarity_audit"] = write(tmp_path/"audit.json", audit)
    power = {"schema": "pilotproxy-measured-rf-power-v1", "settings_sha256": p.settings_identity(result),
             "power_metric": "average_eirp_dbm", "covers_all_slots_markers_transitions": True,
             "measurement_id": "FICTIONAL", "method": "fixture only; no physical measurement",
             "conducted_power_dbm": -30., "antenna_gain_dbi": 3., "cable_loss_db": 1.,
             "total_uncertainty_db": 2.}
    result["power_conversion"] = write(tmp_path/"power.json", power)
    return result


def fake(request):
    response = {k: request["radio"][k] for k in ("frequency_hz", "sample_rate_hz", "tx_gain_setting_db", "rx_gain_setting_db")}
    response.update(samples=np.full(64, .01+.02j, dtype=np.complex64), clock_locked=True,
                    lo_locked=True, measurement_start_sample=48, marker_correlation=.9)
    response.update({k: 0 for k in ("rx_underrun", "rx_overrun", "rx_dropped_packets", "tx_underrun", "tx_overrun", "tx_dropped_packets", "timestamp_discontinuities")})
    return response


def test_unknown_rf_constraints_cannot_freeze(spec, tmp_path):
    spec["rf_limits"] = None
    with pytest.raises(ValueError, match="unresolved"):
        p.freeze_protocol(tmp_path/"run", spec)
    assert not (tmp_path/"run").exists()


def test_gain_does_not_substitute_for_measured_power(spec):
    spec["power_conversion"] = None
    with pytest.raises(ValueError, match="gain is not dBm"):
        p.validate_spec(spec)


def test_power_convention_and_uncertainty(spec):
    assert p.validate_spec(spec)["upper_power_dbm"] == -26.
    spec["rf_limits"]["maximum_power_dbm"] = -27.
    with pytest.raises(ValueError, match="exceeds"):
        p.validate_spec(spec)


@pytest.mark.parametrize("key,value", [("tx_gain_setting_db", 1.), ("tx_rms", .02), ("frequency_hz", 101000000.)])
def test_measured_conversion_must_match_settings(spec, key, value):
    spec["radio"][key] = value
    if key == "tx_gain_setting_db":
        spec["radio"]["native_tx_gain_db"] = int(value) + 12
    with pytest.raises(ValueError, match="bind"):
        p.validate_spec(spec)


def test_band_limit_applies_to_emission_edges(spec):
    spec["rf_limits"]["frequency_high_hz"] = 102000000.
    with pytest.raises(ValueError, match="emission interval"):
        p.validate_spec(spec)


def test_exact_waveform_and_stationarity_identity(spec):
    path = Path(spec["waveform"]["path"])
    np.full(96, .11, dtype=np.complex64).tofile(path)
    with pytest.raises(ValueError, match="digest changed"):
        p.validate_spec(spec)
    spec["waveform"]["sha256"] = p.digest(path)
    with pytest.raises(ValueError, match="stationarity evidence"):
        p.validate_spec(spec)


def test_stationarity_report_requires_beginning_and_end(spec, tmp_path):
    audit = json.loads(Path(spec["stationarity_audit"]["path"]).read_text())
    audit["segments"][-1]["stop_sample"] = 95
    spec["stationarity_audit"] = write(tmp_path/"incomplete-audit.json", audit)
    with pytest.raises(ValueError, match="beginning and end"):
        p.validate_spec(spec)


def test_one_slot_per_call_and_completed_history_preserved(spec, tmp_path):
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    calls = []
    def worker(request):
        calls.append(request["slot"]["slot_index"])
        return fake(request)
    first = p.simulate_next_slot(folder, worker)
    prior = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert first["status"] == "accepted_simulation"
    assert p.next_simulation_request(folder)["slot"]["slot_index"] == 1
    p.simulate_next_slot(folder, worker)
    assert calls == [0, 1]
    for name, value in prior.items():
        assert (folder/name).read_bytes() == value
    p.simulate_next_slot(folder, worker)
    p.simulate_next_slot(folder, worker)
    with pytest.raises(ValueError, match="complete"):
        p.next_simulation_request(folder)


@pytest.mark.parametrize("failure", ["overload", "stream", "readback", "settling", "alignment", "clock", "nan"])
def test_failure_preserves_capture_and_prevents_next_slot(spec, tmp_path, failure):
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    def worker(request):
        response = fake(request)
        if failure == "overload": response["samples"][:] = 1.
        if failure == "stream": response["rx_dropped_packets"] = 1
        if failure == "readback": response["tx_gain_setting_db"] += 1
        if failure == "settling": response["measurement_start_sample"] = 47
        if failure == "alignment": response["marker_correlation"] = .1
        if failure == "clock": response["clock_locked"] = False
        if failure == "nan": response["samples"][0] = complex(float("nan"), 0)
        return response
    with pytest.raises(ValueError):
        p.simulate_next_slot(folder, worker)
    assert (folder/"slot-000000-capture.cfile").exists()
    receipt = json.loads((folder/"slot-000000-receipt.json").read_text())
    assert receipt["status"] == "halted"
    with pytest.raises(ValueError, match="halted"):
        p.simulate_next_slot(folder, fake)


def test_worker_exception_is_a_permanent_failed_attempt(spec, tmp_path):
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    def worker(_request):
        raise RuntimeError("fixture interrupted")
    with pytest.raises(RuntimeError):
        p.simulate_next_slot(folder, worker)
    with pytest.raises(ValueError, match="halted"):
        p.next_simulation_request(folder)


def test_pending_intent_and_modified_receipts_fail_closed(spec, tmp_path):
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    p.simulate_next_slot(folder, fake)
    receipt = folder/"slot-000000-receipt.json"
    receipt.write_text(receipt.read_text()+" ")
    with pytest.raises(ValueError, match="digest changed"):
        p.next_simulation_request(folder)
    other = tmp_path/"pending"
    p.freeze_protocol(other, spec)
    (other/"slot-000000-intent.json").write_text("{}")
    with pytest.raises(ValueError, match="interrupted"):
        p.next_simulation_request(other)


def test_modified_capture_refuses_resume(spec, tmp_path):
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    p.simulate_next_slot(folder, fake)
    (folder/"slot-000000-capture.cfile").write_bytes(b"changed")
    with pytest.raises(ValueError, match="saved capture changed"):
        p.next_simulation_request(folder)


def test_awgn_is_signal_independent_and_legacy_model_is_labelled():
    signal = np.exp(1j*np.arange(10000)/13)
    independent = p.draw_noise(signal, seed=57, model="independent_complex_gaussian")
    other = p.draw_noise(signal[::-1], seed=57, model="independent_complex_gaussian")
    assert np.array_equal(independent, other)
    assert not np.isclose(np.mean(np.abs(independent)**2), 1., rtol=1e-6)
    conditioned = p.draw_noise(signal, seed=57, model="orthogonal_unit_power")
    assert abs(np.vdot(signal, conditioned))/signal.size < 1e-7
    assert np.isclose(np.mean(np.abs(conditioned)**2), 1., atol=1e-6)
    assert abs(np.vdot(signal, independent))/signal.size > 1e-5


def test_changed_implementation_refuses_new_requests(spec, tmp_path):
    source = tmp_path/"fixture-worker.py"
    source.write_text("# fixture implementation v1")
    spec["implementation_sources"] = [{"path": str(source), "sha256": p.digest(source)}]
    folder = tmp_path/"run"
    p.freeze_protocol(folder, spec)
    source.write_text("# modified implementation")
    with pytest.raises(ValueError, match="implementation source digest changed"):
        p.next_simulation_request(folder)
