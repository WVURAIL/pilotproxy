"""Estimator plot workflows preserve trial grouping, conditioning, and provenance."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.result_schema import RESULT_SCHEMA_TOKEN
from pilot_proxy.testbench import plot_results as plots


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("backends,label", [
    (["cpu-reference", "cpu-reference"], "CPU fixed-point"),
    (["cuda", "cuda"], "GPU fixed-point"),
    (["cpu-reference", "cuda"], "CPU/GPU fixed-point"),
    (["cpu-reference", None], "Fixed-point (backend unspecified)"),
])
def test_plot_cli_preserves_independent_passes_and_conditioning_record(
    tmp_path, monkeypatch, backends, label,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setitem(plt.rcParams, "text.usetex", False)
    monkeypatch.setattr(plots, "setup_matplotlib", lambda **kwargs: plt)
    figures = []
    original_subplots = plt.subplots

    def subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        figures.append((fig, ax))
        return fig, ax

    monkeypatch.setattr(plt, "subplots", subplots)
    summaries, trials, metadata_paths = [], [], []
    for source, (backend, ratio) in enumerate(zip(backends, (2, 4), strict=True)):
        directory = tmp_path / f"shard{source}"
        directory.mkdir()
        summary = directory / "dtv_snr_summary.csv"
        trial = directory / "dtv_snr_eval.csv"
        metadata = directory / "dtv_snr_eval.json"
        _write_csv(summary, [{
            "requested_data_shelf_snr_db": snr,
            "pilot_below_data_db": 10, "bin_enbw_hz": 1000,
            "dtv_bandwidth_hz": 10000, "pilot_capture_efficiency": 1,
        } for snr in (0, 10)])
        _write_csv(trial, [{
            "requested_data_shelf_snr_db": snr,
            "pass_index": 0,
            "normalized_coarse_power_ratio": ratio,
            "cpu_packed_normalized_coarse_power_ratio": ratio,
            "p_ref_sum_u64": (2 + 2 * noise) * 16,
            "p_target_u64": (10 + 3 * noise) * 8,
            "quantization_scale": 2,
            "measured_in_band_noise_power": noise,
            "measured_data_shelf_power": 5,
            "target_weight_norm_sq": 2,
            "reference_weight_norm_sum_sq": 4,
        } for snr in (0, 10) for noise in (1, 3)])
        if backend is not None:
            metadata.write_text(json.dumps({"testbench": {"detector_backend": backend}}))
        summaries.append(summary)
        trials.append(trial)
        metadata_paths.append(metadata)

    clean_iq, weights = tmp_path / "clean.cfile", tmp_path / "weights.bin"
    clean_iq.write_bytes(b"controlled clean waveform")
    weights.write_bytes(b"controlled weight bank")
    manifest = Path(str(weights) + ".manifest.json")
    manifest.write_text("{}\n")
    clean = plots._CleanPackedTerms(2, 80, 15, 17, 2, 4, clean_iq, weights)
    monkeypatch.setattr(plots, "_clean_packed_terms", lambda *args, **kwargs: clean)
    png, pdf, record = (tmp_path / "out" / name for name in (
        "transfer.png", "transfer.pdf", "conditioning.json"))
    argv = [
        "--output-png", str(png), "--output-pdf", str(pdf),
        "--conditioning-json", str(record), "--conditioning-trial-csv", str(trials[0]),
        "--bootstrap-samples", "128", "--title", "Controlled transfer",
    ]
    for summary in summaries:
        argv += ["--input-csv", str(summary)]
    assert plots.main(argv) == 0
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF-")
    _, axis = figures[0]
    lines = {line.get_label(): line for line in axis.get_lines()}
    measured = lines[label + ", 0 Hz"]
    np.testing.assert_array_equal(measured.get_xdata(), [0, 10])
    # Both shards carry equal total reference power. Pooling gives Q=3.
    np.testing.assert_allclose(measured.get_ydata(), [10 * math.log10(2)] * 2)
    intervals = [container for container in axis.containers
                 if container.get_label() == "95% pass-bootstrap CI"]
    assert len(intervals) == 1
    # Each shard has pass 0; the independent units have excess 1 and 3.
    # Merging them by pass number alone would leave no bootstrap interval.
    segments = intervals[0].lines[2][0].get_segments()
    np.testing.assert_allclose(segments, [
        [[x, 0], [x, 10 * math.log10(3)]] for x in (0, 10)
    ], atol=1e-12)
    assert [text.get_text() for text in axis.texts] == [
        "Packed CPU reference agrees exactly"
    ]
    conditioned = lines[plots.CONDITIONED_TRANSFER_LABEL]
    x_values = np.asarray(conditioned.get_xdata())
    linear = 10 ** (x_values / 10)
    np.testing.assert_allclose(conditioned.get_ydata(),
                               10 * np.log10((0.5 + 0.8 * linear) / (1 + 0.2 * linear)))
    payload = json.loads(record.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert payload["coefficients"] == {"C_db": 0, "delta": 0.5, "a": 0.8, "b": 0.2}
    assert payload["clean_waveform"]["input_iq_sha256"] == _digest(clean_iq)
    assert payload["clean_waveform"]["weights_sha256"] == _digest(weights)
    assert payload["clean_waveform"]["weight_manifest_sha256"] == _digest(manifest)
    conditioning = payload["lower_trial_conditioning"]
    assert conditioning["trial_csv"] == [{"path": str(trials[0]), "sha256": _digest(trials[0])}]
    assert conditioning["metadata_sha256"] == _digest(metadata_paths[0])
    assert conditioning["rows"] == 4
    assert conditioning["target_noise_response"] == 3
    assert conditioning["reference_noise_response"] == 2
    assert (conditioning["snr_min_db"], conditioning["snr_max_db"]) == (0, 10)
    assert not plt.fignum_exists(figures[0][0].number)


def _clean_input(tmp_path, monkeypatch, *, quantization, zone):
    iq = tmp_path / "clean.cfile"
    np.ones(337, dtype=np.complex64).tofile(iq)
    weights_path = tmp_path / "weights.bin"
    weights_path.write_bytes(b"controlled weights")
    metadata = {
        "result_schema": {"schema_version": RESULT_SCHEMA_TOKEN},
        "input_iq": str(iq),
        "detector_geometry": {
            "input_layout": {"frame_size_samples": 128,
                             "detector_window_samples": 128, "num_input_streams": 2},
            "rf_center_hz": 470e6, "selected_channel_index": 1,
            "dtv_pilot_hz": 470309441, "reference_archive_phase": True, **zone,
        },
        "testbench": {"quantization": {"bits_per_component": 4, "clip_sigma": 3,
                                       **quantization}},
        "atsc_waveform_audit": {"sample_rate_hz": 1e6},
        "selected_weight_layout": {"fixture": True},
    }
    monkeypatch.setattr(plots, "_conditioning_weight_bank_path", lambda *args: weights_path)
    observed = {}

    def adc(values, **kwargs):
        observed["iq"] = values.copy()
        assert kwargs["n_blocks"] == 131
        return np.zeros((131, 2048))

    monkeypatch.setattr(plots, "complex_envelope_to_real_adc_blocks", adc)
    stream = np.tile(np.asarray([1 + 2j, 2 + 1j, -1 - 2j, -2 - 1j]), 32)
    monkeypatch.setattr(plots, "channelize_real_blocks_to_reference_channels",
                        lambda *args, **kwargs: stream[np.newaxis].copy())
    weights = np.full((3, 128), 16, dtype=np.int8)

    class Bank:
        def __init__(self, explicit_path):
            assert explicit_path == weights_path

        def layout_for_pilot_frequency(self, frequency):
            assert frequency == 470.309441
            return {"fixture": True}

        def get_weights_for_pilot_frequency(self, frequency):
            return weights, True

    monkeypatch.setattr(plots, "DetectorWeightBank", Bank)

    def powers(packed, selected_weights, bits):
        observed["packed"] = packed.copy()
        np.testing.assert_array_equal(selected_weights, weights)
        assert bits == 4
        return 5.0, np.asarray([80, 15, 17], dtype=np.uint64)

    monkeypatch.setattr(plots, "coarse_power_ratio_cpu_reference_packed", powers)
    return metadata, stream, observed


@pytest.mark.parametrize("quantization,scale", [
    ({"scale_policy": "explicit", "explicit_scale": 1}, 1),
    ({"scale_policy": "per-trial-estimated", "explicit_scale": None}, 7 / (3 * math.sqrt(2.5))),
])
@pytest.mark.parametrize("zone,reverse", [
    ({"nyquist_zone": 1}, False), ({"spectral_sense": "inverted"}, True),
])
def test_clean_conditioning_preserves_recorded_scale_and_coordinates(
    tmp_path, monkeypatch, quantization, scale, zone, reverse,
):
    metadata, stream, observed = _clean_input(tmp_path, monkeypatch,
                                            quantization=quantization, zone=zone)
    terms = plots._clean_packed_terms(
        metadata, tmp_path / "dtv_snr_eval.json", frequency_offset_hz=0,
        channel_gain_db=20 * math.log10(2), channel_phase_deg=90,
    )
    assert terms.scale == pytest.approx(scale)
    np.testing.assert_allclose(observed["iq"], np.full(337, 2j), atol=1e-15)
    values = [complex(value).conjugate() * (-1 if n % 2 else 1)
              for n, value in enumerate(stream)]
    if reverse:
        values.reverse()
    encoded = []
    for value in values:
        real = min(7, max(-7, round(value.real * scale)))
        imag = min(7, max(-7, round(value.imag * scale)))
        byte = ((real & 15) << 4) | (imag & 15)
        encoded.append(byte - 256 if byte > 127 else byte)
    np.testing.assert_array_equal(observed["packed"], [encoded, encoded])
    assert (terms.p_target, terms.p_ref_lower, terms.p_ref_upper) == (80, 15, 17)
    assert (terms.target_norm_sq, terms.reference_norm_sum_sq) == (128, 256)


@pytest.mark.parametrize("scale", [None, True, "1", 0, -1, math.inf, math.nan])
def test_clean_conditioning_refuses_invalid_recorded_explicit_scale(tmp_path, monkeypatch, scale):
    metadata, _, _ = _clean_input(tmp_path, monkeypatch,
                                  quantization={"scale_policy": "explicit", "explicit_scale": scale},
                                  zone={"nyquist_zone": 1})
    with pytest.raises(ValueError, match="explicit.*scale"):
        plots._clean_packed_terms(metadata, tmp_path / "dtv_snr_eval.json",
                                 frequency_offset_hz=0, channel_gain_db=0, channel_phase_deg=0)


def test_clean_conditioning_refuses_iq_missing_the_final_interpolation_sample(tmp_path, monkeypatch):
    metadata, _, observed = _clean_input(tmp_path, monkeypatch,
                                         quantization={"scale_policy": "explicit", "explicit_scale": 1},
                                         zone={"nyquist_zone": 1})
    # 131*2048 ADC samples at 800 MHz end at IQ position 335.35875 at 1 MHz.
    # Interpolation needs indices 335 and 336, so 336 input samples are short.
    np.ones(336, dtype=np.complex64).tofile(metadata["input_iq"])
    with pytest.raises(ValueError, match="IQ is too short"):
        plots._clean_packed_terms(metadata, tmp_path / "dtv_snr_eval.json",
                                 frequency_offset_hz=0, channel_gain_db=0, channel_phase_deg=0)
    assert observed == {}
