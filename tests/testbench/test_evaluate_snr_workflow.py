"""Bounded real evaluator trials with independent arithmetic and saved products."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.testbench import evaluate_snr as evaluator


def _decode(values):
    return [complex(int(value) >> 4, (int(value) & 15) - (16 if int(value) & 8 else 0))
            for value in values]


def _independent_packed_powers(packed, weights, feeds):
    samples = [_decode(row) for row in packed]
    terms = [_decode(row) for row in weights]
    projections = np.asarray([
        [(sum(x.real * w.real + x.imag * w.imag for x, w in zip(row, term, strict=True)),
          sum(x.imag * w.real - x.real * w.imag for x, w in zip(row, term, strict=True)))
         for row in samples]
        for term in terms
    ], dtype=np.int64)
    # The frozen FFT has its own exact gate; this independently constructs its inputs.
    fine = fine_power_fx(projections, num_streams=feeds)
    return [sum(int(value) for value in row) for row in fine]


@pytest.mark.parametrize("noise_source,scale,feeds,trials,threshold", [
    pytest.param("python", None, 2, 2, -1, id="automatic-two-feed-two-trial"),
    pytest.param("python", 0.002, 1, 1, None, id="explicit-scale"),
    pytest.param("gnuradio", 0.002, 1, 1, -1, id="gnuradio-saved-iq"),
])
def test_real_evaluator_trials_preserve_powers_truth_and_saved_results(
    tmp_path, monkeypatch, noise_source, scale, feeds, trials, threshold,
):
    interpreter = os.environ.get("PILOT_PROXY_GNURADIO_PYTHON")
    if noise_source == "gnuradio" and interpreter is None:
        pytest.skip("set PILOT_PROXY_GNURADIO_PYTHON to exercise GNU Radio")
    rate = 4_500_000.0 / 286.0 * 684.0
    # The last ADC sample needs the two adjacent complex-envelope samples.
    count = math.floor(((16384 + 3) * 2048 - 1) * rate / 800e6) + 2
    assert count == 451485
    source = np.exp(2j * np.pi * (309441 - 3e6) * np.arange(count) / rate).astype(np.complex64)
    iq = tmp_path / "tone.cfile"
    source.tofile(iq)
    audit = tmp_path / "waveform-audit.json"
    audit.write_text(json.dumps({"measured_pilot_below_data_db": 10.0,
                                 "measured_pilot_to_data_power_db": -10.0,
                                 "sample_rate_hz": rate}))
    output = tmp_path / "result"
    noise_calls, packed_calls = [], []
    noise_name = "add_complex_awgn_for_snr" if noise_source == "python" else "add_gnuradio_awgn_for_snr"
    add_noise = getattr(evaluator, noise_name)
    pack = evaluator._pack_streams_for_kernel

    def record_noise(clean, **kwargs):
        result = add_noise(clean, **kwargs)
        noise_calls.append((clean.copy(), result[0].copy()))
        return result

    def record_packing(streams, **kwargs):
        result = pack(streams, **kwargs)
        packed_calls.append((streams.copy(), result.copy()))
        return result

    monkeypatch.setattr(evaluator, noise_name, record_noise)
    monkeypatch.setattr(evaluator, "_pack_streams_for_kernel", record_packing)
    argv = [
        "--input-iq", str(iq), "--output-dir", str(output), "--physical-channel", "14",
        "--noise-source", noise_source, "--detector-backend", "cpu-reference",
        "--synthesis-backend", "cpu", "--frame-size-samples", "16384",
        "--num-input-streams", str(feeds), "--noise-trials", str(trials),
        "--requested-data-shelf-snr-db", "-12", "--seed", "12345",
        "--waveform-audit-json", str(audit), "--pilot-below-data-db-from-audit",
        "--frequency-offset-hz", "100", "--channel-gain-db", "3", "--channel-phase-deg", "30",
    ]
    if scale is not None:
        argv += ["--scale", str(scale)]
    if threshold is not None:
        argv += ["--threshold-data-shelf-snr-db", str(threshold)]
    if noise_source == "gnuradio":
        argv += ["--gnuradio-python", interpreter, "--save-noisy-iq"]
    assert evaluator.main(argv) == 0
    report = json.loads((output / "dtv_snr_eval.json").read_text(),
                        parse_constant=lambda value: pytest.fail(value))
    with (output / "dtv_snr_eval.csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    with (output / "dtv_snr_summary.csv").open() as handle:
        csv_summaries = list(csv.DictReader(handle))
    rows = report["results"]
    assert len(rows) == len(csv_rows) == trials
    assert len(noise_calls) == feeds * trials
    assert len(packed_calls) == trials
    assert len(report["summary"]) == len(csv_summaries) == 1
    expected_clean = source.astype(np.complex128) * 10 ** (3 / 20) * np.exp(
        1j * (2 * np.pi * 100 * np.arange(count) / rate + np.pi / 6))
    for clean, _ in noise_calls:
        np.testing.assert_allclose(clean, expected_clean, rtol=3e-7, atol=3e-7)
    if feeds * trials > 1:
        # Resetting the generator per feed or trial would duplicate these samples.
        noises = [noisy - clean for clean, noisy in noise_calls]
        assert all(not np.array_equal(noises[0], other) for other in noises[1:])
    bank = evaluator.DetectorWeightBank(explicit_path=evaluator.DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_pilot_frequency(470.309441)
    assert valid
    norms = [sum(int(w.real) ** 2 + int(w.imag) ** 2 for w in _decode(term)) for term in weights]
    layout = report["selected_weight_layout"]
    frequencies = [layout[name] for name in (
        "target_normalized_frequency", "lower_reference_normalized_frequency",
        "upper_reference_normalized_frequency",
    )]
    steering = np.exp(-2j * np.pi * np.asarray(frequencies)[:, None] * np.arange(128))

    for trial, (row, csv_row, (streams, packed)) in enumerate(zip(rows, csv_rows, packed_calls, strict=True)):
        assert row["trial"] == trial and row["detector_backend"] == "cpu-reference"
        assert row["detector_rows_per_frame"] == feeds * 128
        clean_noisy = noise_calls[trial * feeds:(trial + 1) * feeds]
        clean_power = sum(float(np.mean(np.abs(clean.astype(np.complex128)) ** 2))
                          for clean, _ in clean_noisy)
        noise_power = sum(float(np.mean(np.abs(noisy.astype(np.complex128) - clean) ** 2))
                          for clean, noisy in clean_noisy)
        in_band = noise_power * 6e6 / rate
        truth = 10 * math.log10(clean_power / 1.1 / in_band)
        assert row["measured_composite_atsc_power"] == pytest.approx(clean_power, rel=2e-6)
        assert row["measured_noise_power"] == pytest.approx(noise_power, rel=2e-6)
        assert row["measured_truth_data_shelf_snr_db"] == pytest.approx(truth, abs=2e-5)
        expected_noise = (clean_power / feeds) / (10 ** (-12 / 10) * 1.1) * rate / 6e6
        assert row["requested_noise_power"] == pytest.approx(expected_noise, rel=2e-6)
        assert row["requested_combined_in_band_noise_power"] == pytest.approx(
            expected_noise * feeds * 6e6 / rate, rel=2e-6)
        expected_scale = scale if scale is not None else 7 / (3 * np.std(streams.real, dtype=np.float64))
        assert row["quantization_scale"] == pytest.approx(expected_scale, rel=2e-6)
        encoded = []
        for value in streams.flat:
            real = min(7, max(-7, round(float(value.real) * row["quantization_scale"])))
            imag = min(7, max(-7, round(float(value.imag) * row["quantization_scale"])))
            byte = ((real & 15) << 4) | (imag & 15)
            encoded.append(byte - 256 if byte > 127 else byte)
        expected_packed = np.asarray(encoded, dtype=np.int8).reshape(feeds * 128, 128)
        np.testing.assert_array_equal(packed, expected_packed)
        powers = _independent_packed_powers(expected_packed, weights, feeds)
        assert [row[name] for name in ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")] == powers
        assert [int(csv_row[name]) for name in ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")] == powers
        ratio = powers[0] * (norms[1] + norms[2]) / ((powers[1] + powers[2]) * norms[0])
        estimate = 10 * math.log10(ratio - 1) + 10 - 10 * math.log10(6e6 / 3051.7578125)
        assert row["normalized_coarse_power_ratio"] == pytest.approx(ratio)
        assert row["estimated_data_shelf_snr_db"] == pytest.approx(estimate)
        assert row["snr_error_db"] == pytest.approx(estimate - truth, abs=2e-5)
        assert row["cpu_gpu_abs_diff"] == 0
        assert row["normalized_positive_excess_decision"] == int(ratio > 1)
        float_rows = streams.reshape(feeds * 128, 128).astype(np.complex128)
        float_powers = [sum(abs(np.sum(values * term.conjugate())) ** 2 for values in float_rows)
                        for term in steering]
        np.testing.assert_allclose([row[name] for name in (
            "cpu_float_p_target", "cpu_float_p_ref_lower", "cpu_float_p_ref_upper")],
            float_powers, rtol=2e-12)
        if threshold is None:
            assert "mask" not in row and report["threshold"] is None
        else:
            expected_mask = int(powers[0] * row["threshold_half_den"]
                                > (powers[1] + powers[2]) * row["threshold_half_num"])
            assert row["mask"] == expected_mask
            assert row["rational_overflow_count"] == 0
        if noise_source == "gnuradio":
            assert row["gnuradio_block"] == "analog.noise_source_c"
            assert len(row["noise_iq_paths"]) == len(row["noise_seeds"]) == feeds
            for feed, (path, (_, noisy), seed) in enumerate(zip(
                row["noise_iq_paths"], clean_noisy, row["noise_seeds"], strict=True,
            )):
                assert Path(path).parent == output / "noisy_iq"
                np.testing.assert_array_equal(np.fromfile(path, dtype=np.complex64), noisy)
                assert Path(path).name == (
                    "snr_shelf_m12p000_freq_offset_p100p000_hz_"
                    f"trial_{trial:04d}_feed_{feed:04d}_seed_{seed}.cfile"
                )
        else:
            assert row["noise_seeds"] == row["noise_iq_paths"] == []

    summary = report["summary"][0]
    target = sum(row["p_target_u64"] for row in rows)
    reference = sum(row["p_ref_sum_u64"] for row in rows)
    pooled = target * (norms[1] + norms[2]) / (reference * norms[0])
    assert summary["gpu_pooled_p_target"] == target
    assert summary["gpu_pooled_normalized_coarse_power_ratio"] == pytest.approx(pooled)
    assert summary["gpu_pooled_estimated_data_shelf_snr_db"] == pytest.approx(
        10 * math.log10(pooled - 1) + 10 - 10 * math.log10(6e6 / 3051.7578125))
    assert int(csv_summaries[0]["trials"]) == summary["trials"] == trials
    assert report["testbench"]["kernel_version"] == "cpu-reference"
    assert report["detector_geometry"]["selected_channel_index"] == 179
    assert report["testbench"]["iq_samples_used"] == count
    assert report["testbench"]["quantization"]["explicit_scale"] == scale
    assert report["calibration"]["pilot_below_data_db_source"] == "waveform_audit"
    assert report["weight_bank"]["sha256"] == hashlib.sha256(bank.path.read_bytes()).hexdigest()
    assert report["selected_weight_coefficients"]["sha256"] == hashlib.sha256(weights.tobytes()).hexdigest()
    assert {path.name for path in output.iterdir()} == {
        "dtv_snr_eval.csv", "dtv_snr_summary.csv", "dtv_snr_eval.json",
        *({"noisy_iq"} if noise_source == "gnuradio" else set()),
    }
