"""Standalone detector artifacts, fixed-point results, and failure cleanup."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy import detect
from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.detector_reference import (
    frame_powers_cpu_reference_packed,
    matched_filter_row_projections_cpu_reference_packed,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.kernel import KernelSpecs
from pilot_proxy.paths import DEFAULT_LIB_PATH, DEFAULT_WEIGHTS_PATH


def _cpu_device(monkeypatch, weights):
    """Exercise the host detector with a CPU implementation of its device calls."""
    arrays = {}
    handles = {}
    destroyed = []

    def device_array(values):
        array = np.asarray(values)
        pointer = id(array)
        arrays[pointer] = array
        return SimpleNamespace(data=SimpleNamespace(ptr=pointer), reshape=array.reshape)

    def create(rows, batch, pointer):
        handle = len(handles) + 1
        handles[handle] = (rows, batch, arrays[pointer])
        return handle

    def projections(handle, weights_pointer, output_pointer):
        rows, batch, packed = handles[handle]
        assert weights_pointer != 0
        output = arrays[output_pointer].reshape(batch, 3, rows, 2)
        for index, block in enumerate(packed.reshape(batch, rows, 128)):
            output[index] = matched_filter_row_projections_cpu_reference_packed(
                block, weights, 4
            )

    def fine_powers(input_pointer, streams, windows, batch, output_pointer):
        assert windows == 128
        rows = arrays[input_pointer].reshape(batch, 3, streams * windows, 2)
        output = arrays[output_pointer].reshape(batch, 3, 256)
        for index, block in enumerate(rows):
            output[index] = fine_power_fx(block.astype(np.int64), num_streams=streams)

    cp = SimpleNamespace(
        asarray=device_array,
        zeros=lambda size, dtype: device_array(np.zeros(size, dtype=dtype)),
        asnumpy=lambda array: arrays[array.data.ptr].copy(),
        int32=np.int32,
        uint64=np.uint64,
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(synchronize=lambda: None)),
    )
    kernel = SimpleNamespace(
        specs=KernelSpecs(128, 3, 4, 2),
        version=SimpleNamespace(as_string=lambda: "cpu-reference"),
        supports_row_projections=lambda: True,
        supports_fine_powers=lambda: True,
        create_raw=lambda rows, pointer: create(rows, 1, pointer),
        create_raw_batch=create,
        compute_row_projections_i32=projections,
        compute_fine_powers_u64=fine_powers,
        destroy=destroyed.append,
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(detect, "PilotProxyKernel", lambda path: kernel)
    return cp, kernel, destroyed


def _weights():
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    weights, valid = bank.get_weights_for_physical_channel(14)
    assert valid and weights is not None
    return weights


@pytest.mark.parametrize(
    "backend", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("input_mode", ["explicit_matrix", "sidecar_default_blocks"])
def test_detect_main_writes_exact_powers_and_calibrated_artifact(
    tmp_path, monkeypatch, capsys, backend, input_mode
):
    weights = _weights()
    if backend == "cuda":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device required")
        if not DEFAULT_LIB_PATH.is_file():
            pytest.skip("Native detector library required")
        expected_version = detect.PilotProxyKernel(DEFAULT_LIB_PATH).version.as_string()
        destroyed = None
    else:
        _, _, destroyed = _cpu_device(monkeypatch, weights)
        expected_version = "cpu-reference"

    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(1701)
    block = np.tile(weights[0], (256, 1))
    block[::3] = rng.integers(-128, 128, size=block[::3].shape, dtype=np.int8)
    if input_mode == "explicit_matrix":
        input_path = tmp_path / "packed.npy"
        packed = block
        arguments = [
            "--input-detector-matrix",
            str(input_path),
            "--physical-channel",
            "14",
        ]
        expected_requested = str(input_path)
    else:
        input_path = tmp_path / "generated/detector_input/detector_blocks_i4.npy"
        input_path.parent.mkdir(parents=True)
        packed = np.stack((block, np.zeros_like(block)))
        (input_path.parent / "metadata.json").write_text(
            json.dumps({"dtv_pilot_hz": physical_channel_to_pilot_hz(14)})
        )
        arguments = []
        expected_requested = "generated/detector_input/detector_matrix_i4.npy"
    np.save(input_path, packed)
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_path = tmp_path / "results/detect.json"
    calibration = {
        "pilot_below_data_db": 13.5,
        "bin_enbw_hz": 3200.0,
        "dtv_bandwidth_hz": 5_500_000.0,
        "pilot_capture_efficiency": 0.8,
    }
    arguments += [
        "--output-json",
        str(output_path),
        "--num-input-streams",
        "2",
        "--frame-size-samples",
        "16384",
        "--lib-path",
        str(DEFAULT_LIB_PATH),
        "--weights-path",
        str(DEFAULT_WEIGHTS_PATH),
    ]
    for name, value in calibration.items():
        arguments += ["--" + name.replace("_", "-"), str(value)]

    assert detect.main(arguments) == 0
    text = output_path.read_text()
    payload = json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_hash
    assert payload["schema_version"] == "pilotproxy_detection_result_v1"
    assert payload["input_detector_matrix"] == expected_requested
    assert (
        tmp_path / payload["resolved_input_detector_matrix"]
    ).resolve() == input_path
    assert payload["dtv_pilot_mhz"] == physical_channel_to_pilot_hz(14) / 1e6
    assert payload["physical_channel"] == (14 if packed.ndim == 2 else None)
    selected = payload["selected_weight_layout"]
    assert selected["physical_channel"] == 14
    assert selected["dtv_pilot_hz"] == physical_channel_to_pilot_hz(14)
    assert selected["coarse_channel_index"] == 843
    assert selected["reference_offset_bins_selected"] == 2
    assert payload["result_schema"]["calibration"] == calibration
    assert all(
        value is None for value in payload["result_schema"]["threshold"].values()
    )
    for layout in (payload["layout"], payload["result_schema"]["layout"]):
        assert layout["num_input_streams"] == 2
        assert layout["frame_size_samples"] == 16384
        assert layout["detector_window_samples"] == 128
        assert layout["windows_per_stream"] == 128
        assert layout["detector_rows_per_frame"] == 256
    assert payload["kernel_specs"] == {
        "detector_window_samples": 128,
        "num_weight_terms": 3,
        "sample_bits_per_component": 4,
        "reference_offset_bins": 2,
    }
    assert payload["kernel_version"] == expected_version
    assert payload["matched_filter_row_projections"] is None

    # Decode both signed nibbles independently of the detector norm helpers.
    unsigned = weights.view(np.uint8).astype(np.int64)
    real = ((unsigned >> 4) + 8) % 16 - 8
    imag = ((unsigned & 15) + 8) % 16 - 8
    norms = (real**2 + imag**2).sum(axis=1)
    target_norm, reference_norm = int(norms[0]), int(norms[1] + norms[2])
    assert payload["target_norm_sq"] == target_norm
    assert payload["reference_norm_sum_sq"] == reference_norm
    assert payload["null_power_ratio"] == pytest.approx(
        2 * target_norm / reference_norm
    )
    blocks = packed.reshape(-1, 256, 128)
    assert payload["batch"] == len(blocks)
    assert payload["detector_rows_per_block"] == 256
    assert len(payload["results"]) == len(blocks)
    powers_by_block = []
    for index, (block, result) in enumerate(
        zip(blocks, payload["results"], strict=True)
    ):
        powers = frame_powers_cpu_reference_packed(block, weights, 4)
        powers_by_block.append(powers)
        target, lower, upper = map(int, powers)
        reference = lower + upper
        assert result["block_index"] == index
        assert [
            result[key]
            for key in ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")
        ] == list(powers)
        assert result["p_ref_sum_u64"] == reference
        positive = target * reference_norm > reference * target_norm
        assert result["mask"] == result["normalized_positive_excess_mask"] == positive
        if reference == 0:
            assert all(
                result[key] is None
                for key in (
                    "coarse_power_ratio",
                    "normalized_coarse_power_ratio_db",
                    "raw_pilot_excess",
                    "normalized_pilot_excess",
                    "pilot_excess_db",
                    "estimated_data_shelf_snr_db",
                )
            )
            continue
        ratio = 2 * target / reference
        normalized = target * reference_norm / (reference * target_norm)
        excess_db = 10 * math.log10(normalized - 1)
        assert result["coarse_power_ratio"] == pytest.approx(ratio)
        assert result["raw_pilot_excess"] == pytest.approx(ratio - 1)
        assert result["normalized_coarse_power_ratio_db"] == pytest.approx(
            10 * math.log10(normalized)
        )
        assert result["normalized_pilot_excess"] == pytest.approx(normalized - 1)
        assert result["pilot_excess_db"] == pytest.approx(excess_db)
        expected_shelf = (
            excess_db + 13.5 - 10 * math.log10(5_500_000 / 3200) - 10 * math.log10(0.8)
        )
        assert result["estimated_data_shelf_snr_db"] == pytest.approx(expected_shelf)
    np.testing.assert_array_equal(
        np.asarray(payload["fine_powers_u64"], dtype=np.uint64).sum(axis=-1),
        np.asarray(powers_by_block),
    )
    stdout = capsys.readouterr().out
    assert f"Wrote {output_path}" in stdout
    assert "blocks, positive_excess_set, coarse_power_ratio_mean" in stdout
    if destroyed is not None:
        assert destroyed == [1]


@pytest.mark.parametrize(
    "stage", ["allocate_rows", "allocate_fine", "compute_rows", "compute_fine"]
)
@pytest.mark.parametrize("batched", [False, True])
def test_detect_releases_handle_after_device_failure(monkeypatch, stage, batched):
    weights = _weights()
    cp, kernel, destroyed = _cpu_device(monkeypatch, weights)
    original_zeros = cp.zeros
    allocations = 0

    def fail(*args, **kwargs):
        raise RuntimeError("injected device failure")

    def zeros(size, dtype):
        nonlocal allocations
        allocations += 1
        if (stage, allocations) in (("allocate_rows", 1), ("allocate_fine", 2)):
            fail()
        return original_zeros(size, dtype)

    cp.zeros = zeros
    if stage == "compute_rows":
        kernel.compute_row_projections_i32 = fail
    elif stage == "compute_fine":
        kernel.compute_fine_powers_u64 = fail
    packed = np.zeros((2, 128, 128) if batched else (128, 128), dtype=np.int8)
    with pytest.raises(RuntimeError, match="injected device failure"):
        detect.detect_packed_detector_input(
            packed=packed, weights=weights, kernel=kernel
        )
    assert destroyed == [1]


@pytest.mark.parametrize("streams, frame", [(1, None), (4, 8192)])
def test_detect_main_refuses_layout_that_changes_fine_stream_grouping(
    tmp_path, monkeypatch, streams, frame
):
    _cpu_device(monkeypatch, _weights())
    input_path = tmp_path / "packed.npy"
    np.save(input_path, np.zeros((256, 128), dtype=np.int8))
    output_path = tmp_path / "existing.json"
    output_path.write_text("previous result\n")

    def unexpected_device_call(**kwargs):
        pytest.fail("Invalid stream grouping must be rejected before detection")

    monkeypatch.setattr(detect, "detect_packed_detector_input", unexpected_device_call)
    arguments = [
        "--input-detector-matrix",
        str(input_path),
        "--physical-channel",
        "14",
        "--output-json",
        str(output_path),
        "--num-input-streams",
        str(streams),
        "--weights-path",
        str(DEFAULT_WEIGHTS_PATH),
    ]
    if frame is not None:
        arguments += ["--frame-size-samples", str(frame)]
    with pytest.raises(SystemExit, match="128 windows per input stream"):
        detect.main(arguments)
    assert output_path.read_text() == "previous result\n"


@pytest.mark.parametrize(
    "name, value",
    [
        ("pilot-below-data-db", "nan"),
        ("pilot-below-data-db", "-inf"),
        ("bin-enbw-hz", "nan"),
        ("bin-enbw-hz", "0"),
        ("dtv-bandwidth-hz", "inf"),
        ("dtv-bandwidth-hz", "-1"),
        ("pilot-capture-efficiency", "nan"),
        ("pilot-capture-efficiency", "0"),
    ],
)
def test_detect_main_refuses_invalid_calibration_before_loading_kernel(
    tmp_path, monkeypatch, name, value
):
    input_path = tmp_path / "packed.npy"
    np.save(input_path, np.zeros((128, 128), dtype=np.int8))
    output_path = tmp_path / "existing.json"
    output_path.write_text("previous result\n")

    def unexpected_kernel(path):
        pytest.fail("Calibration must be validated before loading the kernel")

    monkeypatch.setattr(detect, "PilotProxyKernel", unexpected_kernel)
    with pytest.raises(SystemExit, match=f"--{name} must be"):
        detect.main(
            [
                "--input-detector-matrix",
                str(input_path),
                "--physical-channel",
                "14",
                "--output-json",
                str(output_path),
                f"--{name}={value}",
            ]
        )
    assert output_path.read_text() == "previous result\n"


@pytest.mark.parametrize(
    "case",
    ["missing", "no_pilot", "mismatched_pilot", "ambiguous_pilot", "bad_tolerance"],
)
def test_detect_main_refuses_invalid_request_before_loading_kernel(
    tmp_path, monkeypatch, case
):
    input_path = tmp_path / "packed.npy"
    if case != "missing":
        np.save(input_path, np.zeros((128, 128), dtype=np.int8))
    arguments = ["--input-detector-matrix", str(input_path)]
    expected = {
        "missing": "Input detector matrix does not exist",
        "no_pilot": "no --physical-channel/--dtv-pilot-mhz",
        "mismatched_pilot": "disagrees with the matrix sidecar",
        "ambiguous_pilot": "not both",
        "bad_tolerance": "must be finite and nonnegative",
    }[case]
    if case == "mismatched_pilot":
        (tmp_path / "metadata.json").write_text(
            json.dumps({"dtv_pilot_hz": physical_channel_to_pilot_hz(15)})
        )
        arguments += ["--physical-channel", "14"]
    elif case == "ambiguous_pilot":
        arguments += ["--physical-channel", "14", "--dtv-pilot-mhz", "470.31"]
    elif case == "bad_tolerance":
        arguments += ["--pilot-frequency-tolerance-hz", "nan"]
    output_path = tmp_path / "existing.json"
    output_path.write_text("previous result\n")
    arguments += ["--output-json", str(output_path)]

    def unexpected_kernel(path):
        pytest.fail("Request must be rejected before loading the kernel")

    monkeypatch.setattr(detect, "PilotProxyKernel", unexpected_kernel)
    with pytest.raises(SystemExit, match=expected):
        detect.main(arguments)
    assert output_path.read_text() == "previous result\n"
