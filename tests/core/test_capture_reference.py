"""Tests of the native capture ABI checks and of the exact capture replay.

The replay of the detector arithmetic is compared with an independent scalar
computation in exact integers.
"""
from __future__ import annotations

import ctypes
import dataclasses
import struct

import numpy as np
import pytest

from pilot_proxy.capture_reference import (
    CHORD_COARSE_FREQ_OFFSET,
    CHORD_METADATA_SIZE,
    CHORD_NFREQ_OFFSET,
    CaptureGeometry,
    detector_rows_from_voltage,
    expected_products,
    expected_matched_filter_row_projections,
    read_raw_frames,
    validate_frames,
)
from pilot_proxy.fine_decision import (
    ALWAYS_MASKED_Q16,
    fine_mask_decision,
    fine_required_multiplier_q16,
)
from pilot_proxy.fxfft import fxfft256_scalar


class BeamCoord(ctypes.LittleEndianStructure):
    _fields_ = [("ra", ctypes.c_float * 20), ("dec", ctypes.c_float * 20), ("scaling", ctypes.c_uint32 * 20)]


class NativeMetadata(ctypes.LittleEndianStructure):
    # copy of the native declaration, so the fixtures do not reuse the parser offsets
    _fields_ = [
        ("limits", ctypes.c_int32 * 6), ("frame_counter", ctypes.c_int32),
        ("fpga_seq_num", ctypes.c_int64), ("time_downsampling_fpga", ctypes.c_int32),
        ("name", ctypes.c_char * 24), ("type", ctypes.c_int32), ("dims", ctypes.c_int32),
        ("dim", ctypes.c_int32 * 10), ("dim_name", (ctypes.c_char * 24) * 10),
        ("dim_scaling", ctypes.c_int64 * 10), ("stride", ctypes.c_int64 * 10),
        ("offset", ctypes.c_int64), ("nfreq", ctypes.c_int32),
        ("coarse_freq", ctypes.c_int32 * 12288),
        ("freq_upchan_factor", ctypes.c_int32 * 12288),
        ("freq_upchan_index", ctypes.c_int32 * 12288),
        ("num_stream_ids", ctypes.c_int32), ("stream_ids", ctypes.c_uint32 * 128),
        ("rfi_frame_excision_enabled", ctypes.c_int32),
        ("num_rfi_frame_excision_thresholds", ctypes.c_int32),
        ("rfi_frame_excision_thresholds", (ctypes.c_float * 2) * 8),
        ("rfi_flagged_samples", ctypes.c_int32), ("lost_timesamples", ctypes.c_int32),
        ("have_beam_coord", ctypes.c_int32), ("beam_coord", BeamCoord),
        ("stream_id", ctypes.c_uint64), ("dataset_id", ctypes.c_char * 32),
        ("first_packet_recv_time", ctypes.c_int64 * 2),
    ]


def metadata(geometry, kind="voltage", sequence=100, downsampling=4):
    m = NativeMetadata()
    m.limits[:] = (10, 24, 24, 12288, 128, 8)
    m.frame_counter = 17
    m.fpga_seq_num, m.time_downsampling_fpga = sequence, downsampling
    layouts = {
        "voltage": (b"E", 8, (geometry.num_times, geometry.num_freq, 2, geometry.num_dishes), (b"T", b"F", b"P", b"D")),
        "mask": (b"dtv_mask", 9, (geometry.num_freq,), (b"F",)),
        "power": (b"dtv_powers", 6, (geometry.num_freq, 3), (b"F", b"W")),
    }
    m.name, m.type, dims, names = layouts[kind]
    m.dims = len(dims)
    stride = 1
    for i in reversed(range(m.dims)):
        m.dim[i], m.dim_name[i].value, m.dim_scaling[i], m.stride[i] = dims[i], names[i], 1, stride
        stride *= dims[i]
    m.nfreq = geometry.num_freq
    for i, frequency in enumerate(geometry.frequency_ids):
        m.coarse_freq[i] = frequency
    m.freq_upchan_index[0] = -1
    m.num_stream_ids = -1
    m.rfi_frame_excision_enabled = -1
    m.num_rfi_frame_excision_thresholds = -1
    m.rfi_flagged_samples = m.lost_timesamples = -1
    return m


def write_frames(path, rows):
    with path.open("wb") as out:
        for m, payload in rows:
            out.write(struct.pack("<I", ctypes.sizeof(m)))
            out.write(bytes(m))
            out.write(payload)
    return path


@pytest.fixture
def geometry():
    return CaptureGeometry(num_times=4096, k=64, frequency_ids=(2408, 1600), num_dishes=1)


def captured_frames(tmp_path, geometry):
    voltage = write_frames(tmp_path / "voltage.raw", [
        (metadata(geometry, sequence=100 + i * geometry.num_times * 4), bytes([0x88]) * geometry.voltage_frame_size)
        for i in range(geometry.frames_per_block)
    ])
    mask = write_frames(tmp_path / "mask.raw", [
        (metadata(geometry, "mask", downsampling=geometry.block_samples * 4), bytes(geometry.mask_frame_size)),
    ])
    power = write_frames(tmp_path / "power.raw", [
        (metadata(geometry, "power", downsampling=geometry.block_samples * 4), bytes(geometry.power_frame_size)),
    ])
    return (
        read_raw_frames([voltage], geometry.voltage_frame_size),
        read_raw_frames([mask], geometry.mask_frame_size),
        read_raw_frames([power], geometry.power_frame_size),
    )


def test_native_metadata_abi_and_frame_offsets(tmp_path, geometry):
    assert ctypes.sizeof(NativeMetadata) == CHORD_METADATA_SIZE == 148888
    assert NativeMetadata.nfreq.offset == CHORD_NFREQ_OFFSET == 528
    assert NativeMetadata.coarse_freq.offset == CHORD_COARSE_FREQ_OFFSET == 532
    voltage, masks, powers = captured_frames(tmp_path, geometry)
    assert validate_frames(voltage, masks, powers, 2, geometry=geometry) == 1
    assert voltage[0].frequency_ids == (2408, 1600)
    assert voltage[0].payload_offset == 4 + CHORD_METADATA_SIZE
    assert voltage[1].frame_offset == 4 + CHORD_METADATA_SIZE + geometry.voltage_frame_size
    assert voltage[1].provenance()["payload_offset"] == 2 * (4 + CHORD_METADATA_SIZE) + geometry.voltage_frame_size
    assert voltage[0].provenance()["source_path"] == str((tmp_path / "voltage.raw").resolve())
    assert not voltage[0].payload.flags.writeable
    assert voltage[0].provenance()["lost_timesamples"] == -1


@pytest.mark.parametrize("changes", [
    {"k": 65}, {"k": True}, {"num_times": False}, {"num_times": 100},
    {"num_times": 16384}, {"num_polarizations": 1}, {"windows_per_stream": 127},
    {"num_dishes": 1025}, {"num_dishes": 1.5}, {"frequency_ids": ()},
    {"frequency_ids": (1, 1)}, {"frequency_ids": (True,)}, {"frequency_ids": (12288,)},
])
def test_geometry_refusals(geometry, changes):
    with pytest.raises(ValueError):
        dataclasses.replace(geometry, **changes)


def test_geometry_converts_frequency_list_to_immutable_tuple():
    ids = [2408]
    geometry = CaptureGeometry(16384, 128, ids, 1)
    ids.append(1)
    assert geometry.frequency_ids == (2408,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        geometry.k = 64


@pytest.mark.parametrize("part", ["length", "metadata", "payload", "empty"])
def test_truncated_files_refused(tmp_path, geometry, part):
    path = tmp_path / "bad.raw"
    data = struct.pack("<I", CHORD_METADATA_SIZE) + bytes(metadata(geometry)) + bytes(geometry.voltage_frame_size)
    path.write_bytes({"length": data[:2], "metadata": data[:100], "payload": data[:-1], "empty": b""}[part])
    with pytest.raises(ValueError, match="truncated|empty"):
        read_raw_frames([path], geometry.voltage_frame_size)


@pytest.mark.parametrize("field,value", [
    ("limits", (10, 24, 24, 8192, 128, 8)), ("fpga_seq_num", -1),
    ("time_downsampling_fpga", 0), ("dims", -1), ("nfreq", 0), ("nfreq", 12289),
    ("num_stream_ids", 129), ("num_rfi_frame_excision_thresholds", 9),
    ("rfi_frame_excision_enabled", 2), ("lost_timesamples", -2),
    ("coarse_freq", (-1, 1600)), ("coarse_freq", (2408, 2408)),
    ("freq_upchan_factor", (2, 2)), ("freq_upchan_index", (1, 1)),
])
def test_bad_metadata_refused(tmp_path, geometry, field, value):
    m = metadata(geometry)
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            getattr(m, field)[index] = item
    else:
        setattr(m, field, value)
    path = write_frames(tmp_path / "bad.raw", [(m, bytes(geometry.voltage_frame_size))])
    with pytest.raises(ValueError):
        read_raw_frames([path], geometry.voltage_frame_size)


def test_unknown_abi_and_duplicate_files_refused(tmp_path, geometry):
    path = tmp_path / "bad.raw"
    path.write_bytes(struct.pack("<I", 100))
    with pytest.raises(ValueError, match="metadata size"):
        read_raw_frames([path], geometry.voltage_frame_size)
    with pytest.raises(ValueError, match="unique"):
        read_raw_frames([path, path], geometry.voltage_frame_size)
    with pytest.raises(ValueError, match="ordered sequence"):
        read_raw_frames(path, geometry.voltage_frame_size)


@pytest.mark.parametrize("kind,field,value", [
    (0, "fpga_seq_num", 103), (0, "time_downsampling_fpga", 5),
    (1, "fpga_seq_num", 101), (2, "time_downsampling_fpga", 1),
    (0, "frequency_ids", (1600, 2408)), (1, "frequency_ids", (2408, 2623)),
    (2, "frequency_ids", (1600, 2408)), (0, "data_type", 7),
    (0, "dimension_names", ("T", "F", "D", "P")),
    (0, "strides", (1, 2, 3, 4)), (0, "array_offset", 1),
    (0, "dimensions", (4096, 2, 1, 2)), (1, "data_type", 3),
    (2, "dimension_scaling", (1, 2)), (2, "array_name", "other"),
])
def test_frame_configuration_and_timing_refusals(tmp_path, geometry, kind, field, value):
    frames = captured_frames(tmp_path, geometry)
    index = 1 if kind == 0 else 0
    frames[kind][index] = dataclasses.replace(frames[kind][index], **{field: value})
    with pytest.raises(ValueError):
        validate_frames(*frames, 2, geometry=geometry)


def test_whole_blocks_and_complete_counts_required(tmp_path, geometry):
    frames = captured_frames(tmp_path, geometry)
    with pytest.raises(ValueError, match="whole"):
        validate_frames(*frames, 1, geometry=geometry)
    with pytest.raises(ValueError, match="counts"):
        validate_frames(*frames, 4, geometry=geometry)
    with pytest.raises(ValueError, match="boolean"):
        validate_frames(*frames, True, geometry=geometry)


def bundle(k, *, reverse=False, rank=1):
    return {
        "input_preprocessing": {"time_reverse_detector_windows_before_kernel": reverse},
        "profiles": [{
            "chord_channel_id": 2408, "weight_bank_offset_bytes": 0, "weight_bank_nbytes": 3 * k,
            "fine_calibration": {
                "status": "calibrated", "decision_version": "fine_decision_v1",
                "anchor_bin": 255, "designated_half_width": 1,
                "bulk_mask_words_hex": ["000000000000001c", "0", "0", "0"],
                "cfar_rank": rank, "cfar_multiplier_q16": 65536,
            },
        }],
    }


@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_reference_matches_scalar_projection_and_fft(k, reverse):
    g = CaptureGeometry(k * 128, k, (2408, 1600), 1)
    rng = np.random.default_rng(504)
    raw = rng.integers(0, 256, g.block_samples * g.num_freq * g.num_streams, dtype=np.uint8)
    weights = rng.integers(0, 256, 3 * k, dtype=np.uint8).tobytes()
    real, imag = detector_rows_from_voltage(raw, 0, reverse, geometry=g)
    projections = expected_matched_filter_row_projections(real, imag, weights)
    oracle = np.zeros((3, 256, 2), dtype=np.int64)
    scalar_weights = list(weights)
    raw_view = raw.reshape(g.block_samples, 2, 2, 1)
    for term in range(3):
        for stream in range(2):
            for window in range(128):
                zr = zi = 0
                for pos in range(k):
                    time = window * k + (k - pos - 1 if reverse else pos)
                    x = int(raw_view[time, 0, stream, 0])
                    xr, xi = (x >> 4) - 8, (x & 15) - 8
                    w = scalar_weights[term * k + pos]
                    wr = (w >> 4) if (w >> 4) < 8 else (w >> 4) - 16
                    wi = (w & 15) if (w & 15) < 8 else (w & 15) - 16
                    zr += xr * wr + xi * wi
                    zi += xi * wr - xr * wi
                oracle[term, stream * 128 + window] = zr, zi
    np.testing.assert_array_equal(projections, oracle)
    fine = np.zeros((3, 256), dtype=np.uint64)
    for term in range(3):
        for stream in range(2):
            transformed = fxfft256_scalar(oracle[term, stream * 128:(stream + 1) * 128])
            fine[term] += np.array([r * r + i * i for r, i in transformed], dtype=np.uint64)
    result = expected_products(raw, bundle(k, reverse=reverse), weights, geometry=g)
    args = dict(anchor_bin=255, designated_half_width=1, bulk_mask=[28, 0, 0, 0], cfar_rank=1)
    assert result.decisions[0] == fine_required_multiplier_q16(fine, **args)
    assert result.masks[0] == fine_mask_decision(fine, multiplier_q16=65536, **args).mask
    assert result.decisions[0].valid
    np.testing.assert_array_equal(result.powers[0], fine.sum(axis=1, dtype=np.uint64))
    assert result.decisions[1] is None and result.masks[1] == 0
    assert result.powers[1].tolist() == [0, 0, 0]


def test_invalid_rows_are_retained_distinct_from_unbound(geometry):
    raw = np.full(geometry.block_samples * geometry.num_freq * geometry.num_streams, 0x88, dtype=np.uint8)
    result = expected_products(raw, bundle(64), bytes([0x10]) * 192, geometry=geometry)
    invalid = result.decisions[0]
    assert not invalid.valid and invalid.n_bulk == 0 and invalid.multiplier_q16 is None
    assert result.decisions[1] is None
    assert result.masks.tolist() == [0, 0]


def test_valid_always_masked_score_is_distinct_from_invalid():
    g = CaptureGeometry(8192, 64, (2408,), 1)
    raw = np.full((8192, 1, 2, 1), 0x88, dtype=np.uint8)
    rng = np.random.default_rng(18)
    for stream in range(2):
        for window in range(128):
            value = int(rng.integers(-2, 3))
            raw[window * 64, 0, stream, 0] = ((9 + value) << 4) | 8
            raw[window * 64 + 1, 0, stream, 0] = ((9 - value) << 4) | 8
    weights = np.zeros((3, 64), dtype=np.uint8)
    weights[0, :2] = 0x10
    weights[1:, 0] = 0x10
    config = bundle(64, rank=0)
    config["profiles"][0]["fine_calibration"].update(anchor_bin=0, designated_half_width=0, bulk_mask_words_hex=["4", "0", "0", "0"])
    result = expected_products(raw.ravel(), config, weights.tobytes(), geometry=g)
    assert result.decisions[0].valid
    assert result.decisions[0].n_bulk == 1
    assert result.decisions[0].multiplier_q16 == ALWAYS_MASKED_Q16
    assert result.masks.tolist() == [1]


def test_exact_threshold_equality_is_kept(geometry):
    rng = np.random.default_rng(301)
    raw = rng.integers(0, 256, geometry.block_samples * geometry.num_freq * geometry.num_streams, dtype=np.uint8)
    term = rng.integers(0, 256, geometry.k, dtype=np.uint8).tobytes()
    config = bundle(64)
    equal = expected_products(raw, config, term * 3, geometry=geometry)
    assert equal.decisions[0].valid and equal.decisions[0].multiplier_q16 == 65536
    assert equal.masks[0] == 0
    config["profiles"][0]["fine_calibration"]["cfar_multiplier_q16"] = 65535
    below = expected_products(raw, config, term * 3, geometry=geometry)
    assert below.masks[0] == 1
    assert below.decisions[0] == equal.decisions[0]


@pytest.mark.parametrize("field,value", [
    ("status", "pending"), ("decision_version", "other"), ("cfar_rank", True),
    ("cfar_rank", 256), ("anchor_bin", -1), ("cfar_multiplier_q16", 0),
    ("designated_half_width", 128), ("bulk_mask_words_hex", ["0"]),
])
def test_bound_calibration_configuration_refused(geometry, field, value):
    config = bundle(64)
    config["profiles"][0]["fine_calibration"][field] = value
    raw = np.full(geometry.block_samples * geometry.num_freq * geometry.num_streams, 0x99, dtype=np.uint8)
    with pytest.raises((ValueError, TypeError)):
        expected_products(raw, config, bytes([0x10]) * 192, geometry=geometry)


def test_weight_and_byte_geometry_refused(geometry):
    raw = np.full(geometry.block_samples * geometry.num_freq * geometry.num_streams, 0x99, dtype=np.uint8)
    with pytest.raises(ValueError, match="weight"):
        expected_products(raw, bundle(64), bytes(191), geometry=geometry)
    with pytest.raises(ValueError, match="uint8"):
        expected_products(raw[:-1], bundle(64), bytes(192), geometry=geometry)
    config = bundle(64)
    config["profiles"].append(config["profiles"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        expected_products(raw, config, bytes(192), geometry=geometry)
