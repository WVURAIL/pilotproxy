# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

# noinspection PyProtectedMember
from pilot_proxy.detect import (
    DEFAULT_DETECTOR_BLOCKS_FILENAME,
    DEFAULT_DETECTOR_INPUT_DIR,
    DEFAULT_DETECTOR_MATRIX_FILENAME,
    KernelLike,
    _load_detector_input,
    _resolve_layout_metadata,
    _resolve_detector_input_path,
    _validate_kernel_inputs,
)

LOCKED_DETECTOR_WINDOW_SAMPLES = 128
LOCKED_WEIGHT_TERMS = 3
VALID_DETECTOR_ROWS = 4
BATCH_SIZE = 2
WRONG_DETECTOR_WINDOW_SAMPLES = 64
WRONG_WEIGHT_TERMS = 2
MISSING_INPUT_NAME = "missing_detector_matrix.npy"
TWO_INPUT_STREAMS = 2


def _fake_kernel() -> KernelLike:
    return cast(
        KernelLike,
        cast(
            object,
            SimpleNamespace(
                specs=SimpleNamespace(
                    K=LOCKED_DETECTOR_WINDOW_SAMPLES,
                    N=LOCKED_WEIGHT_TERMS,
                )
            ),
        ),
    )


def test_validate_kernel_inputs_accepts_locked_int4_shapes() -> None:
    _validate_kernel_inputs(
        packed=np.zeros(
            (VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
            dtype=np.int8,
        ),
        weights=np.zeros(
            (LOCKED_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
            dtype=np.int8,
        ),
        kernel=_fake_kernel(),
    )
    _validate_kernel_inputs(
        packed=np.zeros(
            (BATCH_SIZE, VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
            dtype=np.int8,
        ),
        weights=np.zeros(
            (LOCKED_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
            dtype=np.int8,
        ),
        kernel=_fake_kernel(),
    )


def test_validate_kernel_inputs_rejects_wrong_packed_dtype() -> None:
    with pytest.raises(ValueError, match="must be dtype int8"):
        _validate_kernel_inputs(
            packed=np.zeros(
                (VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int16,
            ),
            weights=np.zeros(
                (LOCKED_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            kernel=_fake_kernel(),
        )


def test_validate_kernel_inputs_rejects_wrong_detector_window_length() -> None:
    with pytest.raises(ValueError, match="wrong detector-window length"):
        _validate_kernel_inputs(
            packed=np.zeros(
                (VALID_DETECTOR_ROWS, WRONG_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            weights=np.zeros(
                (LOCKED_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            kernel=_fake_kernel(),
        )


def test_validate_kernel_inputs_rejects_wrong_weights_shape() -> None:
    with pytest.raises(ValueError, match="weights have wrong shape"):
        _validate_kernel_inputs(
            packed=np.zeros(
                (VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            weights=np.zeros(
                (WRONG_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            kernel=_fake_kernel(),
        )


def test_validate_kernel_inputs_rejects_wrong_weights_dtype() -> None:
    with pytest.raises(ValueError, match="weights must be dtype int8"):
        _validate_kernel_inputs(
            packed=np.zeros(
                (VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int8,
            ),
            weights=np.zeros(
                (LOCKED_WEIGHT_TERMS, LOCKED_DETECTOR_WINDOW_SAMPLES),
                dtype=np.int16,
            ),
            kernel=_fake_kernel(),
        )


def test_load_detector_input_missing_file_reports_quantize_step(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        _load_detector_input(tmp_path / MISSING_INPUT_NAME)

    message = str(exc.value)
    assert "Input detector matrix does not exist" in message
    assert "Generate packed detector input before running detect" in message


def test_resolve_detector_input_uses_batched_default_when_matrix_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_dir = DEFAULT_DETECTOR_INPUT_DIR
    input_dir.mkdir(parents=True)
    blocks = input_dir / DEFAULT_DETECTOR_BLOCKS_FILENAME
    np.save(
        blocks,
        np.zeros(
            (BATCH_SIZE, VALID_DETECTOR_ROWS, LOCKED_DETECTOR_WINDOW_SAMPLES),
            dtype=np.int8,
        ),
    )

    resolved = _resolve_detector_input_path(
        input_dir / DEFAULT_DETECTOR_MATRIX_FILENAME
    )
    packed, batch, rows = _load_detector_input(
        input_dir / DEFAULT_DETECTOR_MATRIX_FILENAME
    )

    assert resolved == blocks
    assert batch == BATCH_SIZE
    assert rows == VALID_DETECTOR_ROWS
    assert packed.shape == (
        BATCH_SIZE,
        VALID_DETECTOR_ROWS,
        LOCKED_DETECTOR_WINDOW_SAMPLES,
    )


def test_resolve_layout_metadata_records_input_streams_and_row_rule() -> None:
    layout = _resolve_layout_metadata(
        rows=VALID_DETECTOR_ROWS,
        detector_window_samples=LOCKED_DETECTOR_WINDOW_SAMPLES,
        frame_size_samples=None,
        num_input_streams=TWO_INPUT_STREAMS,
    )

    assert layout["frame_size_samples"] == (
        VALID_DETECTOR_ROWS // TWO_INPUT_STREAMS
    ) * LOCKED_DETECTOR_WINDOW_SAMPLES
    assert layout["num_input_streams"] == TWO_INPUT_STREAMS
    assert layout["windows_per_stream"] == VALID_DETECTOR_ROWS // TWO_INPUT_STREAMS
    assert layout["detector_rows_per_frame"] == VALID_DETECTOR_ROWS
    assert layout["combine_mode"] == "all_rows_summed_before_ratio"
