# coding=utf-8
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pilot_proxy import kernel as kernel_module
from pilot_proxy.kernel import PilotProxyKernel, KernelFeatures


class _Function:
    def __init__(self, callback=lambda *args: None):
        self.callback = callback
        self.argtypes = None
        self.restype = object()

    def __call__(self, *args):
        return self.callback(*args)


def test_void_kernel_call_surfaces_last_error() -> None:
    error = b""

    def fail(*_args):
        nonlocal error
        error = b"PilotProxy API error: handle is null."

    lib = SimpleNamespace(
        PilotProxy_Compute_FusedFine_U64=_Function(fail),
        PilotProxy_LastError=_Function(lambda: error),
    )
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = lib
    wrapper._has_last_error = True
    wrapper._has_fused_fine_u64 = True

    with pytest.raises(RuntimeError, match="handle is null"):
        wrapper.compute_fused_fine_u64(None, 1, 2, 3)


_CORE_NAMES = {
    "PilotProxy_GetSpecs",
    "PilotProxy_GetVersion",
    "PilotProxy_LastError",
    "PilotProxy_Create",
    "PilotProxy_Create_Batch",
    "PilotProxy_Destroy",
    "PilotProxy_Compute_RowSums_I32",
    "PilotProxy_Compute_FusedFine_U64",
}


def _report_version(major: int, minor: int = 0, patch: int = 0):
    def report(c_major, c_minor, c_patch):
        c_major._obj.value, c_minor._obj.value, c_patch._obj.value = major, minor, patch

    return report


def _setup_current_wrapper(monkeypatch, *, major: int = 4) -> SimpleNamespace:
    lib = SimpleNamespace(**{name: _Function() for name in _CORE_NAMES})
    lib.PilotProxy_GetVersion = _Function(_report_version(major, 5 if major == 2 else 0))
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = lib
    monkeypatch.setattr(
        kernel_module,
        "_has_symbol",
        lambda _lib, name: name in _CORE_NAMES,
    )
    wrapper._setup_signatures()
    return lib


def test_setup_signatures_refuses_a_library_with_only_the_legacy_api(monkeypatch):
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = SimpleNamespace()
    monkeypatch.setattr(kernel_module, "_has_symbol", lambda _lib, name: False)
    with pytest.raises(RuntimeError, match="Rebuild libpilotproxy.so"):
        wrapper._setup_signatures()


@pytest.mark.parametrize("major", [2, 3, 5])
def test_setup_signatures_refuses_a_library_of_another_core_major(
    monkeypatch, major: int
) -> None:
    # Reject another ABI even if it exports the expected symbol names.
    with pytest.raises(RuntimeError, match=f"reports core {major}\\."):
        _setup_current_wrapper(monkeypatch, major=major)


def test_setup_signatures_marks_void_functions_as_void(monkeypatch) -> None:
    lib = _setup_current_wrapper(monkeypatch)
    for name in (
        "PilotProxy_GetSpecs",
        "PilotProxy_GetVersion",
        "PilotProxy_Destroy",
        "PilotProxy_Compute_RowSums_I32",
        "PilotProxy_Compute_FusedFine_U64",
    ):
        assert getattr(lib, name).restype is None


def test_setup_signatures_does_not_require_the_removed_coarse_entries(monkeypatch) -> None:
    """A kernel core 3.0.0 library without the coarse entries loads.

    Core 3.0.0 has no float diagnostic or uint64 power entries. The handle
    constructors take only the packed input and the geometry, with no float
    output pointer.
    """
    import ctypes

    lib = _setup_current_wrapper(monkeypatch)
    assert lib.PilotProxy_Create.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert lib.PilotProxy_Create_Batch.argtypes == [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ]
    for removed in (
        "compute_powers",
        "compute_powers_u64",
        "compute_diagnostic_float",
        "compute_numden_mask_rational_half",
        "compute_numden_mask_rational_half_checked",
    ):
        assert not hasattr(PilotProxyKernel, removed), removed


def _fused_mask_wrapper(callback=lambda *args: None) -> PilotProxyKernel:
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = SimpleNamespace(
        PilotProxy_Compute_FusedFineMask_U64=_Function(callback),
    )
    wrapper._has_fused_fine_mask_u64 = True
    wrapper._has_last_error = False
    return wrapper


def _support_wrapper(callback=lambda *args: None) -> PilotProxyKernel:
    wrapper = _fused_mask_wrapper()
    wrapper._has_fused_fine_mask_with_support_u64 = True
    wrapper._lib.PilotProxy_Compute_FusedFineMaskWithSupport_U64 = _Function(callback)
    return wrapper


def test_support_outputs_use_additive_entry_point_in_declared_order():
    calls = []
    wrapper = _support_wrapper(lambda *args: calls.append(args))
    wrapper.compute_fused_fine_mask_with_support_u64(
        None, 1, 62, 2, [0, 0, 0, 0], 4, 65536, 2, 3, 4, 5, 6, 7)
    assert wrapper.supports_fused_fine_mask_with_support()
    assert len(calls) == 1
    assert calls[0][7:] == (2, 3, 4, 5, 6, 7)


@pytest.mark.parametrize("pointers", [(0, 5), (4, 0), (-1, 5), (4, True)])
def test_support_outputs_require_real_nonzero_pointers(pointers):
    calls = []
    wrapper = _support_wrapper(lambda *args: calls.append(args))
    with pytest.raises((ValueError, TypeError), match="ptr"):
        wrapper.compute_fused_fine_mask_with_support_u64(
            None, 1, 62, 2, [0, 0, 0, 0], 4, 65536, 2, 3, *pointers)
    assert calls == []


def test_legacy_library_cannot_claim_rank_support():
    wrapper = _fused_mask_wrapper()
    assert wrapper.supports_fused_fine_mask()
    assert not wrapper.supports_fused_fine_mask_with_support()
    with pytest.raises(RuntimeError, match="FusedFineMaskWithSupport"):
        wrapper.compute_fused_fine_mask_with_support_u64(
            None, 1, 62, 2, [0, 0, 0, 0], 4, 65536, 2, 3, 4, 5)


def test_support_entry_keeps_exact_threshold_validation():
    wrapper = _support_wrapper()
    for multiplier in (0, 1 << 64):
        with pytest.raises(ValueError, match="multiplier_q16"):
            wrapper.compute_fused_fine_mask_with_support_u64(
                None, 1, 62, 2, [0, 0, 0, 0], 4, multiplier, 2, 3, 4, 5)


@pytest.mark.parametrize("invalid_word", (-1, 1 << 64))
def test_fused_mask_wrapper_rejects_mask_word_outside_uint64(
    invalid_word: int,
) -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))
    words = [0, 0, 0, 0]
    words[2] = invalid_word

    with pytest.raises(ValueError, match=r"bulk_mask_words\[2\].*uint64"):
        wrapper.compute_fused_fine_mask_u64(
            None, 1, 62, 2, words, 4, 65_536, 2, 3
        )

    assert calls == []


@pytest.mark.parametrize("invalid_multiplier", (-1, 0, 1 << 64))
def test_fused_mask_wrapper_rejects_multiplier_outside_positive_uint64(
    invalid_multiplier: int,
) -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))

    with pytest.raises(ValueError, match="multiplier_q16.*uint64"):
        wrapper.compute_fused_fine_mask_u64(
            None, 1, 62, 2, [0, 0, 0, 0], 4, invalid_multiplier, 2, 3
        )

    assert calls == []


def test_fused_mask_wrapper_accepts_uint64_boundaries_without_wrapping() -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))

    wrapper.compute_fused_fine_mask_u64(
        None,
        1,
        62,
        2,
        [0, (1 << 64) - 1, 0, (1 << 64) - 1],
        4,
        (1 << 64) - 1,
        2,
        3,
    )

    assert len(calls) == 1
    assert list(calls[0][4]) == [0, (1 << 64) - 1, 0, (1 << 64) - 1]
    assert calls[0][6] == (1 << 64) - 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("anchor_bin", -1, "anchor_bin.*range"),
        ("anchor_bin", 256, "anchor_bin.*range"),
        ("anchor_bin", 1 << 32, "anchor_bin.*range"),
        ("designated_half_width", -1, "designated_half_width.*range"),
        ("designated_half_width", 128, "designated_half_width.*range"),
        ("cfar_rank", -1, "cfar_rank.*range"),
        ("cfar_rank", 256, "cfar_rank.*range"),
        ("cfar_rank", 1 << 31, "cfar_rank.*range"),
    ],
)
def test_fused_mask_wrapper_rejects_c_int_semantic_range_errors(
    field: str,
    value: int,
    match: str,
) -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))
    arguments = {
        "anchor_bin": 62,
        "designated_half_width": 2,
        "cfar_rank": 4,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        wrapper.compute_fused_fine_mask_u64(
            None,
            1,
            arguments["anchor_bin"],
            arguments["designated_half_width"],
            [1, 0, 0, 0],
            arguments["cfar_rank"],
            65_536,
            2,
            3,
        )

    assert calls == []


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 1.0, "1"])
@pytest.mark.parametrize(
    "field", ["anchor_bin", "designated_half_width", "cfar_rank"]
)
def test_fused_mask_wrapper_requires_exact_integer_c_int_fields(
    field: str,
    invalid: object,
) -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))
    arguments = {
        "anchor_bin": 62,
        "designated_half_width": 2,
        "cfar_rank": 0,
    }
    arguments[field] = invalid

    with pytest.raises(TypeError, match=field):
        wrapper.compute_fused_fine_mask_u64(
            None,
            1,
            arguments["anchor_bin"],
            arguments["designated_half_width"],
            [1, 0, 0, 0],
            arguments["cfar_rank"],
            65_536,
            2,
            3,
        )

    assert calls == []


def test_fused_mask_wrapper_accepts_c_int_semantic_boundaries() -> None:
    calls = []
    wrapper = _fused_mask_wrapper(lambda *args: calls.append(args))

    wrapper.compute_fused_fine_mask_u64(
        None,
        1,
        255,
        127,
        [1, 0, 0, 0],
        255,
        65_536,
        2,
        3,
    )

    assert len(calls) == 1
    assert calls[0][2:4] == (255, 127)
    assert calls[0][5] == 255


def _create_wrapper(callback=lambda *_args: 123) -> PilotProxyKernel:
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = SimpleNamespace(
        PilotProxy_Create=_Function(callback),
        PilotProxy_Create_Batch=_Function(callback),
    )
    wrapper._has_batch_create = True
    wrapper._has_last_error = False
    return wrapper


class _DeviceBuffer:
    data = SimpleNamespace(ptr=1)


@pytest.mark.parametrize(
    "invalid",
    [0, -1, 1 << 31, 1 << 32, True, np.bool_(True), 1.0, "128"],
)
@pytest.mark.parametrize(
    "method",
    ["create_handle", "create_detector_matrix_handle", "create_raw"],
)
def test_create_entry_points_require_positive_exact_c_int_rows(
    method: str,
    invalid: object,
) -> None:
    calls = []
    wrapper = _create_wrapper(lambda *args: calls.append(args) or 123)

    with pytest.raises((TypeError, ValueError), match="detector_rows_per_block"):
        if method == "create_raw":
            wrapper.create_raw(invalid, 1)
        else:
            getattr(wrapper, method)(invalid, _DeviceBuffer())

    assert calls == []


@pytest.mark.parametrize("field", ["detector_rows_per_block", "batch"])
@pytest.mark.parametrize(
    "invalid",
    [0, -1, 1 << 31, 1 << 32, True, np.bool_(True), 1.0, "1"],
)
@pytest.mark.parametrize(
    "method",
    ["create_batch_handle", "create_detector_matrix_batch_handle", "create_raw_batch"],
)
def test_batch_create_entry_points_require_positive_exact_c_int_geometry(
    method: str,
    field: str,
    invalid: object,
) -> None:
    calls = []
    wrapper = _create_wrapper(lambda *args: calls.append(args) or 123)
    rows = invalid if field == "detector_rows_per_block" else 128
    batch = invalid if field == "batch" else 2

    with pytest.raises((TypeError, ValueError), match=field):
        if method == "create_raw_batch":
            wrapper.create_raw_batch(rows, batch, 1)
        else:
            getattr(wrapper, method)(rows, batch, _DeviceBuffer())

    assert calls == []


def test_create_entry_points_preserve_c_int_boundaries() -> None:
    calls = []
    wrapper = _create_wrapper(lambda *args: calls.append(args) or 123)

    assert wrapper.create_raw((1 << 31) - 1, 1) == 123
    assert wrapper.create_raw_batch((1 << 31) - 1, (1 << 31) - 1, 1) == 123

    assert calls == [
        (1, (1 << 31) - 1),
        (1, (1 << 31) - 1, (1 << 31) - 1),
    ]


def test_context_manager_handles_pass_input_and_geometry_only() -> None:
    calls = []
    wrapper = _create_wrapper(lambda *args: calls.append(args) or 123)
    wrapper._lib.PilotProxy_Destroy = _Function()
    with wrapper.create_handle(128, _DeviceBuffer()) as handle:
        assert handle == 123
    with wrapper.create_batch_handle(256, 2, _DeviceBuffer()) as handle:
        assert handle == 123
    assert calls == [(1, 128), (1, 256, 2)]


def _fine_powers_wrapper(callback=lambda *args: None) -> PilotProxyKernel:
    wrapper = object.__new__(PilotProxyKernel)
    wrapper._lib = SimpleNamespace(
        PilotProxy_Compute_FinePowers_U64=_Function(callback),
    )
    wrapper._has_fine_powers_u64 = True
    wrapper._has_last_error = False
    wrapper.specs = SimpleNamespace(num_weight_terms=3)
    wrapper.features = KernelFeatures(
        use_dp4a=True,
        use_uint64_power_accumulation=True,
        block_threads=64,
        grid_max_blocks=4,
    )
    return wrapper


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("num_streams", 0),
        ("num_streams", 257),
        ("num_streams", 1 << 32),
        ("num_streams", 1.0),
        ("num_streams", True),
        ("windows_per_stream", 127),
        ("windows_per_stream", 129),
        ("windows_per_stream", 128.0),
        ("batch", 0),
        ("batch", 1 << 32),
        ("batch", 1.0),
        ("batch", True),
    ],
)
def test_fine_power_entry_point_rejects_invalid_geometry_before_call(
    field: str,
    invalid: object,
) -> None:
    calls = []
    wrapper = _fine_powers_wrapper(lambda *args: calls.append(args))
    arguments = {
        "num_streams": 1,
        "windows_per_stream": 128,
        "batch": 1,
    }
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError), match=field):
        wrapper.compute_fine_powers_u64(1, **arguments, fine_powers_ptr=2)

    assert calls == []


def test_fine_power_entry_point_preserves_launch_capacity_boundary() -> None:
    calls = []
    wrapper = _fine_powers_wrapper(lambda *args: calls.append(args))

    wrapper.compute_fine_powers_u64(1, 256, 128, (1 << 31) - 1, 2)

    assert calls == [(1, 3, 256, 128, (1 << 31) - 1, 2)]


