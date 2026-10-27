# coding=utf-8
"""Check that the CUDA transform and C reference index the same twiddle table.

``cuda/fxfft_ref.c`` is compared bit for bit with the Python reference at
every supported size in ``test_fxfft_ref_family.py``. The checks here require
``cuda/pilotproxy_detector.cu`` to use the same index and stage-bound
expressions against that shared table.

These source checks catch an edit made on only one side, even on a host
without CUDA. Golden-vector runs on a GPU still test the device arithmetic.
"""

from __future__ import annotations

import pathlib
import re

import pytest

CUDA_DIR = pathlib.Path(__file__).resolve().parents[2] / "cuda"
KERNEL = (CUDA_DIR / "pilotproxy_detector.cu").read_text()
REF = (CUDA_DIR / "fxfft_ref.c").read_text()
CONFIG = (CUDA_DIR / "config.h").read_text()
TABLE = (CUDA_DIR / "fxfft_master_twiddle.h").read_text()


def test_both_read_the_shared_master_table():
    assert '#include "fxfft_master_twiddle.h"' in KERNEL
    assert '#include "fxfft_master_twiddle.h"' in REF
    # The kernel uses only the initializer. A host array in a device translation
    # unit is never read and would raise a warning.
    assert "#define FX_TW_MASTER_NO_HOST_ARRAY" in KERNEL
    assert "FX_TW_MASTER_INIT" in KERNEL


def test_twiddle_index_expression_matches_the_reference():
    """Both must fold the decimation stride into the shift by FX_MASTER_LOG2."""
    # match the array access, since the header comments give the same index in prose
    kernel_idx = re.findall(
        r"pilotproxy_fxfft_twiddle_q15\[t << \(\(unsigned\)FX_MASTER_LOG2 - stage\)\]", KERNEL
    )
    ref_idx = re.findall(r"FX_TW_MASTER\[t << \(FX_MASTER_LOG2 - s\)\]", REF)
    assert len(kernel_idx) == 2, f"expected 2 kernel twiddle reads, found {len(kernel_idx)}"
    assert len(ref_idx) == 1, f"expected 1 reference twiddle read, found {len(ref_idx)}"


def test_no_hardcoded_transform_length_remains_in_the_kernel():
    """The kernel has no fixed eight-stage bound or shift (`stage <= 8u`, `t << (8u - stage)`)."""
    assert "stage <= 8u" not in KERNEL
    assert "8u - stage" not in KERNEL
    assert "bitrev8" not in KERNEL, "bit reversal must be parameterized on PILOTPROXY_FINE_LOG2"


def test_stage_bound_is_derived():
    bounds = re.findall(
        r"for \(unsigned stage = 1; stage <= \(unsigned\)PILOTPROXY_FINE_LOG2; \+\+stage\)",
        KERNEL,
    )
    assert len(bounds) == 2, f"expected 2 parameterized stage loops, found {len(bounds)}"


def test_config_asserts_the_stage_count_relation():
    assert "(1 << PILOTPROXY_FINE_LOG2) == PILOTPROXY_FINE_NUM_BINS" in CONFIG


def test_table_is_large_enough_for_the_configured_transform():
    master_n = int(re.search(r"#define FX_MASTER_N (\d+)", TABLE).group(1))
    bins = int(re.search(r"#define PILOTPROXY_FINE_WINDOWS_PER_STREAM (\d+)", CONFIG).group(1))
    pad = int(re.search(r"#define PILOTPROXY_FINE_PAD_FACTOR (\d+)", CONFIG).group(1))
    assert pad * bins <= master_n, (
        f"fine transform length {pad * bins} exceeds the master table {master_n}"
    )


@pytest.mark.parametrize("name", ["FX_MASTER_N", "FX_MASTER_LOG2", "FX_MASTER_HALF"])
def test_generated_header_exports_what_consumers_use(name):
    assert f"#define {name} " in TABLE
