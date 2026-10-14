# coding=utf-8
"""Source-level consistency between the CUDA transform and the C reference.

The device transform in ``cuda/f_statistic.cu`` and the C reference in
``cuda/fxfft_ref.c`` must index the shared master twiddle table the same way.
The C reference is proven bit-identical to the Python reference at every
supported size (``test_fxfft_ref_family.py``), so if the CUDA source computes
its twiddle index and stage bound with the same expressions against the same
table, the index arithmetic is covered by that proof.

This is a cheap standing guard rather than a substitute for the golden-vector run: it
catches one side being edited without the other, which is the failure mode a
GPU-only gate would only reveal on a machine with a device attached.
"""

from __future__ import annotations

import pathlib
import re

import pytest

CUDA_DIR = pathlib.Path(__file__).resolve().parents[2] / "cuda"
KERNEL = (CUDA_DIR / "f_statistic.cu").read_text()
REF = (CUDA_DIR / "fxfft_ref.c").read_text()
CONFIG = (CUDA_DIR / "config.h").read_text()
TABLE = (CUDA_DIR / "fxfft_master_twiddle.h").read_text()


def test_both_read_the_shared_master_table():
    assert '#include "fxfft_master_twiddle.h"' in KERNEL
    assert '#include "fxfft_master_twiddle.h"' in REF
    # The kernel wants the initializer only; a host array in a device TU is dead
    # weight and would warn.
    assert "#define FX_TW_MASTER_NO_HOST_ARRAY" in KERNEL
    assert "FX_TW_MASTER_INIT" in KERNEL


def test_twiddle_index_expression_matches_the_reference():
    """Both must fold the decimation stride into the shift by FX_MASTER_LOG2."""
    # Match the array access itself rather than the prose in the header comments.
    kernel_idx = re.findall(
        r"fstat_fxfft_twiddle_q15\[t << \(\(unsigned\)FX_MASTER_LOG2 - stage\)\]", KERNEL
    )
    ref_idx = re.findall(r"FX_TW_MASTER\[t << \(FX_MASTER_LOG2 - s\)\]", REF)
    assert len(kernel_idx) == 2, f"expected 2 kernel twiddle reads, found {len(kernel_idx)}"
    assert len(ref_idx) == 1, f"expected 1 reference twiddle read, found {len(ref_idx)}"


def test_no_hardcoded_transform_length_remains_in_the_kernel():
    """The old form was `stage <= 8u` and `t << (8u - stage)`."""
    assert "stage <= 8u" not in KERNEL
    assert "8u - stage" not in KERNEL
    assert "bitrev8" not in KERNEL, "bit reversal must be parameterized on FSTAT_FINE_LOG2"


def test_stage_bound_is_derived():
    bounds = re.findall(
        r"for \(unsigned stage = 1; stage <= \(unsigned\)FSTAT_FINE_LOG2; \+\+stage\)",
        KERNEL,
    )
    assert len(bounds) == 2, f"expected 2 parameterized stage loops, found {len(bounds)}"


def test_config_asserts_the_stage_count_relation():
    assert "(1 << FSTAT_FINE_LOG2) == FSTAT_FINE_NUM_BINS" in CONFIG


def test_table_is_large_enough_for_the_configured_transform():
    master_n = int(re.search(r"#define FX_MASTER_N (\d+)", TABLE).group(1))
    bins = int(re.search(r"#define FSTAT_FINE_WINDOWS_PER_STREAM (\d+)", CONFIG).group(1))
    pad = int(re.search(r"#define FSTAT_FINE_PAD_FACTOR (\d+)", CONFIG).group(1))
    assert pad * bins <= master_n, (
        f"fine transform length {pad * bins} exceeds the master table {master_n}"
    )


@pytest.mark.parametrize("name", ["FX_MASTER_N", "FX_MASTER_LOG2", "FX_MASTER_HALF"])
def test_generated_header_exports_what_consumers_use(name):
    assert f"#define {name} " in TABLE
