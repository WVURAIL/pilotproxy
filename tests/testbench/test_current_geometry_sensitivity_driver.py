# coding=utf-8
from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np

from pilot_proxy.fine_reduction import independent_bin_mask


DRIVER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "current_geometry_sensitivity.py")
)


def test_production_defaults_to_attainable_sufficient_statistic_design() -> None:
    parser = DRIVER["build_parser"]()
    args = parser.parse_args(
        [
            "--mode",
            "production",
            "--stage",
            "null",
            "--input-iq",
            "unused-in-default-test.cfile",
        ]
    )
    DRIVER["_mode_defaults"](args)

    assert args.simulation_backend == "sufficient-statistic"
    assert args.num_streams == 2048
    assert args.sufficient_pool_streams == 256
    assert args.physical_channel == list(range(14, 37))
    assert args.audit_physical_channel == [14, 25, 36]
    assert args.run_audit_physical_channel == [14, 25, 36]
    assert args.audit_offset_fine_bins == [0.0, 0.5]
    assert args.audit_snr_db == [-54.0, -50.0, -46.0, -42.0]
    assert args.run_audit_snr_db == [-54.0, -50.0, -46.0, -42.0]


def test_multiplier_aggregate_is_deterministic_and_preserves_constant_powers() -> None:
    stages = DRIVER["FLOAT_RESPONSE_STAGES"]
    pool_streams = 4
    base = np.ones((pool_streams, 3, 256), dtype=np.float64)
    base[:, 0, :] = 2.0
    powers = {stage: base.copy() for stage in stages}
    profile = SimpleNamespace(
        physical_channel=14,
        offset_fine_bins=0.5,
        designated=np.asarray([254, 255, 0, 1, 2]),
        bulk_mask=independent_bin_mask(
            256,
            designated_bins=np.asarray([254, 255, 0, 1, 2]),
        ),
        cfar_rank=60,
    )
    args = SimpleNamespace(
        sufficient_pool_streams=pool_streams,
        num_streams=2048,
        trials=3,
        trial_start=7,
        seed=20260820,
    )

    first = DRIVER["_aggregate_sufficient_pool"](
        args,
        profile,
        purpose="h1",
        powers_by_stage=powers,
        multiplier_q16=65536,
    )
    second = DRIVER["_aggregate_sufficient_pool"](
        args,
        profile,
        purpose="h1",
        powers_by_stage=powers,
        multiplier_q16=65536,
    )

    assert first["trial_seed"] == second["trial_seed"]
    assert first["trial_index"] == [7, 8, 9]
    for stage in stages:
        np.testing.assert_array_equal(first["ratios"][stage], [1.0, 1.0, 1.0])
        assert first["negative_fraction_by_stage"][stage] == 0.0
    assert first["cpu_q16"] == [0, 0, 0]
