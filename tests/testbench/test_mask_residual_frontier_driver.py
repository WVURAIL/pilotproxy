# coding=utf-8
from __future__ import annotations

import math
from pathlib import Path
import runpy

import numpy as np
import pytest

from pilot_proxy.dtv_units import pilot_excess_db_to_data_shelf_snr_db
from pilot_proxy.testbench.mask_frontier import ALWAYS_MASKED_Q16


DRIVER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "mask_residual_frontier.py")
)


def _args(**overrides):
    argv = ["--stage", "report", "--output-dir", "unused"]
    for name, value in overrides.items():
        flag = "--" + name.replace("_", "-")
        argv.append(flag)
        argv.extend(str(item) for item in np.atleast_1d(value))
    return DRIVER["build_parser"]().parse_args(argv)


def test_production_defaults_are_the_production_geometry() -> None:
    args = _args()
    assert args.num_streams == 2048
    assert args.allow_reduced_geometry is False
    assert args.shelf_snr_db == [-10.0, -44.0, -55.0]
    assert args.duty_cycle == [0.10, 0.25, 0.50]
    assert args.rho == [12, 31, 62, 93, 112]


def test_a_reduced_geometry_is_refused_unless_it_is_declared() -> None:
    with pytest.raises(SystemExit):
        DRIVER["_prepare"](_args(num_streams=256))


def test_the_shelf_offset_is_the_declared_geometry_transfer() -> None:
    args = _args()
    offset = DRIVER["_shelf_offset_db"](args)
    assert offset == pytest.approx(
        float(
            pilot_excess_db_to_data_shelf_snr_db(
                0.0,
                pilot_below_data_db=args.pilot_below_data_db,
                bin_enbw_hz=args.bin_enbw_hz,
                dtv_bandwidth_hz=args.dtv_bandwidth_hz,
                pilot_capture_efficiency=args.pilot_capture_efficiency,
            )
        )
    )


def test_the_always_masked_sentinel_survives_the_stored_uint64_column() -> None:
    stored = np.asarray([[0, 5], [7, 0]], dtype=np.uint64)
    assert DRIVER["_required_column"](stored, 1) == [ALWAYS_MASKED_Q16, 7]
    assert DRIVER["_required_column"](stored, 2) == [5, ALWAYS_MASKED_Q16]


def _bank(frames: int, *, injected_linear: float, shelf: float):
    return {
        "meta": {
            "injected_linear": injected_linear,
            "injected_shelf_snr_db": shelf,
        },
        "required": np.arange(frames, dtype=np.uint64).reshape(frames, 1) + 1,
        "shelf": np.full(frames, shelf, dtype=np.float64),
        "fine": np.zeros((frames, 3, 256), dtype=np.float64),
    }


def test_a_duty_cycle_is_composed_from_the_labelled_banks_in_order() -> None:
    off = _bank(80, injected_linear=0.0, shelf=math.nan)
    on = _bank(40, injected_linear=0.25, shelf=-6.0)
    population = DRIVER["_duty_population"](
        _args(), off, on, 0.25, frames=40
    )
    assert population["on_frames"] == 10
    assert population["off_frames"] == 30
    assert population["frames"] == 40
    assert population["injected"][:30].tolist() == [0.0] * 30
    assert population["injected"][30:].tolist() == [0.25] * 10
    # A larger duty cycle contains the smaller one's on frames.
    wider = DRIVER["_duty_population"](_args(), off, on, 0.50, frames=40)
    assert np.array_equal(
        population["required"][30:], wider["required"][20:30]
    )


def test_a_duty_cycle_the_banks_cannot_supply_is_refused() -> None:
    off = _bank(10, injected_linear=0.0, shelf=math.nan)
    on = _bank(10, injected_linear=1.0, shelf=0.0)
    with pytest.raises(SystemExit):
        DRIVER["_duty_population"](_args(), off, on, 0.9, frames=40)


def test_the_unmasked_reference_separates_the_transfer_from_the_masking() -> None:
    off = _bank(40, injected_linear=0.0, shelf=math.nan)
    on = _bank(40, injected_linear=0.1, shelf=-10.0)
    population = DRIVER["_duty_population"](_args(), off, on, 0.5, frames=40)
    reference = DRIVER["_unmasked_reference"](population, floor_linear=1e-6)
    assert reference["transfer_closure_on_frames"] == pytest.approx(1.0)
    assert reference["transfer_closure_db"] == pytest.approx(0.0, abs=1e-12)
    assert reference["r_injected_unmasked"] == pytest.approx(0.05)
    # The off frames are booked at the floor, so the claim exceeds the truth.
    assert reference["r_sys_unmasked"] == pytest.approx(0.5 * 0.1 + 0.5 * 1e-6)
    assert reference["claim_over_truth_unmasked"] > 1.0


def test_the_three_spectrum_normalizations_keep_their_denominators_apart() -> None:
    frames = 8
    fine = np.zeros((frames, 3, 256), dtype=np.float64)
    fine[:, 0, 10] = 1.0
    fine[:4, 0, 10] = 3.0
    fine[:, 1, :] = 1.0
    fine[:, 2, :] = 1.0
    population = {"fine": fine}
    required = [1] * 4 + [10] * 4
    spectra = DRIVER["_spectra"](population, required=required, eta_q16=5)
    assert spectra["kept"] == 4
    assert spectra["all_mean"][10] == pytest.approx(2.0)
    assert spectra["keep_given_keep"][10] == pytest.approx(3.0)
    assert spectra["keep_contribution"][10] == pytest.approx(1.5)
    assert spectra["removed"][10] == pytest.approx(0.5)
