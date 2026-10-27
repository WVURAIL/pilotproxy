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


# --stage rederive on frames stored before kernel core 3.0.0

_META = {
    "schema_version": "pilotproxy_mask_residual_frontier_v1",
    "population": "on_m55p0",
    "target_norm_sq": 6338,
    "reference_norm_sum_sq": 12650,
    "shelf_offset_db": -21.6,
    "injected_linear": 1e-6,
}


def _stored_study(tmp_path: Path, *, frames: int = 6) -> Path:
    """A stored v1 study whose excess was formed from row-domain marginals."""
    import json

    from pilot_proxy.testbench.mask_frontier import (
        coarse_normalized_excess,
        shelf_db_from_excess,
    )

    source = tmp_path / "stored"
    (source / "frames").mkdir(parents=True)
    (source / "signal_cache").mkdir()
    (source / "signal_cache" / "cache.npz").write_bytes(b"signal cache bytes")
    config = {
        "schema_version": "pilotproxy_mask_residual_frontier_v1",
        "geometry": {"bulk_size": 124},
        "software": {"numpy": "2.4.2"},
    }
    config["config_sha256"] = DRIVER["_config_digest"](config)
    (source / "study_config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(5)
    fine = rng.integers(0, 1 << 40, size=(frames, 3, 256), dtype=np.uint64)
    # row-domain powers of the earlier core, offset by 3 from the fine sums / 256
    marginals = (fine.sum(axis=2, dtype=np.uint64) // np.uint64(256)) + np.uint64(3)
    meta = dict(_META, config_sha256=config["config_sha256"])
    excess = np.asarray([
        coarse_normalized_excess(*map(int, row), target_norm_sq=6338,
                                 reference_norm_sum_sq=12650)
        for row in marginals
    ])
    shelf = np.asarray([shelf_db_from_excess(v, offset_db=-21.6) for v in excess])
    np.savez_compressed(
        source / "frames" / "on_m55p0.npz",
        required_multiplier_q16=np.arange(frames * 4, dtype=np.uint64).reshape(frames, 4),
        fine_power_u64=fine,
        coarse_marginals_u64=marginals,
        normalized_excess=excess,
        shelf_estimate_db=shelf,
        clip_fraction=np.zeros(frames),
        frame_seed=np.arange(frames, dtype=np.uint64),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return source


def test_rederive_writes_fine_sum_frame_powers_and_their_excess(tmp_path) -> None:
    import json

    from pilot_proxy.testbench.mask_frontier import (
        MASK_FRONTIER_SCHEMA,
        coarse_normalized_excess,
        shelf_db_from_excess,
    )

    source = _stored_study(tmp_path)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "rederived"
    assert DRIVER["main"](
        ["--stage", "rederive", "--source-dir", str(source), "--output-dir", str(out)]
    ) == 0
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    old = np.load(source / "frames" / "on_m55p0.npz")
    new = np.load(out / "frames" / "on_m55p0.npz")
    assert "coarse_marginals_u64" not in new.files
    fine = old["fine_power_u64"]
    exact = [[sum(int(v) for v in fine[n, t]) for t in range(3)] for n in range(len(fine))]
    assert new["frame_powers_u64"].dtype == np.uint64
    assert new["frame_powers_u64"].tolist() == exact
    expected = [
        coarse_normalized_excess(*row, target_norm_sq=6338, reference_norm_sum_sq=12650)
        for row in exact
    ]
    assert new["normalized_excess"].tolist() == expected
    assert new["shelf_estimate_db"].tolist() == pytest.approx(
        [shelf_db_from_excess(v, offset_db=-21.6) for v in expected], nan_ok=True
    )
    assert not np.array_equal(new["normalized_excess"], old["normalized_excess"])
    for name in ("required_multiplier_q16", "fine_power_u64", "clip_fraction", "frame_seed"):
        np.testing.assert_array_equal(new[name], old[name])
    config = json.loads((out / "study_config.json").read_text())
    stored = json.loads((source / "study_config.json").read_text())
    assert config["schema_version"] == MASK_FRONTIER_SCHEMA
    assert config["rederived_from"]["config_sha256"] == stored["config_sha256"]
    assert config["config_sha256"] == DRIVER["_config_digest"](
        {k: v for k, v in config.items() if k != "config_sha256"}
    )
    meta = json.loads(str(new["meta_json"].item()))
    assert meta["config_sha256"] == config["config_sha256"]
    assert meta["schema_version"] == MASK_FRONTIER_SCHEMA
    assert meta["rederived"]["source_frame_powers"] == "coarse_marginals_u64"
    assert (out / "signal_cache" / "cache.npz").read_bytes() == b"signal cache bytes"
    receipt = json.loads((out / "rederive.json").read_text())
    (row,) = receipt["shards"]
    assert row["stored_excess_and_shelf_reproduced_from_stored_frame_powers"] is True
    assert row["frames"] == 6
    # A second pass into the same output is refused, so nothing is overwritten.
    with pytest.raises(SystemExit):
        DRIVER["main"](
            ["--stage", "rederive", "--source-dir", str(source), "--output-dir", str(out)]
        )


def test_rederive_refuses_a_record_its_own_powers_do_not_reproduce(tmp_path) -> None:
    source = _stored_study(tmp_path)
    path = source / "frames" / "on_m55p0.npz"
    with np.load(path) as archive:
        members = {name: archive[name] for name in archive.files}
    members["normalized_excess"] = members["normalized_excess"] + 1e-12
    np.savez_compressed(path, **members)
    out = tmp_path / "rederived"
    with pytest.raises(SystemExit, match="not reproduced"):
        DRIVER["main"](
            ["--stage", "rederive", "--source-dir", str(source), "--output-dir", str(out)]
        )
    assert not (out / "frames").exists() or not any((out / "frames").iterdir())


def test_a_stored_study_is_the_same_study_on_another_machine() -> None:
    stored = {
        "schema_version": "v",
        "geometry": {"bulk_size": 124},
        "calibration": {"shelf_offset_db": -21.6},
        "campaign": {"shelf_snr_db": [-10.0, -44.0, -50.0, -55.0]},
        "policy": {"q16_scale": 65536},
        "inputs": {"input_iq": "/old/iq.cfile", "input_iq_sha256": "a",
                   "waveform_audit_sha256": "b", "weights_path": "/old/w.bin",
                   "weights_sha256": "c", "iq_sample_rate_hz": 1.0},
        "software": {"numpy": "2.4.2", "platform": "old"},
    }
    moved = {
        **stored,
        "inputs": {**stored["inputs"], "input_iq": "/new/iq.cfile", "weights_path": "/new/w.bin"},
        "software": {"numpy": "2.5.2", "platform": "new"},
    }
    identity = DRIVER["_study_identity"]
    assert identity(moved) == identity(stored)
    changed = {**moved, "campaign": {"shelf_snr_db": [-10.0, -44.0, -55.0]}}
    assert identity(changed) != identity(stored)
    other_bytes = {**moved, "inputs": {**moved["inputs"], "weights_sha256": "d"}}
    assert identity(other_bytes) != identity(stored)
