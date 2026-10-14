# coding=utf-8
"""End-to-end: the analyzer writes the exact fine terms into the product.

Complements ``tests/core/test_fine_power_retention.py``, which pins the
arithmetic. This one drives the real ``PilotProxyDetectorAnalyzer`` over a
synthetic baseband file and asserts the saved product carries
``fine_power_u64`` as exact uint64 -- covering both the device path (kernel
supplies the terms) and the host fallback (kernel does not).
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.detector_contract import (
    normalized_positive_excess,
    weight_term_norms_sq,
)
from pilot_proxy.detector_geometry import SPECTRAL_SENSE_INVERTED
from pilot_proxy.detector_reference import REFERENCE_WEIGHT_TERMS
from pilot_proxy.fine_reduction import exact_coarse_power_by_term
from pilot_proxy.fine_decision import FINE_BINS
from pilot_proxy.fxfft import fine_power_fx
from pilot_proxy.integration.receiver_profile import default_reference_receiver_profile
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.product_contract import validate_current_product_identity

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
PHYS_CH = 14
F_CENTER_MHZ = 470.3125
FREQ_ID = 844
WINDOWS = NFFT // K


def _make_detector_fn(
    *,
    supply_fine_powers: bool,
    omit_projections: bool = False,
    missing_power_field: str | None = None,
    reference_sum_delta: int = 0,
):
    """CPU reference that emits row projections, optionally with fine terms."""

    def detector_fn(*, packed, weights, kernel, emit_row_projections=False):
        pk = np.asarray(packed)
        if pk.ndim == 2:
            pk = pk[None, ...]
        w_packed = np.asarray(weights, dtype=np.int8)
        nt, nl, nu = weight_term_norms_sq(w_packed)
        nrs = int(nl + nu)

        batch = int(pk.shape[0])
        rows = int(pk.shape[1])
        streams = rows // WINDOWS

        # Deterministic row sums, and coarse powers derived *from them*, so the
        # analyzers exact marginal identity holds. A fake that reports
        # powers unrelated to its row sums is correctly rejected upstream.
        rng = np.random.default_rng(7)
        rs = rng.integers(
            -14336, 14337, size=(batch, REFERENCE_WEIGHT_TERMS, rows, 2)
        ).astype(np.int32)

        results = []
        for b in range(batch):
            marginal = exact_coarse_power_by_term(
                rs[b], num_weight_terms=REFERENCE_WEIGHT_TERMS
            )
            num = int(marginal[0])
            lo = int(marginal[1])
            up = int(marginal[2])
            result = {
                "block_index": b,
                "mask": normalized_positive_excess(
                    num, lo + up, target_norm_sq=nt, reference_norm_sum_sq=nrs
                ),
                "p_target_u64": num,
                "p_ref_lower_u64": lo,
                "p_ref_upper_u64": up,
                "p_ref_sum_u64": lo + up + reference_sum_delta,
            }
            if missing_power_field is not None:
                result.pop(missing_power_field)
            results.append(result)

        out = {
            "batch": batch,
            "detector_rows_per_block": rows,
            "rational_overflow_count": 0,
            "results": results,
        }
        if emit_row_projections and not omit_projections:
            out["matched_filter_row_projections"] = rs
            if supply_fine_powers:
                out["fine_powers_u64"] = np.stack([
                    fine_power_fx(
                        rs[b], num_streams=streams, windows_per_stream=WINDOWS
                    )
                    for b in range(batch)
                ])
        return out

    return detector_fn


def _stub_kernel():
    specs = SimpleNamespace(
        K=K, N=3, bits=4, reference_offset_bins=2,
        as_descriptive_dict=lambda: {
            "detector_window_samples": K,
            "num_weight_terms": 3,
            "sample_bits_per_component": 4,
            "reference_offset_bins": 2,
        },
    )
    return SimpleNamespace(
        specs=specs,
        version=SimpleNamespace(as_string=lambda: "test"),
        supports_row_projections=lambda: True,
    )


def _run(
    tmp_path,
    *,
    supply_fine_powers: bool,
    omit_projections: bool = False,
    missing_power_field: str | None = None,
    reference_sum_delta: int = 0,
    detector_fn=None,
    weights: np.ndarray | None = None,
):
    rng = np.random.default_rng(11)
    if weights is None:
        weights = rng.integers(-120, 121, size=(3, K)).astype(np.int8)

    input_dir = tmp_path / "data"
    input_dir.mkdir(parents=True)
    synth = input_dir / f"baseband_evt_{FREQ_ID}.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=5)

    profile = dataclasses.replace(
        default_reference_receiver_profile(
            frame_size_samples=NFFT, num_input_streams=N_FEEDS
        ),
        spectral_sense=SPECTRAL_SENSE_INVERTED,
        stream_map_required=False,
    )
    (tmp_path / "receiver_profile.json").write_text(
        json.dumps(profile.to_dict()), encoding="utf-8"
    )

    ctx = RunContext(
        instrument=load_instrument("chime"),
        selection=[FREQ_ID],
        options={
            "detector_fn": detector_fn or _make_detector_fn(
                supply_fine_powers=supply_fine_powers,
                omit_projections=omit_projections,
                missing_power_field=missing_power_field,
                reference_sum_delta=reference_sum_delta,
            ),
            "kernel": _stub_kernel(),
            "weights": weights,
            "fine_products": "on",
        },
    )
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = "synth:dtv14"
    meta["scope"] = "local"
    red = PilotProxyDetectorAnalyzer()
    red.begin(ctx, meta)
    red.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    out = tmp_path / "out" / "14.npz"
    red.save(str(out))
    return np.load(out)


@pytest.mark.parametrize("supply_fine_powers", [True, False])
def test_product_carries_exact_fine_powers(tmp_path, supply_fine_powers):
    got = _run(tmp_path, supply_fine_powers=supply_fine_powers)
    validate_current_product_identity(got)

    assert "fine_power_u64" in got.files, "exact fine terms were not retained"
    S = np.asarray(got["fine_power_u64"])
    assert S.dtype == np.dtype(np.uint64), S.dtype
    assert S.shape == (N_FRAMES, REFERENCE_WEIGHT_TERMS, FINE_BINS), S.shape
    # A real reduction, not a placeholder block of zeros.
    assert int(S.sum()) > 0


def test_device_and_host_paths_agree_bit_for_bit(tmp_path):
    """The fallback must reproduce the device terms exactly, not approximately."""
    dev = np.asarray(_run(tmp_path / "dev", supply_fine_powers=True)["fine_power_u64"])
    host = np.asarray(_run(tmp_path / "host", supply_fine_powers=False)["fine_power_u64"])
    assert np.array_equal(dev, host)


def test_reference_split_is_retained(tmp_path):
    """The lower/upper split must survive; the sum alone hides asymmetry."""
    got = _run(tmp_path, supply_fine_powers=True)
    lo = np.asarray(got["p_ref_lower_u64"], dtype=np.uint64)
    up = np.asarray(got["p_ref_upper_u64"], dtype=np.uint64)
    total = np.asarray(got["p_ref_sum_u64"], dtype=np.uint64)
    assert lo.shape == up.shape == total.shape == (N_FRAMES, 1)
    assert np.array_equal(lo + up, total)


@pytest.mark.parametrize("field", ["p_ref_lower_u64", "p_ref_upper_u64"])
def test_missing_reference_split_is_fatal(tmp_path, field):
    with pytest.raises(ValueError, match="missing exact power"):
        _run(
            tmp_path,
            supply_fine_powers=True,
            missing_power_field=field,
        )


def test_reference_sum_mismatch_is_fatal(tmp_path):
    with pytest.raises(ValueError, match="does not equal"):
        _run(
            tmp_path,
            supply_fine_powers=True,
            reference_sum_delta=1,
        )


def test_reference_sum_overflow_is_fatal(tmp_path):
    def detector_fn(*, packed, weights, kernel, emit_row_projections=False):
        rows = int(np.asarray(packed).shape[-2])
        return {
            "batch": 1,
            "detector_rows_per_block": rows,
            "rational_overflow_count": 0,
            "results": [
                {
                    "block_index": 0,
                    "mask": 0,
                    "p_target_u64": 1,
                    "p_ref_lower_u64": np.iinfo(np.uint64).max,
                    "p_ref_upper_u64": 1,
                    "p_ref_sum_u64": 0,
                }
            ],
            "matched_filter_row_projections": np.zeros(
                (1, REFERENCE_WEIGHT_TERMS, rows, 2), dtype=np.int32
            ),
            "fine_powers_u64": np.zeros(
                (1, REFERENCE_WEIGHT_TERMS, FINE_BINS), dtype=np.uint64
            ),
        }

    with pytest.raises(ValueError, match="exceeds uint64"):
        _run(
            tmp_path,
            supply_fine_powers=True,
            detector_fn=detector_fn,
        )


def test_enabled_fine_measurement_requires_projections(tmp_path):
    with pytest.raises(RuntimeError, match="returned no matched-filter"):
        _run(
            tmp_path,
            supply_fine_powers=False,
            omit_projections=True,
        )


def test_both_excess_signs_keep_exact_terms(tmp_path):
    state = {"frame": 0}

    def detector_fn(*, packed, weights, kernel, emit_row_projections=False):
        rows = int(np.asarray(packed).shape[-2])
        frame = state["frame"]
        state["frame"] += 1
        target, lower, upper = (
            (5, 6, 6) if frame == 0 else (10, 4, 4)
        )
        fine = np.full(
            (1, REFERENCE_WEIGHT_TERMS, FINE_BINS),
            frame + 1,
            dtype=np.uint64,
        )
        return {
            "batch": 1,
            "detector_rows_per_block": rows,
            "rational_overflow_count": 0,
            "results": [
                {
                    "block_index": 0,
                    "mask": frame,
                    "p_target_u64": target,
                    "p_ref_lower_u64": lower,
                    "p_ref_upper_u64": upper,
                    "p_ref_sum_u64": lower + upper,
                }
            ],
            "matched_filter_row_projections": np.zeros(
                (1, REFERENCE_WEIGHT_TERMS, rows, 2), dtype=np.int32
            ),
            "fine_powers_u64": fine,
        }

    row = np.ones((1, K), dtype=np.int8)
    product = _run(
        tmp_path,
        supply_fine_powers=True,
        detector_fn=detector_fn,
        weights=np.repeat(row, REFERENCE_WEIGHT_TERMS, axis=0),
    )
    assert np.asarray(product["p_target_u64"]).reshape(-1).tolist() == [5, 10]
    assert np.asarray(product["p_ref_lower_u64"]).reshape(-1).tolist() == [6, 4]
    assert np.asarray(product["p_ref_upper_u64"]).reshape(-1).tolist() == [6, 4]
    assert np.asarray(product["p_ref_sum_u64"]).reshape(-1).tolist() == [12, 8]
    excess = np.asarray(product["normalized_pilot_excess"]).reshape(-1)
    assert excess[0] < 0.0 < excess[1]
    assert np.asarray(product["reject_mask"]).reshape(-1).tolist() == [0, 1]
    assert np.isnan(np.asarray(product["pilot_excess_db"]).reshape(-1)[0])
    fine = np.asarray(product["fine_power_u64"])
    assert np.all(fine[0] == 1)
    assert np.all(fine[1] == 2)
