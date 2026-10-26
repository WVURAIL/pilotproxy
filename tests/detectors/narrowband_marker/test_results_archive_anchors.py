"""The archive results layer's fine-axis anchor: estimator, fallback, coordinates, bootstrap, bulk."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.characterization import blocks
from pilot_proxy.detectors.narrowband_marker import anchors
from pilot_proxy.products.reader import (
    FINE_BIN_HZ, FINE_BINS, Geometry, Product,
)

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("v5_fixture", ROOT / "tests" / "products" / "test_pilotproxy_v5.py")
v5_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5_fixture)

FRAMES = 120
BASE = 1000                     # every exact term; T = 2 S_0 / (S_1 + S_2) = 1 on a flat null
NOMINAL = 62                    # channel 36's grid residual, +739.7 Hz, the worked example's f_a
PEDESTAL_BIN = 129              # odd bin, not in the independent null bulk and far from every line


def _product(tmp_path, *, on_lines=(), static_lines=(), late_on_lines=()) -> Path:
    """A channel-36 fixture whose fine terms are a flat null plus injected lines.

    ``on_lines`` are ``(bin, T)`` pairs present only in coarse-detected frames
    (the fixture's exact decision flags 60 of its 120 frames), ``static_lines``
    in every frame, ``late_on_lines`` only in coarse-detected frames of the
    last six of the twelve acquisitions. The health gate reads a per-frame
    baseband power the threshold fixture lacks, so an in-range value is added.
    """
    path = v5_fixture._write_product(tmp_path / "506.npz", 36)
    v5_fixture._replace(path, baseband_power_linear=np.full((FRAMES, 1), 4.0))
    with np.load(path, allow_pickle=False) as z:
        # the coarse-detected frames the fixture's levels make (60 of 120)
        rejected = np.asarray(z["reject_mask"])[:, 0].astype(bool)
        units = np.asarray(z["frame_unit_index"]).astype(np.int64)
        nt = int(z["target_norm_sq"][0])
        nrs = int(z["reference_norm_sum_sq"][0])
    fine = np.full((FRAMES, 3, FINE_BINS), BASE, dtype=np.uint64)
    for b, level in static_lines:
        fine[:, 0, b] = int(level * BASE)
    for b, level in on_lines:
        fine[rejected, 0, b] = int(level * BASE)
    for b, level in late_on_lines:
        fine[rejected & (units >= 6), 0, b] = int(level * BASE)
    # The coarse flag is decided by the frame powers, which are the all-bin sums
    # of these fine powers (kernel core 3.0.0). A reference pedestal is added at
    # an odd bin that is not in the independent null bulk or on any line, with the
    # same value in every frame. It places the decision between the target sum
    # of the quiet frames and the larger sum of the on frames, so the flag is
    # exactly the coarse-detected set above. The contrast of the pedestal
    # between cohorts is zero.
    quiet_target = int(fine[~rejected, 0].sum(axis=1).max())
    reference_sum = (quiet_target + 2 * BASE) * nrs // nt
    fine[:, 1, PEDESTAL_BIN] += np.uint64((reference_sum - 2 * FINE_BINS * BASE) // 2)
    fine[:, 2, PEDESTAL_BIN] += np.uint64((reference_sum - 2 * FINE_BINS * BASE) // 2)
    v5_fixture._replace(path, fine_power_u64=fine)
    with Product(path) as p:
        assert np.array_equal(p.rejected, rejected)
    return path


def test_on_minus_quiet_anchor_cancels_static_structure(tmp_path):
    # a strong fixed line at 200 in every frame, the pilot lobe at 65 only when the coarse flag is up
    path = _product(tmp_path, on_lines=[(65, 5.0)], static_lines=[(200, 50.0)])
    with Product(path) as p:
        g = p.geometry
        assert g.nominal_fine_bin == NOMINAL
        r = anchors.anchor(p, p.selected, "archive", replicates=0)
    assert r.status == "ok" and r.method == "on_minus_quiet" and r.fallback_cohort == ""
    assert (r.frames_masked, r.frames_on, r.frames_quiet) == (120, 60, 60)
    assert r.anchor_bin == 65 and r.anchor_offset_bins == 3
    assert r.anchor_contrast == pytest.approx(4.0)
    assert r.median_on[200] == 50.0 and r.median_quiet[200] == 50.0 and r.contrast[200] == 0.0
    assert r.window_anchor_bin == 65 and not r.aliased_out_of_window
    assert r.runner_up_bin not in r.designated_bins and r.runner_up_contrast == 0.0 and r.separation == pytest.approx(4.0)
    assert r.designated_bins == (63, 64, 65, 66, 67)
    assert r.bulk_size == 126 and not r.bulk[list(r.designated_bins)].any()
    assert r.anchor_fine_hz == pytest.approx(65 * FINE_BIN_HZ)
    # under sense -1, three bins above nominal is a small negative RF offset from the pilot. The nominal bin
    # is 0.566 Hz below the grid residual, so bin 65 is -(3 * 11.92 - 0.566) Hz = -35.20 Hz
    assert r.anchor_rf_offset_hz == pytest.approx(-(3 * FINE_BIN_HZ) + (g.grid_residual_hz - NOMINAL * FINE_BIN_HZ), abs=1e-9)
    assert -3 * FINE_BIN_HZ - FINE_BIN_HZ / 2 < r.anchor_rf_offset_hz < -3 * FINE_BIN_HZ + FINE_BIN_HZ / 2
    assert r.boot_status == "skipped" and r.boot_replicates == 0 and np.isnan(r.boot_offset_bins_q16)


def test_rf_conversion_sign_follows_the_receiver_sense():
    def geometry(sense):
        residual = -float(np.sign(sense)) * 739.6640625   # sense * (pilot - center), reduced to the grid
        return Geometry(36, 506, 602_309_441.0, 602_343_750.0, sense, residual,
                        int(round(residual / FINE_BIN_HZ)) % FINE_BINS, 0.0, 0.0, 0)
    g_inverted, g_normal = geometry(-1), geometry(1)
    assert g_inverted.nominal_fine_bin == NOMINAL and g_normal.nominal_fine_bin == FINE_BINS - NOMINAL
    up = anchors.rf_offset_of_bin(g_inverted.nominal_fine_bin + 3, g_inverted)
    assert up < 0 and abs(up + 3 * FINE_BIN_HZ) < FINE_BIN_HZ / 2
    up = anchors.rf_offset_of_bin(g_normal.nominal_fine_bin + 3, g_normal)
    assert up > 0 and abs(up - 3 * FINE_BIN_HZ) < FINE_BIN_HZ / 2
    # unwrapping: bins are circular, offsets are signed
    assert anchors.unwrap_bins(2, 250) == 8 and anchors.unwrap_bins(250, 2) == -8
    np.testing.assert_array_equal(anchors.unwrap_bins([0, 127, 128, 255], 0), [0, 127, -128, -1])


def test_alias_flag_when_a_stronger_on_line_sits_outside_the_window(tmp_path):
    # the channel 33 case, where the strongest transmitter-state feature aliases far from the nominal lobe
    path = _product(tmp_path, on_lines=[(65, 5.0), (200, 20.0)])
    with Product(path) as p:
        r = anchors.anchor(p, p.selected, replicates=0)
    assert r.method == "on_minus_quiet"
    assert r.anchor_bin == 200 and r.anchor_offset_bins == 200 - NOMINAL - FINE_BINS
    assert r.window_anchor_bin == 65 and r.window_anchor_offset_bins == 3
    assert r.aliased_out_of_window
    assert r.runner_up_bin == 65 and r.separation == pytest.approx(19.0 - 4.0)
    assert r.designated_bins == (198, 199, 200, 201, 202) and r.bulk_size == 125
    assert -3 * FINE_BIN_HZ - FINE_BIN_HZ / 2 < r.window_anchor_rf_offset_hz < -3 * FINE_BIN_HZ + FINE_BIN_HZ / 2


def test_fallback_to_the_plain_median_without_a_quiet_cohort(tmp_path):
    # an era with nearly all frames coarse-detected, the channel-36 case of the text,
    # "the marked fallback median estimator rather than an on-minus-off difference"
    path = _product(tmp_path, on_lines=[(65, 5.0)], static_lines=[(64, 3.0), (200, 50.0)])
    early = np.arange(FRAMES) < 12
    with Product(path) as p:
        mostly_on = p.selected & (p.rejected | early)
        r = anchors.anchor(p, mostly_on, replicates=0)
    assert r.frames_masked == 66 and r.frames_on == 60 and 0 < r.frames_quiet < 30
    assert r.method == "median_fallback" and r.fallback_cohort == "on" and r.is_fallback
    assert r.anchor_bin == 200 and r.anchor_contrast == 50.0        # the plain median picks the strongest line
    assert r.median_on[65] == 5.0 and r.median_quiet[65] == 1.0 and r.median_on[64] == 3.0
    # with too few coarse-detected frames, the plain median is taken over all masked frames
    with Product(path) as p:
        mostly_quiet = p.selected & (~p.rejected | early)
        r = anchors.anchor(p, mostly_quiet, replicates=0)
    assert r.frames_masked == 66 and 0 < r.frames_on < 30 and r.frames_quiet == 60
    assert r.method == "median_fallback" and r.fallback_cohort == "all"
    assert r.anchor_bin == 200 and r.contrast[64] == 3.0 and r.contrast[65] == 1.0
    # the minimum is a parameter, so 60 quiet frames under a 61-frame minimum take the fallback
    with Product(path) as p:
        r = anchors.anchor(p, p.selected, replicates=0, min_cohort_frames=61)
    assert r.method == "median_fallback" and r.fallback_cohort == "all" and r.anchor_bin == 200
    assert r.min_cohort_frames == 61
    # and with both cohorts at the minimum the estimator of record cancels the static lines
    with Product(path) as p:
        r = anchors.anchor(p, p.selected, replicates=0, min_cohort_frames=60)
    assert r.method == "on_minus_quiet" and r.anchor_bin == 65


def test_bulk_mask_parity_guard_and_census():
    even = anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1)
    odd = anchors.bulk_mask(65, pad_factor=2, guard_fine_bins=1)
    assert even.sum() == 125 and odd.sum() == 126
    for anchor_bin, mask in ((64, even), (65, odd)):
        assert not mask[list(anchors.designated_set(anchor_bin))].any()
        assert not mask[1::2].any()                       # alternate (even) bins only
    # a census-excluded independent bin leaves the bulk, and an odd one was never in it
    assert anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, census_excluded_bins=[10]).sum() == 124
    assert anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, census_excluded_bins=[11]).sum() == 125
    # a guard narrower than the lobe still leaves D out (the bundle refuses otherwise)
    narrow = anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=0)
    assert narrow.sum() == 125 and not narrow[list(anchors.designated_set(64))].any()
    # wrap at the axis edge
    assert anchors.designated_set(1) == (255, 0, 1, 2, 3)
    assert anchors.bulk_mask(0, pad_factor=2, guard_fine_bins=1).sum() == 125


def test_independent_bin_mask_matches_pilot_proxy():
    fine_reduction = pytest.importorskip("pilot_proxy.fine_reduction")
    rng = np.random.default_rng(3)
    for _ in range(20):
        designated = rng.integers(0, FINE_BINS, size=rng.integers(0, 4)).tolist()
        census = rng.integers(0, FINE_BINS, size=rng.integers(0, 6)).tolist()
        pad, guard = int(rng.choice([1, 2, 4])), int(rng.integers(0, 3))
        ours = anchors.independent_bin_mask(FINE_BINS, pad_factor=pad, designated_bins=designated,
                                            guard_fine_bins=guard, census_excluded_bins=census)
        theirs = fine_reduction.independent_bin_mask(FINE_BINS, pad_factor=pad, designated_bins=designated,
                                                     guard_fine_bins=guard, census_excluded_bins=census)
        np.testing.assert_array_equal(ours, theirs)


def test_sorted_cohort_median_matches_blocks_weighted_quantile():
    rng = np.random.default_rng(11)
    n, bins = 50, 7
    values = np.round(rng.normal(size=(n, bins)), 1)        # rounding makes ties common
    rows = np.arange(10, 10 + n)                              # the cohort is a slice of a longer frame axis
    cohort = anchors.SortedCohort(values, rows)
    for trial in range(30):
        weights = np.zeros(200, dtype=np.int64)
        weights[rows] = rng.integers(0, 4, size=n)            # zero weight for acquisitions not drawn
        if trial == 0:
            weights[rows] = 1                                 # the point estimate
        got = cohort.weighted_median(weights)
        want = [blocks.weighted_quantile(weights[rows], values[:, b], 0.5) for b in range(bins)]
        np.testing.assert_array_equal(got, want)
    assert np.isnan(cohort.weighted_median(np.zeros(200, dtype=np.int64))).all()
    assert np.isnan(anchors.SortedCohort(np.zeros((0, bins)), np.zeros(0, dtype=int)).weighted_median(np.ones(200))).all()
    with pytest.raises(ValueError, match="finite"):
        anchors.SortedCohort(np.array([[np.nan, 1.0]]), np.array([0]))


def test_bootstrap_is_deterministic_and_reports_the_mass_at_the_mode(tmp_path):
    path = _product(tmp_path, on_lines=[(65, 5.0)], static_lines=[(200, 50.0)])
    with Product(path) as p:
        ratio = anchors.fine_ratio(p)
        a = anchors.anchor(p, p.selected, ratio=ratio, replicates=150, seed=5)
        b = anchors.anchor(p, p.selected, ratio=ratio, replicates=150, seed=5)
        c = anchors.anchor(p, p.selected, replicates=150, seed=6)
        few = anchors.anchor(p, p.selected, ratio=ratio, replicates=20, min_blocks=20)
        half = anchors.anchor(p, p.frame_unit_index < 6, "early", ratio=ratio, replicates=20)
    assert a.boot_status == "ok" and a.boot_blocks == 12 and a.boot_replicates == a.boot_replicates_used == 150
    np.testing.assert_array_equal(a.boot_samples, b.boot_samples)
    assert anchors.anchor_row(a) == anchors.anchor_row(b)
    assert c.boot_seed == 6 and c.anchor_bin == a.anchor_bin
    # a line in every acquisition is recovered by every replicate, giving a zero-width interval and unit mass
    assert a.boot_mode_bin == 65 and a.boot_mode_mass == 1.0 and a.boot_mass_at_anchor == 1.0
    for name in ("q025", "q16", "q84", "q975"):
        assert getattr(a, f"boot_offset_bins_{name}") == 3.0
        assert getattr(a, f"boot_rf_hz_{name}") == pytest.approx(a.anchor_rf_offset_hz)
    # with too few acquisitions for the block bootstrap, the point estimate is kept and the interval is blank
    assert few.boot_status == "insufficient_blocks" and few.anchor_bin == 65 and np.isnan(few.boot_mode_mass)
    assert few.boot_mode_bin == -1 and few.boot_samples.size == 0
    # a second mask is a second era, with the same function and its own label and cohorts
    assert half.label == "early" and half.frames_masked == 60 and half.method == "on_minus_quiet"
    assert half.anchor_bin == 65 and anchors.anchor_shift_bins(half, a) == 0


def test_bootstrap_interval_moves_when_acquisitions_disagree(tmp_path):
    # the lobe is at 65 in every coarse-detected frame, and a stronger line is at 67 in the last six acquisitions
    # only, so 67 wins a replicate exactly when those acquisitions carry more than half the drawn on frames
    path = _product(tmp_path, on_lines=[(65, 5.0)], late_on_lines=[(67, 8.0)])
    with Product(path) as p:
        r = anchors.anchor(p, p.selected, replicates=300)
        rf65, rf67 = (float(anchors.rf_offset_of_bin(b, p.geometry)) for b in (65, 67))
    assert r.method == "on_minus_quiet" and r.anchor_bin == 65     # with a six/six split, the lower median at 67 is null
    assert r.boot_offset_bins_q025 <= r.boot_offset_bins_q16 <= r.boot_offset_bins_q84 <= r.boot_offset_bins_q975
    assert r.boot_offset_bins_q025 == 3.0 and r.boot_offset_bins_q975 == 5.0
    assert set(np.unique(r.boot_samples)) == {3.0, 5.0}
    assert 0.0 < r.boot_mass_at_anchor < 1.0 and r.boot_mode_bin in (65, 67)
    assert r.boot_mode_mass == max(r.boot_mass_at_anchor, 1.0 - r.boot_mass_at_anchor)
    # RF percentiles are ordered in RF, and under sense -1 the higher bin is the lower RF offset
    assert r.boot_rf_hz_q025 == pytest.approx(rf67) and r.boot_rf_hz_q975 == pytest.approx(rf65)
    assert rf67 < rf65 < 0


def test_empty_mask_gives_an_empty_row(tmp_path):
    path = _product(tmp_path, on_lines=[(65, 5.0)])
    with Product(path) as p:
        r = anchors.anchor(p, np.zeros(p.n_frames, dtype=bool), "none")
    assert r.status == "empty" and r.method == "none" and r.anchor_bin == -1 and r.bulk_size == 0
    assert r.designated_bins == () and r.boot_status == "empty"
    row = anchors.anchor_row(r)
    assert row["anchor_rf_offset_hz"] == "" and row["designated_bins"] == "" and row["anchor_bin"] == "-1"


def test_anchor_table_and_contrast_curves_round_trip(tmp_path):
    path = _product(tmp_path, on_lines=[(65, 5.0)], static_lines=[(200, 50.0)])
    with Product(path) as p:
        ratio = anchors.fine_ratio(p)
        whole = anchors.anchor(p, p.selected, "archive", ratio=ratio, replicates=20)
        early = anchors.anchor(p, p.frame_unit_index < 6, "early", ratio=ratio, replicates=0)
    table = tmp_path / "anchors.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(anchors.ANCHOR_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(anchors.anchor_row(r) for r in (whole, early))
    with table.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert tuple(rows[0].keys()) == anchors.ANCHOR_COLUMNS
    assert [row["label"] for row in rows] == ["archive", "early"]
    assert rows[0]["channel"] == "36" and rows[0]["freq_id"] == "506"
    assert rows[0]["anchor_bin"] == "65" and rows[0]["designated_bins"] == "63;64;65;66;67"
    assert rows[0]["aliased_out_of_window"] == "False" and rows[0]["bulk_size"] == "126"
    assert rows[0]["anchor_rf_offset_hz"] == repr(whole.anchor_rf_offset_hz)
    assert rows[0]["boot_mode_mass"] == "1.0" and rows[1]["boot_mode_mass"] == ""      # blank for NaN
    assert table.read_text(encoding="utf-8").count("\r") == 0
    curves = anchors.write_contrast_curves([whole, early], tmp_path / "contrast.csv")
    with curves.open(newline="", encoding="utf-8") as fh:
        crows = list(csv.DictReader(fh))
    assert tuple(crows[0].keys()) == anchors.CONTRAST_COLUMNS and len(crows) == 2 * FINE_BINS
    archive_rows = [row for row in crows if row["label"] == "archive"]
    assert sum(row["in_designated"] == "True" for row in archive_rows) == 5
    assert sum(row["in_bulk"] == "True" for row in archive_rows) == 126
    assert sum(row["in_window"] == "True" for row in archive_rows) == 61
    at_anchor = archive_rows[65]
    assert at_anchor["bin"] == "65" and at_anchor["offset_bins"] == "3" and at_anchor["contrast"] == "4.0"
    assert at_anchor["rf_offset_hz"] == repr(whole.anchor_rf_offset_hz)
    assert archive_rows[200]["median_on"] == "50.0" and archive_rows[200]["contrast"] == "0.0"


def test_anchor_from_ratio_validates_shapes(tmp_path):
    path = _product(tmp_path, on_lines=[(65, 5.0)])
    with Product(path) as p:
        g = p.geometry
        ratio = anchors.fine_ratio(p)
        kw = dict(valid=p.valid, rejected=p.rejected, unit_index=p.frame_unit_index, geometry=g, channel=36,
                  freq_id=506, label="x", pad_factor=2, guard_fine_bins=1, replicates=0)
        with pytest.raises(ValueError, match="frames"):
            anchors.anchor_from_ratio(ratio[:, :10], mask=p.selected, **kw)
        with pytest.raises(ValueError, match="frames"):
            anchors.anchor_from_ratio(ratio, mask=p.selected[:-1], **kw)
        r = anchors.anchor_from_ratio(ratio, mask=p.selected, **kw)
        assert r.anchor_bin == 65 and r.label == "x"


def test_fine_and_psd_axes_agree_on_the_rf_offset_of_a_tone(tmp_path):
    """A tone at RF offset D from the nominal pilot lands on the fine bin and the PSD bin that both convert back to D."""
    from pilot_proxy.products.reader import COARSE_BIN_HZ, NFFT, PSD_BIN_HZ
    path = v5_fixture._write_product(tmp_path / "506.npz", 36)
    with Product(path) as p:
        g = p.geometry
    assert g.sense == -1 and g.nominal_fine_bin == NOMINAL
    for d in (500.0, -500.0, 1200.0, -1200.0, 0.0):
        # receiver-frame offset of the tone from the coarse-channel center, then the fine axis measures it modulo one coarse bin
        receiver = g.sense * (g.pilot_hz + d - g.centre_hz)
        fine_hz = receiver % COARSE_BIN_HZ
        fine_bin = int(round(fine_hz / FINE_BIN_HZ)) % FINE_BINS
        assert abs(anchors.rf_offset_of_bin(fine_bin, g) - d) <= FINE_BIN_HZ / 2 + 1e-9
        psd_bin = (receiver / PSD_BIN_HZ) % NFFT
        assert g.psd_rf_offset_hz(np.array([psd_bin]))[0] == pytest.approx(d, abs=1e-6)
    # for the station itself, the nominal bin converts to +0.566 Hz, the grid residual above the bin center
    assert anchors.rf_offset_of_bin(NOMINAL, g) == pytest.approx(g.grid_residual_hz - NOMINAL * FINE_BIN_HZ)


def test_rf_offset_shares_one_wrap_with_the_bin_offset(tmp_path):
    """Offsets -128 and -127 are one fine bin apart in RF, since the wrap is about the nominal bin."""
    path = v5_fixture._write_product(tmp_path / "506.npz", 36)
    with Product(path) as p:
        g = p.geometry
    assert g.grid_residual_hz > NOMINAL * FINE_BIN_HZ       # residual above the bin center, where a misplaced wrap jumps a coarse bin
    bins = np.array([(NOMINAL + k) % FINE_BINS for k in (-128, -127, 0, 127)])
    offsets = anchors.unwrap_bins(bins, NOMINAL)
    assert offsets.tolist() == [-128, -127, 0, 127]
    rf = anchors.rf_offset_of_bin(bins, g)
    assert np.allclose(np.diff(rf), [-FINE_BIN_HZ, -127 * FINE_BIN_HZ, -127 * FINE_BIN_HZ])
    assert rf[0] == pytest.approx(-(offsets[0] * FINE_BIN_HZ) + (g.grid_residual_hz - NOMINAL * FINE_BIN_HZ))


@pytest.mark.parametrize("anchor", [64, 65])
def test_marker_guard_band_keeps_a_line_outside_the_markers_out_of_the_bulk(anchor):
    """A line 3 to 5 bins outside the markers (as on channels 22 and 28) enters the
    default bulk. A 5-bin guard band keeps it out, so the rank ignores it."""
    from pilot_proxy.detectors.narrowband_marker.scores import required_eta_q16_by_rank

    width = anchors.DESIGNATED_HALF_WIDTH
    line = [anchor + width + k for k in (3, 4, 5)]
    default = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1)
    guarded = anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1, marker_guard_bins=5)
    assert default[line].any() and not guarded[line].any()
    # the guard band removes only the bins within h + g of the anchor
    removed = np.flatnonzero(default & ~guarded)
    assert all(width < abs(int(anchors.unwrap_bins(b, anchor))) <= width + 5 for b in removed)
    assert anchors.bulk_mask(anchor, pad_factor=2, guard_fine_bins=1, marker_guard_bins=0).tolist() == default.tolist()

    rng = np.random.default_rng(anchor)
    quiet = rng.integers(900, 1100, size=(3, 256)).astype(np.uint64)
    loud = quiet.copy()
    loud[0, line] = 40 * quiet[0, line]            # target-term line, references unchanged
    rank = {}
    for name, bulk in (("default", default), ("guarded", guarded)):
        rho = (int(bulk.sum()) + 1) // 2
        rank[name] = [required_eta_q16_by_rank(powers, anchor_bin=anchor, designated_half_width=width,
                                               bulk_mask=bulk)[rho - 1] for powers in (quiet, loud)]
    assert rank["guarded"][0] == rank["guarded"][1]   # the line cannot move the guarded reference
    assert rank["default"][0] != rank["default"][1]   # it moves the default one


def test_marker_guard_band_reaches_the_anchor_result(tmp_path):
    path = _product(tmp_path, on_lines=[(65, 5.0)])
    with Product(path) as product:
        default = anchors.anchor(product, product.selected, replicates=0)
        guarded = anchors.anchor(product, product.selected, replicates=0, marker_guard_bins=4)
    assert guarded.anchor_bin == default.anchor_bin == 65
    assert (default.bulk_size, guarded.bulk_size) == (126, 122)
    assert not guarded.bulk[[60, 62, 68, 70]].any() and default.bulk[[60, 62, 68, 70]].all()
    with pytest.raises(ValueError, match="marker_guard_bins"):
        anchors.bulk_mask(64, pad_factor=2, guard_fine_bins=1, marker_guard_bins=-1)
