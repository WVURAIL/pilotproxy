# coding=utf-8
"""Transmitter-census case study: association, dispersion, correlation.

Synthetic fixture encodes the physical expectation the analysis is built to
test: primaries hold tight offsets (disciplined exciters), non-primaries
scatter wide, and channels with more translators show larger spread.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.testbench.transmitter_census import (
    associate,
    bootstrap_ci,
    load_census,
    load_lines,
    mad_hz,
    main,
    normalize_class,
    per_channel_stats,
    spearman,
)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _fixture(tmp_path, rng=None):
    """14 channels; channel c has c-13 translators; primary tight, tx wide."""
    rng = rng or np.random.default_rng(7)
    census, lines = [], []
    for ch in range(14, 28):
        census.append({"rf_channel": ch, "callsign": f"K{ch}AA",
                       "service_class": "full_service", "erp_kw": 1000.0,
                       "distance_km": 60.0})
        lines.append({"rf_channel": ch,
                      "offset_hz": float(rng.normal(0, 8)),
                      "snr_db": 30.0})
        for k in range(ch - 14):  # ch14 has none: anchors the low end
            census.append({"rf_channel": ch, "callsign": f"K{ch}T{k}",
                           "service_class": "translator", "erp_kw": 1.0,
                           "distance_km": 120.0 + 10 * k})
            lines.append({"rf_channel": ch,
                          "offset_hz": float(rng.normal(0, 150.0 * (1 + k))),
                          "snr_db": 12.0 - k})
    return (_write_csv(tmp_path / "census.csv", census),
            _write_csv(tmp_path / "lines.csv", lines))


requires_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="census figure rendering needs matplotlib (the [plot] extra); "
    "these run in the cpu-release CI job",
)


def test_class_normalization_aliases():
    assert normalize_class("Full Service") == "primary"
    assert normalize_class("TRANSLATOR") == "non_primary"
    assert normalize_class("LPTV") == "non_primary"
    assert normalize_class("experimental") == "other"
    # real-world census strings (FCC/ISED exports)
    assert normalize_class("Translator (LPTV)") == "non_primary"
    assert normalize_class("Low-power (LPTV)") == "non_primary"
    assert normalize_class("Relay") == "non_primary"
    assert normalize_class("Class A") == "non_primary"
    assert normalize_class("Full-power") == "primary"


def test_scored_census_ranks_by_detectability_then_distance(tmp_path):
    path = _write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "A", "service_class": "Full-power",
         "detectability_db": "84.1", "distance_km": "60"},
        {"rf_channel": 20, "callsign": "B", "service_class": "Translator (LPTV)",
         "detectability_db": "", "distance_km": "90"},
        {"rf_channel": 20, "callsign": "C", "service_class": "Relay",
         "detectability_db": "", "distance_km": "40"},
    ])
    census = load_census(path)
    order = [t.callsign for t in
             sorted(census, key=lambda t: (-t.detectability, t.distance_km))]
    assert order == ["A", "C", "B"]  # scored first; unscored by distance


def test_association_tiers_and_unassociated(tmp_path):
    census = load_census(_write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "KPRI", "service_class": "primary",
         "erp_kw": 1000.0, "distance_km": 50.0},
        {"rf_channel": 20, "callsign": "KTRX", "service_class": "translator",
         "erp_kw": 0.5, "distance_km": 150.0},
    ]))
    lines = load_lines(_write_csv(tmp_path / "l.csv", [
        {"rf_channel": 20, "offset_hz": 3.0, "snr_db": 30.0},
        {"rf_channel": 20, "offset_hz": -450.0, "snr_db": 12.0},
        {"rf_channel": 20, "offset_hz": 900.0, "snr_db": 8.0},
    ]))
    recs = associate(lines, census, strategy="ranked")
    by_rank = {r["line_rank"]: r for r in recs}
    assert by_rank[0]["tier"] == "dominant" and by_rank[0]["callsign"] == "KPRI"
    assert by_rank[1]["service_class"] == "non_primary"
    assert by_rank[2]["tier"] == "unassociated"  # more lines than census: a finding


def test_dominant_secondary_pools_the_rest(tmp_path):
    census = load_census(_write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "KPRI", "service_class": "primary",
         "erp_kw": 1000.0, "distance_km": 50.0}]))
    lines = load_lines(_write_csv(tmp_path / "l.csv", [
        {"rf_channel": 20, "offset_hz": 1.0, "snr_db": 30.0},
        {"rf_channel": 20, "offset_hz": 600.0, "snr_db": 9.0}]))
    recs = associate(lines, census, strategy="dominant_secondary")
    assert recs[0]["tier"] == "dominant"
    assert recs[1]["tier"] == "pooled"
    assert recs[1]["service_class"] == "non_primary"


def test_spearman_and_bootstrap_recover_monotone_relation():
    x = np.arange(20, dtype=float)
    y = x ** 2 + 0.1
    rho = spearman(x, y)
    assert rho == pytest.approx(1.0)
    lo, hi = bootstrap_ci(x, y, n_boot=300, seed=1)
    assert lo > 0.8 and hi <= 1.0


def test_mad_is_robust_to_one_outlier():
    tight = np.asarray([0.0, 1.0, -1.0, 2.0, -2.0, 1000.0])
    assert mad_hz(tight) < 10.0


@requires_matplotlib
def test_end_to_end_recovers_class_and_correlation(tmp_path):
    census_path, lines_path = _fixture(tmp_path)
    out = tmp_path / "out"
    rc = main(["--census", str(census_path), "--lines", str(lines_path),
               "--output-dir", str(out), "--bootstrap", "300"])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text())
    by = summary["by_class"]
    # the constructed physics: primaries tight, translators wide
    assert by["primary"]["mad_hz"] < 40.0 < by["non_primary"]["mad_hz"]
    # more translators -> larger spread, by construction. The all-channels
    # statistic sees the full constructed relation; the qualifying statistic
    # (pre-registered n>=3 rule) drops the 1- and 2-line anchor channels and
    # is correspondingly weaker but still positive.
    assert summary["spearman_rho_all_channels"] > 0.6
    assert summary["n_channels_qualifying"] == 12
    assert summary["spearman_rho"] > 0.4
    for name in ("association.csv", "threshold_stability.csv",
                 "fig_offset_by_class.png", "fig_offset_by_class.pdf",
                 "fig_spread_vs_composition.png"):
        assert (out / name).exists(), name
    # per-channel table covers all channels with finite spread
    chans = per_channel_stats(
        associate(load_lines(lines_path), load_census(census_path)))
    assert len(chans) == 14
    assert all(np.isfinite(c["mad_hz"]) for c in chans if c["n_lines"] > 1)


@requires_matplotlib
def test_snr_threshold_filters_before_analysis(tmp_path):
    census_path, lines_path = _fixture(tmp_path)
    out = tmp_path / "out_thresh"
    main(["--census", str(census_path), "--lines", str(lines_path),
          "--output-dir", str(out), "--snr-threshold-db", "25",
          "--bootstrap", "50"])
    summary = json.loads((out / "summary.json").read_text())
    # only the 30 dB primaries survive a 25 dB cut
    assert summary["by_class"]["non_primary"]["n"] == 0
    assert summary["by_class"]["primary"]["n"] == 14


# -- line extraction from scan products ---------------------------------------

def _fake_spectrum_product(dirpath, ch, freq_id, spikes, sense=1, nfft=4096,
                           pilot_minus_center_hz=0.0, kwin=128):
    """Per-pilot npz with just the spectrum keys the extractor reads.

    Encodes REAL product semantics: fs = 390625 Hz, bin_enbw_hz = fs/K (the
    DETECTOR bin rather than the spectrum spacing), detector_window_samples = K,
    spectrum spacing = fs/nfft. ``spikes``: [(offset_hz, snr_db), ...] about
    the nominal pilot; the pilot can sit off-center so the full
    center/sense/spacing mapping is exercised.
    """
    fs = 390625.0
    spacing = fs / nfft
    center = 500e6
    pilot = center + pilot_minus_center_hz
    spec = np.ones(nfft, dtype=np.float64)
    for off, snr in spikes:
        f_bb = sense * (off + pilot - center)
        k = int(round(f_bb / spacing)) % nfft
        spec[k] = 10.0 ** (snr / 10.0)
    path = Path(dirpath) / f"{freq_id}.npz"
    np.savez(path,
             integrated_spectrum_before_mask=spec,
             integrated_spectrum_after_mask=spec * 0.5,
             nfft=np.asarray([nfft]), sense=np.asarray([sense]),
             chime_frequency_hz=np.asarray([center]),
             pilot_frequency_hz=np.asarray([pilot]),
             physical_channel=np.asarray([ch]),
             bin_enbw_hz=np.asarray([fs / kwin]),
             detector_window_samples=np.asarray([kwin]))
    return path


def test_extract_matches_real_product_convention(tmp_path):
    """Regression pinned to production numbers: ch26 (freq_id 660) has
    sense=-1 and pilot 121941 Hz above center; the dominant carrier observed
    at spectrum bin 11270 of 16384 must decode to about -16 Hz offset."""
    from pilot_proxy.testbench.transmitter_census import extract_lines_from_run
    fs, nfft = 390625.0, 16384
    spacing = fs / nfft
    spec = np.ones(nfft)
    spec[11270] = 1e3
    np.savez(tmp_path / "660.npz",
             integrated_spectrum_before_mask=spec,
             integrated_spectrum_after_mask=spec,
             nfft=np.asarray([nfft]), sense=np.asarray([-1]),
             chime_frequency_hz=np.asarray([500e6]),
             pilot_frequency_hz=np.asarray([500e6 + 121941.0]),
             physical_channel=np.asarray([26]),
             bin_enbw_hz=np.asarray([fs / 128.0]),
             detector_window_samples=np.asarray([128]))
    (line,) = extract_lines_from_run(tmp_path)
    expected = 500e6 + (-1) * (11270 - nfft) * spacing - (500e6 + 121941.0)
    assert line.offset_hz == pytest.approx(expected, abs=1e-6)
    assert abs(line.offset_hz - (-16.0)) < spacing


def test_extract_never_calls_the_dc_spur_a_carrier(tmp_path):
    from pilot_proxy.testbench.transmitter_census import extract_lines_from_run
    # pilot 18.7 kHz from center (the ch25 geometry): a huge DC spike sits
    # inside the search window but must be guarded out; the true pilot line
    # next to it must survive
    _fake_spectrum_product(tmp_path, 25, 675,
                           [(0.0, 20)], sense=-1,
                           pilot_minus_center_hz=-18_684.0)
    with np.load(tmp_path / "675.npz") as z:
        spec = np.asarray(z["integrated_spectrum_before_mask"]).copy()
        keep = {k: z[k] for k in z.files}
    spec[0] = 1e6  # the DC spur
    keep["integrated_spectrum_before_mask"] = spec
    np.savez(tmp_path / "675.npz", **keep)
    lines = extract_lines_from_run(tmp_path)
    offs = [l.offset_hz for l in lines]
    assert all(abs(o - 18_684.0) > 200 for o in offs), offs  # DC excluded
    assert any(abs(o) < 100 for o in offs)                   # pilot kept


def test_prominence_rejects_sidelobe_skirt_keeps_isolated_line(tmp_path):
    """A 40 dB carrier's leakage skirt (smoothly decaying, with few-dB
    ripples) must yield exactly one line; an isolated 12 dB carrier away
    from the skirt must survive."""
    from pilot_proxy.testbench.transmitter_census import extract_lines_from_run
    fs, nfft, kwin = 390625.0, 4096, 128
    spacing = fs / nfft
    spec = np.ones(nfft, dtype=np.float64)
    center, pilot_off = 500e6, 60_000.0
    c = int(round(pilot_off / spacing)) % nfft   # carrier bin (sense=+1)
    spec[c] = 1e4                                # 40 dB carrier
    spec[c - 1] = spec[c + 1] = 10 ** 3.6        # leakage shoulder (36 dB)
    for k in range(2, 60, 2):                    # skirt: ripple maxima every
        for side in (-1, 1):                     # 2 bins, ~3 dB over valleys
            skirt_db = 34.0 - 0.5 * k
            if skirt_db <= 3:
                break
            spec[c + side * k] = 10 ** (skirt_db / 10)
            spec[c + side * (k + 1)] = 10 ** ((skirt_db - 3.0) / 10)
    iso = int(round((pilot_off + 8000.0) / spacing)) % nfft
    spec[iso] = 10 ** (12.0 / 10)                # isolated real carrier
    np.savez(tmp_path / "700.npz",
             integrated_spectrum_before_mask=spec,
             integrated_spectrum_after_mask=spec,
             nfft=np.asarray([nfft]), sense=np.asarray([1]),
             chime_frequency_hz=np.asarray([center]),
             pilot_frequency_hz=np.asarray([center + pilot_off]),
             physical_channel=np.asarray([20]),
             bin_enbw_hz=np.asarray([fs / kwin]),
             detector_window_samples=np.asarray([kwin]))
    lines = extract_lines_from_run(tmp_path)
    offs = sorted(l.offset_hz for l in lines)
    assert len(offs) == 2, offs                  # carrier + isolated only
    assert abs(offs[0] - 0.0) < spacing
    assert abs(offs[1] - 8000.0) < spacing


def test_extract_lines_recovers_offsets_both_senses(tmp_path):
    from pilot_proxy.testbench.transmitter_census import extract_lines_from_run
    spacing = 390625.0 / 4096
    _fake_spectrum_product(tmp_path, 20, 700, [(0.0, 30), (5000.0, 15),
                                               (-12000.0, 10)], sense=1,
                           pilot_minus_center_hz=60_000.0)
    _fake_spectrum_product(tmp_path, 21, 690, [(0.0, 28), (-7000.0, 12)],
                           sense=-1, pilot_minus_center_hz=-45_000.0)
    lines = extract_lines_from_run(tmp_path)
    by_ch = {}
    for l in lines:
        by_ch.setdefault(l.rf_channel, []).append(l)
    assert sorted(by_ch) == [20, 21]
    offs20 = sorted(l.offset_hz for l in by_ch[20])
    assert len(offs20) == 3
    for got, want in zip(offs20, (-12000.0, 0.0, 5000.0)):
        assert abs(got - want) <= spacing
    snr20 = {round(l.offset_hz / 1000): l.snr_db for l in by_ch[20]}
    assert snr20[0] == pytest.approx(30.0, abs=0.6)
    offs21 = sorted(l.offset_hz for l in by_ch[21])
    for got, want in zip(offs21, (-7000.0, 0.0)):
        assert abs(got - want) <= spacing


def test_extract_lines_window_threshold_separation(tmp_path):
    from pilot_proxy.testbench.transmitter_census import extract_lines_from_run
    _fake_spectrum_product(tmp_path, 22, 660, pilot_minus_center_hz=70_000.0,
                           spikes=[
        (0.0, 30),          # kept
        (50_000.0, 25),     # outside +/-30 kHz window
        (2000.0, 3),        # below 6 dB floor threshold
        (500.0, 20), (540.0, 14),   # within 100 Hz: weaker one thinned
    ])
    lines = extract_lines_from_run(tmp_path)
    offs = sorted(l.offset_hz for l in lines)
    assert len(offs) == 2
    assert abs(offs[0] - 0.0) < 100 and abs(offs[1] - 500.0) < 100


@requires_matplotlib
def test_end_to_end_lines_from_run(tmp_path):
    (tmp_path / "work").mkdir()
    _fake_spectrum_product(tmp_path / "work", 20, 700,
                           [(0.0, 30), (4000.0, 12)],
                           pilot_minus_center_hz=55_000.0)
    census = _write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "KPRI", "service_class": "Full-power",
         "detectability_db": "84", "distance_km": "60"},
        {"rf_channel": 20, "callsign": "KTRX",
         "service_class": "Translator (LPTV)", "detectability_db": "",
         "distance_km": "150"},
    ])
    out = tmp_path / "out"
    rc = main(["--census", str(census), "--lines-from-run",
               str(tmp_path / "work"), "--output-dir", str(out),
               "--bootstrap", "50"])
    assert rc == 0
    assert (out / "extracted_lines.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["line_source"].startswith("extracted:")
    assert summary["by_class"]["primary"]["n"] == 1
    assert summary["by_class"]["non_primary"]["n"] == 1


def test_lines_sources_are_mutually_exclusive(tmp_path):
    census = _write_csv(tmp_path / "c.csv", [
        {"rf_channel": 20, "callsign": "K", "service_class": "Full-power",
         "detectability_db": "80", "distance_km": "60"}])
    with pytest.raises(SystemExit):
        main(["--census", str(census), "--output-dir", str(tmp_path / "o")])


@requires_matplotlib
def test_min_channel_lines_gates_figure_b_statistic(tmp_path):
    """The pre-registered n>=3 rule: single-line channels are excluded from
    the qualifying Spearman but still reported in the all-channels variant
    and counted as excluded."""
    census_rows = []
    (tmp_path / "work").mkdir()
    # three qualifying channels (3 lines each), one single-line channel
    for ch, freq in ((20, 700), (21, 690), (22, 680)):
        _fake_spectrum_product(tmp_path / "work", ch, freq,
                               [(0.0, 30), (900.0, 12), (-700.0, 10)],
                               pilot_minus_center_hz=60_000.0)
        census_rows += [
            {"rf_channel": ch, "callsign": f"K{ch}", "service_class":
             "Full-power", "detectability_db": "80", "distance_km": "60"},
            {"rf_channel": ch, "callsign": f"K{ch}T", "service_class":
             "Translator (LPTV)", "detectability_db": "", "distance_km": "150"},
            {"rf_channel": ch, "callsign": f"K{ch}U", "service_class":
             "Translator (LPTV)", "detectability_db": "", "distance_km": "200"},
        ]
    _fake_spectrum_product(tmp_path / "work", 23, 670, [(0.0, 25)],
                           pilot_minus_center_hz=55_000.0)
    census_rows.append({"rf_channel": 23, "callsign": "K23", "service_class":
                        "Full-power", "detectability_db": "80",
                        "distance_km": "60"})
    census = _write_csv(tmp_path / "c.csv", census_rows)
    out = tmp_path / "out"
    main(["--census", str(census), "--lines-from-run", str(tmp_path / "work"),
          "--output-dir", str(out), "--bootstrap", "50"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["min_channel_lines"] == 3
    assert summary["n_channels_qualifying"] == 3
    assert summary["n_channels_excluded"] == 1
    assert "spearman_rho_all_channels" in summary
