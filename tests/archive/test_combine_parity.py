# coding=utf-8
"""The combine step stacks per-pilot detector products into PilotProxy's canonical products.

Two synthetic channels are processed by the archive detector analyzer per
pilot, followed by ``combine_detector_products``. The combined
``chime_detector_outputs`` powers must equal the per-pilot products' powers and
those computed directly from the files' bytes
(``cpu_detector.reference_fine_powers``). No GPU is needed, because the
CPU-reference ``detector_fn`` of the other archive tests is injected.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.chime.products import SCAN_INPUT_MANIFEST_SCHEMA_TOKEN
from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.archive.combine import combine_detector_products
from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz

from cpu_detector import exact_cpu_detector_fn, reference_fine_powers, stub_kernel  # noqa: E402

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.171875}  # ATSC channel to coarse-channel center (MHz)
FREQ_IDS = {ch: chime_freq_id_from_hz(mhz * 1e6) for ch, mhz in CHANNELS.items()}


def test_combine_stacks_the_per_pilot_products(tmp_path):
    rng = np.random.default_rng(7)
    weights_by_channel = {
        ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8) for ch in CHANNELS
    }

    input_dir = tmp_path / "data"
    input_dir.mkdir()
    files = {}
    for ch, mhz in CHANNELS.items():
        p = input_dir / f"baseband_evt_{FREQ_IDS[ch]}.h5"
        fmt.make_synth_file(str(p), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                            f_center_mhz=mhz, f_tone_bb=1200.0 + 10 * ch, seed=ch)
        with h5py.File(p, "a") as h:   # timed inputs, so the combined sample_rate is 1/delta_time
            h.attrs["delta_time"] = 1.0 / fmt.FS
            h.attrs["time0_ctime"] = 1.0e9
        files[ch] = p

    # the analyzer per pilot, followed by a combine
    inst = load_instrument("chime")
    reader = ChimeBasebandPackedReader()
    per_pilot = []
    for ch in sorted(CHANNELS):
        ctx = RunContext(instrument=inst, selection=[FREQ_IDS[ch]], options={
            "detector_fn": exact_cpu_detector_fn, "kernel": stub_kernel(K),
            "weights": weights_by_channel[ch],
        })
        meta = dict(reader.probe(str(files[ch])))
        meta["unit_key"] = str(files[ch])  # real path, so the event is shared once freq_id is stripped
        meta["scope"] = "local"
        red = PilotProxyDetectorAnalyzer()
        red.begin(ctx, meta)
        red.consume_file(reader.iter_arrays(str(files[ch]), ctx), meta)
        out = tmp_path / "per_pilot" / f"{FREQ_IDS[ch]}.npz"
        red.save(str(out))
        per_pilot.append(out)

    out_dir = tmp_path / "combined"
    combine_detector_products(per_pilot, out_dir)

    got = np.load(out_dir / "chime_detector_outputs.npz")
    for pilot, ch in enumerate(sorted(CHANNELS)):
        z = np.load(per_pilot[pilot])
        powers = reference_fine_powers(files[ch], weights_by_channel[ch]).sum(axis=-1, dtype=np.uint64)
        for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
            np.testing.assert_array_equal(got[name][:, pilot], powers[:, term], err_msg=f"{ch} {name}")
        for name in ("p_ref_sum_u64", "valid", "coarse_power_ratio", "pilot_excess_db"):
            np.testing.assert_array_equal(got[name][:, pilot], z[name][:, 0], err_msg=f"{ch} {name}")
        np.testing.assert_array_equal(got["mask"][:, pilot], z["reject_mask"][:, 0])
    for name in ("chime_spectrogram_cache.npz", "chime_reductions_10s.npz"):
        assert (out_dir / name).is_file(), name

    # The integrated_spectra stack holds the spectra of each per-pilot product in
    # the sorted channel order of the combined product, with the frequency-axis
    # metadata.
    spec = np.load(out_dir / "chime_integrated_spectra.npz")
    assert spec["integrated_spectrum_before_mask"].shape == (len(CHANNELS), NFFT)
    per_by_ch = {int(np.load(p)["physical_channel"][0]): np.load(p) for p in per_pilot}
    for i, ch in enumerate(np.asarray(spec["physical_channel"])):
        z = per_by_ch[int(ch)]
        assert np.array_equal(
            spec["integrated_spectrum_before_mask"][i],
            np.asarray(z["integrated_spectrum_before_mask"]).reshape(-1)), ch
        assert np.array_equal(
            spec["integrated_spectrum_after_mask"][i],
            np.asarray(z["integrated_spectrum_after_mask"]).reshape(-1)), ch
    assert int(spec["nfft"]) == NFFT
    assert float(spec["sample_rate_hz"]) == pytest.approx(fmt.FS)
    assert list(np.asarray(spec["freq_id"])) == [FREQ_IDS[ch] for ch in sorted(CHANNELS)]

    # validate-products must accept the combined scan output unchanged
    from pilot_proxy.chime.validate_products import validate_products
    report = validate_products(run_dir=out_dir)
    assert report["valid"], report["errors"]

    # The coarse-channel freq_id of each pilot is recorded in stats.json, in the
    # physical_channel order of the stacked detector products (844 for ch14 and
    # 829 for ch15). This record is kept for the deferred 6 MHz mask expansion.
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["freq_id_by_pilot"] == [FREQ_IDS[ch] for ch in sorted(CHANNELS)]
    input_manifest = json.loads(
        (out_dir / "input_manifest.json").read_text(encoding="utf-8")
    )
    assert input_manifest["schema_version"] == SCAN_INPUT_MANIFEST_SCHEMA_TOKEN
