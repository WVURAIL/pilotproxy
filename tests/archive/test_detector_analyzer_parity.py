# coding=utf-8
"""GPU-free parity test for the `pilot-proxy-detector` analyzer.

The CUDA kernel runs only on a GPU, so a CPU-reference ``detector_fn``, a
stub kernel, and either explicit weights or the shipped weight bank (whose
manifest the analyzer checks) are used. The analyzer streams a
synthetic file, and its fine and frame powers must equal those computed
directly from the file's bytes (``cpu_detector.reference_fine_powers``). This
tests the packing, windowing, accumulation, and output schema of the analyzer.
Parity of the fixed-point kernel on real data is tested in its own step on a
CANFAR GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")

from pilot_proxy.chime import baseband_format as fmt
from pilot_proxy.archive.instruments import load_instrument
from pilot_proxy.archive.interfaces import RunContext

from pilot_proxy.archive.detector import PilotProxyDetectorAnalyzer
from pilot_proxy.archive.packed_reader import ChimeBasebandPackedReader
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH

from cpu_detector import exact_cpu_detector_fn, reference_fine_powers, stub_kernel  # noqa: E402

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
PHYS_CH = 14
F_CENTER_MHZ = 470.3125
FREQ_ID = 844  # coarse channel for 470.3125 MHz (== chime_freq_id_from_hz)


@pytest.mark.parametrize("from_bank", [False, True], ids=["explicit-weights", "weight-bank"])
def test_detector_analyzer_matches_cpu_reference(tmp_path, from_bank):
    options = {"detector_fn": exact_cpu_detector_fn, "kernel": stub_kernel(K)}
    if from_bank:
        weights, valid = DetectorWeightBank(
            explicit_path=DEFAULT_WEIGHTS_PATH
        ).get_weights_for_physical_channel(PHYS_CH)
        assert valid
    else:
        weights = np.random.default_rng(11).integers(-120, 121, size=(3, K)).astype(np.int8)
        options["weights"] = weights

    # synthetic native baseband (offset-binary 4+4-bit), one channel
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    synth = input_dir / f"baseband_evt_{FREQ_ID}.h5"
    fmt.make_synth_file(str(synth), n_time=NFFT * N_FRAMES, n_feeds=N_FEEDS,
                        f_center_mhz=F_CENTER_MHZ, f_tone_bb=1500.0, seed=5)

    ctx = RunContext(instrument=load_instrument("chime"), selection=[FREQ_ID], options=options)
    reader = ChimeBasebandPackedReader()
    meta = dict(reader.probe(str(synth)))
    meta["unit_key"] = "synth:dtv14"
    meta["scope"] = "local"
    red = PilotProxyDetectorAnalyzer()
    red.begin(ctx, meta)
    red.consume_file(reader.iter_arrays(str(synth), ctx), meta)
    out = tmp_path / "out" / "14.npz"
    red.save(str(out))
    got = np.load(out)

    assert int(got["physical_channel"][0]) == PHYS_CH
    assert int(got["freq_id"][0]) == FREQ_ID
    fine = reference_fine_powers(synth, weights)
    np.testing.assert_array_equal(got["fine_power_u64"], fine)
    powers = fine.sum(axis=-1, dtype=np.uint64)
    for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        np.testing.assert_array_equal(got[name][:, 0], powers[:, term], err_msg=name)
    np.testing.assert_array_equal(got["p_ref_sum_u64"][:, 0], powers[:, 1] + powers[:, 2])
    np.testing.assert_array_equal(got["valid"][:, 0], powers[:, 1] + powers[:, 2] != 0)
    np.testing.assert_array_equal(got["frame_index"], np.arange(N_FRAMES))
