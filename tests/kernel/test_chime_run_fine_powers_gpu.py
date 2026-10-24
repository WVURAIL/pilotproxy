"""chime-run on the GPU stores the fine powers of its frame powers, bit for bit."""

import dataclasses
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from cpu_detector import exact_cpu_detector_fn  # noqa: E402
from gpu import cuda_available  # noqa: E402

from pilot_proxy.archive.chime_coarse import chime_freq_id_from_hz  # noqa: E402
from pilot_proxy.chime import baseband_format as fmt  # noqa: E402
from pilot_proxy.chime.runner import run_chime_analysis  # noqa: E402
from pilot_proxy.chime.validate_products import validate_products  # noqa: E402
from pilot_proxy.detector_geometry import MIRRORED_NYQUIST_ZONE  # noqa: E402
from pilot_proxy.integration.receiver_profile import (  # noqa: E402
    default_reference_receiver_profile,
)
from pilot_proxy.kernel import PilotProxyKernel  # noqa: E402
from pilot_proxy.paths import DEFAULT_LIB_PATH  # noqa: E402

pytestmark = pytest.mark.cuda

NFFT = 16384
K = 128
N_FRAMES = 2
N_FEEDS = 4
CHANNELS = {14: 470.3125, 15: 476.3125}


def test_gpu_chime_run_matches_the_cpu_chain_including_fine_powers(tmp_path):
    pytest.importorskip("cupy")
    ok, reason = cuda_available()
    if not ok:
        pytest.skip(reason)
    if not DEFAULT_LIB_PATH.is_file():
        pytest.skip("build the default K=128 CUDA library first")
    kernel = PilotProxyKernel(DEFAULT_LIB_PATH)
    rng = np.random.default_rng(29)
    weights = {ch: rng.integers(-120, 121, size=(3, K)).astype(np.int8) for ch in CHANNELS}
    source = tmp_path / "input"
    source.mkdir()
    for ch, mhz in CHANNELS.items():
        fmt.make_synth_file(
            str(source / f"baseband_1000_{chime_freq_id_from_hz(mhz * 1e6)}.h5"),
            n_time=NFFT * N_FRAMES,
            n_feeds=N_FEEDS, f_center_mhz=mhz, f_tone_bb=1300.0 + 7 * ch, seed=ch,
        )
    profile = dataclasses.replace(
        default_reference_receiver_profile(frame_size_samples=NFFT, num_input_streams=N_FEEDS),
        nyquist_zone=MIRRORED_NYQUIST_ZONE,
        stream_map_required=False,
    )
    profile_path = tmp_path / "receiver_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict()), encoding="utf-8")

    def run(output, **backend):
        run_chime_analysis(
            input_dir=source, output_dir=output, receiver_profile_path=profile_path,
            stream_map_path=None, physical_channels=sorted(CHANNELS),
            frame_size_samples=NFFT, detector_window_samples=K, frames_per_chunk=1,
            max_frames=N_FRAMES, kernel=kernel, weights_by_channel=weights,
            lib_path=DEFAULT_LIB_PATH, **backend,
        )
        with np.load(output / "chime_detector_outputs.npz") as data:
            return {name: data[name].copy() for name in data.files}

    gpu = run(tmp_path / "gpu")
    cpu = run(tmp_path / "cpu", detector_fn=exact_cpu_detector_fn)
    assert set(gpu) == set(cpu)
    for name in gpu:
        np.testing.assert_array_equal(gpu[name], cpu[name], err_msg=name)
    fine = gpu["fine_power_u64"]
    assert fine.shape == (N_FRAMES, len(CHANNELS), 3, 256) and fine.dtype == np.uint64
    assert np.all(fine.sum(axis=-1) > 0)
    for term, name in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        exact = [[sum(int(v) for v in fine[f, p, term]) for p in range(len(CHANNELS))]
                 for f in range(N_FRAMES)]
        assert gpu[name].tolist() == exact
    assert gpu["p_ref_sum_u64"].tolist() == (
        gpu["p_ref_lower_u64"].astype(object) + gpu["p_ref_upper_u64"].astype(object)
    ).tolist()
    report = validate_products(run_dir=tmp_path / "gpu")
    assert report["valid"] is True, report["errors"]
    assert report["detector_contract_revision"] == 2
