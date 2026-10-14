"""Noise power and optional real GNU Radio flowgraph tests."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pilot_proxy.testbench.add_awgn import _signal_and_noise_power_for_snr


def test_noise_power_uses_the_requested_measurement_bandwidth():
    signal, noise = _signal_and_noise_power_for_snr(
        np.full(16, 3 + 4j),
        snr_db=10,
        sample_rate_hz=12e6,
        snr_bandwidth_hz=6e6,
    )
    assert signal == 25
    assert noise == pytest.approx(5)


@pytest.mark.parametrize(
    "field,invalid",
    [
        (field, value)
        for field in ("snr_db", "sample_rate_hz", "snr_bandwidth_hz")
        for value in (float("nan"), float("inf"), -float("inf"))
    ]
    + [
        ("sample_rate_hz", 0),
        ("snr_bandwidth_hz", -1),
        ("snr_db", 1e6),
        ("snr_db", -1e6),
    ],
)
def test_noise_power_refuses_invalid_requests(field, invalid):
    options = dict(snr_db=10, sample_rate_hz=12e6, snr_bandwidth_hz=6e6)
    options[field] = invalid
    with pytest.raises(ValueError):
        _signal_and_noise_power_for_snr(np.ones(8), **options)


@pytest.mark.parametrize("count", [3, 4, 100003])
def test_real_gnuradio_noise_length_seed_and_power(tmp_path, count):
    python = os.environ.get("PILOT_PROXY_GNURADIO_PYTHON")
    if python is None:
        pytest.skip("set PILOT_PROXY_GNURADIO_PYTHON to exercise GNU Radio")
    source = tmp_path / "clean.cfile"
    np.full(count, 1 + 0j, dtype=np.complex64).tofile(source)
    env = dict(
        os.environ,
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
    )
    outputs = []
    for index, seed in enumerate((1234, 1234, 4321)):
        output = tmp_path / f"noisy-{index}.cfile"
        metadata = tmp_path / f"metadata-{index}.json"
        result = subprocess.run(
            [
                python,
                "-m",
                "pilot_proxy.testbench.add_awgn",
                "--input-iq",
                str(source),
                "--output-iq",
                str(output),
                "--metadata-json",
                str(metadata),
                "--num-samples",
                str(count),
                "--snr-db",
                "10",
                "--sample-rate-hz",
                "12000000",
                "--snr-bandwidth-hz",
                "6000000",
                "--seed",
                str(seed),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        samples = np.fromfile(output, dtype=np.complex64)
        assert samples.size == count
        meta = json.loads(metadata.read_text())
        assert meta["num_samples"] == count
        assert meta["flowgraph_samples"] == ((count + 3) // 4) * 4
        outputs.append(samples)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])
    if count > 1000:
        noise = outputs[0] - 1
        assert np.mean(np.abs(noise) ** 2) == pytest.approx(0.2, rel=0.02)
        assert abs(np.mean(noise)) < 0.005
