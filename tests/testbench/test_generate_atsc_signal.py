"""Transport packets and the real ATSC transmitter's emitted pilot line."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pilot_proxy.testbench.generate_atsc_signal import make_transport_stream_packets


def test_transport_packets_have_stable_headers_and_seeded_payloads():
    data = make_transport_stream_packets(40, 781)
    assert data == make_transport_stream_packets(40, 781)
    assert data != make_transport_stream_packets(40, 782)
    packets = np.frombuffer(data, dtype=np.uint8).reshape(40, 188)
    np.testing.assert_array_equal(packets[:, 0], np.full(40, 0x47))
    np.testing.assert_array_equal(packets[:, 1], np.full(40, 0x40))
    np.testing.assert_array_equal(packets[:, 2], np.arange(40))
    np.testing.assert_array_equal(packets[:, 3], 0x10 | (np.arange(40) % 16))
    assert np.unique(packets[:, 4:]).size == 256


def test_real_transmitter_places_pilot_at_requested_frequency(tmp_path):
    python = os.environ.get("PILOT_PROXY_GNURADIO_PYTHON")
    if python is None:
        pytest.skip("set PILOT_PROXY_GNURADIO_PYTHON to exercise GNU Radio")
    output = tmp_path / "signal.cfile"
    transport = tmp_path / "signal.ts"
    sample_rate = 4_500_000 / 286 * 684
    # Shift the pilot away from its default to test that the request takes effect.
    pilot_offset = 400_000.0
    count = 262_144
    result = subprocess.run(
        [
            python,
            "-m",
            "pilot_proxy.testbench.generate_atsc_signal",
            "--output-iq",
            str(output),
            "--output-ts",
            str(transport),
            "--num-iq-samples",
            str(count),
            "--num-ts-packets",
            "4096",
            "--seed",
            "571",
            "--pilot-offset-hz",
            str(pilot_offset),
        ],
        env=dict(
            os.environ,
            PYTHONNOUSERSITE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"),
        ),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    samples = np.fromfile(output, dtype=np.complex64)
    assert samples.size == count and np.isfinite(samples).all()
    assert transport.stat().st_size == 4096 * 188
    metadata = json.loads(output.with_suffix(".cfile.json").read_text())
    assert metadata["complex_power"] == pytest.approx(float(np.mean(abs(samples) ** 2)))
    # Discard transmitter startup and locate the continuous pilot spectrally.
    data = samples[-131072:]
    spectrum = abs(np.fft.fft(data * np.hanning(data.size))) ** 2
    frequencies = np.fft.fftfreq(data.size, 1 / sample_rate)
    peak = frequencies[np.argmax(spectrum)]
    assert abs(peak - (-3_000_000 + pilot_offset)) < 2 * sample_rate / data.size
    assert spectrum.max() > 100 * np.median(spectrum)
