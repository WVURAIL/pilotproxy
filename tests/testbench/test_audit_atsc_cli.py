"""Waveform auditing publishes its metrics and propagates the quality result."""

import json

import numpy as np
import pytest

from pilot_proxy.testbench import audit_atsc_signal as audit


def _reject_constant(value):
    raise AssertionError(f"nonstandard JSON constant: {value}")


@pytest.mark.parametrize("fail_on_quality", [False, True])
def test_audit_cli_publishes_failed_quality_before_returning_status(tmp_path, capsys, fail_on_quality):
    n = 8192
    time = np.arange(n) / 1024
    rng = np.random.default_rng(1234)
    samples = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    samples += 8 * np.exp(-2j * np.pi * 192 * time)
    source = tmp_path / "input.c64"
    samples.astype(np.complex64).tofile(source)
    original = source.read_bytes()
    output = tmp_path / "audit" / "quality.json"
    args = ["--input-iq", str(source), "--output-json", str(output),
            "--sample-rate-hz", "1024", "--channel-width-hz", "512",
            "--pilot-offset-hz", "64", "--max-samples", str(n),
            "--pilot-search-half-width-hz", "20", "--pilot-window-half-width-hz", "1",
            "--pilot-exclusion-hz", "8", "--edge-exclusion-hz", "16",
            "--expected-pilot-below-data-db", "100", "--pilot-below-data-tolerance-db", "0",
            "--max-pilot-frequency-error-hz", "0", "--min-occupied-bandwidth-hz", "0",
            "--max-occupied-bandwidth-hz", "1024", "--max-shelf-flatness-db", "100",
            "--max-edge-rolloff-db", "100"]
    if fail_on_quality:
        args.append("--fail-on-quality")
    assert audit.main(args) == int(fail_on_quality)
    result = json.loads(output.read_text(), parse_constant=_reject_constant)
    assert result["num_samples_used"] == n
    assert result["sample_rate_hz"] == 1024
    assert result["expected_pilot_frequency_hz"] == result["measured_pilot_frequency_hz"] == -192
    assert result["fft_bin_width_hz"] == 0.125
    checks = {item["name"]: item for item in result["quality"]["quality_checks"]}
    assert [name for name, item in checks.items() if not item["passed"]] == ["pilot_level"]
    assert checks["pilot_level"]["target"] == 100
    assert result["quality_score"] == 0.8
    assert result["quality_passed"] is False
    assert "quality_check.pilot_level=FAIL" in capsys.readouterr().out
    assert source.read_bytes() == original


@pytest.mark.parametrize("option,value,message", [
    ("--max-samples", "-1", "max_samples must be at least"),
    ("--max-samples", "512", "max_samples must be at least"),
    ("--sample-rate-hz", "0", "sample_rate_hz must be finite and positive"),
    ("--sample-rate-hz", "nan", "sample_rate_hz must be finite and positive"),
    ("--channel-width-hz", "-1", "channel_width_hz must be finite and positive"),
    ("--occupied-power-fraction", "1.1", "occupied_power_fraction must be"),
    ("--occupied-power-fraction", "0", "occupied_power_fraction must be"),
])
def test_invalid_audit_geometry_refuses_before_reading_samples(tmp_path, option, value, message):
    output = tmp_path / "quality.json"
    with pytest.raises(SystemExit, match=message):
        audit.main(["--input-iq", str(tmp_path / "missing.c64"), "--output-json", str(output), option, value])
    assert not output.exists()
