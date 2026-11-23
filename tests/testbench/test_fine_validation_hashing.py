"""Validation source hashes support the oldest declared Python version."""
import hashlib
import runpy
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.mark.parametrize("name", [
    "audit_fine_native_coordinates.py",
    "prepare_fine_validation_cache.py",
    "prepare_fine_validation_waveforms.py",
    "reduce_fine_validation.py",
    "run_fine_validation_campaign.py",
    "validate_fine_detector.py",
])
@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 8193])
def test_sha_matches_standard_digest_without_file_digest(tmp_path, monkeypatch, name, payload):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    module = runpy.run_path(str(TOOLS / name))
    assert module["sha"](source) == hashlib.sha256(payload).hexdigest()


def test_waveform_generator_imports_pilot_proxy(tmp_path, monkeypatch):
    import json
    import subprocess

    import numpy as np

    run = runpy.run_path(str(TOOLS / "prepare_fine_validation_waveforms.py"))["run"]
    old = tmp_path / "results/channel29_residual_controls_2026-09-09/preflight"
    old.mkdir(parents=True)
    np.savez(old / "seed-exclusions.npz", raw_seed=[1], effective_seed63=[1], conservative_seed32=[1])
    np.savez(old / "planned-frame-seeds.npz", raw_seed=[2], effective_seed63=[2])
    monkeypatch.setitem(run.__globals__, "RAIL", tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "{}")
    # with no GNU Radio the generator stops at its own runtime probe, after importing pilot_proxy
    run(type("Args", (), dict(output=tmp_path / "out", gnuradio_python="/bin/false")))
    log = (tmp_path / "out/payload-00.log").read_text()
    assert "CalledProcessError" in log and "/bin/false" in log, log
    plan = json.loads((tmp_path / "out/plan.json").read_text())
    assert "tools/generate_settled_atsc.py" in plan["source_sha256"]
