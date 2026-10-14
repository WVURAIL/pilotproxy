"""The public example must work independently of the private test fixture."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_public_example_cli_handoff_and_source_tampering(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    inputs = tmp_path / "example inputs"
    review = tmp_path / "policy review"

    def run(*args, success=True):
        result = subprocess.run(
            [sys.executable, *map(str, args)], cwd=tmp_path, env=env,
            text=True, capture_output=True, timeout=90,
        )
        assert (result.returncode == 0) is success, result.stdout + result.stderr
        return result

    def cli(*args, **kwargs):
        return run("-m", "pilot_proxy.cli", *args, **kwargs)

    generator = ROOT / "tools/make_operating_policy_example.py"
    run(generator, "--output-dir", inputs)
    cli("validate-runtime-weight-bundle", "--bundle-dir", inputs / "bundle")
    cli("compile-operating-policy", "--plan", inputs / "policy_plan.json",
        "--output-dir", review, "--require-complete")
    cli("validate-operating-policy", "--policy-dir", review)
    report = json.loads((review / "policy.json").read_text())
    assert report["deployment_configuration_complete"]
    assert report["receiver_channel_id_namespace"] == "chime_freq_id"
    assert report["kotekan_parameters"] == {"permanent_mask_freq_ids": [844], "require_fine_freq_ids": [614]}
    assert report["required_threshold_bindings"][0]["eta_q16"] == "65536"
    assert [row["physical_channel"] for row in report["not_in_policy"]] == [37]
    assert not any(report[key] for key in (
        "activation_authorized", "science_certification", "consumer_capability_implementation_verified",
    ))
    # Re-running cannot overwrite evidence. A changed source must invalidate it.
    plan_before = (inputs / "policy_plan.json").read_bytes()
    run(generator, "--output-dir", inputs, success=False)
    assert (inputs / "policy_plan.json").read_bytes() == plan_before
    evidence = inputs / "synthetic_evidence.json"
    evidence.write_text(evidence.read_text() + "\n")
    cli("validate-operating-policy", "--policy-dir", review, success=False)
