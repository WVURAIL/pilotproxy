"""The required-hardware gate must fail rather than silently skip."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "body",
    [
        'import pytest\ndef test_device():\n    pytest.skip("no device")\n',
        'import pytest\npytest.skip("no dependency", allow_module_level=True)\n',
        'import pytest\n@pytest.fixture\ndef device():\n    pytest.skip("no compiler")\ndef test_device(device):\n    pass\n',
    ],
)
def test_required_gpu_gate_reports_skips_as_failures(tmp_path, body):
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "conftest.py", tmp_path / "conftest.py"
    )
    tests = tmp_path / "kernel"
    tests.mkdir()
    (tests / "test_device.py").write_text(body)
    for required in (False, True):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *(["--require-cuda"] if required else []),
            ],
            cwd=tmp_path,
            env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
            capture_output=True,
            text=True,
            timeout=20,
        )
        if required:
            assert result.returncode in (1, 2), result.stdout + result.stderr
            assert "Required CUDA" in result.stdout
        else:
            assert result.returncode in (0, 5), result.stdout + result.stderr
