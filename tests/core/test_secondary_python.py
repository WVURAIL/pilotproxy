from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pilot_proxy.paths import PACKAGE_ROOT
import pilot_proxy.secondary_python as secondary_python
from pilot_proxy.secondary_python import (
    is_secondary_interpreter,
    package_only_pythonpath,
    prepend_pythonpath,
)

DIFFERENT_PYTHON = "/definitely/a/different/python"


def test_current_interpreter_needs_no_bridge() -> None:
    assert is_secondary_interpreter(sys.executable) is False
    with package_only_pythonpath(sys.executable) as bridge:
        assert bridge is None


def test_system_launcher_is_secondary_even_when_venv_launcher_symlinks_to_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    system_launcher = tmp_path / "usr" / "bin" / "python3"
    system_launcher.parent.mkdir(parents=True)
    system_launcher.write_text("base interpreter\n", encoding="utf-8")
    venv_launcher = tmp_path / "venv" / "bin" / "python"
    venv_launcher.parent.mkdir(parents=True)
    venv_launcher.symlink_to(system_launcher)
    monkeypatch.setattr(secondary_python.sys, "executable", str(venv_launcher))

    assert venv_launcher.resolve() == system_launcher.resolve()
    assert is_secondary_interpreter(str(venv_launcher)) is False
    assert is_secondary_interpreter(str(system_launcher)) is True


def test_bridge_imports_real_target_with_secondary_dependency(
    tmp_path: Path,
) -> None:
    secondary_dependencies = tmp_path / "secondary-dependencies"
    secondary_dependencies.mkdir()
    (secondary_dependencies / "numpy.py").write_text(
        "ORIGIN = 'secondary-interpreter'\n",
        encoding="utf-8",
    )

    with package_only_pythonpath(DIFFERENT_PYTHON) as bridge:
        assert bridge is not None
        assert sorted(path.name for path in bridge.iterdir()) == ["pilot_proxy"]
        assert (bridge / "pilot_proxy").resolve() == PACKAGE_ROOT
        assert not (bridge / "numpy.py").exists()
        env = {"PYTHONPATH": str(secondary_dependencies)}
        prepend_pythonpath(env, bridge)
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "import pilot_proxy.testbench.generate_atsc_signal; "
                    "import numpy; print(numpy.ORIGIN); print(numpy.__file__)"
                ),
            ],
            cwd=tmp_path,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    output = result.stdout.splitlines()
    assert output[0] == "secondary-interpreter"
    assert Path(output[1]).resolve().is_relative_to(secondary_dependencies)


def test_bridge_copies_package_when_symlinks_are_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def unavailable(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", unavailable)

    with package_only_pythonpath(DIFFERENT_PYTHON) as bridge:
        assert bridge is not None
        package_entry = bridge / "pilot_proxy"
        assert package_entry.is_dir()
        assert not package_entry.is_symlink()
        assert (package_entry / "__init__.py").is_file()
        assert not (package_entry / "__pycache__").exists()
        resources = package_entry / "_resources"
        assert (
            resources / "configs" / "receiver_profiles" / "chime_dtv_fengine.json"
        ).is_file()
        assert (resources / "weights" / "chime_dtv_weights_k128.bin").is_file()
        assert (resources / "weights" / "chord_dtv_weights_k64.bin").is_file()
        assert (resources / "projects" / "chime_atsc" / "project.yaml").is_file()

        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "from pilot_proxy.paths import (CONFIGS_DIR, "
                    "DEFAULT_CHORD_WEIGHTS_PATH, DEFAULT_WEIGHTS_PATH); "
                    "assert (CONFIGS_DIR / 'receiver_profiles' / "
                    "'chime_dtv_fengine.json').is_file(); "
                    "assert DEFAULT_WEIGHTS_PATH.is_file(); "
                    "assert DEFAULT_CHORD_WEIGHTS_PATH.is_file(); "
                    "from pilot_proxy.paths import DATA_ROOT; "
                    "assert (DATA_ROOT / 'projects' / 'chime_atsc' / "
                    "'project.yaml').is_file()"
                ),
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(bridge)},
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
