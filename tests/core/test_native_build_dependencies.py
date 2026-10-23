"""Incremental native builds must notice changes to their included headers."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("target, header, mode", [
    ("libpilotproxy.so", "fxfft_master_twiddle.h", "release"),
    ("libpilotproxy_debug.so", "fxfft_master_twiddle.h", "debug"),
    ("test_pilotproxy_reference", "pilotproxy_detector.h", "release"),
])
def test_changed_header_requires_a_rebuild(tmp_path, target, header, mode):
    if shutil.which("make") is None or shutil.which("true") is None:
        pytest.skip("requires make and a POSIX shell")
    source = Path(__file__).resolve().parents[2] / "cuda"
    for path in source.rglob("*"):
        if path.is_file() and (path.suffix in {".h", ".c", ".cpp", ".cu"}
                               or path.name == "Makefile"):
            dest = tmp_path / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    args = ["make", "-s", "SM=89", "NVCC=true"]
    stamp = f".config_{mode}_sm89_dp4a1_k128_bt64_gmb4096_cwl0_swl1.stamp"
    subprocess.run([*args, stamp], cwd=tmp_path, check=True)
    (tmp_path / target).touch()
    assert subprocess.run([*args, "-q", target], cwd=tmp_path).returncode == 0
    assert subprocess.run([*args, "-q", "-W", header, target], cwd=tmp_path).returncode == 1
