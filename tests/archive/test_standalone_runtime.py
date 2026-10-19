# coding=utf-8
"""Standalone archive runtime and package-data checks."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
INSTRUMENT_NAMES = {"chime", "gbo", "hco", "kko"}


def _run_python(code: str, *args: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *(str(arg) for arg in args)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_archive_imports_injection_and_local_scan_are_standalone(tmp_path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("yaml")

    result = _run_python(
        r"""
        import importlib
        import importlib.abc
        from pathlib import Path
        import sys
        from types import SimpleNamespace

        import numpy as np


        blocked = []


        class RejectExternalPackage(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "datatrawl" or fullname.startswith("datatrawl."):
                    blocked.append(fullname)
                    raise ModuleNotFoundError(fullname)
                return None


        sys.meta_path.insert(0, RejectExternalPackage())
        for module_name in (
            "pilot_proxy.archive.interfaces",
            "pilot_proxy.archive.pipeline",
            "pilot_proxy.archive.selection",
            "pilot_proxy.archive.instruments",
            "pilot_proxy.archive.invpaths",
            "pilot_proxy.archive.accel",
            "pilot_proxy.archive.commands",
            "pilot_proxy.archive.datatrail_client",
            "pilot_proxy.archive.inventory",
            "pilot_proxy.archive.recon",
            "pilot_proxy.archive.survey_state",
            "pilot_proxy.archive.sources.local",
            "pilot_proxy.archive.sources.cadc",
            "pilot_proxy.archive.sources.cadc_inventory",
            "pilot_proxy.archive.sources.cadc_transport",
            "pilot_proxy.chime.baseband_format",
            "pilot_proxy.chime.baseband_reader",
            "pilot_proxy.chime.injection",
            "pilot_proxy.archive.detector",
            "pilot_proxy.archive.packed_reader",
            "pilot_proxy.archive.scan",
        ):
            importlib.import_module(module_name)

        import pilot_proxy
        from pilot_proxy.chime import baseband_format as fmt
        from pilot_proxy.chime.injection import inject_directory
        from pilot_proxy.archive.scan import run_chime_scan

        work = Path(sys.argv[1])
        source_root = Path(sys.argv[2]).resolve()
        assert Path(pilot_proxy.__file__).resolve().is_relative_to(source_root)

        input_dir = work / "input"
        input_dir.mkdir()
        input_path = input_dir / "baseband_1_844.h5"
        nfft = 16384
        detector_window = 128
        fmt.make_synth_file(
            str(input_path),
            n_time=nfft * 2,
            n_feeds=4,
            f_center_mhz=470.3125,
            f_tone_bb=1300.0,
            seed=7,
        )

        injected_dir = work / "injected"
        entries = inject_directory(
            [input_path],
            injected_dir,
            amplitude_lsb=0.0,
            phase_seed=11,
            baseband_frequency_hz=1300.0,
        )
        assert len(entries) == 1
        assert entries[0]["byte_identical_to_source"] is True

        def detector_fn(*, packed, weights, kernel):
            blocks = np.asarray(packed)
            if blocks.ndim == 2:
                blocks = blocks[None, ...]
            # Fixed fine spectra (all power in bin 0) whose all-bin sums are
            # the frame powers 10/10/10 (per-pilot v6 invariant).
            fine = np.zeros((int(blocks.shape[0]), 3, 256), dtype=np.uint64)
            fine[:, :, 0] = 10
            return {
                "batch": int(blocks.shape[0]),
                "detector_rows_per_block": int(blocks.shape[1]),
                "rational_overflow_count": 0,
                "fine_powers_u64": fine,
                "results": [
                    {
                        "block_index": index,
                        "mask": 0,
                        "p_target_u64": 10,
                        "p_ref_lower_u64": 10,
                        "p_ref_upper_u64": 10,
                        "p_ref_sum_u64": 20,
                    }
                    for index in range(int(blocks.shape[0]))
                ],
            }

        specs = SimpleNamespace(
            K=detector_window,
            N=3,
            bits=4,
            reference_offset_bins=2,
            as_descriptive_dict=lambda: {
                "detector_window_samples": detector_window,
                "num_weight_terms": 3,
                "sample_bits_per_component": 4,
                "reference_offset_bins": 2,
            },
        )
        kernel = SimpleNamespace(
            specs=specs,
            version=SimpleNamespace(as_string=lambda: "test"),
        )
        weights = np.random.default_rng(3).integers(
            -120, 121, size=(3, detector_window)
        ).astype(np.int8)
        output_dir = work / "scan"
        outputs = run_chime_scan(
            input_dir=injected_dir,
            output_dir=output_dir,
            source="local",
            analyzer="pilot-proxy-detector",
            select="844",
            analyzer_options={
                "detector_fn": detector_fn,
                "kernel": kernel,
                "weights_by_channel": {14: weights},
            },
            verbose=False,
        )

        assert outputs
        assert (output_dir / "chime_detector_outputs.npz").is_file()
        assert (output_dir / "scan_scope.json").is_file()
        assert blocked == []
        assert not any(
            name == "datatrawl" or name.startswith("datatrawl.")
            for name in sys.modules
        )
        """,
        tmp_path,
        SOURCE_ROOT,
        cwd=REPO_ROOT,
    )

    _assert_success(result)


def test_wheel_contains_loadable_instrument_definitions(tmp_path) -> None:
    pytest.importorskip("yaml")

    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / name, project / name)
    for name in ("src", "configs", "weights", "projects"):
        shutil.copytree(REPO_ROOT / name, project / name)

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(wheel_dir),
    ]
    try:
        setuptools_major = int(importlib.metadata.version("setuptools").split(".", 1)[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        setuptools_major = 0
    if setuptools_major >= 77:
        command.append("--no-build-isolation")
    command.append(str(project))
    built = subprocess.run(
        command,
        cwd=project,
        text=True,
        capture_output=True,
        timeout=180,
    )
    _assert_success(built)

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    expected = {
        f"pilot_proxy/instruments/{name}.yaml"
        for name in INSTRUMENT_NAMES
    }
    profile = {
        "pilot_proxy/_resources/" + path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "projects").rglob("*")
        if path.is_file()
    }
    assert profile
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
        assert expected <= wheel_names
        assert profile <= wheel_names
        unpacked = tmp_path / "unpacked"
        archive.extractall(unpacked)

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                from pathlib import Path
                import pilot_proxy
                from pilot_proxy.archive.instruments import (
                    list_instrument_names,
                    load_instrument,
                )

                root = Path.cwd().resolve()
                assert Path(pilot_proxy.__file__).resolve().is_relative_to(root)
                names = set(list_instrument_names())
                assert names == {"chime", "gbo", "hco", "kko"}
                for name in names:
                    instrument = load_instrument(name)
                    assert instrument.name == name
                    assert instrument.n_channels > 0
                    assert instrument.nfft > 0

                from pilot_proxy.config.project import default_project
                import pilot_proxy.dtv_units

                project = default_project().load_all()
                assert project.directory.resolve().is_relative_to(root)
                assert project.directory.parent.parent.name == "_resources"
                assert pilot_proxy.dtv_units.DTV_BANDWIDTH_HZ == 6.0e6
                """
            ),
        ],
        cwd=unpacked,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    _assert_success(loaded)
