"""Setuptools build hooks for package-local runtime resources."""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

PROJECT_ROOT = Path(__file__).resolve().parent


def _runtime_resource_files() -> list[str]:
    files = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "configs").glob("*/*.json")
    )
    # Project profiles (projects/<name>/...) travel with a wheel so that an
    # installed package has its default project. A checkout reads them in place.
    files.extend(
        sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "projects").rglob("*")
            if path.is_file() and path.suffix in {".yaml", ".json"}
        )
    )
    files.extend(
        [
            "weights/chime_dtv_weights_k128.bin",
            "weights/chime_dtv_weights_k128.bin.manifest.json",
            "weights/chord_dtv_weights_k64.bin",
            "weights/chord_dtv_weights_k64.bin.manifest.json",
        ]
    )
    return files


class PackageLocalResourcesBuildPy(_build_py):
    """Copy runtime data under ``pilot_proxy/_resources`` in built wheels."""

    def _get_data_files(self):  # type: ignore[no-untyped-def]
        data_files = super()._get_data_files()
        build_dir = Path(self.build_lib) / "pilot_proxy" / "_resources"
        data_files.append(
            (
                "pilot_proxy",
                str(PROJECT_ROOT),
                str(build_dir),
                _runtime_resource_files(),
            )
        )
        return data_files


setup(cmdclass={"build_py": PackageLocalResourcesBuildPy})
