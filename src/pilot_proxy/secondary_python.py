# coding=utf-8
"""Safely expose PilotProxy to a different Python interpreter."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, MutableMapping

from pilot_proxy.paths import DATA_ROOT, PACKAGE_ROOT


def _invocation_path(command: str | os.PathLike[str]) -> Path:
    """Normalize a launcher path without collapsing venv symlinks."""
    text = str(command)
    located = shutil.which(text)
    return Path(os.path.abspath(os.path.expanduser(located or text)))


def is_secondary_interpreter(command: str | os.PathLike[str] | None) -> bool:
    """Return whether ``command`` names a different Python executable."""
    if command is None:
        return False
    # Do not use realpath/Path.resolve here: venv/bin/python commonly symlinks
    # to /usr/bin/python, but invoking those paths produces different prefixes
    # and site-package visibility.
    requested = os.path.normcase(str(_invocation_path(command)))
    current = os.path.normcase(str(_invocation_path(sys.executable)))
    return requested != current


def _create_package_entry(shim_root: Path, package_root: Path) -> Path:
    entry = shim_root / "pilot_proxy"
    try:
        entry.symlink_to(package_root, target_is_directory=True)
    except (NotImplementedError, OSError):
        shutil.copytree(
            package_root,
            entry,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        # A source checkout stores runtime data beside ``src`` rather than
        # under the import package. A copied fallback must look like an
        # installed wheel so pilot_proxy.paths remains fully functional.
        resource_root = entry / "_resources"
        if not resource_root.is_dir():
            shutil.copytree(DATA_ROOT / "configs", resource_root / "configs")
            packaged_weights = resource_root / "weights"
            packaged_weights.mkdir(parents=True)
            for source in sorted((DATA_ROOT / "weights").glob("*.bin*")):
                shutil.copy2(source, packaged_weights / source.name)
            if (DATA_ROOT / "projects").is_dir():
                shutil.copytree(DATA_ROOT / "projects", resource_root / "projects")
    return entry


@contextmanager
def package_only_pythonpath(
    command: str | os.PathLike[str] | None,
) -> Iterator[Path | None]:
    """Yield a temporary import root containing only ``pilot_proxy``.

    The bridge is needed only for a genuinely different interpreter. It does
    not add the primary interpreter's site-packages directory, so NumPy, GNU
    Radio, and other dependencies continue to resolve from the secondary
    interpreter. The temporary path remains alive for the context duration.
    """
    if not is_secondary_interpreter(command):
        yield None
        return

    with tempfile.TemporaryDirectory(prefix="pilot-proxy-pythonpath-") as tmp:
        shim_root = Path(tmp)
        _create_package_entry(shim_root, PACKAGE_ROOT)
        yield shim_root


def prepend_pythonpath(
    env: MutableMapping[str, str],
    import_root: str | os.PathLike[str],
) -> None:
    """Prepend one import root without discarding caller-supplied entries."""
    root = str(import_root)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing


__all__ = [
    "is_secondary_interpreter",
    "package_only_pythonpath",
    "prepend_pythonpath",
]
