# coding=utf-8
"""Filesystem paths for source checkouts and installed package data."""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
_REPO_CANDIDATE = SRC_ROOT.parent
SOURCE_CHECKOUT_ROOT = (
    _REPO_CANDIDATE
    if (_REPO_CANDIDATE / "pyproject.toml").is_file()
    and (_REPO_CANDIDATE / "src" / "pilot_proxy").resolve() == PACKAGE_ROOT
    else None
)
REPO_ROOT = SOURCE_CHECKOUT_ROOT or PACKAGE_ROOT


def _package_data_root() -> Path:
    """Return package-local data as a normal Path for installed wheels."""
    return Path(str(resources.files("pilot_proxy").joinpath("_resources")))


def _data_root() -> Path:
    if SOURCE_CHECKOUT_ROOT is not None:
        source_weight = SOURCE_CHECKOUT_ROOT / "weights" / "chime_dtv_weights_k128.bin"
        if source_weight.is_file():
            return SOURCE_CHECKOUT_ROOT
    # Wheels install uncompressed package files, so importlib.resources returns
    # a filesystem-backed Traversable that remains usable as a pathlib.Path.
    return _package_data_root()


DATA_ROOT = _data_root()
CUDA_DIR = (
    SOURCE_CHECKOUT_ROOT / "cuda"
    if SOURCE_CHECKOUT_ROOT is not None
    else PACKAGE_ROOT / "cuda"
)
CONFIGS_DIR = DATA_ROOT / "configs"
GENERATED_DIR = (
    SOURCE_CHECKOUT_ROOT / "generated"
    if SOURCE_CHECKOUT_ROOT is not None
    else Path.cwd() / "generated"
)
WEIGHTS_DIR = DATA_ROOT / "weights"

DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "chime_dtv_weights_k128.bin"
DEFAULT_CHORD_WEIGHTS_PATH = WEIGHTS_DIR / "chord_dtv_weights_k64.bin"
DEFAULT_CONFIG_H = CUDA_DIR / "config.h"


def _default_lib_path() -> Path:
    candidate = CUDA_DIR / "libfstatistic.so"
    cached = Path.home() / ".cache" / "pilot_proxy" / "libfstatistic.so"
    # Prefer a locally rebuilt source-tree library except on WSL-mounted paths.
    # Wheel installs have no source-tree library and therefore use the staged cache.
    if candidate.exists() and not str(candidate).startswith("/mnt/"):
        return candidate
    if cached.exists():
        return cached
    return candidate


DEFAULT_LIB_PATH = _default_lib_path()

_MNT_PATH_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")
_WIN_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def normalize_user_path(value: str | os.PathLike[str]) -> Path:
    """Normalize common Windows / WSL path forms into the local Path format."""
    raw = str(value).strip()
    if not raw:
        return Path(raw)

    if os.name == "nt":
        match = _MNT_PATH_RE.match(raw)
        if match:
            drive = match.group(1).upper()
            tail = match.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{tail}")
    else:
        match = _WIN_DRIVE_RE.match(raw)
        if match:
            drive = match.group(1).lower()
            tail = match.group(2).replace("\\", "/")
            return Path(f"/mnt/{drive}/{tail}")
    return Path(raw)


def resolve_user_path(
    value: str | os.PathLike[str],
    *,
    relative_to: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a user path against the caller's directory, not package files."""
    path = normalize_user_path(value).expanduser()
    if not path.is_absolute():
        base = Path.cwd() if relative_to is None else Path(relative_to)
        path = base / path
    return path.resolve(strict=False)


__all__ = [
    "CUDA_DIR",
    "CONFIGS_DIR",
    "DATA_ROOT",
    "DEFAULT_CONFIG_H",
    "DEFAULT_CHORD_WEIGHTS_PATH",
    "DEFAULT_LIB_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "GENERATED_DIR",
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "SOURCE_CHECKOUT_ROOT",
    "SRC_ROOT",
    "WEIGHTS_DIR",
    "normalize_user_path",
    "resolve_user_path",
]
