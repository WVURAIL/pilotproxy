# coding=utf-8
"""Small durable-file primitives shared by product writers.

Temporary files are created with ``open(2)`` mode ``0666`` so the caller's
normal umask is honoured.  When replacing an existing file, its permission
bits are copied to the temporary file before publication.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping


def create_temporary_sibling(
    destination: Path, *, suffix: str = ".tmp"
) -> tuple[int, Path]:
    """Create an exclusive sibling temporary file with destination-like mode."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    for _ in range(128):
        temporary = target.parent / (
            f".{target.name}.{secrets.token_hex(12)}{suffix}"
        )
        try:
            fd = os.open(temporary, flags, 0o666)
        except FileExistsError:
            continue
        try:
            if target.exists():
                os.fchmod(fd, stat.S_IMODE(target.stat().st_mode))
        except BaseException:
            os.close(fd)
            temporary.unlink(missing_ok=True)
            raise
        return fd, temporary
    raise FileExistsError(f"could not allocate a temporary sibling for {target}")


def fsync_file(path: Path) -> None:
    """Flush a completed regular file to its backing store."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Flush directory entry changes where the platform supports it."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(Path(path), flags)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
            return
        raise
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(fd)


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
) -> Path:
    """Durably serialize JSON and atomically replace ``path``."""
    destination = Path(path)
    fd, temporary = create_temporary_sibling(destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


__all__ = [
    "atomic_write_json",
    "create_temporary_sibling",
    "fsync_directory",
    "fsync_file",
]
