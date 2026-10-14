#!/usr/bin/env python3
"""Safety guard for the destructive ``venv --clear`` setup step."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Iterable, Sequence

MARKER_NAME = ".pilot-proxy-managed-venv"
MARKER_CONTENT = "managed by pilot-proxy scripts/setup_env.sh\n"
SIDECAR_SUFFIX = ".pilot-proxy-managed-venv.json"
OWNERSHIP_SCHEMA = "pilot-proxy-managed-venv"
OWNERSHIP_REVISION = 2


class UnsafeVenvTarget(ValueError):
    """Raised when setup must not clear the requested virtual environment."""


def canonical_path(value: str | Path) -> Path:
    """Return a symlink-resolved absolute path, including nonexistent tails."""
    return Path(value).expanduser().resolve(strict=False)


def _is_same_or_ancestor(candidate: Path, path: Path) -> bool:
    return candidate == path or candidate in path.parents


def ownership_sidecar_path(target: str | Path) -> Path:
    """Return the durable ownership record stored outside the cleared target."""
    resolved = canonical_path(target)
    return resolved.parent / f".{resolved.name}{SIDECAR_SUFFIX}"


def _ownership_content(target: Path) -> str:
    try:
        identity = target.stat()
    except FileNotFoundError as exc:
        raise UnsafeVenvTarget(
            f"cannot bind ownership to a missing target directory: {target}"
        ) from exc
    if not stat.S_ISDIR(identity.st_mode):
        raise UnsafeVenvTarget(f"ownership target is not a directory: {target}")
    return (
        json.dumps(
            {
                "device": int(identity.st_dev),
                "inode": int(identity.st_ino),
                "revision": OWNERSHIP_REVISION,
                "schema": OWNERSHIP_SCHEMA,
                "target": str(target),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync for durable ownership-record replacement."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, content: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode,
            )
        except FileExistsError:
            continue
        break
    else:  # pragma: no cover - cryptographically unlikely name collisions
        raise OSError(f"could not allocate temporary marker beside {path}")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _has_valid_sidecar(target: Path) -> bool:
    sidecar = ownership_sidecar_path(target)
    if sidecar.is_symlink():
        raise UnsafeVenvTarget(f"ownership record may not be a symlink: {sidecar}")
    if not sidecar.exists():
        return False
    if not sidecar.is_file():
        raise UnsafeVenvTarget(f"ownership record is not a file: {sidecar}")
    if not target.exists():
        raise UnsafeVenvTarget(
            "ownership record is stale because its target directory is missing: "
            f"{sidecar}"
        )
    expected = _ownership_content(target)
    if sidecar.read_text(encoding="utf-8") != expected:
        raise UnsafeVenvTarget(
            "ownership record does not match the target directory identity "
            f"(it may have been deleted and recreated): {sidecar}"
        )
    return True


def _has_legacy_inside_marker(target: Path) -> bool:
    marker = target / MARKER_NAME
    return (
        marker.is_file()
        and not marker.is_symlink()
        and marker.read_text(encoding="utf-8") == MARKER_CONTENT
    )


def _looks_like_python_venv(target: Path) -> bool:
    """Recognize a complete standard venv before one-time legacy migration."""
    config = target / "pyvenv.cfg"
    if not config.is_file() or config.is_symlink():
        return False
    try:
        fields = {
            key.strip().lower(): value.strip()
            for line in config.read_text(encoding="utf-8").splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
        }
    except (OSError, UnicodeError):
        return False
    if not fields.get("home"):
        return False

    layouts = (
        (target / "bin" / "python", target / "bin" / "activate"),
        (target / "Scripts" / "python.exe", target / "Scripts" / "activate"),
    )
    return any(python.is_file() and activate.is_file() for python, activate in layouts)


def validate_venv_target(
    target: str | Path,
    *,
    home: str | Path,
    checkouts: Iterable[str | Path],
    adopt_legacy: bool = False,
) -> Path:
    """Validate and canonicalize a target before ``python -m venv --clear``.

    An existing non-empty directory is accepted only when a durable ownership
    sidecar or a prior in-directory marker exists. A standard, unmarked venv
    may be adopted only with an explicit opt-in. Checkout containment is
    rejected in both directions: clearing a checkout ancestor would erase the
    checkout, while placing the managed environment inside a checkout risks
    erasing tracked or user-created project content on a later rerun.
    """
    resolved = canonical_path(target)
    resolved_home = canonical_path(home)
    resolved_checkouts = tuple(canonical_path(path) for path in checkouts)

    root = Path(resolved.anchor)
    if resolved == root:
        raise UnsafeVenvTarget(f"refusing filesystem root: {resolved}")
    if _is_same_or_ancestor(resolved, resolved_home):
        raise UnsafeVenvTarget(
            f"refusing the home directory or one of its ancestors: {resolved}"
        )

    for checkout in resolved_checkouts:
        if _is_same_or_ancestor(resolved, checkout) or _is_same_or_ancestor(
            checkout, resolved
        ):
            raise UnsafeVenvTarget(
                "refusing a virtual environment that overlaps checkout "
                f"{checkout}: {resolved}"
            )

    sidecar_owned = _has_valid_sidecar(resolved)
    if resolved.exists():
        if not resolved.is_dir():
            raise UnsafeVenvTarget(f"target exists and is not a directory: {resolved}")
        entries = list(resolved.iterdir())
        previously_marked = _has_legacy_inside_marker(resolved)
        legacy_venv = _looks_like_python_venv(resolved)
        if entries and not sidecar_owned and not previously_marked and legacy_venv:
            if not adopt_legacy:
                raise UnsafeVenvTarget(
                    "target is a validated but unowned Python virtual environment: "
                    f"{resolved}; rerun with --adopt-legacy-venv (setup_env.sh: "
                    "PILOT_PROXY_ADOPT_LEGACY_VENV=1) to adopt and clear it"
                )
        elif entries and not sidecar_owned and not previously_marked:
            raise UnsafeVenvTarget(
                "target is non-empty and is not owned by pilot-proxy setup "
                "and is not an adoptable Python virtual environment: "
                f"{resolved}"
            )
    return resolved


def prepare_venv_target(
    target: str | Path,
    *,
    home: str | Path,
    checkouts: Iterable[str | Path],
    adopt_legacy: bool = False,
) -> Path:
    """Validate a target and durably claim it before creation or clearing.

    Claiming before ``venv --clear`` makes a rerun safe after an interrupted
    initial creation or rebuild. The sidecar is outside the directory that
    ``venv --clear`` erases.
    """
    resolved = validate_venv_target(
        target,
        home=home,
        checkouts=checkouts,
        adopt_legacy=adopt_legacy,
    )
    # Create the directory itself before claiming it so the durable sidecar can
    # bind to st_dev/st_ino. ``venv --clear`` removes children but preserves
    # this root identity, while deletion/recreation invalidates the claim.
    resolved.mkdir(parents=True, exist_ok=True)
    resolved = validate_venv_target(
        resolved,
        home=home,
        checkouts=checkouts,
        adopt_legacy=adopt_legacy,
    )
    _atomic_write(ownership_sidecar_path(resolved), _ownership_content(resolved))
    _has_valid_sidecar(resolved)
    return resolved


def write_ownership_marker(target: str | Path) -> Path:
    """Mark a successfully created environment inside the venv as well."""
    resolved = canonical_path(target)
    if not resolved.is_dir() or not _looks_like_python_venv(resolved):
        raise UnsafeVenvTarget(f"cannot mark an invalid environment: {resolved}")
    if not _has_valid_sidecar(resolved):
        raise UnsafeVenvTarget(
            f"cannot mark an environment without its ownership record: {resolved}"
        )
    marker = resolved / MARKER_NAME
    _atomic_write(marker, MARKER_CONTENT)
    return marker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check")
    check.add_argument("--venv", required=True)
    check.add_argument("--home", required=True)
    check.add_argument("--checkout", action="append", default=[], required=True)
    check.add_argument("--adopt-legacy-venv", action="store_true")

    mark = subparsers.add_parser("mark")
    mark.add_argument("--venv", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "check":
            print(
                prepare_venv_target(
                    args.venv,
                    home=args.home,
                    checkouts=args.checkout,
                    adopt_legacy=args.adopt_legacy_venv,
                )
            )
        else:
            write_ownership_marker(args.venv)
    except (OSError, UnsafeVenvTarget) as exc:
        raise SystemExit(f"ERROR: unsafe VENV_DIR: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
