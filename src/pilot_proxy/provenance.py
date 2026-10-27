# coding=utf-8
"""File and source digests and detector_version tokens for PilotProxy products."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path


def file_sha256(path: Path | str | None) -> str | None:
    """Return the SHA256 hex digest for an existing file, or None otherwise."""
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None

    return sha256_file(file_path)


@functools.lru_cache(maxsize=8)
def package_source_sha256(package_root: Path | str | None = None) -> str:
    """Return the SHA256 digest of the package's ``*.py`` sources.

    The files are hashed in sorted order of their relative POSIX paths, each
    as its path and its bytes with a length prefix, so the digest does not
    depend on where the package is installed. A development version can span
    many commits at one package version, so checkpoint compatibility is tied
    to this digest of the code that produced the product as well as to
    ``__version__``.

    The digest is memoized per process. A long-running scan therefore stamps
    every product with the tree as it was at first use, even when source files
    change on disk mid-run (for example, patches applied while a survey is
    processing). The imported code cannot change in a running process, so the
    first digest is the closest available record of the code that runs. A
    relaunched process hashes the new tree.
    """
    root = (
        Path(__file__).resolve().parent
        if package_root is None
        else Path(package_root).resolve()
    )
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return the SHA256 hex digest of a file. A missing file raises OSError."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def producer_identity(roots=None) -> dict:
    """Return the identity of the code that produced a run.

    The record is for information only, and no consumer requires it. It has
    the repository commit, the dirty flag and dirty files of the package tree
    (``src/pilot_proxy`` only), the digest of the package sources
    (:func:`package_source_sha256`, fixed at first use in the process), the
    package version, and the SHA256 of the default project's detector
    register. The commit is empty when the package is not in a git checkout
    and ``"unknown"`` when git cannot be run. ``roots`` maps names to further
    source roots, and their digests are added under ``other_source_digests``.
    """
    import subprocess

    from pilot_proxy import __version__
    from pilot_proxy.config.project import default_project

    package = Path(__file__).resolve().parent
    commit, dirty, status = "", None, ""
    try:
        top = subprocess.run(["git", "-C", str(package), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=30)
        if top.returncode == 0:
            root = top.stdout.strip()
            commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=30).stdout.strip()
            status = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no", "--",
                                     str(package)], capture_output=True, text=True, timeout=30).stdout.strip()
            dirty = bool(status)
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unknown", None
    out = {"repository": "WVURAIL/pilotproxy", "package": "pilot_proxy", "version": __version__,
           "commit": commit, "dirty": dirty, "dirty_files": status, "source_root": str(package),
           "source_digest": package_source_sha256(), "register_sha256": default_project().register.sha256()}
    if roots:
        out["other_source_digests"] = {name: package_source_sha256(Path(root)) for name, root in sorted(roots.items())}
    return out


def sidecar_manifest_path(path: Path | str | None) -> Path | None:
    """Return the sidecar manifest path ``<path>.manifest.json``."""
    if path is None:
        return None
    return Path(f"{Path(path)}.manifest.json")


# detector_version tokens
#
# A detector_version string has the form
#
#     pilot-proxy/<version> source=<tree hash> kernel=<v> kernel_sha256=<hash>
#     <schema tag> K=<K>
#
# and its tokens fall into two classes. The build tokens
# ``pilot-proxy/<version>`` and ``source=<tree hash>`` name the build that
# produced a product. The first is the release label set by a maintainer, and
# the second is the digest of the code that ran. Neither token determines the
# detector numbers. A release bump (0.3.0.dev0 to 1.0.0) changes both tokens
# and leaves the detector math as it was, and a tree can move many commits at
# one version. The label is therefore the weaker of the two, and the source
# digest names the exact implementation.
#
# The remaining tokens (kernel version, kernel binary hash, schema tag, and K)
# set the geometry. Cross-pilot stacking depends on them and on the weights
# hashes, detector contract JSON, mask rule, and reference placement, which
# are compared in their own checks. Resume is stricter and compares the full
# detector_version with its build tokens, so one checkpoint never appends
# frames from two Python implementations.
#
# Both classes are recorded in full in every product, and combine reports the
# distinct build strings it stacked, so no provenance is lost when the
# stacking gate ignores the build tokens.
DETECTOR_VERSION_BUILD_TOKEN_PREFIXES = ("pilot-proxy/", "source=")


def detector_version_geometry(version: object) -> tuple[str, ...]:
    """Return the geometry tokens of a ``detector_version`` string.

    The build tokens (``DETECTOR_VERSION_BUILD_TOKEN_PREFIXES``) are dropped,
    and the other tokens are kept in order. Two complete per-pilot products
    with equal geometry may be stacked, and each keeps its own build
    provenance. Resume uses an exact match of the full version string instead
    of this helper.
    """
    return tuple(
        token
        for token in str(version).split()
        if not token.startswith(DETECTOR_VERSION_BUILD_TOKEN_PREFIXES)
    )


def detector_version_build_id(version: object, digest_chars: int = 12) -> str:
    """Return a short human-readable build identity, ``<version>@<source>``.

    It is used in the notes printed for the operator when products from
    different builds are resumed or stacked. The source digest is cut to
    ``digest_chars`` characters, and a missing token is printed as ``?``.
    """
    label = "?"
    source = "?"
    for token in str(version).split():
        if token.startswith("pilot-proxy/"):
            label = token[len("pilot-proxy/"):] or "?"
        elif token.startswith("source="):
            source = token[len("source="):][:digest_chars] or "?"
    return f"{label}@{source}"


__all__ = [
    "DETECTOR_VERSION_BUILD_TOKEN_PREFIXES",
    "detector_version_build_id",
    "detector_version_geometry",
    "file_sha256",
    "package_source_sha256",
    "sidecar_manifest_path",
]
