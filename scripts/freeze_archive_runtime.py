#!/usr/bin/env python3
"""Build and verify an offline archive runtime bundle."""
from __future__ import annotations

import argparse
import datetime as dt
import email.parser
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "pilotproxy_archive_runtime_bundle_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FreezeError(RuntimeError):
    """Raised when a runtime cannot be frozen or verified safely."""


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        rendered = " ".join(command)
        raise FreezeError(f"command failed ({result.returncode}): {rendered}\n{result.stdout}")
    return result.stdout


def git_value(*arguments: str) -> str:
    return run(("git", *arguments), cwd=REPO_ROOT).strip()


def clean_source_revision() -> str:
    status = git_value("status", "--porcelain", "--untracked-files=all")
    if status:
        raise FreezeError("the source checkout is not clean")
    revision = git_value("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise FreezeError("git did not return a full source revision")
    return revision


def package_source_sha256() -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from pilot_proxy.provenance import package_source_sha256 as calculate
    finally:
        sys.path.pop(0)
    return calculate()


def distribution_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            raise FreezeError("an installed distribution has no name or version")
        normalized = canonical_name(name)
        if normalized in seen:
            raise FreezeError(f"duplicate installed distribution: {name}")
        seen.add(normalized)
        direct_url = None
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            try:
                direct_url = json.loads(raw_direct_url)
            except json.JSONDecodeError as exc:
                raise FreezeError(f"invalid direct_url.json for {name}: {exc}") from exc
        records.append(
            {
                "name": name,
                "normalized_name": normalized,
                "version": version,
                "direct_url": direct_url,
            }
        )
    return sorted(records, key=lambda item: item["normalized_name"])


def source_archive_name(records: Sequence[Mapping[str, Any]]) -> str:
    matches = [
        item for item in records
        if item.get("normalized_name") == "pilot-proxy"
    ]
    if len(matches) != 1:
        raise FreezeError("pilot-proxy is not installed exactly once")
    version = matches[0].get("version")
    if not isinstance(version, str) or not re.fullmatch(
            r"[0-9A-Za-z][0-9A-Za-z.!+_-]*", version):
        raise FreezeError("pilot-proxy has an invalid installed version")
    return f"pilot-proxy-{version}.tar.gz"


def local_path_from_url(url: str) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file":
        return None
    if parsed.netloc not in ("", "localhost"):
        raise FreezeError(f"unsupported file URL host: {parsed.netloc}")
    return Path(urllib.parse.unquote(parsed.path)).resolve()


def requirement_for(record: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    name = str(record["name"])
    normalized = str(record["normalized_name"])
    version = str(record["version"])
    direct_url = record.get("direct_url")
    if direct_url is None:
        return f"{name}=={version}", None
    if not isinstance(direct_url, dict) or not isinstance(direct_url.get("url"), str):
        raise FreezeError(f"invalid direct source metadata for {name}")

    url = direct_url["url"]
    local_path = local_path_from_url(url)
    directory = direct_url.get("dir_info")
    if normalized == "pilot-proxy":
        if local_path != REPO_ROOT or not isinstance(directory, dict) or not directory.get("editable"):
            raise FreezeError("pilot-proxy is not installed editable from this checkout")
        return None, {"name": name, "kind": "checkout", "url": url}

    vcs = direct_url.get("vcs_info")
    if isinstance(vcs, dict):
        commit_id = vcs.get("commit_id")
        vcs_name = vcs.get("vcs")
        if vcs_name != "git" or not isinstance(commit_id, str) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", commit_id
        ):
            raise FreezeError(f"{name} is not installed from an exact git revision")
        base = url[4:] if url.startswith("git+") else url
        requirement = f"{name} @ git+{base}@{commit_id.lower()}"
        return requirement, {
            "name": name,
            "kind": "git",
            "url": base,
            "commit_id": commit_id.lower(),
        }

    if local_path is not None:
        raise FreezeError(f"unreproducible local direct source for {name}: {local_path}")
    raise FreezeError(f"unsupported direct source for {name}: {url}")


def wheel_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist()
                if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            if len(members) != 1:
                raise FreezeError(f"{path.name} does not contain one wheel METADATA file")
            metadata = email.parser.BytesParser().parsebytes(archive.read(members[0]))
    except zipfile.BadZipFile as exc:
        raise FreezeError(f"invalid wheel: {path}") from exc
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise FreezeError(f"wheel metadata has no name or version: {path}")
    return canonical_name(name), version


def wheel_hashes(wheelhouse: Path) -> dict[tuple[str, str], list[tuple[str, str]]]:
    result: dict[tuple[str, str], list[tuple[str, str]]] = {}
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise FreezeError("the wheelhouse is empty")
    for path in wheels:
        identity = wheel_identity(path)
        result.setdefault(identity, []).append((path.name, sha256_file(path)))
    return result


def lock_text(
    records: Sequence[Mapping[str, Any]],
    wheels: Mapping[tuple[str, str], Sequence[tuple[str, str]]],
) -> str:
    lines = ["# Complete runtime lock. Install only from the bundled wheelhouse."]
    expected = {(str(item["normalized_name"]), str(item["version"])) for item in records}
    extra = set(wheels) - expected
    missing = expected - set(wheels)
    if missing:
        rendered = ", ".join(f"{name}=={version}" for name, version in sorted(missing))
        raise FreezeError(f"wheelhouse is missing installed distributions: {rendered}")
    if extra:
        rendered = ", ".join(f"{name}=={version}" for name, version in sorted(extra))
        raise FreezeError(f"wheelhouse has unexpected distributions: {rendered}")
    by_key = {
        (str(item["normalized_name"]), str(item["version"])): item for item in records
    }
    for key in sorted(expected):
        record = by_key[key]
        hashes = sorted({digest for _filename, digest in wheels[key]})
        line = f"{record['name']}=={record['version']}"
        if len(hashes) == 1:
            lines.append(f"{line} --hash=sha256:{hashes[0]}")
            continue
        lines.append(f"{line} \\")
        for index, digest in enumerate(hashes):
            suffix = " \\" if index + 1 < len(hashes) else ""
            lines.append(f"    --hash=sha256:{digest}{suffix}")
    return "\n".join(lines) + "\n"


def payload_records(root: Path, *, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        if path.is_symlink():
            raise FreezeError(f"bundle payload may not be a symlink: {relative}")
        records.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return records


def checksum_text(root: Path) -> str:
    records = payload_records(root, excluded=("SHA256SUMS",))
    return "".join(f"{item['sha256']}  {item['path']}\n" for item in records)


def safe_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise FreezeError("the runtime bundle must be outside the source checkout")
    if resolved.exists() or resolved.is_symlink():
        raise FreezeError(f"output already exists: {resolved}")
    if resolved == resolved.parent:
        raise FreezeError("invalid output path")
    return resolved


def offline_resolution(root: Path) -> None:
    run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-index",
            "--only-binary=:all:",
            "--find-links",
            str(root / "wheelhouse"),
            "--require-hashes",
            "-r",
            str(root / "requirements.lock"),
        )
    )


def create_bundle(output: Path) -> dict[str, Any]:
    output = safe_output_path(output)
    revision = clean_source_revision()
    records = distribution_records()
    archive_name = source_archive_name(records)

    requirements: list[str] = []
    direct_sources: list[dict[str, Any]] = []
    for record in records:
        requirement, direct_source = requirement_for(record)
        if requirement is not None:
            requirements.append(requirement)
        if direct_source is not None:
            direct_sources.append(direct_source)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        wheelhouse = temporary / "wheelhouse"
        source_dir = temporary / "source"
        wheelhouse.mkdir()
        source_dir.mkdir()
        build_requirements = temporary / "build-requirements.txt"
        write_bytes(build_requirements, ("\n".join(requirements) + "\n").encode("utf-8"))

        source_archive = source_dir / archive_name
        run(
            (
                "git",
                "archive",
                "--format=tar.gz",
                f"--prefix=pilot-proxy-{revision[:12]}/",
                "-o",
                str(source_archive),
                revision,
            ),
            cwd=REPO_ROOT,
        )
        run(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--wheel-dir",
                str(wheelhouse),
                "--no-deps",
                "-r",
                str(build_requirements),
            )
        )
        run(
            (
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--wheel-dir",
                str(wheelhouse),
                "--no-deps",
                "--no-build-isolation",
                str(source_archive),
            )
        )

        wheels = wheel_hashes(wheelhouse)
        write_bytes(temporary / "requirements.lock", lock_text(records, wheels).encode("utf-8"))
        manifest = {
            "schema_version": SCHEMA,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_revision": revision,
            "package_source_sha256": package_source_sha256(),
            "python": {
                "executable": sys.executable,
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "platform": platform.platform(),
            },
            "packages": records,
            "direct_sources": direct_sources,
            "artifacts": payload_records(temporary),
        }
        write_bytes(temporary / "runtime_manifest.json", json_bytes(manifest))
        write_bytes(temporary / "SHA256SUMS", checksum_text(temporary).encode("utf-8"))
        verify_bundle(temporary, run_resolution=True)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "  " not in line:
            raise FreezeError(f"invalid SHA256SUMS line {number}")
        digest, relative = line.split("  ", 1)
        if not SHA256_RE.fullmatch(digest):
            raise FreezeError(f"invalid checksum on SHA256SUMS line {number}")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in checksums:
            raise FreezeError(f"unsafe or duplicate path on SHA256SUMS line {number}")
        checksums[relative] = digest
    return checksums


def verify_bundle(root: Path, *, run_resolution: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise FreezeError(f"bundle is not a plain directory: {root}")
    checksum_path = root / "SHA256SUMS"
    manifest_path = root / "runtime_manifest.json"
    if not checksum_path.is_file() or not manifest_path.is_file():
        raise FreezeError("bundle is missing SHA256SUMS or runtime_manifest.json")
    checksums = parse_checksums(checksum_path)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if actual != set(checksums):
        missing = sorted(set(checksums) - actual)
        extra = sorted(actual - set(checksums))
        raise FreezeError(f"bundle file set differs; missing={missing}, extra={extra}")
    for relative, expected in checksums.items():
        path = root / relative
        if path.is_symlink() or sha256_file(path) != expected:
            raise FreezeError(f"checksum mismatch: {relative}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FreezeError(f"invalid runtime manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA:
        raise FreezeError("unexpected runtime manifest schema")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_revision", ""))):
        raise FreezeError("runtime manifest has an invalid source revision")
    if not SHA256_RE.fullmatch(str(manifest.get("package_source_sha256", ""))):
        raise FreezeError("runtime manifest has an invalid package source hash")

    recorded = {
        item.get("path"): (item.get("sha256"), item.get("size_bytes"))
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    expected_artifacts = payload_records(
        root, excluded=("SHA256SUMS", "runtime_manifest.json")
    )
    expected_recorded = {
        item["path"]: (item["sha256"], item["size_bytes"])
        for item in expected_artifacts
    }
    if recorded != expected_recorded:
        raise FreezeError("runtime manifest artifact inventory differs")
    wheels = wheel_hashes(root / "wheelhouse")
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise FreezeError("runtime manifest package list is invalid")
    expected_lock = lock_text(packages, wheels).encode("utf-8")
    if (root / "requirements.lock").read_bytes() != expected_lock:
        raise FreezeError("requirements.lock does not match the wheelhouse")
    if run_resolution:
        offline_resolution(root)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="build a new runtime bundle")
    create.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a runtime bundle")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    verify.add_argument("--skip-resolution", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "create":
            manifest = create_bundle(arguments.output_dir)
            print(f"runtime bundle created: {arguments.output_dir.resolve()}")
        else:
            manifest = verify_bundle(
                arguments.bundle_dir, run_resolution=not arguments.skip_resolution
            )
            print(f"runtime bundle verified: {arguments.bundle_dir.resolve()}")
        print(f"source revision: {manifest['source_revision']}")
        print(f"packages: {len(manifest['packages'])}")
        return 0
    except (FreezeError, OSError) as exc:
        print(f"runtime freeze failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
