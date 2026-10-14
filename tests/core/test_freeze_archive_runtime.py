from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "freeze_archive_runtime.py"
SPEC = importlib.util.spec_from_file_location("freeze_archive_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
freeze_archive_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze_archive_runtime)


def _wheel(path: Path, name: str = "Example", version: str = "1.2.3") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    wheel = wheelhouse / "example-1.2.3-py3-none-any.whl"
    _wheel(wheel)
    records = [
        {
            "name": "Example",
            "normalized_name": "example",
            "version": "1.2.3",
            "direct_url": None,
        }
    ]
    wheels = freeze_archive_runtime.wheel_hashes(wheelhouse)
    (root / "requirements.lock").write_text(
        freeze_archive_runtime.lock_text(records, wheels), encoding="utf-8"
    )
    (root / "build-requirements.txt").write_text("Example==1.2.3\n", encoding="utf-8")
    source = root / "source" / "pilot-proxy.tar.gz"
    source.parent.mkdir()
    source.write_bytes(b"source")
    manifest = {
        "schema_version": freeze_archive_runtime.SCHEMA,
        "source_revision": "1" * 40,
        "package_source_sha256": "2" * 64,
        "packages": records,
        "artifacts": freeze_archive_runtime.payload_records(root),
    }
    (root / "runtime_manifest.json").write_bytes(
        freeze_archive_runtime.json_bytes(manifest)
    )
    (root / "SHA256SUMS").write_text(
        freeze_archive_runtime.checksum_text(root), encoding="utf-8"
    )
    return root


def test_verify_bundle_rejects_changed_payload(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    manifest = freeze_archive_runtime.verify_bundle(root, run_resolution=False)
    assert manifest["source_revision"] == "1" * 40

    (root / "build-requirements.txt").write_text("Example==9\n", encoding="utf-8")
    with pytest.raises(freeze_archive_runtime.FreezeError, match="checksum mismatch"):
        freeze_archive_runtime.verify_bundle(root, run_resolution=False)


def test_wheel_identity_ignores_vendored_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "setuptools-84.0.0-py3-none-any.whl"
    _wheel(wheel, name="setuptools", version="84.0.0")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(
            "setuptools/_vendor/example-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n",
        )
    assert freeze_archive_runtime.wheel_identity(wheel) == (
        "setuptools", "84.0.0"
    )


def test_exact_git_direct_source_is_rebuilt_from_commit() -> None:
    requirement, source = freeze_archive_runtime.requirement_for(
        {
            "name": "datatrail-cli",
            "normalized_name": "datatrail-cli",
            "version": "0.11.0",
            "direct_url": {
                "url": "https://github.com/WVURAIL/datatrail-cli.git",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            },
        }
    )
    assert requirement == (
        "datatrail-cli @ git+https://github.com/WVURAIL/datatrail-cli.git@" + "a" * 40
    )
    assert source["commit_id"] == "a" * 40


def test_project_archive_uses_installed_version() -> None:
    records = [
        {
            "name": "PilotProxy",
            "normalized_name": "pilot-proxy",
            "version": "2.3.0",
            "direct_url": None,
        }
    ]
    assert freeze_archive_runtime.source_archive_name(records) == (
        "pilot-proxy-2.3.0.tar.gz"
    )


def test_untracked_local_direct_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(freeze_archive_runtime.FreezeError, match="unreproducible"):
        freeze_archive_runtime.requirement_for(
            {
                "name": "local-package",
                "normalized_name": "local-package",
                "version": "1",
                "direct_url": {"url": tmp_path.as_uri(), "dir_info": {"editable": True}},
            }
        )


def test_output_inside_checkout_is_refused() -> None:
    with pytest.raises(freeze_archive_runtime.FreezeError, match="outside"):
        freeze_archive_runtime.safe_output_path(
            freeze_archive_runtime.REPO_ROOT / "runtime-bundle"
        )
