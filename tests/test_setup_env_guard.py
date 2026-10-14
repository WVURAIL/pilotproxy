from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = REPO_ROOT / "scripts" / "setup_env_guard.py"
SPEC = importlib.util.spec_from_file_location("setup_env_guard", GUARD_PATH)
assert SPEC is not None and SPEC.loader is not None
setup_env_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_env_guard)


def _make_minimal_venv(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    bin_dir = target / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text("", encoding="utf-8")
    (bin_dir / "activate").write_text("", encoding="utf-8")


def test_guard_accepts_new_empty_and_owned_targets(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "pilot-proxy"
    home.mkdir()
    checkout.mkdir(parents=True)

    assert setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    ) == target.resolve()
    assert target.is_dir()
    assert setup_env_guard.ownership_sidecar_path(target).is_file()

    _make_minimal_venv(target)
    setup_env_guard.write_ownership_marker(target)
    assert setup_env_guard.validate_venv_target(
        target, home=home, checkouts=[checkout]
    ) == target.resolve()


def test_guard_migrates_a_valid_legacy_venv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "legacy"
    home.mkdir()
    checkout.mkdir(parents=True)
    _make_minimal_venv(target)

    sidecar = setup_env_guard.ownership_sidecar_path(target)
    assert not sidecar.exists()
    with pytest.raises(
        setup_env_guard.UnsafeVenvTarget,
        match="--adopt-legacy-venv",
    ):
        setup_env_guard.prepare_venv_target(
            target, home=home, checkouts=[checkout]
        )
    assert not sidecar.exists()

    assert setup_env_guard.prepare_venv_target(
        target,
        home=home,
        checkouts=[checkout],
        adopt_legacy=True,
    ) == target.resolve()
    assert sidecar.is_file()


def test_guard_migrates_the_previous_in_directory_marker(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "previously-managed"
    home.mkdir()
    checkout.mkdir(parents=True)
    target.mkdir(parents=True)
    (target / setup_env_guard.MARKER_NAME).write_text(
        setup_env_guard.MARKER_CONTENT,
        encoding="utf-8",
    )

    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )

    assert setup_env_guard.ownership_sidecar_path(target).is_file()


def test_guard_allows_retry_after_interrupted_initial_creation(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "interrupted"
    home.mkdir()
    checkout.mkdir(parents=True)

    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )
    partial = target / "partially-created-file"
    partial.write_text("venv creation stopped here\n", encoding="utf-8")

    assert setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    ) == target.resolve()
    assert partial.is_file()


def test_guard_sidecar_survives_clear_before_inside_marker(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "rebuild"
    home.mkdir()
    checkout.mkdir(parents=True)

    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )
    claimed_identity = (target.stat().st_dev, target.stat().st_ino)
    _make_minimal_venv(target)
    setup_env_guard.write_ownership_marker(target)
    for child in tuple(target.iterdir()):
        if child.is_dir():
            for nested in child.iterdir():
                nested.unlink()
            child.rmdir()
        else:
            child.unlink()
    (target / "partial-after-clear").write_text("", encoding="utf-8")

    assert setup_env_guard.validate_venv_target(
        target, home=home, checkouts=[checkout]
    ) == target.resolve()
    assert (target.stat().st_dev, target.stat().st_ino) == claimed_identity


def test_real_venv_clear_preserves_claimed_directory_identity(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "real-clear"
    home.mkdir()
    checkout.mkdir(parents=True)
    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )
    claimed_identity = (target.stat().st_dev, target.stat().st_ino)

    subprocess.run(
        [sys.executable, "-m", "venv", "--clear", str(target)],
        check=True,
    )

    assert (target.stat().st_dev, target.stat().st_ino) == claimed_identity
    assert setup_env_guard.validate_venv_target(
        target, home=home, checkouts=[checkout]
    ) == target.resolve()
    setup_env_guard.write_ownership_marker(target)


def test_guard_refuses_sidecar_after_target_is_deleted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "deleted"
    home.mkdir()
    checkout.mkdir(parents=True)
    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )

    target.rmdir()

    with pytest.raises(setup_env_guard.UnsafeVenvTarget, match="target.*missing"):
        setup_env_guard.prepare_venv_target(
            target, home=home, checkouts=[checkout]
        )
    assert not target.exists()


@pytest.mark.parametrize("foreign_content", [False, True])
def test_guard_refuses_stale_sidecar_for_recreated_target(
    tmp_path: Path,
    foreign_content: bool,
) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "envs" / "recreated"
    home.mkdir()
    checkout.mkdir(parents=True)
    setup_env_guard.prepare_venv_target(
        target, home=home, checkouts=[checkout]
    )
    claimed_identity = (target.stat().st_dev, target.stat().st_ino)

    shutil.rmtree(target)
    # Some filesystems immediately reuse the just-freed inode. Keep that inode
    # occupied so this regression deterministically models a distinct target.
    inode_holder = target.parent / "inode-holder"
    inode_holder.mkdir()
    target.mkdir()
    if foreign_content:
        (target / "keep-me.txt").write_text("important\n", encoding="utf-8")

    assert (target.stat().st_dev, target.stat().st_ino) != claimed_identity
    with pytest.raises(
        setup_env_guard.UnsafeVenvTarget,
        match="does not match the target directory identity",
    ):
        setup_env_guard.prepare_venv_target(
            target, home=home, checkouts=[checkout]
        )
    if foreign_content:
        assert (target / "keep-me.txt").read_text(encoding="utf-8") == "important\n"


def test_guard_refuses_foreign_nonempty_directory_without_modifying_it(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    target = tmp_path / "existing"
    sentinel = target / "keep-me.txt"
    home.mkdir()
    checkout.mkdir(parents=True)
    target.mkdir()
    sentinel.write_text("important\n", encoding="utf-8")

    with pytest.raises(setup_env_guard.UnsafeVenvTarget, match="not owned"):
        setup_env_guard.validate_venv_target(
            target, home=home, checkouts=[checkout]
        )

    assert sentinel.read_text(encoding="utf-8") == "important\n"


@pytest.mark.parametrize("relation", ["same", "ancestor", "descendant"])
def test_guard_refuses_checkout_overlap(tmp_path: Path, relation: str) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "workspace" / "pilot-proxy"
    home.mkdir()
    checkout.mkdir(parents=True)
    targets = {
        "same": checkout,
        "ancestor": checkout.parent,
        "descendant": checkout / ".venv",
    }

    with pytest.raises(setup_env_guard.UnsafeVenvTarget, match="overlaps checkout"):
        setup_env_guard.validate_venv_target(
            targets[relation], home=home, checkouts=[checkout]
        )


def test_guard_refuses_home_and_filesystem_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    checkout = tmp_path / "src" / "pilot-proxy"
    home.mkdir()
    checkout.mkdir(parents=True)

    for target in (home, Path(home.anchor)):
        with pytest.raises(setup_env_guard.UnsafeVenvTarget):
            setup_env_guard.validate_venv_target(
                target, home=home, checkouts=[checkout]
            )
