# coding=utf-8
from __future__ import annotations

import errno
import json
import os
import threading
from pathlib import Path

import pytest

from pilot_proxy.archive import combine as combine_module


def _write_json(path: Path, generation: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"generation": generation}), encoding="utf-8")


def _generation(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["generation"])


def _staged_generation(
    run_dir: Path, *, name: str = "test", generation: str = "new"
) -> tuple[Path, dict[str, Path]]:
    staging = run_dir / f".pilotproxy-combine-transaction.{name}"
    staging.mkdir()
    paths = {
        "first": staging / "run_config.json",
        "new": staging / "input_manifest.json",
        "last": staging / "stats.json",
    }
    for path in paths.values():
        _write_json(path, generation)
    return staging, paths


def test_nth_publish_replace_failure_rolls_back_whole_output_set(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_config.json", "old")
    _write_json(run_dir / "stats.json", "old")
    staging, outputs = _staged_generation(run_dir)

    real_replace = os.replace
    publish_count = 0

    def fail_fourth_canonical_replace(source, destination):
        nonlocal publish_count
        target = Path(destination)
        if target in {
            run_dir / "run_config.json",
            run_dir / "input_manifest.json",
            run_dir / "stats.json",
            run_dir / combine_module.CHIME_COMBINE_GENERATION_MANIFEST_FILENAME,
        }:
            publish_count += 1
            if publish_count == 4:
                raise OSError("simulated fourth canonical replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(combine_module.os, "replace", fail_fourth_canonical_replace)

    with pytest.raises(OSError, match="fourth canonical replace"):
        combine_module._publish_output_set(outputs, staging, run_dir)

    assert _generation(run_dir / "run_config.json") == "old"
    assert _generation(run_dir / "stats.json") == "old"
    assert not (run_dir / "input_manifest.json").exists()
    assert not (
        run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    ).exists()
    marker = json.loads(
        (
            run_dir
            / combine_module.CHIME_COMBINE_GENERATION_MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert marker["state"] == "recovered"
    assert not staging.exists()


def test_journal_recovers_a_process_terminated_mid_publish(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_config.json", "old")
    _write_json(run_dir / "stats.json", "old")
    staging, outputs = _staged_generation(run_dir)

    outputs = combine_module._stage_generation_manifest(outputs, staging)
    with combine_module._exclusive_publish_ownership(run_dir) as ownership:
        entries = combine_module._prepare_publish_transaction(
            outputs, staging, run_dir, ownership
        )
        first = entries[0]
        combine_module._durable_replace(
            staging / first["relative_path"],
            run_dir / first["relative_path"],
        )
        assert _generation(run_dir / "run_config.json") == "new"
        assert (
            run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
        ).exists()

        assert combine_module._recover_interrupted_publish(
            run_dir, ownership
        ) is True

    assert _generation(run_dir / "run_config.json") == "old"
    assert _generation(run_dir / "stats.json") == "old"
    assert not (run_dir / "input_manifest.json").exists()
    assert not (
        run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    ).exists()
    marker = json.loads(
        (
            run_dir
            / combine_module.CHIME_COMBINE_GENERATION_MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert marker["state"] == "recovered"
    assert not staging.exists()


def test_recovery_rejects_symlinked_canonical_parent(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    outside = external / "mask_summary_by_pilot.csv"
    outside.write_text("outside-old", encoding="utf-8")
    (run_dir / "tables").symlink_to(external, target_is_directory=True)

    transaction = run_dir / ".pilotproxy-combine-transaction.attack"
    backup = transaction / "_previous_outputs" / "tables"
    backup.mkdir(parents=True)
    (backup / "mask_summary_by_pilot.csv").write_text(
        "attacker-backup", encoding="utf-8"
    )
    staged = transaction / "tables"
    staged.mkdir()
    (staged / "mask_summary_by_pilot.csv").write_text(
        "attacker-new", encoding="utf-8"
    )
    journal = {
        "schema_version": combine_module._PUBLISH_JOURNAL_SCHEMA,
        "owner_token": "a" * 32,
        "transaction_directory": transaction.name,
        "entries": [
            {
                "label": "mask_summary",
                "relative_path": "tables/mask_summary_by_pilot.csv",
                "had_previous": True,
            }
        ],
    }
    (run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME).write_text(
        json.dumps(journal), encoding="utf-8"
    )

    with combine_module._exclusive_publish_ownership(run_dir) as ownership:
        with pytest.raises(RuntimeError, match="symlinked canonical"):
            combine_module._recover_interrupted_publish(run_dir, ownership)

    assert outside.read_text(encoding="utf-8") == "outside-old"
    assert (
        run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    ).exists()


def test_recovery_rejects_noncanonical_journal_path(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    transaction = run_dir / ".pilotproxy-combine-transaction.attack"
    transaction.mkdir()
    journal = {
        "schema_version": combine_module._PUBLISH_JOURNAL_SCHEMA,
        "owner_token": "a" * 32,
        "transaction_directory": transaction.name,
        "entries": [
            {
                "label": "attack",
                "relative_path": "not-a-canonical-output.txt",
                "had_previous": False,
            }
        ],
    }
    (run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME).write_text(
        json.dumps(journal), encoding="utf-8"
    )

    with combine_module._exclusive_publish_ownership(run_dir) as ownership:
        with pytest.raises(RuntimeError, match="not a canonical output"):
            combine_module._recover_interrupted_publish(run_dir, ownership)


def test_concurrent_publisher_cannot_replace_owner_journal(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_config.json", "old")
    _write_json(run_dir / "stats.json", "old")
    staging_a, outputs_a = _staged_generation(
        run_dir, name="writer-a", generation="writer-a"
    )
    staging_b, outputs_b = _staged_generation(
        run_dir, name="writer-b", generation="writer-b"
    )

    first_replaced = threading.Event()
    let_writer_a_finish = threading.Event()
    real_durable_replace = combine_module._durable_replace

    def pause_writer_a(source: Path, destination: Path) -> None:
        real_durable_replace(source, destination)
        if (
            Path(source).is_relative_to(staging_a)
            and Path(destination) == run_dir / "run_config.json"
        ):
            first_replaced.set()
            assert let_writer_a_finish.wait(timeout=10)

    monkeypatch.setattr(
        combine_module, "_durable_replace", pause_writer_a
    )
    writer_a_error: list[BaseException] = []

    def publish_a() -> None:
        try:
            combine_module._publish_output_set(outputs_a, staging_a, run_dir)
        except BaseException as exc:  # surfaced below in the test thread
            writer_a_error.append(exc)

    thread = threading.Thread(target=publish_a)
    thread.start()
    assert first_replaced.wait(timeout=10)
    journal_path = run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME
    owner_before = json.loads(journal_path.read_text(encoding="utf-8"))[
        "owner_token"
    ]

    with pytest.raises(RuntimeError, match="another process owns"):
        combine_module._publish_output_set(outputs_b, staging_b, run_dir)
    owner_after = json.loads(journal_path.read_text(encoding="utf-8"))[
        "owner_token"
    ]
    assert owner_after == owner_before
    assert combine_module._cleanup_unreferenced_staging(
        run_dir, staging_b
    ) is True
    assert not staging_b.exists()
    assert staging_a.exists()

    let_writer_a_finish.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert writer_a_error == []
    assert _generation(run_dir / "run_config.json") == "writer-a"
    assert _generation(run_dir / "stats.json") == "writer-a"
    assert not journal_path.exists()
    assert not (run_dir / combine_module.CHIME_COMBINE_PUBLISH_LOCK_FILENAME).exists()


@pytest.mark.parametrize("failure_site", ["metadata", "directory_fsync"])
def test_lock_initialization_failure_is_reacquirable(
    tmp_path, monkeypatch, failure_site
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if failure_site == "metadata":
        real = combine_module._write_locked_metadata
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected lock metadata failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(combine_module, "_write_locked_metadata", fail_once)
    else:
        real = combine_module.fsync_directory
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected lock directory fsync failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(combine_module, "fsync_directory", fail_once)

    with pytest.raises(OSError, match="injected lock"):
        with combine_module._exclusive_publish_ownership(run_dir):
            pass
    assert not (run_dir / combine_module.CHIME_COMBINE_PUBLISH_LOCK_FILENAME).exists()
    with combine_module._exclusive_publish_ownership(run_dir) as ownership:
        ownership.assert_owned()


def test_kernel_lock_error_is_closed_and_reacquirable(
    tmp_path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    real_flock = combine_module.fcntl.flock
    failed = False

    def fail_once(fd, operation):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(errno.EIO, "injected flock failure")
        return real_flock(fd, operation)

    monkeypatch.setattr(combine_module.fcntl, "flock", fail_once)
    with pytest.raises(RuntimeError, match="kernel publish-lock acquisition"):
        with combine_module._exclusive_publish_ownership(run_dir):
            pass
    assert not (run_dir / combine_module.CHIME_COMBINE_PUBLISH_LOCK_FILENAME).exists()
    with combine_module._exclusive_publish_ownership(run_dir) as ownership:
        ownership.assert_owned()


def test_hard_linked_publish_lock_cannot_truncate_external_file(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("outside remains intact", encoding="utf-8")
    lock_path = run_dir / combine_module.CHIME_COMBINE_PUBLISH_LOCK_FILENAME
    try:
        os.link(external, lock_path)
    except OSError as exc:
        pytest.skip(f"hard links unavailable on this filesystem: {exc}")

    with pytest.raises(RuntimeError, match="non-hard-linked"):
        with combine_module._exclusive_publish_ownership(run_dir):
            pass
    assert external.read_text(encoding="utf-8") == "outside remains intact"
    assert lock_path.read_text(encoding="utf-8") == "outside remains intact"


def test_invalid_journal_preserves_possible_recovery_staging(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    staging, _outputs = _staged_generation(run_dir)
    (run_dir / combine_module.CHIME_COMBINE_PUBLISH_JOURNAL_FILENAME).write_text(
        "{not valid json\n", encoding="utf-8"
    )

    assert combine_module._cleanup_unreferenced_staging(
        run_dir, staging
    ) is False
    assert staging.is_dir()
