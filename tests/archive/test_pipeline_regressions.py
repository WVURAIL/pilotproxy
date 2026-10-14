"""Focused regressions for pipeline worker, scratch, and product outcomes."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from pilot_proxy.archive import pipeline
from pilot_proxy.archive.interfaces import (
    Analyzer,
    DataSource,
    PluginInfo,
    READY,
    Reader,
    RunContext,
    STREAM_COMPLEX_BASEBAND,
    Unit,
    UnreadableUnitError,
)

# A hang guard, NOT a rendezvous. Every wait that uses it is on an event some
# other thread is already committed to setting, so nothing here schedules work
# by wall clock and the ceiling is never reached; it only turns a genuine
# deadlock into a failure instead of a hung suite. The 2s waits this replaced
# were short enough to expire under CPU load (9 failures in 130 loaded runs).
_RENDEZVOUS_CEILING = 120.0


class _ByteSource(DataSource):
    info = PluginInfo(
        name="bytes",
        kind="source",
        summary="test byte source",
        status=READY,
        instruments=("*",),
    )

    def __init__(self, *, terminal: Optional[BaseException] = None):
        self.terminal = terminal
        self.fetch_calls: list[str] = []

    def enumerate(self, ctx):
        return []

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        with open(dest, "wb") as fh:
            fh.write(b"staged")
        if self.terminal is not None:
            raise self.terminal
        return True, ""


class _BlockingSource(_ByteSource):
    """Keep one fetch active while another unit reaches the analyzer."""

    def __init__(self):
        super().__init__()
        self.blocking_fetch_started = threading.Event()
        self.release_fetch = threading.Event()
        self.blocking_fetch_returned = threading.Event()

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-2":
            with open(dest, "wb") as fh:
                fh.write(b"partial")
            self.blocking_fetch_started.set()
            self.release_fetch.wait(timeout=10)
            self.blocking_fetch_returned.set()
            return True, ""
        assert self.blocking_fetch_started.wait(timeout=2)
        with open(dest, "wb") as fh:
            fh.write(b"staged")
        return True, ""


class _FailedResultSource(_ByteSource):
    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-1":
            return False, "temporary outage"
        with open(dest, "wb") as fh:
            fh.write(b"staged")
        return True, ""


class _OutOfOrderSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.first_started = threading.Event()
        self.second_finished = threading.Event()
        self.completed: list[str] = []
        self.lock = threading.Lock()

    def fetch(self, unit, dest):
        with self.lock:
            self.fetch_calls.append(unit.key)
        if unit.key == "unit-1":
            self.first_started.set()
            assert self.second_finished.wait(timeout=2)
            time.sleep(0.05)
        elif unit.key == "unit-2":
            assert self.first_started.wait(timeout=2)
        Path(dest).write_bytes(unit.key.encode())
        with self.lock:
            self.completed.append(unit.key)
        if unit.key == "unit-2":
            self.second_finished.set()
        return True, ""


class _LaterTransientFailureSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.third_finished = threading.Event()

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-2":
            assert self.third_finished.wait(timeout=2)
            time.sleep(0.05)
            return False, "temporary outage"
        Path(dest).write_bytes(unit.key.encode())
        if unit.key == "unit-3":
            self.third_finished.set()
        return True, ""


class _LaterFatalSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.third_failed = threading.Event()

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key in {"unit-1", "unit-2"}:
            assert self.third_failed.wait(timeout=2)
            time.sleep(0.05)
            Path(dest).write_bytes(unit.key.encode())
            return True, ""
        Path(dest).write_bytes(b"partial")
        self.third_failed.set()
        raise ValueError("source implementation bug")


class _HeadFailureWithBlockedPeerSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.peer_started = threading.Event()
        self.release_peer = threading.Event()
        self.peer_returned = threading.Event()

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-2":
            Path(dest).write_bytes(b"partial")
            self.peer_started.set()
            self.release_peer.wait(timeout=10)
            self.peer_returned.set()
            return True, ""
        assert self.peer_started.wait(timeout=2)
        return False, "temporary outage"


class _LaterFailureWithBlockedPeerSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.peer_started = threading.Event()
        self.release_peer = threading.Event()
        self.peer_returned = threading.Event()

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-3":
            Path(dest).write_bytes(b"partial")
            self.peer_started.set()
            self.release_peer.wait(timeout=10)
            self.peer_returned.set()
            return True, ""
        assert self.peer_started.wait(timeout=2)
        if unit.key == "unit-2":
            return False, "temporary outage"
        Path(dest).write_bytes(unit.key.encode())
        return True, ""


class _JoinablePeerSource(_LaterFailureWithBlockedPeerSource):
    """The blocked-peer source, plus the download worker thread running it.

    `peer_returned` is set from INSIDE fetch(), so it proves only that the
    source call is about to return. The worker still has to hand the result
    back, delete its staged file and release its slot before the thread ends,
    so a test that needs the worker to have FINISHED must join this thread.
    """

    def __init__(self):
        super().__init__()
        self.peer_thread: Optional[threading.Thread] = None

    def fetch(self, unit, dest):
        if unit.key == "unit-3":
            self.peer_thread = threading.current_thread()
        return super().fetch(unit, dest)


class _OutOfOrderTailSource(_ByteSource):
    def __init__(self):
        super().__init__()
        self.early_started = threading.Event()
        self.later_finished = threading.Event()
        self.completed: list[str] = []

    def fetch(self, unit, dest):
        self.fetch_calls.append(unit.key)
        if unit.key == "unit-2":
            self.early_started.set()
            assert self.later_finished.wait(timeout=2)
        elif unit.key == "unit-3":
            assert self.early_started.wait(timeout=2)
        Path(dest).write_bytes(unit.key.encode())
        self.completed.append(unit.key)
        if unit.key == "unit-3":
            self.later_finished.set()
        return True, ""


class _ByteReader(Reader):
    info = PluginInfo(
        name="byte-reader",
        kind="reader",
        summary="test byte reader",
        status=READY,
        instruments=("*",),
        stream_kind=STREAM_COMPLEX_BASEBAND,
    )

    def __init__(self, *, reject_probe: bool = False):
        self.reject_probe = reject_probe

    def probe(self, path):
        if self.reject_probe:
            raise UnreadableUnitError("bad header")
        return {}

    def iter_arrays(self, path, ctx):
        yield b"frame"


class _BuggyProbeReader(_ByteReader):
    def probe(self, path):
        raise ValueError("reader implementation bug")


class _FileAnalyzer(Analyzer):
    info = PluginInfo(
        name="file-analyzer",
        kind="analyzer",
        summary="test file analyzer",
        status=READY,
        instruments=("*",),
        accepts_stream_kinds=(STREAM_COMPLEX_BASEBAND,),
    )

    def __init__(self):
        self.keys: set[str] = set()

    def resume(self, path, ctx):
        return os.path.exists(path)

    def processed_keys(self):
        return self.keys

    def begin(self, ctx, first_meta):
        return None

    def consume_file(self, arrays, meta):
        list(arrays)
        self.keys.add(meta["unit_key"])
        return 1

    def save(self, path):
        with open(path, "w") as fh:
            fh.write("product")

    def summary(self):
        return {"count": len(self.keys)}


class _InvalidCountAnalyzer(_FileAnalyzer):
    def __init__(self, result):
        super().__init__()
        self.result = result

    def consume_file(self, arrays, meta):
        list(arrays)
        return self.result


class _ResumeResultAnalyzer(_FileAnalyzer):
    def __init__(self, result):
        super().__init__()
        self.result = result

    def resume(self, path, ctx):
        return self.result


class _PresetResumeAnalyzer(_FileAnalyzer):
    def __init__(self, keys):
        super().__init__()
        self.keys = set(keys)

    def resume(self, path, ctx):
        return True


class _OrderedPresetResumeAnalyzer(_PresetResumeAnalyzer):
    requires_in_order = True

    def __init__(self, keys):
        super().__init__(keys)
        self.order = list(keys)

    def processed_key_order(self):
        return list(self.order)


class _OrderedRecordingAnalyzer(_FileAnalyzer):
    requires_in_order = True

    def __init__(self):
        super().__init__()
        self.order: list[str] = []

    def processed_key_order(self):
        return list(self.order)

    def consume_file(self, arrays, meta):
        list(arrays)
        key = str(meta["unit_key"])
        self.keys.add(key)
        self.order.append(key)
        return 1

    def save(self, path):
        Path(path).write_text("\n".join(self.order))


class _ReloadingOrderedAnalyzer(_OrderedRecordingAnalyzer):
    def resume(self, path, ctx):
        product = Path(path)
        if not product.exists():
            return False
        self.order = product.read_text().splitlines()
        self.keys = set(self.order)
        return True


class _BlockingOrderedSaveAnalyzer(_OrderedRecordingAnalyzer):
    def __init__(self):
        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save(self, path):
        self.save_started.set()
        assert self.release_save.wait(timeout=_RENDEZVOUS_CEILING)
        super().save(path)


class _BlockingSaveAnalyzer(_FileAnalyzer):
    def __init__(self):
        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save(self, path):
        self.save_started.set()
        assert self.release_save.wait(timeout=5)
        super().save(path)


class _EarlyStopAnalyzer(_FileAnalyzer):
    def consume_file(self, arrays, meta):
        next(iter(arrays))
        self.keys.add(meta["unit_key"])
        return 1


class _CloseTrackingReader(_ByteReader):
    def __init__(self):
        super().__init__()
        self.closed = False

    def iter_arrays(self, path, ctx):
        with open(path, "rb") as fh:
            try:
                yield fh.read(1)
                yield fh.read(1)
            finally:
                self.closed = True


class _MetadataReader(_ByteReader):
    def probe(self, path):
        return {"f_center_hz": 123.0, "fs_hz": 456.0}


class _MetadataAnalyzer(_FileAnalyzer):
    def __init__(self):
        super().__init__()
        self.meta = None

    def consume_file(self, arrays, meta):
        list(arrays)
        self.meta = dict(meta)
        self.keys.add(meta["unit_key"])
        return 1


def _run(tmp_path, *, source, reader, analyzer=None, units=None, quarantine=None,
         workers=1, staged=1, **run_options):
    return pipeline.run(
        source=source,
        reader=reader,
        analyzer=analyzer or _FileAnalyzer(),
        units=units or [Unit(key="unit-1", name="unit-1.dat")],
        out_path=str(tmp_path / "product.out"),
        tmp_dir=str(tmp_path / "scratch"),
        ctx=RunContext(instrument=object(), options={}),
        quarantine_path=(str(quarantine) if quarantine is not None else None),
        download_workers=workers,
        max_staged_files=staged,
        verbose=False,
        **run_options,
    )


@pytest.mark.parametrize(
    "run_options",
    [
        {"checkpoint_every": 0},
        {"checkpoint_every": True},
        {"workers": 0},
        {"staged": -1},
        {"max_files": 0},
        {"max_frames_per_file": 1.5},
    ],
)
def test_direct_engine_limits_fail_closed(tmp_path, run_options):
    with pytest.raises(ValueError, match="positive integer"):
        _run(
            tmp_path,
            source=_ByteSource(),
            reader=_ByteReader(),
            **run_options,
        )


def test_duplicate_unit_keys_are_rejected_before_fetch(tmp_path):
    source = _ByteSource()
    units = [Unit(key="same", name="one.dat"),
             Unit(key="same", name="two.dat")]

    with pytest.raises(ValueError, match="duplicate unit key"):
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            units=units,
        )

    assert source.fetch_calls == []


def test_output_lock_excludes_writer_through_final_save(tmp_path):
    """A second run cannot resume stale state while the first is still saving."""
    first_analyzer = _BlockingSaveAnalyzer()
    first_errors = []

    def first_run():
        try:
            _run(
                tmp_path,
                source=_ByteSource(),
                reader=_ByteReader(),
                analyzer=first_analyzer,
            )
        except BaseException as exc:                         # noqa: BLE001
            first_errors.append(exc)

    background = threading.Thread(target=first_run)
    background.start()
    assert first_analyzer.save_started.wait(timeout=2)
    second_source = _ByteSource()
    try:
        with pytest.raises(
                pipeline.OutputLockedError,
                match=r"already locked.*Wait for that run to finish",
        ):
            _run(
                tmp_path,
                source=second_source,
                reader=_ByteReader(),
            )
        assert second_source.fetch_calls == []
    finally:
        first_analyzer.release_save.set()
        background.join(timeout=5)

    assert not background.is_alive()
    assert first_errors == []
    assert (tmp_path / "product.out").read_text() == "product"


def test_existing_product_is_preserved_when_analyzer_reports_no_resume(tmp_path):
    out = tmp_path / "product.out"
    out.write_text("valuable existing product")
    source = _ByteSource()

    with pytest.raises(RuntimeError, match=r"did not resume.*left unchanged"):
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=_ResumeResultAnalyzer(False),
        )

    assert source.fetch_calls == []
    assert out.read_text() == "valuable existing product"


def test_resume_rejects_keys_outside_current_source_scope(tmp_path):
    out = tmp_path / "product.out"
    out.write_text("valuable existing product")
    source = _ByteSource()

    with pytest.raises(SystemExit, match="outside the current source scope"):
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=_PresetResumeAnalyzer({"retired-unit"}),
            units=[Unit(key="current-unit", name="current.dat")],
        )

    assert source.fetch_calls == []
    assert out.read_text() == "valuable existing product"


def test_resume_scope_check_uses_units_before_max_files_slice(tmp_path):
    out = tmp_path / "product.out"
    out.write_text("existing product")
    source = _ByteSource()
    analyzer = _PresetResumeAnalyzer({"unit-2"})

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=analyzer,
        units=[
            Unit(key="unit-1", name="unit-1.dat"),
            Unit(key="unit-2", name="unit-2.dat"),
        ],
        max_files=1,
    )

    assert result.n_new == 1
    assert source.fetch_calls == ["unit-1"]


def test_ordered_resume_requires_current_sequence_prefix(tmp_path):
    out = tmp_path / "product.out"
    out.write_text("existing product")
    source = _ByteSource()

    with pytest.raises(SystemExit, match="does not match the current source scope"):
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=_OrderedPresetResumeAnalyzer(["unit-2"]),
            units=[
                Unit(key="unit-1", name="unit-1.dat"),
                Unit(key="unit-2", name="unit-2.dat"),
            ],
        )

    assert source.fetch_calls == []
    assert out.read_text() == "existing product"


def test_ordered_run_stops_at_first_transient_fetch_failure(tmp_path):
    source = _FailedResultSource()
    analyzer = _OrderedPresetResumeAnalyzer([])
    analyzer.resume = lambda path, ctx: False

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=analyzer,
        units=[
            Unit(key="unit-1", name="unit-1.dat"),
            Unit(key="unit-2", name="unit-2.dat"),
        ],
    )

    assert source.fetch_calls == ["unit-1"]
    assert result.n_done == 0
    assert result.n_failed == 1
    assert result.product_available is False
    assert not (tmp_path / "product.out").exists()


def test_ordered_run_stops_at_unreadable_unit_without_quarantine(tmp_path):
    source = _ByteSource()
    analyzer = _OrderedPresetResumeAnalyzer([])
    analyzer.resume = lambda path, ctx: False

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(reject_probe=True),
        analyzer=analyzer,
        units=[
            Unit(key="unit-1", name="unit-1.dat"),
            Unit(key="unit-2", name="unit-2.dat"),
        ],
    )

    assert source.fetch_calls == ["unit-1"]
    assert result.n_done == 0
    assert result.n_failed == 1
    assert result.product_available is False
    assert not (tmp_path / "product.out").exists()


def test_parallel_fetch_preserves_ordered_analysis(tmp_path):
    source = _OutOfOrderSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=analyzer,
        units=units,
        workers=2,
        staged=2,
        checkpoint_every=1,
    )

    assert source.completed[:2] == ["unit-2", "unit-1"]
    assert analyzer.order == [unit.key for unit in units]
    assert result.n_new == result.n_done == 3
    assert (tmp_path / "product.out").read_text().splitlines() == analyzer.order
    assert list((tmp_path / "scratch").iterdir()) == []


def test_ordered_failure_discards_later_download(tmp_path):
    source = _LaterTransientFailureSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=analyzer,
        units=units,
        workers=3,
        staged=3,
        checkpoint_every=1,
    )

    assert source.third_finished.is_set()
    assert analyzer.order == ["unit-1"]
    assert result.n_new == result.n_done == result.n_failed == 1
    assert (tmp_path / "product.out").read_text() == "unit-1"
    assert list((tmp_path / "scratch").iterdir()) == []


def test_later_worker_exception_keeps_ordered_prefix(tmp_path):
    source = _LaterFatalSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]

    with pytest.raises(
        RuntimeError,
        match=r"download worker terminated.*source implementation bug",
    ) as caught:
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=analyzer,
            units=units,
            workers=3,
            staged=3,
            checkpoint_every=1,
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert analyzer.order == ["unit-1", "unit-2"]
    assert (tmp_path / "product.out").read_text().splitlines() == analyzer.order
    assert list((tmp_path / "scratch").iterdir()) == []


def test_ordered_failure_reason_survives_active_peer(tmp_path, monkeypatch):
    source = _HeadFailureWithBlockedPeerSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key="unit-1", name="unit-1.dat"),
        Unit(key="unit-2", name="unit-2.dat"),
    ]
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.05)
    try:
        with pytest.raises(
                pipeline.ActiveDownloadWorkersError,
                match=r"Scratch was retained",
        ) as caught:
            _run(
                tmp_path,
                source=source,
                reader=_ByteReader(),
                analyzer=analyzer,
                units=units,
                workers=2,
                staged=2,
            )

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "unit-1.dat" in str(caught.value.__cause__)
        assert "temporary outage" in str(caught.value.__cause__)
        assert (scratch / pipeline._stage_name(units[1])).exists()
        assert not (tmp_path / "product.out").exists()
    finally:
        source.release_peer.set()

    assert source.peer_returned.wait(timeout=2)
    deadline = time.monotonic() + 2
    while any(scratch.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(scratch.iterdir()) == []


def test_ordered_prefix_is_saved_before_active_peer_error(tmp_path, monkeypatch):
    source = _LaterFailureWithBlockedPeerSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.05)
    try:
        with pytest.raises(pipeline.ActiveDownloadWorkersError) as caught:
            _run(
                tmp_path,
                source=source,
                reader=_ByteReader(),
                analyzer=analyzer,
                units=units,
                workers=3,
                staged=3,
                checkpoint_every=50,
            )

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "unit-2.dat" in str(caught.value.__cause__)
        assert analyzer.order == ["unit-1"]
        assert (tmp_path / "product.out").read_text() == "unit-1"
        assert (scratch / pipeline._stage_name(units[2])).exists()
    finally:
        source.release_peer.set()

    assert source.peer_returned.wait(timeout=2)
    deadline = time.monotonic() + 2
    while any(scratch.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(scratch.iterdir()) == []


def test_worker_finishing_during_prefix_save_returns_partial_result(
        tmp_path, monkeypatch):
    source = _JoinablePeerSource()
    analyzer = _BlockingOrderedSaveAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]
    release_errors = []

    def release_during_save():
        # Hold the ordered prefix save open until the blocked download worker
        # has genuinely FINISHED -- that is the scenario this test is named
        # for. Waiting on the thread rather than on peer_returned (which fires
        # while fetch() is still on the stack) is what makes the worker's exit
        # a fact instead of a race against however long save() happens to take.
        try:
            assert analyzer.save_started.wait(timeout=_RENDEZVOUS_CEILING)
            source.release_peer.set()
            assert source.peer_returned.wait(timeout=_RENDEZVOUS_CEILING)
            source.peer_thread.join(timeout=_RENDEZVOUS_CEILING)
            assert not source.peer_thread.is_alive()
        except BaseException as exc:                         # noqa: BLE001
            release_errors.append(exc)
        finally:
            analyzer.release_save.set()

    releaser = threading.Thread(target=release_during_save)
    releaser.start()
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.05)
    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=analyzer,
        units=units,
        workers=3,
        staged=3,
        checkpoint_every=50,
    )
    releaser.join(timeout=_RENDEZVOUS_CEILING)

    assert not releaser.is_alive()
    assert release_errors == []
    assert result.n_new == result.n_done == result.n_failed == 1
    assert analyzer.order == ["unit-1"]
    assert (tmp_path / "product.out").read_text() == "unit-1"
    assert list((tmp_path / "scratch").iterdir()) == []


def test_pending_ordered_file_cleanup_failure_is_surfaced(
        tmp_path, monkeypatch):
    source = _LaterTransientFailureSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]
    retained = tmp_path / "scratch" / pipeline._stage_name(units[2])
    real_remove = pipeline.os.remove

    def deny_later_delete(path):
        if os.path.abspath(path) == str(retained):
            raise PermissionError("file is busy")
        return real_remove(path)

    monkeypatch.setattr(pipeline.os, "remove", deny_later_delete)

    with pytest.raises(
            pipeline.StagedFileCleanupError,
            match=r"slot was not released",
    ) as caught:
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=analyzer,
            units=units,
            workers=3,
            staged=3,
            checkpoint_every=1,
        )

    assert source.third_finished.is_set()
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "unit-2.dat" in str(caught.value.__cause__)
    assert "temporary outage" in str(caught.value.__cause__)
    assert analyzer.order == ["unit-1"]
    assert (tmp_path / "product.out").read_text() == "unit-1"
    assert retained.exists()


def test_parallel_checkpoint_resume_preserves_exact_order(tmp_path):
    units = [
        Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
        for index in range(1, 4)
    ]
    first = _OrderedRecordingAnalyzer()

    result = _run(
        tmp_path,
        source=_LaterTransientFailureSource(),
        reader=_ByteReader(),
        analyzer=first,
        units=units,
        workers=3,
        staged=3,
        checkpoint_every=1,
    )

    assert result.n_new == 1
    assert first.order == ["unit-1"]

    source = _OutOfOrderTailSource()
    resumed = _ReloadingOrderedAnalyzer()
    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        analyzer=resumed,
        units=units,
        workers=2,
        staged=2,
        checkpoint_every=1,
    )

    assert source.completed == ["unit-3", "unit-2"]
    assert resumed.order == [unit.key for unit in units]
    assert len(resumed.order) == len(set(resumed.order))
    assert result.n_new == 2
    assert result.n_done == 3
    assert (tmp_path / "product.out").read_text().splitlines() == resumed.order
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("resume_result", [None, 0, ""])
def test_resume_result_must_be_an_exact_boolean(tmp_path, resume_result):
    source = _ByteSource()

    with pytest.raises(TypeError, match=r"must return exactly True or False"):
        _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=_ResumeResultAnalyzer(resume_result),
        )

    assert source.fetch_calls == []


def test_download_worker_count_is_clamped_to_staging_capacity_and_work(
        tmp_path, monkeypatch):
    real_thread = threading.Thread
    created = []

    def recording_thread(*args, **kwargs):
        worker = real_thread(*args, **kwargs)
        created.append(worker)
        return worker

    monkeypatch.setattr(pipeline.threading, "Thread", recording_thread)
    units = [Unit(key=f"unit-{index}", name=f"unit-{index}.dat")
             for index in range(3)]

    result = _run(
        tmp_path,
        source=_ByteSource(),
        reader=_ByteReader(),
        units=units,
        workers=50,
        staged=2,
    )

    assert result.n_new == 3
    assert len(created) == 2


def test_partial_worker_startup_retains_scratch_under_active_writer(
        tmp_path, monkeypatch):
    source = _BlockingSource()
    # Put the blocking unit first so the only successfully started worker is
    # demonstrably writing when creation of the second worker fails.
    units = [Unit(key="unit-2", name="unit-2.dat"),
             Unit(key="unit-1", name="unit-1.dat")]
    real_start = threading.Thread.start
    calls = 0

    def fail_second_start(worker):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert source.blocking_fetch_started.wait(timeout=2)
            raise RuntimeError("thread capacity exhausted")
        return real_start(worker)

    monkeypatch.setattr(pipeline.threading.Thread, "start", fail_second_start)
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.05)
    scratch = tmp_path / "scratch"
    try:
        with pytest.raises(
                pipeline.ActiveDownloadWorkersError,
                match=r"Scratch was retained",
        ) as caught:
            _run(
                tmp_path,
                source=source,
                reader=_ByteReader(),
                units=units,
                workers=2,
                staged=2,
            )

        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "thread capacity exhausted" in str(caught.value.__cause__)
        assert (scratch / pipeline._stage_name(units[0])).exists()
        assert not (tmp_path / "product.out").exists()
    finally:
        source.release_fetch.set()

    assert source.blocking_fetch_returned.wait(timeout=2)
    deadline = time.monotonic() + 2
    while any(scratch.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(scratch.iterdir()) == []
    deadline = time.monotonic() + 2
    while any(scratch.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("max_frames", [None, 1])
def test_reader_iterator_closes_before_staged_file_delete(
        tmp_path, monkeypatch, max_frames):
    reader = _CloseTrackingReader()
    scratch = tmp_path / "scratch"
    observed_closed = []
    real_remove = pipeline.os.remove

    def observe_remove(path):
        if os.path.dirname(os.path.abspath(path)) == str(scratch):
            observed_closed.append(reader.closed)
        return real_remove(path)

    monkeypatch.setattr(pipeline.os, "remove", observe_remove)
    result = _run(
        tmp_path,
        source=_ByteSource(),
        reader=reader,
        analyzer=_EarlyStopAnalyzer(),
        max_frames_per_file=max_frames,
    )

    assert result.n_new == 1
    assert reader.closed is True
    assert observed_closed == [True]


def test_reader_probe_metadata_is_authoritative_over_inventory(tmp_path):
    analyzer = _MetadataAnalyzer()
    unit = Unit(
        key="unit-1",
        name="unit-1.dat",
        meta={
            "f_center_hz": 999.0,
            "fs_hz": 888.0,
            "inventory_only": "preserved",
        },
    )

    _run(
        tmp_path,
        source=_ByteSource(),
        reader=_MetadataReader(),
        analyzer=analyzer,
        units=[unit],
    )

    assert analyzer.meta["f_center_hz"] == 123.0
    assert analyzer.meta["fs_hz"] == 456.0
    assert analyzer.meta["inventory_only"] == "preserved"


def test_delete_failure_is_surfaced_without_freeing_scratch_slot(
        tmp_path, monkeypatch):
    """An undeleted file remains the sole resident; the next fetch never starts."""
    source = _ByteSource()
    units = [Unit(key="unit-1", name="unit-1.dat"),
             Unit(key="unit-2", name="unit-2.dat")]
    scratch = tmp_path / "scratch"
    real_remove = pipeline.os.remove

    def deny_staged_delete(path):
        if os.path.dirname(os.path.abspath(path)) == str(scratch):
            raise PermissionError("file is busy")
        return real_remove(path)

    monkeypatch.setattr(pipeline.os, "remove", deny_staged_delete)

    with pytest.raises(pipeline.StagedFileCleanupError, match="slot was not released"):
        _run(tmp_path, source=source, reader=_ByteReader(), units=units,
             workers=1, staged=1)

    assert source.fetch_calls == ["unit-1"]
    assert [path.name for path in scratch.iterdir()] == [
        pipeline._stage_name(units[0])
    ]
    assert not (tmp_path / "product.out").exists()


def test_system_exit_in_fetch_is_reported_and_does_not_hang(tmp_path):
    source = _ByteSource(terminal=SystemExit("source requested exit"))

    with pytest.raises(RuntimeError, match=(
            r"download worker terminated.*SystemExit: source requested exit"
    )) as caught:
        _run(tmp_path, source=source, reader=_ByteReader())

    assert isinstance(caught.value.__cause__, SystemExit)
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not (tmp_path / "product.out").exists()


def test_ordinary_fetch_exception_is_not_downgraded_to_file_failure(tmp_path):
    source = _ByteSource(terminal=ValueError("source implementation bug"))

    with pytest.raises(
            RuntimeError,
            match=r"download worker terminated.*ValueError: source implementation bug",
    ) as caught:
        _run(tmp_path, source=source, reader=_ByteReader())

    assert isinstance(caught.value.__cause__, ValueError)
    assert list((tmp_path / "scratch").iterdir()) == []
    assert not (tmp_path / "product.out").exists()


def test_active_fetch_retains_scratch_and_reports_cancellation_timeout(tmp_path):
    source = _BlockingSource()
    units = [Unit(key="unit-1", name="unit-1.dat"),
             Unit(key="unit-2", name="unit-2.dat")]
    scratch = tmp_path / "scratch"
    started = time.monotonic()
    try:
        with pytest.raises(
                pipeline.ActiveDownloadWorkersError,
                match=r"source\.fetch\(\).*Scratch was retained",
        ) as caught:
            _run(
                tmp_path,
                source=source,
                reader=_ByteReader(),
                analyzer=_InvalidCountAnalyzer(0),
                units=units,
                workers=2,
                staged=2,
            )

        assert time.monotonic() - started < 4
        assert caught.value.scratch_dir == str(scratch)
        assert isinstance(caught.value.__cause__, pipeline.AnalyzerConsumptionError)
        assert scratch.is_dir()
        assert (scratch / pipeline._stage_name(units[1])).exists()
    finally:
        source.release_fetch.set()

    assert source.blocking_fetch_returned.wait(timeout=2)


def test_planned_checkpoint_stop_drains_active_fetch_and_resumes(
    tmp_path, monkeypatch
):
    source = _BlockingSource()
    analyzer = _OrderedRecordingAnalyzer()
    units = [
        Unit(key="unit-1", name="unit-1.dat"),
        Unit(key="unit-2", name="unit-2.dat"),
    ]
    product = tmp_path / "product.out"
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.01)

    def release_after_checkpoint():
        deadline = time.monotonic() + 2
        while not product.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert product.exists()
        time.sleep(0.05)
        source.release_fetch.set()

    release = threading.Thread(target=release_after_checkpoint)
    release.start()
    try:
        result = _run(
            tmp_path,
            source=source,
            reader=_ByteReader(),
            analyzer=analyzer,
            units=units,
            workers=2,
            staged=2,
            checkpoint_every=1,
            stop_after_checkpoint=True,
        )
    finally:
        source.release_fetch.set()
        release.join(timeout=2)

    assert not release.is_alive()
    assert source.blocking_fetch_returned.is_set()
    assert result.stopped_after_checkpoint is True
    assert result.n_new == result.n_done == 1
    assert product.read_text() == "unit-1"
    assert list((tmp_path / "scratch").iterdir()) == []

    resumed = _run(
        tmp_path,
        source=_ByteSource(),
        reader=_ByteReader(),
        analyzer=_ReloadingOrderedAnalyzer(),
        units=units,
        workers=2,
        staged=2,
    )

    assert resumed.stopped_after_checkpoint is False
    assert resumed.resumed is True
    assert resumed.n_new == 1
    assert resumed.n_done == 2
    assert product.read_text().splitlines() == ["unit-1", "unit-2"]
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("invalid_count", [0, -1, None, True, 1.5])
def test_invalid_analyzer_count_does_not_commit_empty_unit(
        tmp_path, invalid_count):
    analyzer = _InvalidCountAnalyzer(invalid_count)

    with pytest.raises(
            pipeline.AnalyzerConsumptionError,
            match=r"must return a positive integer.*not marked done",
    ):
        _run(
            tmp_path,
            source=_ByteSource(),
            reader=_ByteReader(),
            analyzer=analyzer,
        )

    assert analyzer.processed_keys() == set()
    assert not (tmp_path / "product.out").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


def test_all_newly_quarantined_reports_no_product(tmp_path, capsys):
    result = _run(
        tmp_path,
        source=_ByteSource(),
        reader=_ByteReader(reject_probe=True),
        quarantine=tmp_path / "quarantine.jsonl",
    )

    assert result.n_quarantined == 1
    assert result.product_available is False
    assert result.product_written is False
    assert result.resumed is False
    assert not (tmp_path / "product.out").exists()
    assert "product: not created" in capsys.readouterr().out


def test_unclassified_probe_error_aborts_without_quarantine(tmp_path):
    quarantine = tmp_path / "quarantine.jsonl"

    with pytest.raises(RuntimeError, match=(
            r"reader implementation bug.*not quarantined"
    )):
        _run(
            tmp_path,
            source=_ByteSource(),
            reader=_BuggyProbeReader(),
            quarantine=quarantine,
        )

    assert not quarantine.exists()
    assert not (tmp_path / "product.out").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("existing_product", [False, True])
def test_prequarantined_product_outcome_reflects_resume(
        tmp_path, existing_product, capsys):
    unit = Unit(key="unit-1", name="unit-1.dat")
    quarantine = tmp_path / "quarantine.jsonl"
    pipeline._append_quarantine(str(quarantine), unit, "known bad")
    out = tmp_path / "product.out"
    if existing_product:
        out.write_text("existing product")
    source = _ByteSource()

    result = _run(
        tmp_path,
        source=source,
        reader=_ByteReader(),
        units=[unit],
        quarantine=quarantine,
    )

    assert source.fetch_calls == []
    assert result.n_quarantined == 1
    assert result.product_available is existing_product
    assert result.product_written is False
    assert result.resumed is existing_product
    output = capsys.readouterr().out
    if existing_product:
        assert f"product: {out}" in output
    else:
        assert "no product exists" in output
        assert f"product: {out}" not in output


class _InterruptingAnalyzer(_FileAnalyzer):
    """Raises KeyboardInterrupt on the first unit, as an operator's Ctrl-C would."""

    def consume_file(self, arrays, meta):
        list(arrays)
        raise KeyboardInterrupt


def _release_after(source, seconds):
    threading.Timer(seconds, source.release_fetch.set).start()


def test_interrupt_drains_an_active_fetch_instead_of_retaining_scratch(tmp_path, monkeypatch):
    """An interrupt asks for a drain: a fetch in flight finishes within the
    interrupt window, so scratch is cleaned and no writer is left active."""
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 0.05)
    monkeypatch.setattr(pipeline, "_INTERRUPT_JOIN_SECONDS", 3.0)
    source = _BlockingSource()
    units = [Unit(key="unit-1", name="unit-1.dat"), Unit(key="unit-2", name="unit-2.dat")]
    _release_after(source, 0.4)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, source=source, reader=_ByteReader(), analyzer=_InterruptingAnalyzer(),
             units=units, workers=2, staged=2)
    assert source.blocking_fetch_returned.is_set()
    assert list((tmp_path / "scratch").iterdir()) == []


def test_interrupt_window_is_the_interrupt_one(tmp_path, monkeypatch):
    """With the interrupt window shorter than the fetch, the interrupt still
    reports the active worker and retains scratch -- the failure window is
    not what governs an interrupt."""
    monkeypatch.setattr(pipeline, "_WORKER_JOIN_SECONDS", 3.0)
    monkeypatch.setattr(pipeline, "_INTERRUPT_JOIN_SECONDS", 0.05)
    source = _BlockingSource()
    units = [Unit(key="unit-1", name="unit-1.dat"), Unit(key="unit-2", name="unit-2.dat")]
    try:
        with pytest.raises(pipeline.ActiveDownloadWorkersError, match=r"0\.05 seconds") as caught:
            _run(tmp_path, source=source, reader=_ByteReader(), analyzer=_InterruptingAnalyzer(),
                 units=units, workers=2, staged=2)
        assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    finally:
        source.release_fetch.set()
    assert source.blocking_fetch_returned.wait(timeout=2)
