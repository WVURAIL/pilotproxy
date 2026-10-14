#!/usr/bin/env python3
"""
Regression tests for the engine/resume hardening (the external review findings).

Each test pins a behavior that previously could silently violate one of the
headline guarantees -- bounded scratch, product compatibility, "resume means
complete" -- or a source-selection / readiness bug. They run fully offline.

Coverage map (review finding -> tests):
  #1 bounded scratch          -> test_staged_file_bound_* , scratch-empty asserts
  #2 unique scratch names      -> test_stage_name_unique , test_duplicate_basenames_*
  #3/#4 resume compatibility   -> the detector analyzer's resume tests (test_detector_resume.py)
  #5 analysis order            -> test_default_delivers_source_order
  #7 local freq_id selection   -> test_local_source_44_not_844 , _duplicate_basenames_*
  #8 doctor readiness          -> test_geometry_only_telescope_not_archive_ready
  #9 failure boundary          -> test_reader_iteration_failure_quarantines_and_aborts,
                                  test_analyzer_failure_aborts_without_quarantine_and_cleans_scratch
  #10 quarantine identity      -> test_quarantine_uses_stable_unit_identity

Run:  PYTHONPATH=src python -m pytest tests/test_engine_safety.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

from pilot_proxy.archive import instruments as inst_mod
from pilot_proxy.archive import pipeline
from pilot_proxy.archive.pipeline import (_append_quarantine, _load_quarantine,
                                  _quarantine_key, _stage_name)
from pilot_proxy.archive.interfaces import (DataSource, Reader, Analyzer, RunContext, Unit,
                                  PluginInfo, READY, STREAM_COMPLEX_BASEBAND,
                                  UnreadableUnitError)
from pilot_proxy.chime.baseband_format import NFFT, make_synth_file
from pilot_proxy.chime.baseband_reader import ChimeBasebandReader
from pilot_proxy.archive.sources.local import LocalDirectorySource

F_TONE_BB = 12000.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tmp(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=f"dt_{prefix}_")


class _FakeSynthSource(DataSource):
    """A source that synthesizes a baseband file per unit on fetch().

    Lets a test control unit identity (key/name) and fetch timing independently
    of the local filesystem, and records the peak number of files co-resident on
    scratch so the storage bound can be asserted.
    """
    info = PluginInfo(name="fake", kind="source",
                      summary="synthetic in-test source", status=READY,
                      instruments=("*",))

    def __init__(self, fcen_hz, n_frames=4, delay_keys=()):
        self._fcen = fcen_hz
        self._nframes = n_frames
        self._delay = set(delay_keys)
        self._lock = threading.Lock()
        self.max_on_disk = 0
        self.fetch_order = []

    def enumerate(self, ctx):                       # not used by pipeline.run
        return []

    def fetch(self, unit, dest):
        if unit.key in self._delay:
            time.sleep(0.25)                        # make this unit "slow"
        make_synth_file(dest, self._nframes * NFFT, 16, self._fcen / 1e6,
                        F_TONE_BB, seed=(abs(hash(unit.key)) % 997) + 1)
        with self._lock:
            self.fetch_order.append(unit.key)
            n = len(os.listdir(os.path.dirname(dest)))
            self.max_on_disk = max(self.max_on_disk, n)
        return True, ""


class _RecordingAnalyzer(Analyzer):
    """Minimal analyzer that records the order files are delivered in."""
    info = PluginInfo(name="rec", kind="analyzer", summary="records delivery order",
                      status=READY, instruments=("*",),
                      accepts_stream_kinds=(STREAM_COMPLEX_BASEBAND,))

    def __init__(self):
        self.order = []

    def resume(self, path, ctx):
        return False

    def processed_keys(self):
        return set()

    def begin(self, ctx, first_meta):
        pass

    def consume_file(self, arrays, meta):
        n = len(list(arrays))                       # drain so the engine deletes it
        self.order.append(meta["unit_key"])
        return n

    def save(self, path):
        np.savez(path, order=np.array(self.order))

    def summary(self):
        return {"count": len(self.order)}


class _BrokenReader(Reader):
    """Yield one frame, then fail as a truncated streaming reader might."""
    info = PluginInfo(name="broken-reader", kind="reader",
                      summary="fails during iteration", status=READY,
                      instruments=("*",),
                      stream_kind=STREAM_COMPLEX_BASEBAND)

    def probe(self, path):
        return {}

    def iter_arrays(self, path, ctx):
        yield np.zeros(8, dtype=np.float32)
        raise UnreadableUnitError("truncated stream")


class _ExplodingAnalyzer(_RecordingAnalyzer):
    info = PluginInfo(name="explode", kind="analyzer",
                      summary="raises a science-code error", status=READY,
                      instruments=("*",),
                      accepts_stream_kinds=(STREAM_COMPLEX_BASEBAND,))

    def consume_file(self, arrays, meta):
        raise ValueError("science bug")


# ---------------------------------------------------------------------------
# #2  unique scratch names
# ---------------------------------------------------------------------------
def test_stage_name_unique_for_same_basename_different_key():
    u1 = Unit(key="src://A/baseband_dup_844.h5", name="baseband_dup_844.h5")
    u2 = Unit(key="src://B/baseband_dup_844.h5", name="baseband_dup_844.h5")
    assert _stage_name(u1) != _stage_name(u2)
    # the human-readable basename is preserved as a suffix for provenance
    assert _stage_name(u1).endswith("baseband_dup_844.h5")


def test_duplicate_basenames_distinct_keys_both_processed():
    """Two units sharing a basename must not collide on scratch (provenance + data)."""
    work = _tmp("dup")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    u1 = Unit(key="src://A/baseband_dup_844.h5", name="baseband_dup_844.h5",
              meta={"f_center_hz": fcen})
    u2 = Unit(key="src://B/baseband_dup_844.h5", name="baseband_dup_844.h5",
              meta={"f_center_hz": fcen})
    src = _FakeSynthSource(fcen, n_frames=4)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "dup.npz"); tmp = os.path.join(work, "tmp")
    analyzer = _RecordingAnalyzer()
    res = pipeline.run(source=src, reader=ChimeBasebandReader(),
                       analyzer=analyzer, units=[u1, u2], out_path=out,
                       tmp_dir=tmp, ctx=ctx, verbose=False)
    assert res.n_new == 2
    assert sorted(analyzer.order) == sorted([u1.key, u2.key])  # both, distinct
    assert os.listdir(tmp) == []                               # scratch clean


# ---------------------------------------------------------------------------
# #1  bounded scratch
# ---------------------------------------------------------------------------
def _bound_run(work, n_units, workers, staged):
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    units = [Unit(key=f"src://{i}/f_{i}_844.h5", name=f"f_{i}_844.h5",
                  meta={"f_center_hz": fcen}) for i in range(n_units)]
    src = _FakeSynthSource(fcen, n_frames=2)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "b.npz"); tmp = os.path.join(work, "tmp")
    pipeline.run(source=src, reader=ChimeBasebandReader(),
                 analyzer=_RecordingAnalyzer(), units=units, out_path=out,
                 tmp_dir=tmp, ctx=ctx, download_workers=workers,
                 max_staged_files=staged, verbose=False)
    return src.max_on_disk, tmp


def test_staged_file_bound_one_by_default():
    """Even with 4 download workers, the default holds exactly one file on scratch."""
    work = _tmp("bound1")
    peak, tmp = _bound_run(work, n_units=6, workers=4, staged=1)
    assert peak <= 1
    assert os.listdir(tmp) == []


def test_staged_file_bound_respects_max_staged_files():
    """Raising the bound to N never exceeds N files on scratch."""
    work = _tmp("bound3")
    peak, tmp = _bound_run(work, n_units=8, workers=4, staged=3)
    assert peak <= 3
    assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# #5  analysis order under the default
# ---------------------------------------------------------------------------
def test_default_delivers_source_order_even_if_later_file_is_faster():
    work = _tmp("order")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    u1 = Unit(key="k1", name="f1_844.h5", meta={"f_center_hz": fcen})
    u2 = Unit(key="k2", name="f2_844.h5", meta={"f_center_hz": fcen})
    # u1 is slow, u2 is fast -- under the default (1 worker) u1 still arrives first.
    src = _FakeSynthSource(fcen, n_frames=2, delay_keys={"k1"})
    analyzer = _RecordingAnalyzer()
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "o.npz"); tmp = os.path.join(work, "tmp")
    pipeline.run(source=src, reader=ChimeBasebandReader(), analyzer=analyzer,
                 units=[u1, u2], out_path=out, tmp_dir=tmp, ctx=ctx, verbose=False)
    assert analyzer.order == ["k1", "k2"]


# ---------------------------------------------------------------------------
# #9  reader failures are file-level; analyzer failures are run-level
# ---------------------------------------------------------------------------
def test_reader_iteration_failure_quarantines_and_aborts():
    work = _tmp("reader_fail")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    unit = Unit(key="src://bad", name="bad_844.h5", meta={"f_center_hz": fcen})
    src = _FakeSynthSource(fcen, n_frames=2)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "reader.npz")
    tmp = os.path.join(work, "tmp")
    quarantine = os.path.join(work, "quarantine.jsonl")

    with pytest.raises(RuntimeError, match="not checkpointed"):
        pipeline.run(source=src, reader=_BrokenReader(),
                     analyzer=_RecordingAnalyzer(), units=[unit], out_path=out,
                     tmp_dir=tmp, ctx=ctx, quarantine_path=quarantine,
                     verbose=False)

    assert os.listdir(tmp) == []
    assert not os.path.exists(out)
    with open(quarantine) as fh:
        assert "truncated stream" in fh.read()

    # The next run skips the quarantined unit and can resume from the last clean
    # checkpoint (there is no product in this one-file case).
    res = pipeline.run(source=src, reader=_BrokenReader(),
                       analyzer=_RecordingAnalyzer(), units=[unit], out_path=out,
                       tmp_dir=tmp, ctx=ctx, quarantine_path=quarantine,
                       verbose=False)
    assert res.n_new == 0


def test_analyzer_failure_aborts_without_quarantine_and_cleans_scratch():
    work = _tmp("analyzer_fail")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    units = [Unit(key=f"src://{i}", name=f"f_{i}_844.h5",
                  meta={"f_center_hz": fcen}) for i in range(4)]
    src = _FakeSynthSource(fcen, n_frames=2)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "analyzer.npz")
    tmp = os.path.join(work, "tmp")
    quarantine = os.path.join(work, "quarantine.jsonl")

    with pytest.raises(RuntimeError, match="science bug"):
        pipeline.run(source=src, reader=ChimeBasebandReader(),
                     analyzer=_ExplodingAnalyzer(), units=units, out_path=out,
                     tmp_dir=tmp, ctx=ctx, quarantine_path=quarantine,
                     download_workers=2, max_staged_files=2, verbose=False)

    assert not os.path.exists(quarantine)
    assert not os.path.exists(out)
    assert os.path.isdir(tmp)
    assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# #7  local source freq_id selection
# ---------------------------------------------------------------------------
def _touch(path):
    with open(path, "wb") as fh:
        fh.write(b"\x00")


def test_local_source_44_not_844():
    work = _tmp("sel")
    _touch(os.path.join(work, "baseband_s0_44.h5"))
    _touch(os.path.join(work, "baseband_s0_844.h5"))
    inst = inst_mod.load_instrument("chime")
    ctx = RunContext(instrument=inst, selection=[44],
                     options={"source_root": work, "source_glob": "*.h5"})
    units = list(LocalDirectorySource().enumerate(ctx))
    names = sorted(u.name for u in units)
    assert names == ["baseband_s0_44.h5"]              # 844 must NOT match 44
    assert all(u.meta["scope"] == "local" for u in units)


def test_local_source_duplicate_basenames_distinct_keys():
    work = _tmp("dupsel")
    a = os.path.join(work, "a"); b = os.path.join(work, "b")
    os.makedirs(a); os.makedirs(b)
    _touch(os.path.join(a, "baseband_x_844.h5"))
    _touch(os.path.join(b, "baseband_x_844.h5"))
    inst = inst_mod.load_instrument("chime")
    ctx = RunContext(instrument=inst, selection=[844],
                     options={"source_root": work, "source_glob": "*.h5"})
    units = list(LocalDirectorySource().enumerate(ctx))
    assert len(units) == 2                             # both subdirs enumerated
    assert len({u.key for u in units}) == 2            # distinct keys
    assert len({_stage_name(u) for u in units}) == 2   # -> distinct scratch names


def test_local_source_rejects_bytes_changed_after_enumeration():
    work = _tmp("local_changed")
    source_path = os.path.join(work, "baseband_x_844.h5")
    _touch(source_path)
    inst = inst_mod.load_instrument("chime")
    ctx = RunContext(
        instrument=inst,
        selection=[844],
        options={"source_root": work, "source_glob": "*.h5"},
    )
    source = LocalDirectorySource()
    unit = list(source.enumerate(ctx))[0]
    with open(source_path, "ab") as stream:
        stream.write(b"x")

    ok, error = source.fetch(unit, os.path.join(work, "staged.h5"))
    assert ok is False
    assert "changed after enumeration" in error


def test_quarantine_uses_stable_unit_identity_for_duplicate_basenames():
    work = _tmp("quarantine_identity")
    ledger = os.path.join(work, "quarantine.jsonl")
    bad = Unit(key="src://bad/same.h5", name="same.h5")
    good = Unit(key="src://good/same.h5", name="same.h5")

    _append_quarantine(ledger, bad, "bad header")
    keys = _load_quarantine(ledger)

    assert _quarantine_key(bad) in keys
    assert _quarantine_key(good) not in keys


# ---------------------------------------------------------------------------
# #8  doctor readiness
# ---------------------------------------------------------------------------
def test_geometry_only_telescope_not_archive_ready():
    """The readiness rule `doctor` enforces: a telescope with geometry + `nyquist_zone`
    but no declared baseband `scopes` is usable with a LOCAL source and counts as
    'geometry-only' -- not an out-of-the-box archive combo (it still works against an
    archive source if you pass --scope).

    Hermetic by construction -- it writes a throwaway geometry-only instrument and
    checks the readiness logic directly, so it depends neither on which shipped
    telescopes happen to declare scopes (all current ones do) nor on local CADC
    packages / proxy-cert state."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "geomonly.yaml"), "w") as f:
            f.write(
                "name: geomonly\n"
                "band: {f0_mhz: 800.0, bandwidth_mhz: 400.0, n_channels: 1024}\n"
                "nyquist_zone: 2\n"   # geometry + Nyquist zone set, but no scopes declared
            )
        rd = inst_mod.instrument_readiness("geomonly", directory=d)
        assert rd.nyquist_zone_set is True
        assert rd.scopes_set is False
        assert rd.ready is False
        assert rd.status == "geometry-only"


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q"])
    sys.exit(rc)


# ---------------------------------------------------------------------------
# #13 staging failures are retried, not quarantined: a staged copy that has
# vanished or is short of the archive's recorded size says nothing about the
# archived bytes (2026-09-01: a staging directory removed under a live scan
# quarantined 14 readable units).
# ---------------------------------------------------------------------------
class _VanishingReader(ChimeBasebandReader):
    """Loses the staged copy before it can read it."""

    def probe(self, path):
        os.remove(path)
        raise UnreadableUnitError(f"[Errno 2] Unable to open file {path}")


class _TruncatingReader(ChimeBasebandReader):
    """Finds a staged copy cut short."""

    def probe(self, path):
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) // 2)
        raise UnreadableUnitError("CHIME packed baseband is shorter than one transform")


def _synth_size(work, fcen):
    """Size of the file _FakeSynthSource will stage, to declare as the archive's."""
    probe = os.path.join(work, "size_probe.h5")
    make_synth_file(probe, 2 * NFFT, 16, fcen / 1e6, F_TONE_BB, seed=1)
    size = os.path.getsize(probe)
    os.remove(probe)
    return size


@pytest.mark.parametrize("reader_cls, declare_short", [
    (_VanishingReader, False),
    (_TruncatingReader, True),
])
def test_incomplete_staged_copy_is_a_transient_failure_not_a_quarantine(reader_cls, declare_short):
    work = _tmp("staging_failure")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    meta = {"f_center_hz": fcen}
    if declare_short:
        meta["size_bytes"] = _synth_size(work, fcen)   # the archive's size; the copy will be shorter
    unit = Unit(key="src://staged", name="staged_844.h5", meta=meta)
    src = _FakeSynthSource(fcen, n_frames=2)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "prod.npz")
    tmp = os.path.join(work, "tmp")
    quarantine = os.path.join(work, "quarantine.jsonl")

    res = pipeline.run(source=src, reader=reader_cls(), analyzer=_RecordingAnalyzer(),
                       units=[unit], out_path=out, tmp_dir=tmp, ctx=ctx,
                       quarantine_path=quarantine, verbose=False)
    assert res.n_failed == 1 and res.n_quarantined == 0
    assert not os.path.exists(quarantine) or open(quarantine).read().strip() == ""

    # The next run fetches it again and, read cleanly this time, completes it.
    res = pipeline.run(source=src, reader=ChimeBasebandReader(), analyzer=_RecordingAnalyzer(),
                       units=[unit], out_path=out, tmp_dir=tmp, ctx=ctx,
                       quarantine_path=quarantine, verbose=False)
    assert res.n_new == 1 and res.n_failed == 0 and res.n_quarantined == 0


def test_short_but_archive_sized_copy_is_still_quarantined():
    """A file the archive itself stores short is a property of the object."""
    work = _tmp("archive_short")
    inst = inst_mod.load_instrument("chime")
    fcen = inst.freq_of_freq_id(844) * 1e6
    unit = Unit(key="src://short", name="short_844.h5", meta={"f_center_hz": fcen, "size_bytes": 1})
    src = _FakeSynthSource(fcen, n_frames=2)
    ctx = RunContext(instrument=inst, selection=[844], options={})
    out = os.path.join(work, "prod.npz"); tmp = os.path.join(work, "tmp")
    quarantine = os.path.join(work, "quarantine.jsonl")
    res = pipeline.run(source=src, reader=_TruncatingReader(), analyzer=_RecordingAnalyzer(),
                       units=[unit], out_path=out, tmp_dir=tmp, ctx=ctx,
                       quarantine_path=quarantine, verbose=False)
    assert res.n_quarantined == 1 and res.n_failed == 0
    assert "shorter than one transform" in open(quarantine).read()
