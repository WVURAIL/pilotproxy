# coding=utf-8
"""Tests that a study's identity is unchanged by the files the study writes.

``_study_config`` hashes the configuration, and ``_write_or_validate_config``
requires later stages to use the same digest. The prepare stage creates the
output directory, so untracked files must not change that identity. Modified
tracked sources are still part of the identity, so that shards produced by
different code are never pooled as one study.
"""
from __future__ import annotations

from pathlib import Path
import runpy


DRIVER = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "tools" / "current_geometry_sensitivity.py")
)
_identity_software = DRIVER["_identity_software"]


def _software(dirty_paths):
    return {
        "git": {
            "commit": "0" * 40,
            "branch": "master",
            "dirty": bool(dirty_paths),
            "dirty_paths": list(dirty_paths),
        },
        "numpy": "2.4.2",
    }


def test_untracked_paths_do_not_bind_the_study() -> None:
    """Creating the run directory must not change the identity."""
    before = _identity_software(_software(["?? generated/"]))
    after = _identity_software(_software(["?? generated/", "?? results/"]))
    assert before == after


def test_untracked_only_tree_reads_as_clean_for_identity() -> None:
    ident = _identity_software(_software(["?? generated/"]))
    assert ident["git"]["dirty_paths"] == []
    assert ident["git"]["dirty"] is False


def test_modified_tracked_sources_still_bind() -> None:
    """An edited working tree must still give a different identity."""
    clean = _identity_software(_software(["?? generated/"]))
    edited = _identity_software(
        _software(["?? generated/", " M src/pilot_proxy/detect.py"])
    )
    assert clean != edited
    assert edited["git"]["dirty"] is True
    assert edited["git"]["dirty_paths"] == [" M src/pilot_proxy/detect.py"]


def test_written_config_keeps_the_full_dirty_list() -> None:
    """Untracked paths are dropped from the identity and kept in the config."""
    software = _software(["?? generated/", " M src/pilot_proxy/detect.py"])
    ident = _identity_software(software)
    # the input mapping is not mutated, so the written config lists both paths
    assert software["git"]["dirty_paths"] == [
        "?? generated/",
        " M src/pilot_proxy/detect.py",
    ]
    assert ident["git"]["dirty_paths"] == [" M src/pilot_proxy/detect.py"]
