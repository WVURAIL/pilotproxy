# coding=utf-8
"""A study must not invalidate its own run directory.

``_build_config`` hashes the configuration into ``config_sha256``, and
``_write_or_validate_config`` refuses any later stage whose digest differs.
Because git provenance came from ``git status --short``, that digest included
untracked paths -- and ``--stage prepare`` *creates* the output directory. The
documented production sequence, with the documented
``--output-dir results/current_geometry_sensitivity``, therefore died at stage
two with "output directory is bound to a different config".

Untracked build products are not scientific identity. Modified *tracked* sources
still are: shards must not be pooled across an edited working tree.
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
    """An edited working tree must still separate two studies."""
    clean = _identity_software(_software(["?? generated/"]))
    edited = _identity_software(
        _software(["?? generated/", " M src/pilot_proxy/detect.py"])
    )
    assert clean != edited
    assert edited["git"]["dirty"] is True
    assert edited["git"]["dirty_paths"] == [" M src/pilot_proxy/detect.py"]


def test_written_config_keeps_the_full_dirty_list() -> None:
    """Identity drops untracked paths; the recorded provenance keeps them."""
    software = _software(["?? generated/", " M src/pilot_proxy/detect.py"])
    ident = _identity_software(software)
    # The input mapping is not mutated -- the written config still shows both.
    assert software["git"]["dirty_paths"] == [
        "?? generated/",
        " M src/pilot_proxy/detect.py",
    ]
    assert ident["git"]["dirty_paths"] == [" M src/pilot_proxy/detect.py"]
