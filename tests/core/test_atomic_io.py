# coding=utf-8
from __future__ import annotations

import os
import stat

import pytest

from pilot_proxy.atomic_io import atomic_write_json


def test_atomic_json_new_file_honours_process_umask(tmp_path) -> None:
    probe = tmp_path / "mode-probe"
    previous_umask = os.umask(0o027)
    try:
        probe.write_bytes(b"")
    finally:
        os.umask(previous_umask)
    if stat.S_IMODE(probe.stat().st_mode) != 0o640:
        pytest.skip("filesystem does not expose normal POSIX umask semantics")
    probe.unlink()

    destination = tmp_path / "scan_scope.json"
    previous_umask = os.umask(0o027)
    try:
        atomic_write_json(destination, {"complete": False})
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
