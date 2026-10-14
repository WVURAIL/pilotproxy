from __future__ import annotations

from pathlib import Path

import pytest

from pilot_proxy.testbench import quantize


class _StopAfterPathResolution(RuntimeError):
    pass


def test_quantize_resolves_relative_io_against_caller_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: list[Path] = []

    def stop_at_read(path: Path):
        observed.append(path)
        raise _StopAfterPathResolution

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quantize, "_read_complex64", stop_at_read)

    with pytest.raises(_StopAfterPathResolution):
        quantize.main(
            [
                "--input-iq",
                "relative-input.cfile",
                "--output-dir",
                "relative-output",
            ]
        )

    assert observed == [(tmp_path / "relative-input.cfile").resolve()]
    assert (tmp_path / "relative-output").is_dir()
