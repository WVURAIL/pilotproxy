from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_proxy import cli
from pilot_proxy.archive.datatrail_client import Datatrail
from pilot_proxy.archive.inventory import inventory_meta_path
from pilot_proxy.archive.sources import CadcDatatrailSource
from pilot_proxy.chime.baseband_reader import ChimeBasebandReader


def _inventory_row(freq_id: int, event: str = "123456") -> dict[str, object]:
    return {
        "scope": "chime.event.baseband.raw",
        "event": event,
        "name": f"baseband_{event}_{freq_id}.h5",
        "size_bytes": 2048,
        "common_path": "cadc:CHIMEFRB/raw/2026/08/24/123456",
        "obs_date": "2026-08-24",
        "freq_id": freq_id,
    }


def test_chime_survey_writes_compatible_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "survey"

    monkeypatch.setattr(Datatrail, "installed", staticmethod(lambda: True))
    monkeypatch.setattr(
        Datatrail,
        "api_available",
        staticmethod(lambda: (True, "")),
    )
    monkeypatch.setattr(
        CadcDatatrailSource,
        "preflight",
        lambda self, ctx: (True, [], []),
    )

    def fake_survey(self, ctx, out):
        assert isinstance(ctx.reader, ChimeBasebandReader)
        path = Path(out) / "inventory.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_inventory_row(591)) + "\n", encoding="utf-8")
        return str(path)

    monkeypatch.setattr(CadcDatatrailSource, "survey", fake_survey)

    assert (
        cli.main(
            [
                "chime-survey",
                "--out",
                str(out_dir),
                "--name",
                "controls",
                "--freq-ids",
                "591",
            ]
        )
        == 0
    )

    meta = json.loads(
        inventory_meta_path(out_dir / "inventory.jsonl").read_text(encoding="utf-8")
    )
    assert meta["datatrawl_inventory"] == 1
    assert meta["name"] == "controls"
    assert meta["telescope"] == "chime"
    assert meta["source"] == "cadc-datatrail"
    assert meta["reader"] == "chime-baseband"


def test_chime_survey_rejects_two_output_roots(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="either --out or --root"):
        cli.main(
            [
                "chime-survey",
                "--out",
                str(tmp_path / "out"),
                "--root",
                str(tmp_path / "root"),
                "--dry-run",
            ]
        )


def test_chime_inventory_summarizes_archive_rows(
    tmp_path: Path,
    capsys,
) -> None:
    inventory = tmp_path / "inventory.jsonl"
    rows = [_inventory_row(591), _inventory_row(591, "123457"), _inventory_row(745)]
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert cli.main(["chime-inventory", "--inventory", str(inventory)]) == 0

    output = capsys.readouterr().out
    assert "files          : 3" in output
    assert "2 present (591..745)" in output
    assert "591      2" in output
    assert "745      1" in output
    assert "pilot-proxy chime-scan" in output
