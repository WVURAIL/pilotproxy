"""The release ``characterize archive`` writes is the one ``characterize histogram-frames`` reads."""
from __future__ import annotations

import json

from characterization_fixtures import characterize
from pilot_proxy.characterization import histogram_frames


def test_the_release_feeds_the_frame_export(tmp_path):
    summary, release = characterize(tmp_path)
    assert summary["channels"] == [29, 34]
    preflight = {row["filename"]: row
                 for row in json.loads((release / "products-preflight.json").read_text())["products"]}
    output = tmp_path / "frames"
    output.mkdir()
    for ledger_path in sorted((release / "ledger" / "channels").glob("ch*_fid*.json")):
        ledger = json.loads(ledger_path.read_text())
        eras = release / "channels" / f"ch{ledger['channel']}" / "eras.json"
        metadata = histogram_frames.process_channel(tmp_path / "products" / ledger["product"], ledger_path, eras,
                                                    output, preflight[ledger["product"]])
        assert metadata["channel"] == ledger["channel"]
        assert metadata["product"]["sha256"] == ledger["product_sha256"] == preflight[ledger["product"]]["sha256"]
        assert metadata["counts"]["selected"] == ledger["sections"]["product"]["n_selected"]
        assert metadata["counts"]["current_selected"] == ledger["sections"]["era"]["current_frames"] > 0
