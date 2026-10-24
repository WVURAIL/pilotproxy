"""Shared setup: the release's code (clones at the recorded commits), its era, blocks, anchor and bulk.

Release of record: results/archive_no_split_2026-09-24_r2 (RFIsher a281112, pilot-proxy bdfcdce,
RadioFisher 12a1616), the release the dissertation vendors. The clones under clones/ were checked
against the release's source_snapshots (no .py file differs).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLONES = HERE / "clones"
sys.path[:0] = [str(CLONES / "RFIsher/src"), str(CLONES / "RadioFisher"), str(CLONES / "pilot-proxy/src")]

import numpy as np  # noqa: E402

import rfisher  # noqa: E402
import rfisher_results  # noqa: E402
from rfisher_results.archive import anchors, blocks, eras, selection  # noqa: E402
from rfisher_results.archive.products import Product, fine_power_ratio  # noqa: E402

for _mod in (rfisher, rfisher_results):
    assert Path(_mod.__file__).resolve().is_relative_to(CLONES), _mod.__file__

RELEASE = Path("/home/djg/rail/results/archive_no_split_2026-09-24_r2")
ARCHIVE = RELEASE / "archive"
PLAN = json.loads((RELEASE / "analysis-plan.json").read_text())
PRODUCTS = Path(PLAN["products"])
REPLICATES, SEED = PLAN["bootstrap"]["replicates"], PLAN["bootstrap"]["seed"]
OVERRIDES = {int(c): [tuple(e) for e in v] for c, v in PLAN["era_overrides"]["channels"].items()}
SELECTION_ROWS = {int(r["channel"]): r for r in csv.DictReader(open(ARCHIVE / "tables/selection.csv", encoding="utf-8"))}
CHANNELS = list(range(14, 37))
TOL_CHANNELS = [18, 19, 21, 22, 31, 32, 33]


def ledger(ch: int) -> dict:
    return json.loads(next((ARCHIVE / "ledger/channels").glob(f"ch{ch:02d}_fid*.json")).read_text())


def campaign_last(config=None):
    config = config or eras.DEFAULT_CONFIG
    opened = [Product(p, require_health=True) for p in sorted(PRODUCTS.glob("*.npz"))]
    try:
        return eras.campaign_last_populated_month(opened, config)
    finally:
        for p in opened:
            p.close()


def check_run_identity(last, config=None):
    config = config or eras.DEFAULT_CONFIG
    run_json = json.loads((ARCHIVE / "ledger/run.json").read_text())
    assert blocks.month_label(last) == run_json["campaign_last_month"], (blocks.month_label(last), run_json["campaign_last_month"])
    assert config.digest == run_json["era_config_digest"]


def channel_setup(ch: int, last: int, config=None) -> dict:
    """Product, current era, calibration/evaluation split, selector anchor and bulk, as run.py builds them."""
    config = config or eras.DEFAULT_CONFIG
    rec = ledger(ch)
    s = rec["sections"]
    p = Product(PRODUCTS / rec["product"], require_health=True)
    assert p.geometry.physical_channel == ch
    table = eras.era_table(p, config=config, campaign_last_month=last, station_record={})
    if ch in OVERRIDES:
        table = eras.impose_eras(table, p, OVERRIDES[ch])
    timed = np.isfinite(p.frame_time)
    era_mask = table.current_era_mask(p) & timed
    split = blocks.split_blocks(p.frame_unit_index, p.unit_time, era_mask, frame_time=p.frame_time, minimum_months=1,
                                month_kwargs=dict(min_frames=config.min_frames, min_units=config.min_units, min_days=config.min_days))
    assert split.calibration_frames == s["blocks"]["calibration_frames"], (ch, split.calibration_frames, s["blocks"])
    assert split.evaluation_frames == s["blocks"]["evaluation_frames"], ch
    assert split.boundary_time == s["blocks"]["boundary_time"], ch
    # the selector's anchor (run.py step 7), which the ledger's null section records; selection.csv agrees
    anchor_bin, anchor_source = int(s["null"]["anchor_bin"]), "ledger null.anchor_bin"
    row = SELECTION_ROWS.get(ch)
    if row is not None:
        assert int(row["anchor_bin"]) == anchor_bin, (ch, row["anchor_bin"], anchor_bin)
        anchor_source = "selection.csv anchor_bin (= ledger null.anchor_bin)"
    bulk = anchors.bulk_mask(anchor_bin, pad_factor=p.fine_pad_factor, guard_fine_bins=p.fine_guard_bins,
                             census_excluded_bins=p.fine_census_excluded_bins)
    designated = np.array(anchors.designated_set(anchor_bin, selection.DESIGNATED_HALF_WIDTH), dtype=int)
    return dict(ch=ch, rec=rec, s=s, p=p, table=table, era_mask=era_mask, split=split, anchor_bin=anchor_bin,
                anchor_source=anchor_source, bulk=bulk, designated=designated,
                census=[int(b) for b in np.atleast_1d(p.fine_census_excluded_bins)])
