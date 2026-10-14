"""``pilot-proxy characterize archive``: the release the frame export is keyed to.

Per screened band's product: its frame accounting and health reasons, and its
eras (the predeclared rule on the selected frames, replaced by the project's
author-dated era list where it dates the band). Nothing here sets a threshold.
The outputs are the files ``characterize histogram-frames --release`` reads:

    <out>/characterization/channels/chNN/eras.json
    <out>/characterization/ledger/{channels/chNN_fidF.json, run.json}
    <out>/characterization/products-preflight.json

    pilot-proxy characterize archive --project projects/chime_atsc --products DIR --out DIR
        [--bands 14,29] [--campaign-last-month YYYY-MM]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pilot_proxy.config._files import ProfileError
from pilot_proxy.config.project import Project, default_project, load_project
from pilot_proxy.products.reader import Product, sha256_of

from . import blocks, eras

LEDGER_SCHEMA = {"name": "pilot_proxy_characterization_ledger", "version": 2}


def producer_identity(roots: Mapping[str, Path] | None = None) -> dict:
    """The producing code, informational: its commit, dirty flag and a digest of the package sources.

    Delegates to :func:`pilot_proxy.provenance.producer_identity`.
    """
    from pilot_proxy.provenance import producer_identity as identity
    return identity(roots)


def _require_default_layers(project: Project) -> None:
    """Refuse a project whose register, detector configuration or instrument differ from the default project's.

    The era rule reads those parts from the default project when it is
    imported, so another project's values would otherwise be mixed silently
    with the default's. A project that differs only in its frequency plan,
    eras, transmitter-off record, interference template or integration model
    is threaded through and accepted.
    """
    default = default_project()
    theirs, ours = project.file_sha256(), default.file_sha256()
    differ = [k for k in ("register", "detector_config") if theirs[k] != ours[k]]
    if project.instrument_name != default.instrument_name:
        differ.append("instrument")
    if differ:
        raise ProfileError(f"project {project.name!r} at {project.directory}: its {', '.join(differ)} differ from the "
                           f"default project's, which the characterization binds at import; refused rather than mixed")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _month_index(label: str) -> int:
    y, m = (int(x) for x in label.split("-"))
    return y * 12 + m - 1


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=1) + "\n", encoding="utf-8")


def release_band(p: Product, out: Path, *, campaign_last_month: int, era_config: eras.EraConfig,
                 era_spec: Sequence[Sequence[str]] | None = None) -> dict:
    """Write one band's eras.json and return its ledger record and preflight row."""
    g = p.geometry
    ch = g.physical_channel
    months_all = eras.frame_months(p)
    sel_months = months_all[p.selected & (months_all >= 0)]
    product = {
        "n_frames": p.n_frames, "n_valid": int(p.valid.sum()), "n_selected": int(p.selected.sum()),
        "n_units": int(np.unique(p.frame_unit_index[p.selected]).size) if p.selected.any() else 0,
        "n_rejected_flag": int((p.selected & p.rejected).sum()),
        "n_with_shelf_estimate": int((p.selected & np.isfinite(p.shelf_db)).sum()),
        "n_without_time": int((p.selected & ~np.isfinite(p.frame_time)).sum()),
        "health_schema": p.health.schema, "health_excluded": int((p.valid & ~p.health.include).sum()),
        "health_reasons": ";".join(f"{k}:{v}" for k, v in sorted(p.health.reason_counts.items())),
        "first_month": blocks.month_label(int(sel_months.min())) if sel_months.size else "",
        "last_month": blocks.month_label(int(sel_months.max())) if sel_months.size else "",
        "per_frame_spectra": "psd_frame_db_i16" in p.archive.files,
    }
    table = eras.era_table(p, config=era_config, campaign_last_month=campaign_last_month, station_record={})
    if era_spec:
        # an author-dated era list replaces the rule's eras; the rule's month record is kept beside it
        table = eras.impose_eras(table, p, era_spec)
    eras.write_era_json(table, out / "channels" / f"ch{ch:02d}" / "eras.json")
    era = {k: v for k, v in eras.channel_row(table).items() if k not in ("channel", "freq_id")}
    digest = sha256_of(p.path)
    record = {"channel": ch, "freq_id": g.freq_id, "product": p.path.name, "product_sha256": digest,
              "sections": {"product": product, "era": era}, "notes": []}
    preflight = {"filename": p.path.name, "bytes": p.path.stat().st_size, "sha256": digest, "channel": ch,
                 "frames": int(p.n_frames), "valid": int(p.valid.sum()), "health_selected": int(p.selected.sum()),
                 "health_schema": p.health.schema}
    return {"record": record, "preflight": preflight}


def characterize_archive(products_dir: Path | str, out_dir: Path | str, *, project_dir: Path | str,
                         bands: Sequence[int] | None = None, campaign_last_month: str | None = None,
                         era_config: eras.EraConfig | None = None, generated: str | None = None) -> dict:
    """Write the release of every screened band's product (``bands`` restricts it)."""
    project = load_project(project_dir)
    _require_default_layers(project)
    products_dir, out = Path(products_dir), Path(out_dir) / "characterization"
    out.mkdir(parents=True, exist_ok=True)
    producer = producer_identity()                   # read once, before any work: the code that runs is the code named
    screened = {int(b.label) for b in project.frequency_plan.bands("screened")}
    opened = [Product(p, require_health=True) for p in sorted(products_dir.glob("*.npz"))]
    try:
        all_bands = {p.geometry.physical_channel: p for p in opened if p.geometry.physical_channel in screened}
        config = era_config or eras.DEFAULT_CONFIG
        # the campaign snapshot is the last populated month over every product present, whatever subset runs
        campaign_last = (_month_index(campaign_last_month) if campaign_last_month
                         else eras.campaign_last_populated_month(list(all_bands.values()), config))
        by_band = {c: all_bands[c] for c in bands} if bands else all_bands
        overrides = project.eras.overrides_spec()
        results = [release_band(p, out, campaign_last_month=campaign_last, era_config=config, era_spec=overrides.get(c))
                   for c, p in sorted(by_band.items())]
        products = {p.path.name: sha256_of(p.path) for c, p in sorted(all_bands.items())}
    finally:
        for p in opened:
            p.close()
    overrides_json = {str(c): [list(e) for e in v] for c, v in sorted(overrides.items())}
    generated = generated or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    paths = []
    for result in results:
        record = result["record"]
        path = f"channels/ch{record['channel']:02d}_fid{record['freq_id']}.json"
        _write_json(out / "ledger" / path, record)
        paths.append(path)
    _write_json(out / "ledger" / "run.json", {
        "schema": LEDGER_SCHEMA, "generated": generated, "products_dir": str(products_dir),
        "channels": paths, "campaign_last_month": blocks.month_label(campaign_last),
        "era_config": json.loads(config.canonical_json()), "era_config_digest": config.digest,
        "era_overrides": overrides_json,
        "era_overrides_sha256": hashlib.sha256(json.dumps(overrides_json, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest(),
        "producer": producer, "products": products})
    _write_json(out / "products-preflight.json",
                {"generated_utc": generated, "products": [result["preflight"] for result in results]})
    return {"channels": [result["record"]["channel"] for result in results], "out": str(out)}


def main(argv=None) -> int:
    import argparse

    from pilot_proxy.config.project import default_project_dir

    parser = argparse.ArgumentParser(prog="pilot-proxy characterize archive", description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, default=None, help="project profile directory (default: the repository's)")
    parser.add_argument("--products", type=Path, required=True, help="directory of per-pilot products (*.npz)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (characterization/ is written in it)")
    parser.add_argument("--bands", default=None, help="comma-separated band labels (default: every screened band)")
    parser.add_argument("--campaign-last-month", default=None, help="YYYY-MM of the campaign snapshot")
    args = parser.parse_args(argv)
    result = characterize_archive(
        args.products, args.out, project_dir=args.project or default_project_dir(),
        bands=[int(b) for b in args.bands.split(",")] if args.bands else None,
        campaign_last_month=args.campaign_last_month)
    print(json.dumps(result, indent=1, default=str), flush=True)
    return 0


__all__ = ["characterize_archive", "main", "producer_identity", "release_band"]
