"""Save one final fine score per frame beside an authenticated coarse export.

    pilot-proxy characterize fine-histogram-frames --frames DIR --output NEW_DIR

The score Z63 is max(T in the designated set) divided by the 63rd smallest
usable bulk T, where T = 2*S0/(S1+S2). It is one median-normalized score per
frame, and the export holds no per-bin F statistic. The saved current-era
anchor is reused for all archive frames, and the control uses its nominal
marker. No anchor or era is fitted here.

The marker geometry is set by two options of the detector v2 calibration,
``--designated-half-width`` h and ``--marker-guard-bins`` g
(:func:`anchors.bulk_mask`). With either one changed, the rank is the median
of the resulting bulk (``scores.candidate_rho_values``, the rule that gives 63
for the 125- and 126-bin bulks). The score is then Z<rho>, and the field
names carry that rank. The defaults (h = 2, g = 0) are the export of record.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from pilot_proxy.characterization import histogram_frames as coarse
from pilot_proxy.products import reader
from pilot_proxy.products.reader import FINE_BINS, HEALTH_GATE_SCHEMA, NFFT, Product

from . import anchors, scores

RHO = 63
WIDTH = 2
SCHEMA = "fine-histogram-frame-export-v1"
MANIFEST_SCHEMA = "fine-histogram-frames-manifest-v1"


def final_score_arrays(powers, selected, *, anchor_bin, bulk_mask, designated_half_width=WIDTH, rho=None):
    """Return the score arrays over all frames, with unselected frames left uncomputed.

    The exact keep boundaries come from the production scorer. The
    unquantized ratio is ordered and divided in exact rational arithmetic and
    converted to float64 only at the end. With ``rho`` None, the geometry of
    record is required (h = 2 and a 125- or 126-bin bulk, rank 63). Otherwise
    ``rho`` is the one-based rank.
    """
    powers = np.asarray(powers)
    selected = np.asarray(selected)
    bulk = np.asarray(bulk_mask)
    if powers.dtype != np.dtype(np.uint64) or powers.shape != (selected.size, 3, FINE_BINS):
        raise ValueError("fine powers must be uint64 with shape (frames, 3, 256)")
    if selected.dtype != np.dtype(bool) or selected.shape != (powers.shape[0],):
        raise ValueError("selected must be a boolean frame vector")
    if bulk.dtype != np.dtype(bool) or bulk.shape != (FINE_BINS,):
        raise ValueError("bulk_mask must be a boolean vector of length 256")
    if rho is None:
        if (designated_half_width != WIDTH or int(bulk.sum()) not in (125, 126)
                or scores.candidate_rho_values(int(bulk.sum())) != (RHO,)):
            raise ValueError("Z63 requires the intended 125- or 126-bin bulk")
        rho = RHO
    elif type(rho) is not int or not 1 <= rho <= int(bulk.sum()):
        raise ValueError("rho must be a one-based rank within the bulk")
    designated = scores._declared_bins(anchor_bin, designated_half_width)
    if bulk[designated].any():
        raise ValueError("bulk must exclude the designated bins")
    count = selected.size
    z = np.full(count, np.nan, dtype=np.float64)
    required = np.zeros(count, dtype=np.uint64)
    always = np.zeros(count, dtype=bool)
    rank_valid = np.zeros(count, dtype=bool)
    rank_zero = np.zeros(count, dtype=bool)
    bulk_count = np.zeros(count, dtype=np.uint16)
    designated_count = np.zeros(count, dtype=np.uint8)
    for row in np.flatnonzero(selected):
        frame = powers[row]
        denominator = [int(lo) + int(hi) for lo, hi in zip(frame[1], frame[2])]
        usable_bulk = [int(b) for b in np.flatnonzero(bulk) if denominator[b] > 0]
        usable_designated = [int(b) for b in designated if denominator[b] > 0]
        bulk_count[row] = len(usable_bulk)
        designated_count[row] = len(usable_designated)
        if len(usable_bulk) < rho:
            always[row] = True
            continue
        rank_valid[row] = True
        boundary = scores.required_eta_q16_by_rank(
            frame, anchor_bin=anchor_bin, designated_half_width=designated_half_width, bulk_mask=bulk)[rho - 1]
        always[row] = boundary == scores.ALWAYS_MASKED_Q16
        if not always[row]:
            required[row] = boundary
        ranked = sorted(Fraction(2 * int(frame[0, b]), denominator[b]) for b in usable_bulk)
        background = ranked[rho - 1]
        rank_zero[row] = background == 0
        numerator = max((Fraction(2 * int(frame[0, b]), denominator[b]) for b in usable_designated), default=Fraction(0))
        if background:
            z[row] = float(numerator / background)
        elif numerator:
            z[row] = np.inf
        else:
            # The strict detector comparison never fires in this case.
            z[row] = 0.0
    return {
        "fine_source_row_index": np.arange(count, dtype=np.int64),
        f"fine_score_z{rho}": z,
        "fine_score_computed": selected.copy(),
        "fine_score_finite": np.isfinite(z),
        "fine_score_infinite": np.isposinf(z),
        "fine_score_zero": np.isfinite(z) & (z == 0),
        "fine_rank_valid": rank_valid,
        "fine_rank_zero": rank_zero,
        "fine_usable_bulk_count": bulk_count,
        "fine_usable_designated_count": designated_count,
        "fine_required_multiplier_q16": required,
        "fine_always_masked": always,
    }


def _same(name, actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError(f"coarse frame field differs from source product: {name}")


def _check_file(path, digest):
    path = Path(path)
    if coarse.sha(path) != digest:
        raise ValueError(f"source hash differs: {path}")
    return path


def _member(root, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("export members must be filenames within the frames directory")
    return root / name


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _geometry(product, metadata, designated_half_width=WIDTH, marker_guard_bins=0):
    """Return the anchor, the bulk, the geometry record, and the rank of the export geometry.

    The rank is None for the geometry of record, which :func:`final_score_arrays`
    checks itself.
    """
    sources = metadata["source_records"]
    records = {name: json.loads(Path(record["path"]).read_text())
               for name, record in sources.items() if name == "ledger"}
    if metadata.get("role") == "control":
        if "ledger" in records or metadata.get("band", {}).get("role") != "control":
            raise ValueError("control geometry must come from a control export")
        anchor = int(product.geometry.nominal_fine_bin)
        anchor_source = "nominal product marker; no fitted control anchor"
        scope = "Nominal control geometry over the entire control period; no fitted anchor or false-alarm calibration."
    else:
        ledger = records["ledger"]
        if (ledger["channel"], ledger["freq_id"], ledger["product_sha256"]) != (
                metadata["channel"], metadata["freq_id"], metadata["product"]["sha256"]):
            raise ValueError("saved anchor ledger belongs to a different product")
        null = ledger["sections"]["null"]
        anchor = null["anchor_bin"]
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise ValueError("saved selector anchor must be an integer")
        anchor_source = null["anchor_source"]
        scope = "Saved current-era selector geometry applied unchanged to every era; no historical-era anchor refit."
    def bulk_for(width, guard):
        return anchors.bulk_mask(anchor, pad_factor=product.fine_pad_factor,
                                 guard_fine_bins=product.fine_guard_bins,
                                 census_excluded_bins=product.fine_census_excluded_bins,
                                 designated_half_width=width, marker_guard_bins=guard)

    record = designated_half_width == WIDTH and marker_guard_bins == 0
    # the saved selector's bulk size is that of the geometry of record
    if metadata.get("role") != "control" and int(null["bulk_size"]) != int(bulk_for(WIDTH, 0).sum()):
        raise ValueError("saved bulk size differs from product geometry")
    bulk = bulk_for(designated_half_width, marker_guard_bins)
    rho = None if record else scores.candidate_rho_values(int(bulk.sum()))[0]
    rank = RHO if record else rho
    geometry = {
        "anchor_bin": anchor, "anchor_source": anchor_source, "scope": scope,
        "rho_one_based": rank, "cfar_rank_zero_based": rank - 1,
        "designated_half_width": designated_half_width,
        "designated_bins": list(anchors.designated_set(anchor, designated_half_width)),
        "bulk_bins": np.flatnonzero(bulk).tolist(), "intended_bulk_size": int(bulk.sum()),
        "pad_factor": product.fine_pad_factor, "guard_fine_bins": product.fine_guard_bins,
        "census_excluded_bins": product.fine_census_excluded_bins.tolist(),
    }
    if marker_guard_bins:
        geometry["marker_guard_bins"] = marker_guard_bins
    return anchor, bulk, geometry, rho


def process_channel(frames, item, output, *, designated_half_width=WIDTH, marker_guard_bins=0):
    """Add final fine scores without changing any original coarse array."""
    metadata_path = _member(frames, item["metadata"])
    coarse_path = _member(frames, item["npz"])
    metadata = json.loads(metadata_path.read_text())
    if metadata["schema"] != "coarse-histogram-frame-export-v1":
        raise ValueError("expected a coarse frame export")
    if (metadata["channel"], metadata["freq_id"]) != (item["channel"], item["freq_id"]):
        raise ValueError("coarse manifest and channel metadata disagree")
    _check_file(coarse_path, metadata["export_sha256"])
    product_record = metadata["product"]
    product_path = _check_file(product_record["path"], product_record["sha256"])
    stat = product_path.stat()
    if stat.st_size != product_record["bytes"]:
        raise ValueError("source product byte count differs")
    for record in metadata["source_records"].values():
        _check_file(record["path"], record["sha256"])
    with np.load(coarse_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if any(key.startswith("fine_") for key in arrays):
        raise ValueError("input already contains fine export fields")
    with Product(product_path, require_health=True) as product:
        raw_archive = product.archive
        # The coarse exporter's archive wrapper is used, so the frame powers
        # are formed from the fine sums and the spectral members cannot be read.
        product._z = coarse.FrameExportArchive(raw_archive, product_path)
        if (product.view.physical_channel, product.view.freq_id) != (item["channel"], item["freq_id"]):
            raise ValueError("source product identity differs")
        if int(product.scalar("nfft")) != NFFT or metadata["health"]["schema"] != HEALTH_GATE_SCHEMA:
            raise ValueError("source frame geometry or health schema differs")
        for name, expected in {
            "Q": product.statistic, "valid": product.valid,
            "health_include": product.health.include, "selected": product.selected,
            "frame_index": product.frame_column("frame_index"),
            "frame_unit_index": product.frame_unit_index, "frame_in_unit": product.frame_in_unit,
            "frame_time": product.frame_time, "unit_time": product.unit_time,
            "unit_event_id": product.unit_event_id,
            "acquisition_id": product.unit_event_id[product.frame_unit_index],
            "unit_keys": product.archive["unit_keys"], "source_event_keys": product.archive["source_event_keys"],
        }.items():
            _same(name, arrays[name], expected)
        _same("finite_time", arrays["finite_time"], np.isfinite(product.frame_time))
        _same("is_current", arrays["is_current"], arrays["era_id"] == metadata["current_era"])
        anchor, bulk, geometry, rho = _geometry(product, metadata, designated_half_width, marker_guard_bins)
        rank = geometry["rho_one_based"]
        fine = final_score_arrays(raw_archive["fine_power_u64"], product.selected,
                                  anchor_bin=anchor, bulk_mask=bulk,
                                  designated_half_width=designated_half_width, rho=rho)
    fine["fine_histogram_eligible"] = arrays["selected"] & arrays["finite_time"] & fine["fine_score_finite"]
    fine["current_fine_histogram_eligible"] = fine["fine_histogram_eligible"] & arrays["is_current"]
    arrays.update(fine)
    counts = {key: int(value.sum()) for key, value in fine.items() if value.dtype == np.dtype(bool)}
    counts.update(total=int(arrays["selected"].size),
                  insufficient_bulk=int((arrays["selected"] & ~fine["fine_rank_valid"]).sum()),
                  no_designated_reference=int((arrays["selected"] & (fine["fine_usable_designated_count"] == 0)).sum()),
                  undefined_score=int((arrays["selected"] & np.isnan(fine[f"fine_score_z{rank}"])).sum()))
    final_stat = product_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("source product changed during extraction")
    for record in metadata["source_records"].values():
        _check_file(record["path"], record["sha256"])
    name = f"ch{item['channel']:02d}"
    output_path = output / f"{name}.npz"
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result = {
        "schema": SCHEMA, "channel": item["channel"], "freq_id": item["freq_id"],
        "role": metadata.get("role", "screened"), "product": product_record,
        "coarse_export": {"npz": {"path": str(coarse_path), "sha256": coarse.sha(coarse_path)},
                          "metadata": {"path": str(metadata_path), "sha256": coarse.sha(metadata_path)}},
        "source_records": metadata["source_records"], "geometry": geometry,
        "current_era": metadata["current_era"], "era_definitions": metadata["era_definitions"],
        "coarse_counts": metadata["counts"], "counts": counts,
        "statistic": f"Z{rank} = max designated T / {_ordinal(rank)} smallest usable bulk T; T = 2*S0/(S1+S2)",
        "statistic_scope": "Final median-normalized fine score, not a per-bin F statistic or a false-alarm certificate.",
        "ratio_rule": "Positive reference denominators only; exact rational ranking and division, then float64 conversion. Zero scores are retained. A valid zero rank with no positive usable designated power has score zero by the detector's non-firing convention; otherwise positive/zero is +inf. No usable designated reference also gives score zero and is counted explicitly. "
                      f"Fewer than {rank} usable bulk bins is NaN.",
        "histogram_rule": "fine_histogram_eligible = selected & finite_time & fine_score_finite; no positive-score cut or statistical trimming. Undefined and infinite selected scores are counted separately.",
        "exact_boundary": {
            "field": "fine_required_multiplier_q16", "scale": scores.Q16_SCALE,
            "logical_always_masked_sentinel": str(scores.ALWAYS_MASKED_Q16),
            "encoding": "uint64 boundary plus fine_always_masked; word is zero for always-masked or uncomputed frames. Interpret only where fine_score_computed is true.",
            "keep_rule": "On computed frames: not fine_always_masked and multiplier_q16 >= fine_required_multiplier_q16. The minimum deployable word is 1. Insufficient bulk is always-masked in the characterization scorer.",
            "float_warning": f"Use the exact boundary for masking; do not reconstruct it from float64 Z{rank}.",
        },
        "source_row_rule": "All arrays retain the original product row order; original coarse arrays, including frame and acquisition identifiers, are copied without alteration.",
        "per_bin_f_values_saved": False, "psd_payloads_decoded": False,
        "export_sha256": coarse.sha(output_path),
    }
    coarse.write_json(output / f"{name}.json", result)
    return result


def run(frames, output, *, designated_half_width=WIDTH, marker_guard_bins=0):
    frames, output = Path(frames).resolve(), Path(output).resolve()
    manifest_path = frames / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest["schema"] != "coarse-histogram-frames-manifest-v1" or source_manifest["passed"] is not True:
        raise ValueError("a completed coarse frame export is required")
    input_files = {str(manifest_path): coarse.sha(manifest_path)}
    for name, digest in source_manifest["files"].items():
        input_files[str(_check_file(_member(frames, name), digest))] = digest
    if source_manifest["extraction_plan_sha256"] != input_files.get(str(frames / "extraction-plan.json")):
        raise ValueError("coarse extraction plan hash differs")
    items = source_manifest["channels"]
    if not items or len({item["channel"] for item in items}) != len(items):
        raise ValueError("coarse manifest channels must be nonempty and unique")
    for item in items:
        for key in ("npz", "metadata"):
            if str(_member(frames, item[key])) not in input_files:
                raise ValueError("coarse channel file is absent from the source manifest")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("use a new empty fine frames directory")
    sources = {str(path): coarse.sha(path) for path in (
        Path(__file__).resolve(), Path(scores.__file__).resolve(), Path(anchors.__file__).resolve(),
        Path(reader.__file__).resolve(), *coarse.SOURCES)}
    plan = {
        "schema": "fine-histogram-export-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "coarse_manifest": str(manifest_path), "input_files": input_files, "source_files": sources,
        "python": sys.version, "numpy_version": np.__version__,
        "scope": "Read-only final-score export with saved geometry and unchanged coarse populations; no refit, null calibration, operating-point selection, or per-bin export.",
    }
    if designated_half_width != WIDTH or marker_guard_bins:
        plan["geometry_options"] = {"designated_half_width": designated_half_width,
                                    "marker_guard_bins": marker_guard_bins}
    coarse.write_json(output / "extraction-plan.json", plan)
    records = []
    for item in items:
        record = process_channel(frames, item, output, designated_half_width=designated_half_width,
                                 marker_guard_bins=marker_guard_bins)
        records.append(record)
        print(json.dumps({"channel": record["channel"], "counts": record["counts"]}), flush=True)
    for path, digest in {**input_files, **sources}.items():
        _check_file(path, digest)
    result = {
        "schema": MANIFEST_SCHEMA, "passed": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "extraction_plan_sha256": coarse.sha(output / "extraction-plan.json"),
        "channels": [{"channel": r["channel"], "freq_id": r["freq_id"], "role": r["role"],
                      "npz": f"ch{r['channel']:02d}.npz", "metadata": f"ch{r['channel']:02d}.json", "counts": r["counts"]}
                     for r in records],
        "totals": {key: sum(r["counts"][key] for r in records) for key in records[0]["counts"]},
        "files": {path.name: coarse.sha(path) for path in output.iterdir() if path.is_file()},
        "per_bin_f_values_saved": False, "psd_payloads_decoded": False,
    }
    coarse.write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pilot-proxy characterize fine-histogram-frames", description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="authenticated coarse histogram-frames directory")
    parser.add_argument("--output", type=Path, required=True, help="new, empty output directory")
    parser.add_argument("--designated-half-width", type=int, default=WIDTH,
                        help=f"marker half-width h in padded fine bins (default {WIDTH}, the export of record)")
    parser.add_argument("--marker-guard-bins", type=int, default=0,
                        help="bins next to the markers left out of the bulk (default 0, the export of record)")
    args = parser.parse_args(argv)
    run(args.frames, args.output, designated_half_width=args.designated_half_width,
        marker_guard_bins=args.marker_guard_bins)
    return 0
