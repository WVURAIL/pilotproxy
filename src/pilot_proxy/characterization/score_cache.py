"""Build a score cache from one authenticated fine-frame export.

Use the export's selected current-era population. No threshold is inferred from
the distribution; current_word is an explicit reference for replay checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def build(metadata, output, *, current_word=1):
    metadata = Path(metadata).resolve()
    source = metadata.with_suffix(".npz")
    meta = json.loads(metadata.read_text())
    require(sha(source) == meta["export_sha256"], "Fine export hash mismatch")
    require(type(current_word) is int and 1 <= current_word < 2**64, "Invalid current word")
    product = dict(meta["product"])
    product["path"] = str((metadata.parent / product["path"]).resolve())
    require(sha(product["path"]) == product["sha256"], "Product hash mismatch")
    product["bytes"] = Path(product["path"]).stat().st_size
    with np.load(source, allow_pickle=False) as z:
        selected = z["selected"] & z["finite_time"] & z["is_current"]
        rows = np.flatnonzero(selected)
        require(len(rows) > 0, "No selected current-era frames")
        require(np.array_equal(z["fine_source_row_index"], np.arange(len(selected))), "Export row order differs")
        units = z["frame_unit_index"]
        _, event, counts = np.unique(units[rows], return_inverse=True, return_counts=True)
        words = z["fine_required_multiplier_q16"][rows]
        require(words.dtype == np.dtype("uint64"), "Fine words must be uint64")
        support = z["fine_score_computed"][rows] & z["fine_rank_valid"][rows] & ~z["fine_always_masked"][rows]
        weight = 1.0 / (len(counts) * counts[event])
        months = sorted(set(int(m) for m in z["frame_month"][rows]))
        arrays = dict(source_row_index=rows, required_q16=words, supported=support,
                      probability_weight=weight, frame_event_index=event,
                      current_keep=support & (words <= current_word), frame_time=z["frame_time"][rows])
    output = Path(output)
    channel = meta["channel"]
    target = output / f"ch{channel}"
    require(not target.with_suffix(".json").exists() and not target.with_suffix(".npz").exists(), "Cache already exists")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix(".npz"), **arrays)
    result = dict(schema="pilotproxy-score-cache-v2", channel=channel,
                  original_product_identity=product, score_geometry=meta["geometry"],
                  current_threshold={"kind": "finite", "eta_q16": str(current_word)},
                  current_months=[f"{m // 12:04d}-{m % 12 + 1:02d}" for m in months],
                  frames=len(rows), events=len(counts), physical_science_qualified=False,
                  threshold_activated=False, npz_sha256=sha(target.with_suffix(".npz")),
                  source_export={"path": str(source), "sha256": sha(source)})
    write_json(target.with_suffix(".json"), result)
    return result

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--current-word", type=int, default=1)
    args = ap.parse_args(argv)
    result = build(args.metadata, args.output, current_word=args.current_word)
    print(json.dumps({"channel": result["channel"], "frames": result["frames"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
