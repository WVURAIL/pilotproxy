"""Find sustained changes in each channel's monthly pilot-region power.

Each month is given one level. For each event, the median of its frame-maximum
powers in the pilot neighborhood is taken, and the month's level is the median
of these values over the month's events. The levels are segmented by Bayesian
Blocks with the point-measure fitness and false-positive prior of Scargle et
al. (2013, ApJ 764, 167), equations 41 and 21, in the form used by astropy.

The monthly error is the counting error of the event median added in
quadrature to the channel's scatter in each era. That scatter is the robust
spread of the levels about a 13-month running median. It is added so that
seasonal and propagation changes do not split the series into extra blocks.

Power boundaries follow four rules:

1. The block-step sizes of every channel are sorted together. The largest gap
   in this list is the split between large transitions and other changes, and
   a step above that gap starts an era.
2. An excursion is merged away when its block is shorter than both neighbors
   and the level returns to its previous state. The levels before and after it
   must differ by less than the split.
3. Power alone sets these boundaries, so frequency drift does not cut an era.
   A change across a coverage gap is dated at the first month after the gap.
4. The current era extends from the last boundary to the last covered month.

With ``--line-identity``, changes in the pilot's frequency behavior are also
tested. At each block edge, the line is judged on the three covered months on
either side. The marker half-width, denoted h, is ``DESIGNATED_HALF_WIDTH``
fine bins, which is 23.84 Hz or one bin of the review's grid. A line is steady
when the median absolute deviation of its monthly offsets is at most h/2, and
unsteady when it exceeds h.

An edge is a line-boundary candidate if either:

* both sides are steady, their local median offsets differ by more than h,
  and the last-to-first covered-month jump exceeds h/2; or
* the line changes between steady and unsteady.

The jump condition excludes drift of one fine bin per month. In either case,
the power step must also exceed the largest step seen at edges where that
channel's line identity stays the same.

Reversing blocks, defined by the excursion rule above, never create line
boundaries. A reversing block shorter than the local window is also kept out
of the calibration, because it is below the method's resolution. Finally, a
channel whose steady offsets take just two values, each in at least two runs,
is treated as a two-state line, and its handovers do not split eras.

    pilot-proxy characterize era-boundaries --review DIR --output DIR [--previous ERA_LIST]
    pilot-proxy characterize era-boundaries --levels FILE --output DIR [--previous ERA_LIST]

``--review`` reads ``data/ch<N>.npz`` from the era review: each event's month,
level, and peak offset, plus monthly frame counts. Line-identity checks require
this input. ``--levels`` instead reads an existing ``monthly_levels.json``.
The outputs are ``monthly_levels.json``, ``blocks.json`` (steps, boundaries,
excursions, and split), and ``eras.json`` in the format of
``projects/chime_atsc/eras/power_eras_2026-10-04.json``. With
``--line-identity``, the monthly line offsets are also written to
``monthly_lines.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

P0 = 0.05                 # false-positive probability of each change point
MIN_EVENTS = 5            # a month with fewer events takes the channel's pooled spread
CODE_QUANTUM_DB = 0.005   # the stored dB codes are quantized at this step
SCATTER_HALF_WINDOW = 6   # months on each side of the running median (13 months, one seasonal cycle)
MAD_TO_SIGMA = 1.4826     # normal standard deviation per median absolute deviation
MEDIAN_ERROR = 1.2533     # sqrt(pi / 2): error of a median per error of a mean
CHANNELS = tuple(range(14, 38))
SCHEMA = "robust-power-era-threshold-cohorts-v1"
LOCAL_MONTHS = 3          # covered months on each side of a block edge the line is judged in
FINE_BIN_HZ = 390625 / 128 / 256  # the detector's fine bin, 11.92 Hz (a copy of products.reader.FINE_BIN_HZ, pinned by a test, so this file needs no package import)
MARKER_HALF_WIDTH_FINE_BINS = 2   # the detector's marker half-width h (narrowband_marker.anchors.DESIGNATED_HALF_WIDTH, pinned by a test)
MARKER_HALF_WIDTH_HZ = MARKER_HALF_WIDTH_FINE_BINS * FINE_BIN_HZ   # 23.84 Hz: the detector's fine bins are half the review's


def month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def month_name(index: int) -> str:
    return f"{int(index) // 12:04d}-{int(index) % 12 + 1:02d}"


def monthly_levels(event_month, event_level, months, frame_counts) -> list[dict]:
    """Return one row per month with at least one event, giving its level, event count, frame count, and event spread."""
    event_month, event_level = np.asarray(event_month), np.asarray(event_level, float)
    frames = dict(zip(map(int, months), map(int, frame_counts)))
    rows = []
    for month in sorted(frames):
        level = event_level[(event_month == month) & np.isfinite(event_level)]
        if not len(level):
            continue
        median = float(np.median(level))
        spread = float(MAD_TO_SIGMA * np.median(np.abs(level - median))) if len(level) >= MIN_EVENTS else None
        rows.append(dict(month=month_name(month), level_db=median, events=len(level), frames=frames[month],
                         spread_db=spread))
    return rows


def within_era_scatter(t, x) -> float:
    """Robust spread of the levels about a centered 13-month running median."""
    t, x = np.asarray(t), np.asarray(x, float)
    residual = [x[i] - np.median(x[np.abs(t - t[i]) <= SCATTER_HALF_WINDOW]) for i in range(len(t))]
    return float(MAD_TO_SIGMA * np.median(np.abs(residual)))


def errors(rows: list[dict]) -> tuple[np.ndarray, float]:
    """Each month's error (counting error of the median in quadrature with the within-era scatter) and the scatter."""
    t = np.asarray([month_index(r["month"]) for r in rows])
    x = np.asarray([r["level_db"] for r in rows], float)
    n = np.asarray([r["events"] for r in rows], float)
    spread = np.asarray([np.nan if r["spread_db"] is None else r["spread_db"] for r in rows], float)
    pooled = np.median(spread[n >= MIN_EVENTS])
    spread = np.maximum(np.where(n >= MIN_EVENTS, spread, pooled), CODE_QUANTUM_DB / np.sqrt(12))
    scatter = within_era_scatter(t, x)
    return np.hypot(MEDIAN_ERROR * spread / np.sqrt(n), scatter), scatter


def bayesian_blocks(x, sigma, p0: float = P0) -> list[int]:
    """Start index of every block of the optimal partition of point measures x with errors sigma
    (Scargle et al. 2013, section 2 and equation 41, with ncp_prior from equation 21)."""
    x, sigma = np.asarray(x, float), np.asarray(sigma, float)
    n = len(x)
    ncp_prior = 4 - np.log(73.53 * p0 * (n ** -0.478))
    a_raw, b_raw = np.ones_like(x) / sigma ** 2, x / sigma ** 2
    best, last = np.zeros(n), np.zeros(n, dtype=int)
    for r in range(n):
        a = 0.5 * np.cumsum(a_raw[:r + 1][::-1])[::-1]
        b = -np.cumsum(b_raw[:r + 1][::-1])[::-1]
        fitness = b * b / (4 * a) - ncp_prior
        fitness[1:] += best[:r]
        last[r] = int(np.argmax(fitness))
        best[r] = fitness[last[r]]
    starts, index = [], n
    while index > 0:
        index = int(last[index - 1])
        starts.append(index)
    return starts[::-1]


@dataclass(frozen=True)
class Block:
    first: int    # first covered month (month index)
    last: int     # last covered month
    level: float  # error-weighted mean level, dB


def blocks(rows: list[dict], p0: float = P0) -> tuple[list[Block], float]:
    t = np.asarray([month_index(r["month"]) for r in rows])
    x = np.asarray([r["level_db"] for r in rows], float)
    sigma, scatter = errors(rows)
    starts = bayesian_blocks(x, sigma, p0) + [len(x)]
    w = 1 / sigma ** 2
    return [Block(int(t[a]), int(t[b - 1]), float(np.sum(w[a:b] * x[a:b]) / np.sum(w[a:b])))
            for a, b in zip(starts[:-1], starts[1:])], scatter


def steps(channel_blocks: list[Block]) -> list[float]:
    return [b.level - a.level for a, b in zip(channel_blocks[:-1], channel_blocks[1:])]


def largest_gap(step_sizes) -> tuple[float, float]:
    """The largest gap between sorted step sizes: (largest below it, smallest above it)."""
    s = np.sort(np.abs(np.asarray(step_sizes, float)))
    if len(s) < 2:
        raise ValueError("the split needs at least two steps")
    i = int(np.argmax(np.diff(s)))
    return float(s[i]), float(s[i + 1])


def boundaries(channel_blocks: list[Block], upper: float) -> tuple[list[int], list[dict]]:
    """Indices of the blocks that start an era, and the excursions merged away."""
    cut = [k for k, d in enumerate(steps(channel_blocks), start=1) if abs(d) >= upper]
    merged = []
    while True:
        edges = [0] + cut + [len(channel_blocks)]
        eras = [(edges[j], edges[j + 1] - 1) for j in range(len(edges) - 1)]
        span = [channel_blocks[b].last - channel_blocks[a].first + 1 for a, b in eras]
        for j in range(1, len(eras) - 1):
            before, after = channel_blocks[eras[j - 1][1]], channel_blocks[eras[j + 1][0]]
            if span[j] < min(span[j - 1], span[j + 1]) and abs(after.level - before.level) < upper:
                merged.append(dict(first_month=month_name(channel_blocks[eras[j][0]].first),
                                   last_month=month_name(channel_blocks[eras[j][1]].last),
                                   months=span[j], level_before_db=before.level, level_after_db=after.level))
                cut = [k for k in cut if k not in (eras[j][0], eras[j + 1][0])]
                break
        else:
            return cut, merged


def months(first: int, last: int) -> list[str]:
    return [month_name(i) for i in range(first, last + 1)]


def monthly_lines(event_month, event_offset_hz, bin_hz: float) -> dict[int, dict]:
    """Per month with at least MIN_EVENTS events: the median peak offset and its median absolute deviation, in fine bins."""
    event_month = np.asarray(event_month)
    offset = np.round(2 * np.asarray(event_offset_hz, float) / bin_hz) / 2   # the review's peaks sit on a half-bin grid
    rows = {}
    for month in np.unique(event_month):
        o = offset[(event_month == month) & np.isfinite(offset)]
        if len(o) < MIN_EVENTS:
            continue
        median = float(np.median(o))
        rows[int(month)] = dict(median_bins=median, mad_bins=float(np.median(np.abs(o - median))), events=int(len(o)))
    return rows


def _local(lines: dict[int, dict], months: list[int]) -> dict | None:
    """Median offset and median MAD over the covered months of a window, in bins."""
    rows = [lines[m] for m in months if m in lines]
    if not rows:
        return None
    return dict(median_bins=float(np.median([r["median_bins"] for r in rows])),
                mad_bins=float(np.median([r["mad_bins"] for r in rows])), months=len(rows))


def reversing_blocks(channel_blocks: list[Block], upper: float) -> list[int]:
    """Indices of the blocks shorter than both neighbors where the levels on either side differ by less than the split."""
    span = [b.last - b.first + 1 for b in channel_blocks]
    return [j for j in range(1, len(channel_blocks) - 1)
            if span[j] < min(span[j - 1], span[j + 1])
            and abs(channel_blocks[j + 1].level - channel_blocks[j - 1].level) < upper]


def two_state_line(lines: dict[int, dict], h: float) -> bool:
    """True when the steady monthly offsets take exactly two values (to a tolerance of h/2) and each value occurs in at least two runs."""
    months = sorted(m for m, r in lines.items() if r["mad_bins"] <= h / 2)
    if len(months) < 4:
        return False
    values = np.array([lines[m]["median_bins"] for m in months])
    centres: list[float] = []
    label = []
    for v in values:
        for k, c in enumerate(centres):
            if abs(v - c) <= h / 2:
                label.append(k)
                break
        else:
            centres.append(float(v))
            label.append(len(centres) - 1)
    if len(centres) != 2:
        return False
    runs = [sum(1 for i in range(len(label)) if label[i] == k and (i == 0 or label[i - 1] != k)) for k in range(2)]
    return min(runs) >= 2


def line_edges(channel_blocks: list[Block], lines: dict[int, dict], h: float) -> list[dict]:
    """The line's state on each side of every block edge, and the kind of change it makes."""
    edges = []
    for j in range(1, len(channel_blocks)):
        a, b = channel_blocks[j - 1], channel_blocks[j]
        before = _local(lines, [m for m in range(a.last, a.first - 1, -1) if m in lines][:LOCAL_MONTHS])
        after = _local(lines, [m for m in range(b.first, b.last + 1) if m in lines][:LOCAL_MONTHS])
        last_before = next((m for m in range(a.last, a.first - 1, -1) if m in lines), None)
        first_after = next((m for m in range(b.first, b.last + 1) if m in lines), None)
        edge_move = (abs(lines[first_after]["median_bins"] - lines[last_before]["median_bins"])
                     if last_before is not None and first_after is not None else None)
        change = "unknown"
        if before is not None and after is not None:
            steady_a, steady_b = before["mad_bins"] <= h / 2, after["mad_bins"] <= h / 2
            unsteady_a, unsteady_b = before["mad_bins"] > h, after["mad_bins"] > h
            move = abs(after["median_bins"] - before["median_bins"])
            if steady_a and steady_b and move > h:
                # drift moves at most one fine bin (h/2) between covered months, so a larger single step is a jump
                change = "jump" if edge_move is not None and edge_move > h / 2 else "drift"
            elif (steady_a and unsteady_b) or (unsteady_a and steady_b):
                change = "steadiness"
            else:
                change = "same"
        edges.append(dict(block=j, month=month_name(b.first), step_db=b.level - a.level, change=change,
                          before=before, after=after, edge_move_bins=edge_move))
    return edges


def line_boundaries(channel_blocks: list[Block], lines: dict[int, dict], upper: float, h: float) -> tuple[list[int], dict]:
    """Block indices that start an era by the line's identity, and the evidence (edges, reversing blocks, calibration)."""
    reversing = reversing_blocks(channel_blocks, upper)
    touching = {j for r in reversing for j in (r, r + 1)}
    unresolved = {j for r in reversing if channel_blocks[r].last - channel_blocks[r].first + 1 < LOCAL_MONTHS for j in (r, r + 1)}
    two_state = two_state_line(lines, h)
    edges = line_edges(channel_blocks, lines, h)
    calibration = max([abs(e["step_db"]) for e in edges
                       if e["change"] in ("same", "drift") and e["block"] not in unresolved] or [0.0])
    cut = []
    for e in edges:
        j = e["block"]
        kind = e["change"]
        backed = abs(e["step_db"]) > calibration and j not in touching
        e["boundary"] = backed and ((kind == "jump" and not two_state) or kind == "steadiness")
        if e["boundary"]:
            cut.append(j)
    return cut, dict(edges=edges, calibration_db=calibration, two_state_line=two_state,
                     reversing_blocks=[dict(first_month=month_name(channel_blocks[j].first),
                                            last_month=month_name(channel_blocks[j].last),
                                            resolved=j not in unresolved) for j in reversing])


def era_boundaries(levels: dict[int, list[dict]], p0: float = P0, lines: dict[int, dict[int, dict]] | None = None,
                   h_bins: float | None = None) -> dict:
    """Blocks, the computed split, the boundaries and the current era of every channel."""
    fitted = {ch: blocks(rows, p0) for ch, rows in levels.items()}
    every_step = [d for b, _ in fitted.values() for d in steps(b)]
    lower, upper = largest_gap(every_step)
    channels = []
    for ch, (b, scatter) in fitted.items():
        cut, merged = boundaries(b, upper)
        row = dict(channel=ch, scatter_db=scatter,
                   blocks=[dict(first_month=month_name(x.first), last_month=month_name(x.last), level_db=x.level) for x in b])
        if lines is not None:
            line_cut, evidence = line_boundaries(b, lines[ch], upper, h_bins)
            row["line"] = dict(boundaries=[month_name(b[k].first) for k in line_cut], **evidence)
            cut = sorted(set(cut) | set(line_cut))
        start = b[cut[-1]].first if cut else b[0].first
        row.update(
            steps=[dict(month=month_name(x.first), step_db=d, boundary=k in cut)
                   for k, (x, d) in enumerate(zip(b[1:], steps(b)), start=1)],
            boundaries=[month_name(b[k].first) for k in cut], excursions=merged,
            current_first_month=month_name(start), current_last_month=month_name(b[-1].last))
        channels.append(row)
    result = dict(p0=p0, step_count=len(every_step), split=dict(largest_below_db=lower, smallest_above_db=upper), channels=channels)
    if lines is not None:
        result.update(line_identity=dict(marker_half_width_hz=MARKER_HALF_WIDTH_HZ, marker_half_width_review_bins=h_bins,
                                         local_months=LOCAL_MONTHS))
    return result


def era_list(result: dict, provenance: dict, previous: dict | None = None) -> dict:
    """Return the era list in the layout of the 2026-10-04 list, with each status relative to ``previous`` when it is given."""
    before = {r["channel"]: r["current_months"][0] for r in previous["channels"]} if previous else {}
    rows = []
    for r in result["channels"]:
        first = r["current_first_month"]
        status = ("current_era" if r["channel"] not in before else
                  "unchanged_current_era" if before[r["channel"]] == first else "revised_current_era")
        line = set(r.get("line", {}).get("boundaries", []))
        on_off = [m for m in r["boundaries"] if m not in line]
        basis = ("on_off_and_line_identity_transition" if on_off and line else "line_identity_transition" if line
                 else "on_off_transition" if on_off else "single_era_no_on_off_transition")
        rows.append(dict(channel=r["channel"],
                         current_months=months(month_index(first), month_index(r["current_last_month"])),
                         status=status, era_basis=basis, hold_reason=None))
    return dict(schema=SCHEMA, provenance=provenance, channels=rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels_from_review(review: Path, with_lines: bool = False):
    """Each channel's monthly levels from the era review and the sha256 of every file read.

    With ``with_lines``, each channel's monthly line offsets (the review's
    per-event peak offsets, in fine bins) and the marker half-width in review
    bins are also returned."""
    levels, lines, read, bin_hz = {}, {}, {}, None
    for ch in CHANNELS:
        path = Path(review) / "data" / f"ch{ch}.npz"
        read[f"data/ch{ch}.npz"] = _sha(path)
        with np.load(path, allow_pickle=False) as z:
            levels[ch] = monthly_levels(z["event_month"], z["event_peak_db"], z["months"], z["frame_counts"])
            if with_lines:
                step = float(z["sky_offset_hz"][1] - z["sky_offset_hz"][0])
                if bin_hz is not None and abs(step - bin_hz) > 1e-9:
                    raise ValueError("the review's offset grid differs between channels")
                bin_hz = step
                lines[ch] = monthly_lines(z["event_month"], z["event_peak_offset_hz"], bin_hz)
    return (levels, lines, read, MARKER_HALF_WIDTH_HZ / bin_hz) if with_lines else (levels, read)


def _write(path: Path, value) -> str:
    with path.open("x") as handle:
        json.dump(value, handle, indent=1, allow_nan=False)
        handle.write("\n")
    return _sha(path)


def _write_levels(path: Path, header: dict, levels: dict[int, list[dict]]) -> str:
    """monthly_levels.json with one month per line."""
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v, allow_nan=False)}," for k, v in header.items()] + [' "channels": {']
    for i, (ch, rows) in enumerate(levels.items()):
        lines.append(f'  "{ch}": [')
        lines += [f"   {json.dumps(row, allow_nan=False)}{',' if j < len(rows) - 1 else ''}" for j, row in enumerate(rows)]
        lines.append("  ]" + ("," if i < len(levels) - 1 else ""))
    with path.open("x") as handle:
        handle.write("\n".join(lines + [" }", "}", ""]))
    return _sha(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pilot-proxy characterize era-boundaries", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--review", type=Path, help="era-review delivery (data/ch<N>.npz)")
    source.add_argument("--levels", type=Path, help="monthly_levels.json of an earlier run")
    parser.add_argument("--previous", type=Path, help="era list the status is reported against")
    parser.add_argument("--p0", type=float, default=P0, help="false-positive probability per change point")
    parser.add_argument("--line-identity", action="store_true", help="also cut where the pilot line's identity changes (needs --review)")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    lines, h_bins = None, None
    if args.review:
        if args.line_identity:
            levels, lines, read, h_bins = levels_from_review(args.review, with_lines=True)
        else:
            levels, read = levels_from_review(args.review)
        header = dict(source=f"era review {args.review.name}", source_sha256=read)
    else:
        if args.line_identity:
            parser.error("--line-identity needs --review (the per-event peak offsets)")
        saved = json.loads(args.levels.read_text())
        levels = {int(ch): rows for ch, rows in saved["channels"].items()}
        header = dict(source=saved["source"], source_sha256=saved["source_sha256"])
    header["level"] = "median over the month's events of each event's frame-maximum power in the pilot neighbourhood, dB"
    header["spread"] = f"{MAD_TO_SIGMA} x median absolute deviation of the month's event levels; null below {MIN_EVENTS} events"
    args.output.mkdir(parents=True, exist_ok=False)
    level_sha = _write_levels(args.output / "monthly_levels.json", header, levels)
    result = era_boundaries(levels, args.p0, lines, h_bins)
    result["monthly_levels_sha256"] = level_sha
    if lines is not None:
        result["monthly_lines_sha256"] = _write(args.output / "monthly_lines.json",
                                                {str(ch): {month_name(m): r for m, r in rows.items()} for ch, rows in lines.items()})
    blocks_sha = _write(args.output / "blocks.json", result)
    previous = json.loads(args.previous.read_text()) if args.previous else None
    _write(args.output / "eras.json", era_list(result, dict(path="blocks.json", sha256=blocks_sha), previous))
    split = result["split"]
    print(f"split: {split['largest_below_db']:.2f} dB | {split['smallest_above_db']:.2f} dB over {result['step_count']} steps")
    for r in result["channels"]:
        line = f"  line {r['line']['boundaries'] or '-'}" if "line" in r else ""
        print(f"ch{r['channel']}: {r['current_first_month']} .. {r['current_last_month']}"
              f"  boundaries {r['boundaries'] or '-'}{line}  excursions {[e['first_month'] for e in r['excursions']] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
