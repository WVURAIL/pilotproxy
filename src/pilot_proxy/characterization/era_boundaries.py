"""Era boundaries at the switch-on and switch-off transitions of each channel's monthly power.

Each channel's monthly level (the median over the month's events of each event's
frame-maximum power in the pilot neighbourhood, the era review's reduction) is
segmented with Bayesian Blocks for point measures (Scargle et al. 2013, ApJ 764,
167: the fitness of eq. 41 and the false-positive prior of eq. 21, as astropy
implements them). A month's error is the counting error of a median of its
events added in quadrature to the channel's within-era scatter, the robust
spread of its levels about a 13-month running median, so that season and
propagation do not split blocks.

1. A change point is a boundary when its step lies above the largest gap in the
   sorted step sizes over all channels. The data separate switch-on and
   switch-off from every other level change; no step size is typed in.
2. An excursion is not a boundary: an era shorter than both its neighbours,
   across which the level returns to the prior state (the change from the block
   before it to the block after it falls below the split, among the level
   changes the scatter makes).
3. Frequency drift never cuts, since the series is power only. A change across a
   coverage gap is dated at the first covered month after it.
4. The current era runs from the last boundary to the last covered month.

    pilot-proxy characterize era-boundaries --review DIR --output DIR [--previous ERA_LIST]
    pilot-proxy characterize era-boundaries --levels FILE --output DIR [--previous ERA_LIST]

``--review`` is the era-review delivery (``data/ch<N>.npz``: each event's month
and level, each month's frame count); ``--levels`` a ``monthly_levels.json``
written by an earlier run. The output holds ``monthly_levels.json``,
``blocks.json`` (every block, step, boundary and excursion, and the split) and
``eras.json``, the era list in the layout of
``projects/chime_atsc/eras/power_eras_2026-10-04.json``.
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


def month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def month_name(index: int) -> str:
    return f"{int(index) // 12:04d}-{int(index) % 12 + 1:02d}"


def monthly_levels(event_month, event_level, months, frame_counts) -> list[dict]:
    """One row per month with at least one event: its level, events, frames and event spread."""
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
    """Robust spread of the levels about a centred 13-month running median."""
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
    (Scargle et al. 2013, sec. 2 and eq. 41; ncp_prior from eq. 21)."""
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


def era_boundaries(levels: dict[int, list[dict]], p0: float = P0) -> dict:
    """Blocks, the computed split, the boundaries and the current era of every channel."""
    fitted = {ch: blocks(rows, p0) for ch, rows in levels.items()}
    every_step = [d for b, _ in fitted.values() for d in steps(b)]
    lower, upper = largest_gap(every_step)
    channels = []
    for ch, (b, scatter) in fitted.items():
        cut, merged = boundaries(b, upper)
        start = b[cut[-1]].first if cut else b[0].first
        channels.append(dict(
            channel=ch, scatter_db=scatter,
            blocks=[dict(first_month=month_name(x.first), last_month=month_name(x.last), level_db=x.level) for x in b],
            steps=[dict(month=month_name(x.first), step_db=d, boundary=k in cut)
                   for k, (x, d) in enumerate(zip(b[1:], steps(b)), start=1)],
            boundaries=[month_name(b[k].first) for k in cut], excursions=merged,
            current_first_month=month_name(start), current_last_month=month_name(b[-1].last)))
    return dict(p0=p0, step_count=len(every_step), split=dict(largest_below_db=lower, smallest_above_db=upper),
                channels=channels)


def era_list(result: dict, provenance: dict, previous: dict | None = None) -> dict:
    """The era list, in the layout of the 2026-10-04 list; status is relative to ``previous`` when given."""
    before = {r["channel"]: r["current_months"][0] for r in previous["channels"]} if previous else {}
    rows = []
    for r in result["channels"]:
        first = r["current_first_month"]
        status = ("current_era" if r["channel"] not in before else
                  "unchanged_current_era" if before[r["channel"]] == first else "revised_current_era")
        rows.append(dict(channel=r["channel"],
                         current_months=months(month_index(first), month_index(r["current_last_month"])),
                         status=status,
                         era_basis="on_off_transition" if r["boundaries"] else "single_era_no_on_off_transition",
                         hold_reason=None))
    return dict(schema=SCHEMA, provenance=provenance, channels=rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def levels_from_review(review: Path) -> tuple[dict[int, list[dict]], dict[str, str]]:
    """Each channel's monthly levels from the era review, and the sha256 of every file read."""
    levels, read = {}, {}
    for ch in CHANNELS:
        path = Path(review) / "data" / f"ch{ch}.npz"
        read[f"data/ch{ch}.npz"] = _sha(path)
        with np.load(path, allow_pickle=False) as z:
            levels[ch] = monthly_levels(z["event_month"], z["event_peak_db"], z["months"], z["frame_counts"])
    return levels, read


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
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    if args.review:
        levels, read = levels_from_review(args.review)
        header = dict(source=f"era review {args.review.name}", source_sha256=read)
    else:
        saved = json.loads(args.levels.read_text())
        levels = {int(ch): rows for ch, rows in saved["channels"].items()}
        header = dict(source=saved["source"], source_sha256=saved["source_sha256"])
    header["level"] = "median over the month's events of each event's frame-maximum power in the pilot neighbourhood, dB"
    header["spread"] = f"{MAD_TO_SIGMA} x median absolute deviation of the month's event levels; null below {MIN_EVENTS} events"
    args.output.mkdir(parents=True, exist_ok=False)
    level_sha = _write_levels(args.output / "monthly_levels.json", header, levels)
    result = era_boundaries(levels, args.p0)
    result["monthly_levels_sha256"] = level_sha
    blocks_sha = _write(args.output / "blocks.json", result)
    previous = json.loads(args.previous.read_text()) if args.previous else None
    _write(args.output / "eras.json", era_list(result, dict(path="blocks.json", sha256=blocks_sha), previous))
    split = result["split"]
    print(f"split: {split['largest_below_db']:.2f} dB | {split['smallest_above_db']:.2f} dB over {result['step_count']} steps")
    for r in result["channels"]:
        print(f"ch{r['channel']}: {r['current_first_month']} .. {r['current_last_month']}"
              f"  boundaries {r['boundaries'] or '-'}  excursions {[e['first_month'] for e in r['excursions']] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
