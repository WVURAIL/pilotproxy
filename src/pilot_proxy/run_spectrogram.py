"""Fine-power spectrograms of a detect-only run, with and without the detector mask."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.bundle_calibration import (
    _expand,
    _rows,
    iter_run_frames,
    required_words,
)
from pilot_proxy.fine_decision import unpack_bulk_mask

# 8192 samples on the Pathfinder and 16384 on CHIME give the same block
FRAME_SECONDS = 0.04194304
TARGET_ROW = 0


def run_spectrogram(*, bundle_dir: Path, runs: Sequence[str], channel: int, chunk_seconds: float = 10.0) -> dict[str, Any]:
    """Return the mean target fine powers of each chunk over all frames and over the kept frames."""
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    rows = _rows(bundle)
    if channel not in rows:
        raise ValueError(f"channel {channel} is not in the bundle")
    calibration = rows[channel]["fine_calibration"]
    anchor, word = int(calibration["anchor_bin"]), int(calibration["cfar_multiplier_q16"])
    bulk = unpack_bulk_mask([int(w, 16) for w in calibration["bulk_mask_words_hex"]])
    rank = int(calibration["cfar_rank"])
    channel_id = int(rows[channel]["receiver_channel_id"])
    per_chunk = max(1, round(chunk_seconds / FRAME_SECONDS))

    every, kept, frames, kept_frames = [], [], [], []
    pending: list[np.ndarray] = []
    streak = longest = 0

    def flush() -> None:
        nonlocal streak, longest
        # the mask is recomputed so the control, whose stage mask stays zero, can be shown too
        block = np.stack(pending)
        # DtvRfiMask masks a block whose rank has too little support (-1)
        words = required_words(pending, anchor_bin=anchor, bulk=bulk, rank=rank)
        keep = (words >= 0) & (words <= word)
        target = block[:, TARGET_ROW].astype(np.float64)
        every.append(target.mean(axis=0))
        kept.append(target[keep].mean(axis=0) if keep.any() else np.full(target.shape[1], np.nan))
        frames.append(len(pending))
        kept_frames.append(int(keep.sum()))
        # a masked stretch can cross chunks
        for frame_kept in keep:
            streak = 0 if frame_kept else streak + 1
            longest = max(longest, streak)
        pending.clear()

    for path in _expand(runs):
        for frame in iter_run_frames(path, channel_id):
            pending.append(frame)
            if len(pending) == per_chunk:
                flush()
    if pending:
        flush()
    if not frames:
        raise ValueError(f"no recorded frames for channel {channel}")
    frames = np.asarray(frames)
    return dict(channel=channel, anchor_bin=anchor, word_q16=word, chunk_seconds=per_chunk * FRAME_SECONDS,
                all_frames=np.stack(every), kept_frames=np.stack(kept), frames=frames,
                kept=np.asarray(kept_frames), longest_masked_frames=longest,
                minutes=np.cumsum(frames) * FRAME_SECONDS / 60.0)


def write_masked_summary(results: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    """Write one row per channel with its masked percent and longest masked stretch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "masked_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "word_q16", "frames", "masked_frames", "masked_percent", "longest_masked_seconds"])
        for result in results:
            frames = int(result["frames"].sum())
            masked = frames - int(result["kept"].sum())
            writer.writerow([result["channel"], result["word_q16"], frames, masked, f"{100.0 * masked / frames:.2f}",
                             f"{result['longest_masked_frames'] * FRAME_SECONDS:.2f}"])
    return path


def write_run_spectrogram(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Save the arrays and a two-panel figure, and return their paths."""
    import shutil

    import matplotlib.colors as colors

    from pilot_proxy import figure_style as style

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"spectrogram_ch{result['channel']:02d}"
    data = output_dir / f"{name}.npz"
    np.savez_compressed(data, **result)

    style.configure(require_tex=shutil.which("latex") is not None)
    fig, axes = style.plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 3.4), sharex=True, sharey=True,
                                   constrained_layout=True)
    # one reference and one colour scale, so the panels compare directly
    reference = np.median(result["all_frames"])
    panels = [10.0 * np.log10(np.maximum(result[key], 1e-12 * reference) / reference)
              for key in ("all_frames", "kept_frames")]
    top = float(np.nanmax(panels[0]))
    low = min(float(np.nanquantile(panels[0], 0.01)), top - 3.0)
    cmap = style.plt.get_cmap("viridis").with_extremes(bad=style.LIGHT_GRAY)
    bins = result["all_frames"].shape[1]
    extent = (-0.5 - result["anchor_bin"], bins - 0.5 - result["anchor_bin"], result["minutes"][-1], 0.0)
    for index, (ax, panel) in enumerate(zip(axes, panels)):
        image = ax.imshow(panel, aspect="auto", interpolation="nearest", extent=extent, cmap=cmap,
                          norm=colors.Normalize(low, top))
        ax.set_xlabel("Fine bin from the anchor")
        style.panel_label(ax, "ab"[index])
    axes[0].set_ylabel("Recorded time (min)")
    fig.colorbar(image, ax=axes, label="Fine power (dB)")
    figure = style.save(fig, output_dir / f"{name}.pdf", title=f"Channel {result['channel']} spectrogram")
    return data, Path(figure)
