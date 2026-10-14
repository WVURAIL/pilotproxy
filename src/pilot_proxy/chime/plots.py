# coding=utf-8
"""Plot products for CHIME real-data detector runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy import dtype, float64, ndarray

from pilot_proxy.dtv_units import (
    DB_LINEAR_BASE,
    DB_POWER_FACTOR,
    UNIT_NORMALIZED_POWER_RATIO,
    data_shelf_snr_db_to_pilot_excess_db,
    normalized_coarse_power_ratio_to_db,
)
from pilot_proxy.plot_style import setup_matplotlib

from .products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
    spectrum_before_after,
    write_spectrum_table,
)

FIGURE_DPI = 300

# Camera-ready export: comma-separated list of Matplotlib formats, e.g.
# PILOT_PROXY_FIGURE_FORMATS=png,pdf. PNG (300 dpi) stays the default and the
# canonical product name; extra formats are written alongside with the same
# stem, so pdf gives vector line art for the manuscript without changing the
# figure-name contract.
FIGURE_FORMATS_ENV = "PILOT_PROXY_FIGURE_FORMATS"
DEFAULT_FIGURE_FORMATS = ("png",)


def _figure_formats() -> tuple[str, ...]:
    raw = os.environ.get(FIGURE_FORMATS_ENV, "")
    formats = tuple(
        part.strip().lower() for part in raw.split(",") if part.strip()
    )
    return formats or DEFAULT_FIGURE_FORMATS


def _save_figure(fig, path) -> None:
    """Save one figure under every requested format, sharing the PNG stem."""
    base = Path(path)
    for fmt in _figure_formats():
        fig.savefig(base.with_suffix("." + fmt), dpi=FIGURE_DPI)


DATA_SHELF_SNR_TOP_TICKS_DB = np.asarray([-60.0, -30.0, -25.0, -20.0, -15.0, -10.0, 0.0])

KNOWN_CHIME_FIGURES = frozenset({
    "data_shelf_snr_histogram_by_channel.png",
    "coarse_power_ratio_survival_by_channel.png",
    "normalized_coarse_power_ratio_db_spectrogram.png",
    "baseband_spectrogram.png",
    "baseband_spectrum_before_after_mask.png",
    "mask_spectrogram.png",
})

def _setup_matplotlib():
    return setup_matplotlib(force_agg=True)


def _finite_values(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if valid is not None:
        arr = arr[np.asarray(valid) != 0]
    arr = arr[np.isfinite(arr)]
    return arr.astype(np.float64, copy=False)


def _set_snr_shelf_ticks(ax, *, xmin: float, xmax: float) -> None:
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    span = float(xmax - xmin)
    if not np.isfinite(span) or span <= 0.0:
        return
    step = 2.0 if span <= 20.0 else 10.0 if span >= 60.0 else 5.0
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))


def _format_db_tick(value: float) -> str:
    if abs(float(value)) >= 1.0 or float(value) == 0.0:
        return f"{value:g}"
    return f"{value:.3g}"


def _data_shelf_snr_db_to_normalized_coarse_power_ratio_db(values) -> np.ndarray:
    data_shelf_snr = np.asarray(values, dtype=np.float64)
    pilot_excess_db = data_shelf_snr_db_to_pilot_excess_db(data_shelf_snr)
    normalized_pilot_excess = DB_LINEAR_BASE ** (
        pilot_excess_db / DB_POWER_FACTOR
    )
    normalized_ratio = UNIT_NORMALIZED_POWER_RATIO + normalized_pilot_excess
    return np.asarray(
        normalized_coarse_power_ratio_to_db(normalized_ratio),
        dtype=np.float64,
    )


def _add_data_shelf_snr_top_axis_for_normalized_power_ratio(ax) -> None:
    xmin, xmax = ax.get_xlim()
    positions = _data_shelf_snr_db_to_normalized_coarse_power_ratio_db(
        DATA_SHELF_SNR_TOP_TICKS_DB
    )
    keep = np.isfinite(positions) & (positions >= xmin) & (positions <= xmax)
    if not np.any(keep):
        return
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(positions[keep])
    top.set_xticklabels(
        [_format_db_tick(value) for value in DATA_SHELF_SNR_TOP_TICKS_DB[keep]]
    )
    top.tick_params(axis="x", labelsize="small")
    top.set_xlabel(r"Estimated DTV data-shelf SNR $[\mathrm{dB}]$")


def _robust_color_limits(
    values: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float | None, float | None]:
    finite = _finite_values(values)
    if finite.size == 0:
        return None, None
    lo = float(np.percentile(finite, lower_percentile))
    hi = float(np.percentile(finite, upper_percentile))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return lo, hi


def _manifest_frequency_hz(
    run_dir: Path, physical_channel: np.ndarray
) -> np.ndarray | None:
    manifest_path = Path(run_dir) / "input_manifest.json"
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("datasets", [])
    if not isinstance(rows, list):
        return None
    by_channel = {
        int(row["physical_channel"]): float(row["coarse_channel_center_hz"])
        for row in rows
        if isinstance(row, dict)
        and row.get("physical_channel") is not None
        and row.get("coarse_channel_center_hz") is not None
    }
    values: list[float] = []
    for channel in physical_channel:
        if int(channel) not in by_channel:
            return None
        values.append(by_channel[int(channel)])
    return np.asarray(values, dtype=np.float64)


def _run_frequency_hz(run_dir: Path, product: np.lib.npyio.NpzFile) -> np.ndarray:
    if "chime_frequency_hz" in product.files:
        return np.asarray(product["chime_frequency_hz"], dtype=np.float64)
    from_manifest = _manifest_frequency_hz(Path(run_dir), product["physical_channel"])
    if from_manifest is not None:
        return from_manifest
    raise KeyError(
        "coarse-channel-center metadata is unavailable; pilot frequency is not "
        "a receiver-frequency substitute"
    )


def _coordinate_edges(centers: np.ndarray, *, default_step: float = 1.0) -> np.ndarray:
    coords = np.asarray(centers, dtype=np.float64)
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("coordinate centers must be a non-empty 1D array")
    if coords.size == 1:
        half = 0.5 * float(default_step)
        return np.asarray([coords[0] - half, coords[0] + half], dtype=np.float64)
    mid = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mid[0] - coords[0])
    last = coords[-1] + (coords[-1] - mid[-1])
    return np.concatenate(([first], mid, [last])).astype(np.float64)


def _sparse_tick_indices(size: int, *, max_ticks: int = 8) -> np.ndarray:
    if size <= 0:
        return np.asarray([], dtype=np.int64)
    count = min(int(size), int(max_ticks))
    return np.unique(np.round(np.linspace(0, size - 1, count)).astype(np.int64))


def _add_frame_index_axis(
        ax, *, relative_time_s: np.ndarray, frame_index: np.ndarray
) -> None:
    times = np.asarray(relative_time_s, dtype=np.float64)
    frames = np.asarray(frame_index, dtype=np.float64)
    if times.size != frames.size or times.size == 0:
        return
    if times.size == 1:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks([float(times[0])])
        sec.set_xticklabels([str(int(frames[0]))])
        sec.set_xlabel(r"Frame index, $n_{\mathrm{frame}}$")
        return
    dt = float((times[-1] - times[0]) / (frames[-1] - frames[0]))
    if not np.isfinite(dt) or dt == 0.0:
        return

    def time_to_frame(value):
        arr = np.asarray(value, dtype=np.float64)
        return (arr - times[0]) / dt + frames[0]

    def frame_to_time(value):
        arr = np.asarray(value, dtype=np.float64)
        return (arr - frames[0]) * dt + times[0]

    sec = ax.secondary_xaxis("top", functions=(time_to_frame, frame_to_time))
    tick_indices = _sparse_tick_indices(times.size)
    sec.set_xticks(frames[tick_indices])
    sec.set_xticklabels([str(int(value)) for value in frames[tick_indices]])
    sec.set_xlabel(r"Frame index, $n_{\mathrm{frame}}$")


def _add_dtv_channel_axis(
    ax,
    *,
    frequency_mhz: np.ndarray,
    physical_channel: np.ndarray,
    tick_labels: Sequence[str] | None = None,
    ylabel: str = r"DTV physical channel",
) -> None:
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(np.asarray(frequency_mhz, dtype=np.float64))
    right.set_yticklabels(
        list(tick_labels)
        if tick_labels is not None
        else [str(int(ch)) for ch in physical_channel]
    )
    right.set_ylabel(ylabel)


def _set_spectrogram_time_axis(ax, x_edges: np.ndarray) -> None:
    from matplotlib.ticker import MultipleLocator

    edges = np.asarray(x_edges, dtype=np.float64)
    if edges.size < 2:
        return
    left = max(0.0, float(edges[0]))
    right = float(edges[-1])
    if 9.0 <= right < 10.0:
        right = 10.0
    ax.set_xlim(left, right)
    if right >= 9.0:
        ax.xaxis.set_major_locator(MultipleLocator(2.0))


def _write_histogram_summary(
    path: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    estimated_data_shelf_snr_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "num_detector_valid_frames",
                "num_finite_data_shelf_snr_estimates",
                "finite_data_shelf_snr_fraction_of_valid_frames",
                "mean_estimated_data_shelf_snr_db",
                "max_estimated_data_shelf_snr_db",
                "mask_fraction_total",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            values = np.asarray(estimated_data_shelf_snr_db[:, index], dtype=np.float64)
            finite = values[np.isfinite(values)]
            detector_valid = np.asarray(valid[:, index]) != 0
            detector_valid_count = int(np.sum(detector_valid))
            positive_count = int(finite.size)
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(chime_frequency_hz[index]),
                    "num_detector_valid_frames": detector_valid_count,
                    "num_finite_data_shelf_snr_estimates": positive_count,
                    "finite_data_shelf_snr_fraction_of_valid_frames": (
                        float(positive_count / detector_valid_count)
                        if detector_valid_count
                        else float("nan")
                    ),
                    "mean_estimated_data_shelf_snr_db": (
                        float(np.mean(finite)) if finite.size else float("nan")
                    ),
                    "max_estimated_data_shelf_snr_db": (
                        float(np.max(finite)) if finite.size else float("nan")
                    ),
                    "mask_fraction_total": float(np.mean(mask[:, index] != 0)),
                }
            )


def _write_fstat_summary(
    path: Path,
    *,
    physical_channel: np.ndarray,
    pilot_frequency_hz: np.ndarray,
    chime_frequency_hz: np.ndarray,
    normalized_coarse_power_ratio_db: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "physical_channel",
                "pilot_frequency_hz",
                "chime_frequency_hz",
                "num_detector_valid_frames",
                "mean_normalized_coarse_power_ratio_db",
                "max_normalized_coarse_power_ratio_db",
                "mask_fraction",
            ],
        )
        writer.writeheader()
        for index, channel in enumerate(physical_channel):
            detector_valid = np.asarray(valid[:, index]) != 0
            values = _finite_values(normalized_coarse_power_ratio_db[:, index], detector_valid)
            writer.writerow(
                {
                    "physical_channel": int(channel),
                    "pilot_frequency_hz": float(pilot_frequency_hz[index]),
                    "chime_frequency_hz": float(chime_frequency_hz[index]),
                    "num_detector_valid_frames": int(np.sum(detector_valid)),
                    "mean_normalized_coarse_power_ratio_db": (
                        float(np.mean(values)) if values.size else float("nan")
                    ),
                    "max_normalized_coarse_power_ratio_db": (
                        float(np.max(values)) if values.size else float("nan")
                    ),
                    "mask_fraction": float(np.mean(mask[:, index] != 0)),
                }
            )


def _histogram_probability_density(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(int(np.asarray(bins).size) - 1, dtype=np.float64)
    counts, _ = np.histogram(finite, bins=bins)
    bin_width = np.diff(np.asarray(bins, dtype=np.float64))
    return counts.astype(np.float64) / (float(finite.size) * bin_width)


def _survival_probability_on_grid(values: np.ndarray, grid: np.ndarray) -> ndarray[tuple[()], dtype[float64]] | float:
    """Evaluate P(values >= grid) for finite values on a common plot grid."""
    finite = np.sort(_finite_values(values))
    grid_values = np.asarray(grid, dtype=np.float64)
    if finite.size == 0:
        return np.full(grid_values.shape, np.nan, dtype=np.float64)
    count_below = np.searchsorted(finite, grid_values, side="left")
    return (float(finite.size) - count_below.astype(np.float64)) / float(finite.size)


def _fstat_survival_plot_bounds(
    panel_values: Sequence[np.ndarray],
    *,
    include_comparison_range: bool,
) -> tuple[float, float]:
    finite_all = np.concatenate([_finite_values(values) for values in panel_values])
    xmin = float(np.min(finite_all))
    xmax = float(np.max(finite_all))
    span = xmax - xmin
    pad = max(0.05, 0.03 * span) if span > 0.0 else 0.5
    left = xmin - pad
    right = xmax + pad
    if include_comparison_range:
        left = min(left, -1.0)
        right = max(right, 1.0)
    return left, right


def plot_snr_shelf_histogram(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    physical_channel = detector["physical_channel"]
    pilot_frequency_hz = detector["pilot_frequency_hz"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), detector)
    estimated_data_shelf_snr_db = detector["estimated_data_shelf_snr_db"]
    normalized_coarse_power_ratio_db = detector["normalized_coarse_power_ratio_db"]
    mask = detector["mask"]
    valid = (
        detector["valid"]
        if "valid" in detector.files
        else (detector["p_ref_sum_u64"] > 0).astype(np.uint8)
    )

    panels = [
        (r"$\mathrm{SNR}_{\mathrm{shelf}}$ distribution", -90.0, 0.0),
        (
            r"Low range: $\mathrm{SNR}_{\mathrm{shelf}}$ from $-90$ to $-25$ dB",
            -90.0,
            -25.0,
        ),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.5, 8.4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )

    for panel_index, (title, xmin, xmax) in enumerate(panels):
        ax = axes[panel_index]
        bins = np.linspace(xmin, xmax, 100)
        channel_histograms: list[np.ndarray] = []
        for index, channel in enumerate(physical_channel):
            detector_valid = np.asarray(valid[:, index]) != 0
            values = np.asarray(estimated_data_shelf_snr_db[:, index], dtype=np.float64)
            hist = _histogram_probability_density(values[detector_valid], bins)
            channel_histograms.append(hist)
            ax.stairs(
                hist,
                bins,
                color=colors[int(index)],
                label=(
                    rf"DTV {int(channel)}, "
                    rf"$f_{{\mathrm{{CHIME}}}}="
                    rf"{chime_frequency_hz[index] / 1.0e6:.1f}"
                    rf"\,\mathrm{{MHz}}$"
                ),
            )
        if channel_histograms:
            mean_hist = np.mean(np.vstack(channel_histograms), axis=0)
            ax.stairs(
                mean_hist,
                bins,
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean over plotted pilots",
                zorder=20,
            )
        ax.set_xlim(xmin, xmax)
        _set_snr_shelf_ticks(ax, xmin=xmin, xmax=xmax)
        ax.set_ylabel(r"Probability density [1/dB]")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if panel_index == len(panels) - 1:
            ax.set_xlabel(
                r"DTV pilot shelf strength, "
                r"$\mathrm{SNR}_{\mathrm{shelf}}\;[\mathrm{dB}]$"
            )
        if panel_index == 0:
            ax.legend(ncol=3, fontsize="x-small")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "data_shelf_snr_histogram_by_channel.png",
    ]
    for path in outputs:
        _save_figure(fig, path)
    plt.close(fig)

    _write_fstat_summary(
        Path(run_dir) / "tables" / "normalized_coarse_power_ratio_summary_by_channel.csv",
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        normalized_coarse_power_ratio_db=normalized_coarse_power_ratio_db,
        mask=mask,
        valid=valid,
    )
    _write_histogram_summary(
        Path(run_dir) / "tables" / "data_shelf_snr_histogram_summary.csv",
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        estimated_data_shelf_snr_db=estimated_data_shelf_snr_db,
        mask=mask,
        valid=valid,
    )
    return outputs


def plot_coarse_power_ratio_survival(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    physical_channel = detector["physical_channel"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), detector)
    normalized_coarse_power_ratio_db = detector["normalized_coarse_power_ratio_db"]
    valid = (
        detector["valid"]
        if "valid" in detector.files
        else (detector["p_ref_sum_u64"] > 0).astype(np.uint8)
    )

    panels = [(r"All DTV pilot channels", np.ones_like(physical_channel, dtype=bool))]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.5, 4.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("turbo")(
        np.linspace(0.05, 0.95, max(int(physical_channel.size), 1))
    )
    for panel_index, (title, keep) in enumerate(panels):
        ax = axes[panel_index]
        channel_values: list[tuple[int, int, np.ndarray]] = []
        for index, channel in enumerate(physical_channel):
            if not keep[index]:
                continue
            values = np.sort(_finite_values(normalized_coarse_power_ratio_db[:, index], valid[:, index]))
            if values.size == 0:
                continue
            channel_values.append((index, int(channel), values))

        panel_values = [values for _, _, values in channel_values]
        if panel_values:
            xlim = _fstat_survival_plot_bounds(
                panel_values,
                include_comparison_range=False,
            )
            max_count = max(values.size for values in panel_values)
            survival_floor = 0.5 / float(max_count)
            grid = np.linspace(xlim[0], xlim[1], 900)

            for index, channel, values in channel_values:
                survival = np.asarray(
                    _survival_probability_on_grid(values, grid),
                    dtype=np.float64,
                )
                ax.step(
                    grid,
                    np.maximum(survival, survival_floor),
                    where="post",
                    color=colors[int(index)],
                    label=(
                        rf"DTV {channel}, "
                        rf"$f_{{\mathrm{{CHIME}}}}="
                        rf"{chime_frequency_hz[index] / 1.0e6:.1f}"
                        rf"\,\mathrm{{MHz}}$"
                    ),
                )

            mean_survival = np.nanmean(
                np.vstack(
                    [
                        _survival_probability_on_grid(values, grid)
                        for values in panel_values
                    ]
                ),
                axis=0,
            )
            ax.plot(
                grid,
                np.maximum(mean_survival, survival_floor),
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean over plotted pilots",
                zorder=20,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(survival_floor, 1.15)
        ax.set_ylabel(r"$P(R_F \geq T)$")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        _add_data_shelf_snr_top_axis_for_normalized_power_ratio(ax)
        if panel_index == 0:
            ax.legend(ncol=3, fontsize="x-small")
        if panel_index == len(panels) - 1:
            ax.set_xlabel(r"$10\log_{10}Q_\mathrm{coarse}\;[\mathrm{dB}]$")
    fig.suptitle(r"CHIME DTV pilot local-reference power ratio survival curves")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "coarse_power_ratio_survival_by_channel.png",
    ]
    for path in outputs:
        _save_figure(fig, path)
    plt.close(fig)
    return outputs


def _load_json_object(path: Path) -> dict[str, object]:
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _mask_label_from_policy(policy: object) -> str:
    if not isinstance(policy, dict):
        return ""
    source = policy.get("mask_source")
    if source is None:
        return ""
    source_text = str(source)
    if source_text == "normalized_positive_excess_decision":
        return (
            r"Mask rule: positive excess; valid if $P_{\mathrm{ref}}\ne0$, "
            r"mask if $P_t > (P_{\mathrm{ref}}\gg1)$"
        )
    return f"Mask rule: {source_text}"


def _mask_label_for_run(run_dir: Path) -> str:
    run = Path(run_dir)
    stats = _load_json_object(run / "stats.json")
    run_config = _load_json_object(run / "run_config.json")
    return _mask_label_from_policy(
        stats.get("mask_policy") or run_config.get("mask_policy")
    )


def plot_baseband_spectrum(run_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    physical_channel = cache["physical_channel"]
    pilot_frequency_hz = cache["pilot_frequency_hz"]
    chime_frequency_hz = _run_frequency_hz(Path(run_dir), cache)
    baseband_power_linear = cache["baseband_power_linear"]
    mask = cache["mask"]
    valid = (
        cache["valid"]
        if "valid" in cache.files
        else np.ones_like(baseband_power_linear, dtype=np.uint8)
    )
    before_db, after_db = spectrum_before_after(baseband_power_linear, mask, valid)

    x = chime_frequency_hz / 1.0e6
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.plot(x, before_db, marker="o", label=r"before mask")
    ax.plot(x, after_db, marker="o", label=r"after mask")
    ax.set_xlabel(r"CHIME frequency, $f_{\mathrm{CHIME}}\;[\mathrm{MHz}]$")
    ax.set_ylabel(r"$10\log_{10}P_{\mathrm{bb}}\;[\mathrm{dB}]$")
    title = (r"Integrated baseband power at ATSC pilot coarse channels"
             + "\n" + r"Per-frame means: 'before' over valid frames, 'after' "
             r"over unmasked frames; reweighting can move either way")
    mask_label = _mask_label_for_run(Path(run_dir))
    if mask_label:
        title += "\n" + mask_label
    if not np.any(mask != 0):
        title += (
            "\n" + r"No mask applied; before and after are identical by construction."
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(x)
    sec.set_xticklabels([str(int(ch)) for ch in physical_channel], rotation=90)
    sec.set_xlabel(r"DTV physical channel")

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "baseband_spectrum_before_after_mask.png",
    ]
    for path in outputs:
        _save_figure(fig, path)
    plt.close(fig)
    write_spectrum_table(
        Path(run_dir),
        physical_channel=physical_channel,
        pilot_frequency_hz=pilot_frequency_hz,
        chime_frequency_hz=chime_frequency_hz,
        baseband_power_linear=baseband_power_linear,
        mask=mask,
        valid=valid,
    )
    return outputs


def _event_boundaries(run_dir: Path):
    """Frame indices where the source event changes, from the combined
    product's frame-identity sidecar (written by event-keyed combines), plus
    the event count. None when the sidecar is absent (chime-run products, or
    combines of identity-less legacy products). On stitched products the
    frame axis concatenates acquisitions, so 'relative time' is data seconds
    rather than wall clock; the boundaries make that visible."""
    path = Path(run_dir) / "chime_frame_identity.npz"
    if not path.exists():
        return None
    with np.load(str(path)) as z:
        if "frame_event_key" not in z.files:
            return None
        keys = np.asarray(z["frame_event_key"]).astype(str).reshape(-1)
    if keys.size == 0:
        return None
    change = np.nonzero(keys[1:] != keys[:-1])[0] + 1
    return change, len(set(keys.tolist()))


def _plot_spectrogram(
    *,
    run_dir: Path,
    values: np.ndarray,
    physical_channel: np.ndarray,
    frequency_hz: np.ndarray,
    frame_index: np.ndarray,
    relative_time_s: np.ndarray,
    event_boundaries=None,
    title: str,
    colorbar_label: str,
    basename: str,
    cmap: str,
    robust_limits: bool = False,
    discrete_mask_colorbar: bool = False,
) -> list[Path]:
    plt = _setup_matplotlib()
    if discrete_mask_colorbar:
        from matplotlib.colors import BoundaryNorm, ListedColormap

        plot_cmap = ListedColormap(["white", "black"])
    else:
        BoundaryNorm = None
        plot_cmap = cmap
    time_s = np.asarray(relative_time_s, dtype=np.float64)
    freq_hz = np.asarray(frequency_hz, dtype=np.float64)
    freq_mhz = np.asarray(freq_hz / 1.0e6, dtype=np.float64).reshape(-1)
    channel_array = np.asarray(physical_channel).reshape(-1)
    x_edges = _coordinate_edges(time_s)
    value_array = np.asarray(values)
    if event_boundaries is not None:
        _, _n_events = event_boundaries
        title = rf"{title} ({_n_events} events stitched)"
    panels: list[tuple[str, np.ndarray]] = [
        (title, np.ones_like(channel_array, dtype=bool)),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9.8, 5.2 if len(panels) == 1 else 8.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for panel_index, (ax, (panel_title, keep)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        panel_freq_mhz = freq_mhz[keep]
        panel_channels = channel_array[keep]
        panel_values = value_array[:, keep]
        y_edges = _coordinate_edges(panel_freq_mhz)
        vmin, vmax = (
            _robust_color_limits(panel_values) if robust_limits else (None, None)
        )
        norm = (
            BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2)
            if discrete_mask_colorbar and BoundaryNorm is not None
            else None
        )
        if event_boundaries is not None:
            for _b in event_boundaries[0]:
                ax.axvline(x_edges[int(_b)], color="tab:red", lw=0.7,
                           alpha=0.85, linestyle=(0, (2, 2)), zorder=3)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            panel_values.T,
            shading="auto",
            cmap=plot_cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        if panel_index == len(panels) - 1:
            ax.set_xlabel(
                r"Relative time, $t_{\mathrm{rel}}\;[\mathrm{s}]$"
                + (" (data seconds; events stitched)"
                   if event_boundaries is not None else ""))
        ax.set_ylabel(r"CHIME frequency, $f_{\mathrm{CHIME}}\;[\mathrm{MHz}]$")
        ax.set_title(panel_title)
        ax.set_yticks(panel_freq_mhz)
        ax.set_yticklabels([f"{value:.1f}" for value in panel_freq_mhz])
        _set_spectrogram_time_axis(ax, x_edges)
        _add_frame_index_axis(ax, relative_time_s=time_s, frame_index=frame_index)
        _add_dtv_channel_axis(
            ax,
            frequency_mhz=panel_freq_mhz,
            physical_channel=panel_channels,
        )
        cbar = fig.colorbar(
            image,
            ax=ax,
            ticks=[0, 1] if discrete_mask_colorbar else None,
        )
        cbar.set_label(colorbar_label)

    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [figures / f"{basename}.png"]
    for path in outputs:
        _save_figure(fig, path)
    plt.close(fig)
    return outputs


def plot_baseband_spectrogram(run_dir: Path) -> list[Path]:
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=cache["baseband_power_db"],
        physical_channel=cache["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), cache),
        frame_index=cache["frame_index"],
        relative_time_s=cache["relative_time_s"],
        event_boundaries=_event_boundaries(Path(run_dir)),
        title=r"Baseband power spectrogram",
        colorbar_label=r"$10\log_{10}P_{\mathrm{bb}}\;[\mathrm{dB}]$",
        basename="baseband_spectrogram",
        cmap="viridis",
    )


def plot_normalized_coarse_power_ratio_db_spectrogram(run_dir: Path) -> list[Path]:
    detector = np.load(Path(run_dir) / CHIME_DETECTOR_OUTPUTS_FILENAME)
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=detector["normalized_coarse_power_ratio_db"],
        physical_channel=detector["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), detector),
        frame_index=detector["frame_index"],
        relative_time_s=cache["relative_time_s"],
        event_boundaries=_event_boundaries(Path(run_dir)),
        title=r"local-reference power ratio level spectrogram",
        colorbar_label=r"$10\log_{10}Q_\mathrm{coarse}\;[\mathrm{dB}]$",
        basename="normalized_coarse_power_ratio_db_spectrogram",
        cmap="magma",
        robust_limits=True,
    )


def plot_mask_spectrogram(run_dir: Path) -> list[Path]:
    cache = np.load(Path(run_dir) / CHIME_SPECTROGRAM_CACHE_FILENAME)
    title = r"Detector mask spectrogram"
    mask_label = _mask_label_for_run(Path(run_dir))
    if mask_label:
        title += "\n" + mask_label
    if not np.any(cache["mask"] != 0):
        title += "\n" + r"No mask applied; all mask samples are zero."
    return _plot_spectrogram(
        run_dir=Path(run_dir),
        values=cache["mask"],
        physical_channel=cache["physical_channel"],
        frequency_hz=_run_frequency_hz(Path(run_dir), cache),
        frame_index=cache["frame_index"],
        relative_time_s=cache["relative_time_s"],
        event_boundaries=_event_boundaries(Path(run_dir)),
        title=title,
        colorbar_label=r"Mask $M$",
        basename="mask_spectrogram",
        cmap="gray_r",
        discrete_mask_colorbar=True,
    )


def clean_known_figures(run_dir: Path) -> None:
    """Delete all known CHIME figure files from the figures directory."""
    figures = Path(run_dir) / "figures"
    for name in KNOWN_CHIME_FIGURES:
        for candidate in figures.glob(Path(name).stem + ".*"):
            if candidate.is_file():
                candidate.unlink()


def generate_chime_plots(run_dir: Path) -> list[Path]:
    run = Path(run_dir)
    outputs: list[Path] = []
    # Without detector products there is nothing to plot; return instead of
    # failing while trying to load chime_detector_outputs.npz.
    if not (run / CHIME_DETECTOR_OUTPUTS_FILENAME).exists():
        return outputs
    outputs.extend(plot_snr_shelf_histogram(run))
    outputs.extend(plot_coarse_power_ratio_survival(run))
    outputs.extend(plot_normalized_coarse_power_ratio_db_spectrogram(run))
    outputs.extend(plot_baseband_spectrogram(run))
    outputs.extend(plot_baseband_spectrum(run))
    outputs.extend(plot_mask_spectrogram(run))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CHIME real-data PilotProxy figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-figures",
        action="store_true",
        help="Delete known CHIME figures before generating current products.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.clean_figures:
        clean_known_figures(args.run_dir)
    outputs = generate_chime_plots(args.run_dir)
    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
