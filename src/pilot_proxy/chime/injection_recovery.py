# coding=utf-8
"""Analyze an injection ladder for recovery linearity and the radiometer baseline.

The inputs are the products of ``pilot-proxy chime-scan`` runs over
``inject-pilot-tone`` output trees. There is one run per tone amplitude, and
each run directory also holds the ``injection_manifest.json`` of its ladder
point.

For the recovery linearity, the mean corrected pilot excess
``rho_hat = mean(normalized_pilot_excess[valid])`` is computed for each point
with its standard error, and the weighted linear fit
``rho_hat = floor + gain * a^2`` is made against the injected tone power. A
log-log slope of one in the signal-dominated regime shows that the recovered
excess tracks the injected power. The intercept is the ambient floor of the
channel, anchored by the required ``a = 0`` control point, which the injection
harness makes byte-identical to the source data.

For the radiometer baseline, the detection rates of the power ratio
(``coarse_power_ratio``) and of the classical total-power detector
(``baseband_power_linear``) are evaluated on the same frames at matched
false-alarm rates. The thresholds are empirical quantiles of the ``a = 0``
control, and the Wilson 95% intervals come from
``evaluate_snr.wilson_interval``. The horizontal gap between the two detection
curves is the measured sensitivity advantage.

Only stored products are read, and the detector is not run again.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import operator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np

from pilot_proxy import wvu_palette as wvu
from pilot_proxy.characterization import injection
from pilot_proxy.characterization.injection import (
    detection_rates,
    false_alarm_rate_key,
    matched_thresholds,
    signal_dominated_log_slope,
    weighted_linear_fit as _weighted_linear_fit,
)
from pilot_proxy.chime.injection import INJECTION_MANIFEST_FILENAME
from pilot_proxy.chime.products import (
    CHIME_DETECTOR_OUTPUTS_FILENAME,
    CHIME_SPECTROGRAM_CACHE_FILENAME,
    require_current_detector_contract,
)

RECOVERY_CSV_FILENAME = "injection_recovery.csv"
RECOVERY_SUMMARY_FILENAME = "injection_recovery_summary.json"
RECOVERY_FIGURE = "injection_recovery_linearity.png"
BASELINE_FIGURE = "detector_vs_radiometer_pd.png"
DEFAULT_FALSE_ALARM_RATES = (1e-2,)
# An empirical quantile at P_fa is used only when the H0 frames give at least
# this many expected false alarms (detector register entry
# detection.minimum_frames_per_false_alarm).
MIN_FRAMES_PER_FALSE_ALARM = injection.MIN_FRAMES_PER_FALSE_ALARM
STATISTICS = {"power_ratio": "power_ratio", "radiometer": "power"}


def _mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    """Reduce finite samples without summing or squaring at their raw scale."""
    _, exponent = math.frexp(float(np.max(np.abs(values))))
    scaled = np.ldexp(values, -exponent)
    mean = math.ldexp(math.fsum(scaled) / values.size, exponent)
    if values.size < 2:
        return mean, float("nan")
    # Center before squaring so a large common offset does not consume the
    # precision available for the scatter. The power-of-two scale avoids an
    # extra rounding step for normal scaled samples.
    centered = scaled - scaled[0]
    centered -= math.fsum(centered) / values.size
    sem = math.sqrt(math.fsum(centered * centered) / (values.size * (values.size - 1)))
    return mean, math.ldexp(sem, exponent)


def _load_point(point_dir: Path) -> dict[str, Any]:
    point_dir = Path(point_dir)
    detector_path = point_dir / CHIME_DETECTOR_OUTPUTS_FILENAME
    cache_path = point_dir / CHIME_SPECTROGRAM_CACHE_FILENAME
    manifest_path = point_dir / INJECTION_MANIFEST_FILENAME
    for path in (detector_path, cache_path, manifest_path):
        if not path.exists():
            raise SystemExit(
                f"ladder point {point_dir} is missing {path.name}; each "
                "--point directory must hold the run products plus its "
                "injection_manifest.json (copy the manifest from the "
                "injected-input directory)"
            )
    try:
        require_current_detector_contract(point_dir)
    except ValueError as exc:
        raise SystemExit(f"ladder point {point_dir}: {exc}") from None
    with np.load(detector_path) as detector, np.load(cache_path) as cache:
        for key in ("coarse_power_ratio", "normalized_pilot_excess", "valid"):
            if key not in detector:
                raise SystemExit(
                    f"{detector_path} lacks {key!r}; the recovery analysis needs "
                    "norm-corrected products"
                )
        stored_valid = detector["valid"]
        if stored_valid.dtype.kind not in "biuf" or not np.all(
            (stored_valid == 0) | (stored_valid == 1)
        ):
            raise SystemExit(f"{detector_path} requires binary valid values (0 or 1)")
        valid = stored_valid.astype(bool)
        for key in ("normalized_pilot_excess", "coarse_power_ratio"):
            if np.iscomplexobj(detector[key]):
                raise SystemExit(f"{detector_path} requires real-valued {key!r}")
        rho = detector["normalized_pilot_excess"].astype(np.float64)
        power_ratio = detector["coarse_power_ratio"].astype(np.float64)
        if "baseband_power_linear" not in cache:
            raise SystemExit(f"{cache_path} lacks 'baseband_power_linear'")
        if np.iscomplexobj(cache["baseband_power_linear"]):
            raise SystemExit(f"{cache_path} requires real-valued 'baseband_power_linear'")
        power = cache["baseband_power_linear"].astype(np.float64)
        if valid.ndim != 2 or any(array.shape != valid.shape for array in (rho, power_ratio, power)):
            raise SystemExit(f"ladder point {point_dir} has inconsistent statistic shapes")
        # Matching shapes do not establish that a cache describes the same
        # frames or pilots. Refuse stale or reordered products before pairing
        # the radiometer values with the detector's validity mask.
        identity_shapes = {
            "frame_index": (valid.shape[0],),
            "physical_channel": (valid.shape[1],),
            "pilot_frequency_hz": (valid.shape[1],),
            "chime_frequency_hz": (valid.shape[1],),
        }
        for key, shape in identity_shapes.items():
            if key not in detector or key not in cache:
                raise SystemExit(f"ladder point {point_dir} is missing product identity {key!r}")
            if detector[key].shape != shape or cache[key].shape != shape:
                raise SystemExit(f"ladder point {point_dir} has inconsistent identity shapes for {key!r}")
            if not np.array_equal(detector[key], cache[key]):
                raise SystemExit(f"ladder point {point_dir} has mismatched product identity {key!r}")
        if "valid" not in cache:
            raise SystemExit(f"{cache_path} lacks 'valid'")
        if not np.array_equal(detector["valid"], cache["valid"]):
            raise SystemExit(f"ladder point {point_dir} has mismatched product validity masks")
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a nonempty list")
        amplitudes = {float(entry["amplitude_lsb"]) for entry in files}
        frequencies = {float(entry["baseband_frequency_hz"]) for entry in files}
        clip_counts = [operator.index(entry["clip_count"]) for entry in files]
        if (any(not math.isfinite(value) or value < 0 for value in amplitudes)
                or any(not math.isfinite(value) for value in frequencies)
                or any(value < 0 for value in clip_counts)
                or any(isinstance(entry["clip_count"], bool) for entry in files)):
            raise ValueError("invalid injection truth")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SystemExit(
            f"{manifest_path} requires finite nonnegative amplitude and finite frequency "
            "in a nonempty files list, with nonnegative integer clip counts"
        ) from exc
    frequencies = {round(value, 6) for value in frequencies}
    if len(amplitudes) != 1 or len(frequencies) != 1:
        raise SystemExit(
            f"{manifest_path} mixes amplitudes {sorted(amplitudes)} or "
            f"frequencies {sorted(frequencies)}; one ladder point must be a "
            "single (amplitude, frequency) setting"
        )
    # Both detectors and the fitted mean use the same finite frame population.
    valid &= np.isfinite(rho) & np.isfinite(power_ratio) & np.isfinite(power)
    rho = rho[valid]
    if rho.size == 0:
        raise SystemExit(f"ladder point {point_dir} has no valid frames")
    power = power[valid]
    power_ratio = power_ratio[valid]
    amplitude = amplitudes.pop()
    injected_power = amplitude * amplitude
    if not math.isfinite(injected_power) or (amplitude > 0 and injected_power == 0):
        raise SystemExit(
            f"{manifest_path}: injected power (amplitude squared) must be finite "
            "and positive for nonzero amplitude"
        )
    mean, sem = _mean_and_sem(rho)
    return {
        "point_dir": str(point_dir),
        "amplitude_lsb": amplitude,
        "injected_power_lsb2": injected_power,
        "baseband_frequency_hz": frequencies.pop(),
        "n_valid": int(rho.size),
        "rho_hat": mean,
        "rho_sem": sem,
        "power_ratio": power_ratio,
        "power": power,
        "total_clip_count": sum(clip_counts),
    }


def analyze_injection_recovery(
    point_dirs: Sequence[Path],
    *,
    false_alarm_rates: Sequence[float] = DEFAULT_FALSE_ALARM_RATES,
) -> dict[str, Any]:
    if any(not np.isfinite(pfa) or not 0 < pfa < 1 for pfa in false_alarm_rates):
        raise SystemExit("false-alarm rates must be finite and strictly between 0 and 1")
    points = sorted(
        (_load_point(p) for p in point_dirs),
        key=lambda point: point["amplitude_lsb"],
    )
    controls = [p for p in points if p["amplitude_lsb"] == 0.0]
    if len(controls) != 1:
        raise SystemExit(
            "exactly one ladder point must be the a = 0 control (it anchors "
            f"the floor and every threshold); got {len(controls)}"
        )
    control = controls[0]
    if len(points) < 2:
        raise SystemExit("the ladder needs at least one nonzero injection amplitude")
    if len({point["baseband_frequency_hz"] for point in points}) != 1:
        raise SystemExit("all ladder points must use the same baseband frequency")

    # Matched-P_fa thresholds from the control's empirical quantiles.
    thresholds, usable_pfa = matched_thresholds(control, false_alarm_rates, statistics=STATISTICS)
    if not usable_pfa:
        raise SystemExit(
            f"the control point's {control['n_valid']} valid frames cannot "
            "support any requested false-alarm rate (need >= "
            f"{MIN_FRAMES_PER_FALSE_ALARM}/P_fa frames); request a larger "
            "P_fa or scan more control files"
        )

    rows: list[dict[str, Any]] = []
    for point in points:
        row: dict[str, Any] = {
            "amplitude_lsb": point["amplitude_lsb"],
            "injected_power_lsb2": point["injected_power_lsb2"],
            "n_valid": point["n_valid"],
            "rho_hat": point["rho_hat"],
            "rho_sem": point["rho_sem"],
            "total_clip_count": point["total_clip_count"],
        }
        row.update(detection_rates(point, thresholds, usable_pfa, statistics=STATISTICS))
        rows.append(row)

    x = np.asarray([row["injected_power_lsb2"] for row in rows])
    y = np.asarray([row["rho_hat"] for row in rows])
    sem = np.asarray([row["rho_sem"] for row in rows])
    if not np.all(np.isfinite(sem)) or np.any(sem <= 0):
        raise SystemExit("every ladder point needs >= 2 valid frames and a finite, positive SEM")
    try:
        fit = _weighted_linear_fit(x, y, sem)
    except ValueError as exc:
        raise SystemExit(f"injection recovery fit: {exc}") from exc

    # Slope-one check in log space over signal-dominated points.
    log_slope = signal_dominated_log_slope(x, y, fit)

    return {
        "schema_version": "pilotproxy_injection_recovery_v2",
        "points": [
            {k: v for k, v in row.items()} for row in rows
        ],
        "control_point_dir": control["point_dir"],
        "false_alarm_rates": usable_pfa,
        "thresholds": thresholds,
        "fit": fit,
        "signal_dominated_log_slope": log_slope,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> list[Path]:
    """Stage artifacts, then publish the summary as the completion marker.

    A publication failure can leave partial artifacts, but no summary claiming
    that the report completed. Render failures preserve all prior artifacts.
    """
    # Serialize before creating files: nonfinite results cannot be published
    # as a JSON report. Render the complete report before replacing old files.
    summary = json.dumps(report, indent=2, allow_nan=False)
    if not report["points"]:
        raise ValueError("recovery report must contain points")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".injection-recovery-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        csv_path = staging / RECOVERY_CSV_FILENAME
        fieldnames = list(report["points"][0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report["points"])
        json_path = staging / RECOVERY_SUMMARY_FILENAME
        json_path.write_text(summary, encoding="utf-8")
        staged = list(dict.fromkeys([csv_path, json_path, *_write_figures(report, staging)]))
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in staged:
            if (output_dir / path.name).is_dir():
                raise IsADirectoryError(f"report artifact destination is a directory: {output_dir / path.name}")
        # Invalidate a previous completion marker only once staging succeeds.
        # The marker is replaced last so failed publication cannot leave a
        # successful summary beside a mixture of old and new report artifacts.
        (output_dir / RECOVERY_SUMMARY_FILENAME).unlink(missing_ok=True)
        publication_order = [path for path in staged if path != json_path] + [json_path]
        for path in publication_order:
            destination = output_dir / path.name
            path.replace(destination)
        return [output_dir / path.name for path in staged]


def _write_figures(report: dict[str, Any], output_dir: Path) -> list[Path]:
    from pilot_proxy.chime.plots import _figure_formats, _save_figure, _setup_matplotlib

    plt = _setup_matplotlib()
    rows = report["points"]
    fit = report["fit"]
    x = np.asarray([row["injected_power_lsb2"] for row in rows])
    y = np.asarray([row["rho_hat"] for row in rows])
    sem = np.asarray([row["rho_sem"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    try:
        ax.errorbar(x, y, yerr=1.959963984540054 * sem, fmt="o", markersize=4,
                    capsize=3, label=r"Recovered $\hat{\rho}$ (95\% CI)")
        grid = np.linspace(0.0, float(x.max()) * 1.05, 200)
        ax.plot(grid, fit["floor"] + fit["gain_per_lsb2"] * grid, "-",
                label=(r"Fit: $\hat{\rho} = \mathrm{floor} + g\,a^2$"))
        ax.axhline(fit["floor"], linestyle=":", color=wvu.MUTED,
                   label=r"Ambient floor ($a = 0$)")
        ax.set_xlabel(r"Injected tone power $a^2$ [LSB$^2$]")
        ax.set_ylabel(r"Recovered pilot excess $\hat{\rho} = F/\mu_0 - 1$")
        ax.set_title(r"Injection--recovery linearity on real baseband")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        linearity_path = Path(output_dir) / RECOVERY_FIGURE
        _save_figure(fig, linearity_path)
    finally:
        plt.close(fig)

    pfa = false_alarm_rate_key(report["false_alarm_rates"][0])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    try:
        for name, label in (("power_ratio", r"power-ratio"),
                            ("radiometer", r"Radiometer (total power)")):
            pd = np.asarray([row[f"pd_{name}_pfa{pfa}"] for row in rows])
            lo = np.asarray([row[f"pd_{name}_pfa{pfa}_wilson95_lo"] for row in rows])
            hi = np.asarray([row[f"pd_{name}_pfa{pfa}_wilson95_hi"] for row in rows])
            ax.errorbar(x, pd, yerr=np.stack([pd - lo, hi - pd]), fmt="o-",
                        markersize=4, capsize=3, label=label)
        ax.set_xlabel(r"Injected tone power $a^2$ [LSB$^2$]")
        ax.set_ylabel(rf"Detection rate at $P_\mathrm{{fa}} = {pfa}$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(r"Matched-$P_\mathrm{fa}$ detection: power-ratio vs.\ radiometer")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        baseline_path = Path(output_dir) / BASELINE_FIGURE
        _save_figure(fig, baseline_path)
    finally:
        plt.close(fig)
    return [path.with_suffix("." + fmt)
            for path in (linearity_path, baseline_path) for fmt in _figure_formats()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an injection ladder's run products: recovery linearity "
            "(weighted fit of recovered pilot excess vs injected tone power) "
            "and the power-ratio vs radiometer detection comparison at "
            "matched false-alarm rates from the a = 0 control."
        ),
    )
    parser.add_argument("--point", type=Path, action="append", required=True,
                        dest="points",
                        help=(
                            "A ladder-point run directory (repeat per "
                            "amplitude). Must contain the run products and "
                            "that point's injection_manifest.json; exactly "
                            "one point must be the a = 0 control."
                        ))
    parser.add_argument("--false-alarm-rate", type=float, action="append",
                        dest="false_alarm_rates", default=None,
                        help="Matched P_fa (repeatable; default 1e-2).")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_injection_recovery(
        args.points,
        false_alarm_rates=args.false_alarm_rates or DEFAULT_FALSE_ALARM_RATES,
    )
    written = write_outputs(report, args.output_dir)
    fit = report["fit"]
    slope = report["signal_dominated_log_slope"]
    print(
        f"gain {fit['gain_per_lsb2']:.4g} +/- {fit['gain_err']:.2g} per LSB^2, "
        f"floor {fit['floor']:.4g} +/- {fit['floor_err']:.2g}"
        + (f", signal-dominated log-log slope {slope:.3f}" if slope is not None
           else "")
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
