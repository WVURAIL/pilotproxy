"""Process a detect-only run in one step: split, calibrate, test, plot and summarize.

The output directory holds plans/, bundle/ with bundle.report.json, test.json, the spectrograms of
each part under the new bundle, and summary.md and summary.json, one row per channel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pilot_proxy.bundle_calibration import calibrate_bundle, test_bundle
from pilot_proxy.fine_decision import MULTIPLIER_ONE
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.run_plans import CALIBRATION_SHARE, CORE_PROFILE, write_run_plans
from pilot_proxy.run_spectrogram import (
    FRAME_SECONDS,
    run_spectrogram,
    write_masked_summary,
    write_run_spectrogram,
)

PARTS = ("calibrate", "test")


def _masked(result: dict[str, Any]) -> dict[str, float]:
    frames = int(result["frames"].sum())
    return dict(frames=frames, masked_percent=round(100.0 * (frames - int(result["kept"].sum())) / frames, 3),
                longest_masked_seconds=round(result["longest_masked_frames"] * FRAME_SECONDS, 2))


def _summary_rows(report: dict[str, Any], tested: dict[str, Any], masked: dict[str, dict[int, dict]]) -> list[dict]:
    checks = tested["checks"]
    rows = []
    for channel, row in report["channels"].items():
        mine = [c for c in checks if c["check"].endswith(f"channel {channel}")]
        # the spectrograms keep frames at the word the stage applies, which for a v3 channel is the power-limit
        # word where calibration wrote one, so the row names that word
        limit = row.get("power_limit_word_q16") is not None
        word = row["power_limit_word_q16"] if limit else row["word_q16"]
        eta = word / MULTIPLIER_ONE if limit else row["eta"]
        source = row["power_limit_word_source"] if limit else row["word_source"]
        rows.append(dict(
            channel=int(channel), anchor_bin=row["anchor_bin"], anchor_source=row["anchor_source"],
            nominal_bin=row["nominal_bin"], eta=round(eta, 4), word_q16=word,
            word_source=source, pilot_share=None if "pilot_share" not in row else round(row["pilot_share"], 3),
            scale=row.get("scale"),
            **{f"{part}_{key}": value for part in PARTS
               for key, value in masked[part].get(int(channel), {}).items()},
            test_passed=all(c["passed"] for c in mine) if mine else None))
    return rows


def _markdown(epoch: str, report: dict[str, Any], tested: dict[str, Any], rows: list[dict]) -> str:
    control = next((c for c in tested["checks"] if c["check"].startswith("false alarms")), None)
    # fine decision v3 has no search geometry, and its default word is the control word scaled to each channel
    v3 = report.get("decision_version") == "fine_decision_v3"
    word = "Control word" if v3 else "Floor"
    lines = [f"# Run {epoch}", "",
             f"Test {'PASSED' if tested['passed'] else 'FAILED'}. {word} eta {report['floor_eta']:.4f} "
             f"from the {report['floor_source']}.",
             f"Control: {control['detail'] if control else 'no check'}.", "",
             f"| Channel | {'Scale' if v3 else 'Anchor'} | Eta | Word from | Pilot share | Masked, calibrate "
             "| Masked, test | Longest masked, s | Test |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        anchor = (f"{r['scale']:.4f}" if r.get("scale") is not None else "-") if v3 else (
            f"{r['anchor_bin']} ({r['anchor_source']}" + (
                f", nominal {r['nominal_bin']})" if r["anchor_bin"] != r["nominal_bin"] else ")"))
        test = {True: "pass", False: "FAIL", None: "-"}[r["test_passed"]]
        lines.append(f"| {r['channel']} | {anchor} | {r['eta']} | {r['word_source']} | {r['pilot_share']} "
                     f"| {r.get('calibrate_masked_percent', '-')}% | {r.get('test_masked_percent', '-')}% "
                     f"| {r.get('test_longest_masked_seconds', '-')} | {test} |")
    failed = [c for c in tested["checks"] if not c["passed"]]
    if failed:
        lines += ["", "Failed checks:", ""] + [f"- {c['check']}: {c['detail']}" for c in failed]
    if report.get("threshold") == "power_limit":
        tested = report["power_limit"]
        none = ", ".join(str(c) for c in tested["no_passing_word"]) or "none"
        lines += ["", f"Threshold: power limit, against {tested['experiment']}. Channels with no passing word: {none}. "
                  f"The operating policy masks these whole, and the bundle holds the {'default word' if v3 else 'floor'} "
                  "on them. A power-limit word "
                  "is in-sample, and it needs a test on independent data before it masks."]
    source = "the control behind it" if v3 else "the run behind it"
    lines += ["", f"A word from the {'control' if v3 else 'knee or the floor'} is only as good as {source}. "
              "Read each channel's spectrograms under spectrograms/ before the bundle is used."]
    return "\n".join(lines) + "\n"


def process_run(*, bundle_dir: Path, runs: Sequence[str], epoch: str, output_dir: Path,
                calibration_share: float = CALIBRATION_SHARE, null_quantile: float = 0.99,
                receiver_profile: Path | None = None, detector_core_profile: Path = Path(CORE_PROFILE),
                chunk_seconds: float = 10.0, threshold: str = "knee", power_limit: Path | None = None,
                decision_version: str | None = None, night_site: Sequence[float] | None = None,
                log=print) -> dict[str, Any]:
    """Split the run, calibrate a bundle on the first part, test it on the rest, and write the summary."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty")
    plans = write_run_plans(bundle_dir=bundle_dir, runs=list(runs), output_dir=output_dir / "plans", epoch=epoch,
                            receiver_profile=receiver_profile, detector_core_profile=detector_core_profile,
                            null_quantile=null_quantile, calibration_share=calibration_share, threshold=threshold,
                            power_limit=power_limit, decision_version=decision_version, night_site=night_site)
    log("plans written")
    bundle = output_dir / "bundle"
    report = calibrate_bundle(plan_path=output_dir / "plans" / "calibrate.json", output_dir=bundle)
    log(f"bundle calibrated, floor eta {report['floor_eta']:.4f}")
    tested = test_bundle(bundle_dir=bundle, plan_path=output_dir / "plans" / "test.json",
                         output_json=output_dir / "test.json")
    log(f"bundle test {'passed' if tested['passed'] else 'FAILED'}")

    masked: dict[str, dict[int, dict]] = {}
    for part in PARTS:
        results, masked[part] = [], {}
        for channel, paths in plans[part]["channels"].items():
            result = run_spectrogram(bundle_dir=bundle, runs=paths, channel=int(channel), chunk_seconds=chunk_seconds)
            write_run_spectrogram(result, output_dir / "spectrograms" / part)
            results.append(result)
            masked[part][int(channel)] = _masked(result)
        write_masked_summary(results, output_dir / "spectrograms" / part)
        log(f"spectrograms of the {part} part written")

    rows = _summary_rows(report, tested, masked)
    summary = dict(epoch=epoch, passed=tested["passed"], threshold=report["threshold"], floor_eta=report["floor_eta"],
                   floor_source=report["floor_source"], calibration_share=calibration_share, channels=rows)
    write_json_strict(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_markdown(epoch, report, tested, rows))
    return summary
