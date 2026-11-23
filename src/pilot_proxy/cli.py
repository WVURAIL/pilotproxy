# coding=utf-8
"""CLI for the standalone PilotProxy pilot detector testbench."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz
from pilot_proxy.bundle_calibration import (
    THRESHOLDS,
    calibrate_bundle,
    qualify_bundle,
    test_bundle,
)
from pilot_proxy.detector_contract import (
    WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
    WEIGHT_COORDINATE_RAW_INPUT,
    normalize_weight_coordinate_system,
)
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.dtv_units import (
    DTV_BANDWIDTH_HZ,
    EFFECTIVE_BIN_BW_HZ,
    PILOT_BELOW_DATA_DB,
    PILOT_CAPTURE_EFFICIENCY,
)
from pilot_proxy.fine_decision import FINE_DECISION_VERSIONS
from pilot_proxy.integration import (
    DEFAULT_DETECTOR_CORE_PROFILE,
    default_reference_receiver_profile,
    layout_uint64_bound_check,
    load_detector_core_profile,
    load_receiver_profile,
    load_stream_map,
    parse_physical_channel_selection,
    write_weight_bank_from_receiver_profile,
)
from pilot_proxy.integration.stream_layout import validate_integration_compatibility
from pilot_proxy.json_utils import write_json_strict
from pilot_proxy.paths import (
    DEFAULT_LIB_PATH,
    DEFAULT_WEIGHTS_PATH,
    GENERATED_DIR,
    SOURCE_CHECKOUT_ROOT,
    resolve_user_path,
)
from pilot_proxy.process_run import process_run
from pilot_proxy.run_plans import CORE_PROFILE, write_run_plans
from pilot_proxy.run_scores import build_run_scores
from pilot_proxy.run_spectrogram import (
    run_spectrogram,
    write_masked_summary,
    write_run_spectrogram,
)
from pilot_proxy.runtime_bundle import (
    export_runtime_weight_bundle,
    validate_runtime_weight_bundle,
)
from pilot_proxy.secondary_python import (
    package_only_pythonpath,
    prepend_pythonpath,
)

# GNU Radio is typically installed into the system Python on Linux/WSL, while
# CUDA/CuPy is installed in the active detector environment.
DEFAULT_GNURADIO_PYTHON = "/usr/bin/python3"
DEFAULT_CLEAN_ATSC_IQ_SAMPLES = 600_000
DEFAULT_TS_PACKETS = 4096
# A fixed seed keeps generate-atsc output reproducible.
DEFAULT_GENERATOR_SEED = 12345
DEFAULT_FRAME_SIZE_SAMPLES = 16_384
DEFAULT_NUM_INPUT_STREAMS = 1
DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ = 10.0
HZ_PER_MHZ = 1.0e6


def _run_module(
    module_name: str,
    args: list[str],
    *,
    python: str | None = None,
    env_updates: dict[str, str] | None = None,
) -> None:
    with package_only_pythonpath(python) as package_bridge:
        env = os.environ.copy()
        if package_bridge is not None:
            prepend_pythonpath(env, package_bridge)
        elif SOURCE_CHECKOUT_ROOT is not None:
            # Same-interpreter source runs still need the checkout import root
            # when invoked without an editable installation.
            prepend_pythonpath(env, SOURCE_CHECKOUT_ROOT / "src")
        if env_updates:
            env.update(env_updates)

        cmd = [str(python or sys.executable), "-m", module_name, *args]
        print(f"[run] {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=Path.cwd(), env=env)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit={result.returncode}: {cmd}")


def _caller_absolute_path(value: str | os.PathLike[str]) -> str:
    return str(resolve_user_path(value))


def _cmd_generate_atsc(args: argparse.Namespace) -> None:
    argv = [
        "--output-iq",
        _caller_absolute_path(args.output_iq),
        "--output-ts",
        _caller_absolute_path(args.output_ts),
        "--num-iq-samples",
        str(args.num_iq_samples),
        "--num-ts-packets",
        str(args.num_ts_packets),
        "--seed",
        str(args.seed),
    ]
    env_updates = {}
    if not args.python_user_site:
        env_updates["PYTHONNOUSERSITE"] = "1"
    _run_module(
        "pilot_proxy.testbench.generate_atsc_signal",
        argv,
        python=args.gnuradio_python,
        env_updates=env_updates,
    )


def _cmd_audit_atsc(args: argparse.Namespace) -> None:
    argv = [
        "--input-iq",
        args.input_iq,
        "--output-json",
        args.output_json,
    ]
    if args.fail_on_quality:
        argv.append("--fail-on-quality")
    _run_module("pilot_proxy.testbench.audit_atsc_signal", argv)


def _cmd_detect(args: argparse.Namespace) -> None:
    argv = [
        "--input-detector-matrix",
        args.input_detector_matrix,
        "--output-json",
        args.output_json,
        "--lib-path",
        args.lib_path,
        "--weights-path",
        args.weights_path,
        "--pilot-below-data-db",
        str(args.pilot_below_data_db),
        "--bin-enbw-hz",
        str(args.bin_enbw_hz),
        "--pilot-capture-efficiency",
        str(args.pilot_capture_efficiency),
        "--dtv-bandwidth-hz",
        str(args.dtv_bandwidth_hz),
        "--pilot-frequency-tolerance-hz",
        str(args.pilot_frequency_tolerance_hz),
        "--num-input-streams",
        str(args.num_input_streams),
    ]
    if args.frame_size_samples is not None:
        argv.extend(["--frame-size-samples", str(args.frame_size_samples)])
    if args.physical_channel is not None:
        argv.extend(["--physical-channel", str(args.physical_channel)])
    if args.dtv_pilot_mhz is not None:
        argv.extend(["--dtv-pilot-mhz", str(args.dtv_pilot_mhz)])
    _run_module("pilot_proxy.detect", argv)


def _add_optional_arg(
    argv: list[str],
    flag: str,
    value: object | None,
) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _cmd_chime_plot(args: argparse.Namespace) -> None:
    argv = ["--run-dir", str(args.run_dir)]
    if args.clean_figures:
        argv.append("--clean-figures")
    _run_module("pilot_proxy.chime.plots", argv)


def _cmd_validate_products(args: argparse.Namespace) -> None:
    argv = [
        "--run-dir",
        str(args.run_dir),
    ]
    _add_optional_arg(argv, "--output-json", args.output_json)
    _run_module("pilot_proxy.chime.validate_products", argv)


def _cmd_check_layout(args: argparse.Namespace) -> None:
    profile = (
        load_receiver_profile(args.receiver_profile)
        if args.receiver_profile is not None
        else default_reference_receiver_profile(
            frame_size_samples=int(
                DEFAULT_FRAME_SIZE_SAMPLES
                if args.frame_size_samples is None
                else args.frame_size_samples
            ),
            num_input_streams=int(
                DEFAULT_NUM_INPUT_STREAMS
                if args.num_input_streams is None
                else args.num_input_streams
            ),
        )
    )
    stream_map = None if args.stream_map is None else load_stream_map(args.stream_map)
    frame_size_samples = int(
        profile.frame_size_samples
        if args.frame_size_samples is None
        else args.frame_size_samples
    )
    num_input_streams = int(
        profile.num_input_streams
        if args.num_input_streams is None
        else args.num_input_streams
    )
    if stream_map is not None:
        validate_integration_compatibility(profile=profile, stream_map=stream_map)
        num_input_streams = int(stream_map.num_streams)
    detector_window_samples = int(
        profile.detector_window_samples
        if args.detector_window_samples is None
        else args.detector_window_samples
    )
    check = layout_uint64_bound_check(
        frame_size_samples=frame_size_samples,
        detector_window_samples=detector_window_samples,
        num_input_streams=num_input_streams,
        num_selected_channels=int(args.num_selected_channels),
    )
    payload = {
        "schema_version": "pilotproxy_layout_check_v1",
        "receiver_profile": profile.to_dict(),
        "stream_map": None if stream_map is None else stream_map.to_dict(),
        "layout_check": check,
    }
    if args.output_json:
        write_json_strict(Path(args.output_json), payload, indent=2, sort_keys=True)
        print(f"Wrote {args.output_json}")
    print("detector_rows_per_frame, power_sum_fits_uint64, recommended_batching")
    print(
        f"{int(check['detector_rows_per_frame'])}, "
        f"{bool(check['power_sum_fits_uint64'])}, "
        f"{check['recommended_batching']}"
    )


def _cmd_check_profile(args: argparse.Namespace) -> None:
    profile = load_receiver_profile(args.receiver_profile)
    detector_core = (
        None
        if args.detector_core_profile is None
        else load_detector_core_profile(args.detector_core_profile)
    )
    stream_map = None if args.stream_map is None else load_stream_map(args.stream_map)
    validate_integration_compatibility(
        profile=profile,
        detector_core=detector_core,
        stream_map=stream_map,
    )
    payload = {
        "schema_version": "pilotproxy_profile_check_v1",
        "receiver_profile": profile.to_dict(),
        "detector_core_profile": (
            None if detector_core is None else detector_core.to_dict()
        ),
        "stream_map": None if stream_map is None else stream_map.to_dict(),
    }
    if args.output_json:
        write_json_strict(Path(args.output_json), payload, indent=2, sort_keys=True)
        print(f"Wrote {args.output_json}")
    print("receiver_profile_id, num_input_streams, frame_size_samples, nyquist_zone")
    print(
        f"{profile.receiver_profile_id}, {int(profile.num_input_streams)}, "
        f"{int(profile.frame_size_samples)}, {int(profile.nyquist_zone)}"
    )


def _cmd_make_weights(args: argparse.Namespace) -> None:
    profile = load_receiver_profile(args.receiver_profile)
    detector_core = load_detector_core_profile(args.detector_core_profile)
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    pilot_overrides_hz: dict[int, float] = {}
    for item in args.pilot_override or ():
        channel_text, _, hz_text = item.partition("=")
        if not hz_text:
            raise SystemExit(f"--pilot-override expects CH=HZ, got {item!r}")
        pilot_overrides_hz[int(channel_text)] = float(hz_text)
    manifest = write_weight_bank_from_receiver_profile(
        output_path=args.output,
        profile=profile,
        core=detector_core,
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
        pilot_overrides_hz=pilot_overrides_hz or None,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output}.manifest.json")
    print("physical_channels, receiver_profile_hash")
    print(
        f"{','.join(str(channel) for channel in physical_channels)}, "
        f"{manifest['receiver_profile_hash']}"
    )
    _print_adaptive_reference_diagnostics(
        cast(Sequence[Mapping[str, Any]], manifest.get("target_reference_layout", ()))
    )


def _cmd_export_runtime_weight_bundle(args: argparse.Namespace) -> None:
    physical_channels = parse_physical_channel_selection(
        physical_channels=args.physical_channel,
        physical_channel_range=args.physical_channel_range,
    )
    outputs = export_runtime_weight_bundle(
        receiver_profile_path=Path(args.receiver_profile),
        detector_core_profile_path=Path(args.detector_core_profile),
        physical_channels=physical_channels,
        weight_coordinate_system=args.weight_coordinate_system,
        output_dir=Path(args.output_dir),
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


def _cmd_validate_runtime_weight_bundle(args: argparse.Namespace) -> None:
    report = validate_runtime_weight_bundle(
        bundle_dir=Path(args.bundle_dir),
        output_json=args.output_json,
    )
    print("valid, num_errors, bundle_dir", flush=True)
    print(
        f"{bool(report['valid'])}, {int(report['num_errors'])}, {report['bundle_dir']}",
        flush=True,
    )
    for error in report["errors"]:
        print(f"ERROR {error['check']}: {error['message']}", flush=True)
    if not report["valid"]:
        raise SystemExit(1)


def _add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", choices=THRESHOLDS, default="knee",
                        help="Word of each channel. knee: the default word, the knee for v1 and v2 and the control's 1%% word "
                             "scaled to the channel for v3. power_limit: the word that keeps the most frames under a "
                             "checked power limit.")
    parser.add_argument("--power-limit", type=Path, default=None,
                        help="For --threshold power_limit: JSON naming the response table and the checked power limits, "
                             "with the curvature and the pilot passband factor of each channel and the margin.")
    parser.add_argument("--decision-version", choices=FINE_DECISION_VERSIONS, default=None,
                        help="Test the words are set for (default: the calibration default). "
                             "fine_decision_v1 for a Kotekan stage that runs only that test.")
    parser.add_argument("--night", type=float, nargs=2, default=None, metavar=("LATITUDE_DEG", "LONGITUDE_DEG_EAST"),
                        help="Take the control quantile of the default word over the control frames recorded with the "
                             "Sun below the horizon at this site, as CHIME's 21 cm analyses use the data. "
                             "DRAO is 49.3208 -119.6236.")


def _cmd_plan_run(args: argparse.Namespace) -> None:
    plans = write_run_plans(bundle_dir=args.bundle_dir, runs=args.run, output_dir=args.output_dir, epoch=args.epoch,
                            receiver_profile=args.receiver_profile, detector_core_profile=args.detector_core_profile,
                            null_quantile=args.null_quantile, calibration_share=args.calibration_share,
                            threshold=args.threshold, power_limit=args.power_limit,
                            decision_version=args.decision_version, night_site=args.night)
    for channel, paths in plans["calibrate"]["channels"].items():
        print(f"channel {channel}: {len(paths)} files to calibrate, {len(plans['test']['channels'][channel])} to test")


def _cmd_process_run(args: argparse.Namespace) -> None:
    summary = process_run(bundle_dir=args.bundle_dir, runs=args.run, epoch=args.epoch, output_dir=args.output_dir,
                          calibration_share=args.calibration_share, null_quantile=args.null_quantile,
                          receiver_profile=args.receiver_profile, detector_core_profile=args.detector_core_profile,
                          chunk_seconds=args.chunk_seconds, threshold=args.threshold, power_limit=args.power_limit,
                          decision_version=args.decision_version, night_site=args.night,
                          log=lambda line: print(line, flush=True))
    print((args.output_dir / "summary.md").read_text(), end="")
    if not summary["passed"]:
        raise SystemExit(1)


def _cmd_run_spectrogram(args: argparse.Namespace) -> None:
    results = []
    for channel in args.channel:
        result = run_spectrogram(bundle_dir=args.bundle_dir, runs=args.run, channel=channel,
                                 chunk_seconds=args.chunk_seconds)
        data, figure = write_run_spectrogram(result, args.output_dir)
        results.append(result)
        print(f"channel {channel}: {int(result['kept'].sum())} of {int(result['frames'].sum())} frames kept, "
              f"{data}, {figure}")
    print(write_masked_summary(results, args.output_dir))


def _cmd_run_scores(args: argparse.Namespace) -> None:
    for channel in args.channel:
        result = build_run_scores(bundle_dir=args.bundle_dir, runs=args.run, channel=channel, output_dir=args.output_dir,
                                  event_seconds=args.event_seconds, jobs=args.jobs)
        print(f"channel {channel}: {result['frames']} frames, {result['events']} events, "
              f"{result['masked_percent']}% masked at the bundle word")


def _cmd_calibrate_bundle(args: argparse.Namespace) -> None:
    report = calibrate_bundle(plan_path=args.plan, output_dir=args.output_dir)
    print(f"threshold {report['threshold']}")
    print(f"floor {report['floor_word_q16']} (eta {report['floor_eta']:.4f}) from {report['floor_source']}")
    for channel, row in report["channels"].items():
        print(f"channel {channel}: anchor {row['anchor_bin']} ({row['anchor_source']}), "
              f"word {row['word_q16']} (eta {row['eta']:.4f}) from {row['word_source']}")


def _cmd_test_bundle(args: argparse.Namespace) -> None:
    report = test_bundle(bundle_dir=args.bundle_dir, plan_path=args.plan, output_json=args.output_json)
    for row in report["checks"]:
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['check']}: {row['detail']}")
    if not report["passed"]:
        raise SystemExit(1)


def _cmd_qualify_bundle(args: argparse.Namespace) -> None:
    summary = qualify_bundle(bundle_dir=args.bundle_dir, test_report=args.test_report,
                             reviewer=args.reviewer, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


def _cmd_list_channels(args: argparse.Namespace) -> None:
    weights = DetectorWeightBank(explicit_path=args.weights_path)
    layouts: list[Mapping[str, Any]] = []
    print(
        "physical_channel,pilot_mhz,coarse_channel_index,"
        "adaptive_reference_placement"
    )
    for channel in weights.supported_physical_channels():
        layout = weights.layout_for_physical_channel(channel)
        layouts.append(cast(Mapping[str, Any], layout))
        coarse_channel_index = cast(int, layout["coarse_channel_index"])
        adaptive_reference_placement = cast(
            bool,
            layout.get("adaptive_reference_placement", False),
        )
        print(
            f"{channel},"
            f"{physical_channel_to_pilot_hz(channel) / HZ_PER_MHZ:.6f},"
            f"{int(coarse_channel_index)},"
            f"{bool(adaptive_reference_placement)}"
        )
    _print_adaptive_reference_diagnostics(layouts)


# Steps of the characterize group. Each step is a module with its own
# ``main(argv)``, and the group passes the remaining arguments to it.
CHARACTERIZE_STEPS = {
    "archive": "pilot_proxy.characterization.run",
    "histogram-frames": "pilot_proxy.characterization.histogram_frames",
    "fine-histogram-frames": "pilot_proxy.detectors.narrowband_marker.fine_histogram_frames",
    "score-cache": "pilot_proxy.characterization.score_cache",
}


def _cmd_step(steps: dict[str, str]):
    def run(args: argparse.Namespace) -> None:
        import importlib

        module = importlib.import_module(steps[args.step])
        status = module.main(list(args.step_args))
        if status:
            raise SystemExit(status)
    return run


def _print_adaptive_reference_diagnostics(
    layouts: Sequence[Mapping[str, Any]],
) -> None:
    for layout in layouts:
        if not bool(layout.get("adaptive_reference_placement", False)):
            continue
        channel = layout.get("physical_channel", "?")
        coarse = layout.get("coarse_channel_index", "?")
        status = str(layout.get("reference_placement_status", "unknown"))
        reason = str(layout.get("reference_placement_reason", "unknown"))
        lower = layout.get("lower_reference_offset_bins", "?")
        upper = layout.get("upper_reference_offset_bins", "?")
        details: list[str] = []
        if bool(layout.get("lower_reference_edge_wrapped", False)):
            details.append("lower_reference=edge_wrapped")
        if bool(layout.get("upper_reference_edge_wrapped", False)):
            details.append("upper_reference=edge_wrapped")
        if details and "frame_origin_description" not in layout:
            # A manifest written before the origin description was recorded
            # gets a note on the wrap, since "edge" alone would read as the
            # channel edge.
            details.append(
                "edge_wrapped=wrap across the baseband-frame origin nu=0 "
                "(the coarse-channel center/DC for a center-at-DC profile, "
                "not the channel edge)"
            )
        if bool(layout.get("lower_reference_dc_shifted", False)):
            details.append("lower_reference=dc_shifted")
        if bool(layout.get("upper_reference_dc_shifted", False)):
            details.append("upper_reference=dc_shifted")
        warning = str(layout.get("placement_warnings", "")).strip()
        detail_text = f"; {'; '.join(details)}" if details else ""
        warning_text = f"; warning={warning}" if warning else ""
        print(
            "NOTE adaptive reference placement: "
            f"physical_channel={channel}, coarse_channel_index={coarse}, "
            f"status={status}, reason={reason}, "
            f"lower_reference_offset_bins={lower}, "
            f"upper_reference_offset_bins={upper}"
            f"{detail_text}{warning_text}",
            file=sys.stderr,
        )
    _print_channel_edge_diagnostics(layouts)


def _reference_crosses_channel_edge(
    layout: Mapping[str, Any], side: str
) -> bool:
    """True when a reference was requested more than fs/2 from the channel center.

    Manifests that record ``*_crosses_channel_edge`` are read directly. In an
    older manifest the stored reference offset is the circular (aliased)
    offset, so it differs from ``target_offset_hz`` plus the reference's
    relative offset only when the request crossed the physical coarse-channel
    edge.
    """
    key = f"{side}_reference_crosses_channel_edge"
    if key in layout:
        return bool(layout[key])
    try:
        target = float(layout["target_offset_hz"])
        relative = float(layout[f"{side}_reference_relative_to_target_hz"])
        stored = float(layout[f"{side}_reference_offset_hz"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs((target + relative) - stored) > 1.0


def _print_channel_edge_diagnostics(
    layouts: Sequence[Mapping[str, Any]],
) -> None:
    for layout in layouts:
        crossing = [
            side
            for side in ("lower", "upper")
            if _reference_crosses_channel_edge(layout, side)
        ]
        if not crossing:
            continue
        channel = layout.get("physical_channel", "?")
        coarse = layout.get("coarse_channel_index", "?")
        note = str(layout.get("channel_edge_notes", "")).strip()
        note_text = f"; {note}" if note else ""
        print(
            "NOTE reference beyond coarse-channel half-width: "
            f"physical_channel={channel}, coarse_channel_index={coarse}, "
            f"references={','.join(crossing)}; the reference aliases to the "
            "opposite channel edge and samples the channelizer roll-off "
            "(placement unchanged; detection arithmetic unaffected)"
            f"{note_text}",
            file=sys.stderr,
        )


def _cmd_replay_fine_capture(args: argparse.Namespace) -> None:
    from pilot_proxy.capture_replay import write_replay

    try:
        report = write_replay(args.plan, args.output_dir)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Replay written to {args.output_dir}: {report['status']}; {report['invalid_rows']} invalid rows retained.")
    if not report["product_agreement"]:
        raise SystemExit(1)


def _cmd_validate_fine_capture_replay(args: argparse.Namespace) -> None:
    from pilot_proxy.capture_replay import validate_replay

    try:
        report = validate_replay(args.replay_dir)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Replay identities and calculations verified: {report['status']}.")
    if not report["product_agreement"]:
        raise SystemExit(1)


def _cmd_compile_operating_policy(args: argparse.Namespace) -> None:
    from pilot_proxy.operating_policy import write_policy

    try:
        report = write_policy(args.plan, args.output_dir, require_complete=args.require_complete)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    state = "complete" if report["deployment_configuration_complete"] else "pending calibration"
    print(f"Operating policy written to {args.output_dir}: {state}; activation is not authorized.")


def _cmd_validate_operating_policy(args: argparse.Namespace) -> None:
    from pilot_proxy.operating_policy import validate_policy

    try:
        validate_policy(args.policy_dir)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print("Operating policy identities and assignments verified; activation is not authorized.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PilotProxy detector, validation, and CHIME workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    from pilot_proxy import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_command(name: str, summary: str) -> argparse.ArgumentParser:
        # One string is both the one-liner in `pilot-proxy --help` and
        # the description at the top of `pilot-proxy <command> --help`.
        return subparsers.add_parser(name, help=summary, description=summary)

    gen = _add_command(
        "generate-atsc",
        "Generate a reference ATSC 8-VSB IQ capture (and transport stream) "
        "with GNU Radio.",
    )
    gen.add_argument(
        "--output-iq",
        default=str(GENERATED_DIR / "atsc" / "atsc_8vsb_complex64.cfile"),
    )
    gen.add_argument(
        "--output-ts",
        default=str(GENERATED_DIR / "atsc" / "atsc_test_pattern.ts"),
    )
    gen.add_argument(
        "--num-iq-samples", type=int, default=DEFAULT_CLEAN_ATSC_IQ_SAMPLES
    )
    gen.add_argument("--num-ts-packets", type=int, default=DEFAULT_TS_PACKETS)
    gen.add_argument("--seed", type=int, default=DEFAULT_GENERATOR_SEED)
    gen.add_argument(
        "--gnuradio-python",
        default=DEFAULT_GNURADIO_PYTHON,
        help="Python executable that can import GNU Radio.",
    )
    gen.add_argument(
        "--python-user-site",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the GNU Radio Python to load user-site packages.",
    )
    gen.set_defaults(func=_cmd_generate_atsc)

    audit = _add_command(
        "audit-atsc",
        "Verify a generated IQ capture: pilot frequency and level, occupied "
        "bandwidth, shelf flatness, edge rolloff.",
    )
    audit.add_argument(
        "--input-iq",
        default=str(GENERATED_DIR / "atsc" / "atsc_8vsb_complex64.cfile"),
    )
    audit.add_argument(
        "--output-json",
        default=str(GENERATED_DIR / "atsc" / "atsc_waveform_audit.json"),
    )
    audit.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit non-zero if any waveform quality check fails.",
    )
    audit.set_defaults(func=_cmd_audit_atsc)

    detect = _add_command(
        "detect",
        "Run the CUDA detector once over a packed detector matrix and write "
        "a detections JSON.",
    )
    detect.add_argument(
        "--input-detector-matrix",
        default=str(GENERATED_DIR / "detector_input" / "detector_matrix_i4.npy"),
        help="Packed int4 detector matrix.",
    )
    detect.add_argument(
        "--output-json",
        default=str(GENERATED_DIR / "detections" / "detect.json"),
    )
    detect.add_argument("--physical-channel", type=int, default=None)
    detect.add_argument("--dtv-pilot-mhz", type=float, default=None)
    detect.add_argument(
        "--pilot-below-data-db",
        dest="pilot_below_data_db",
        type=float,
        default=PILOT_BELOW_DATA_DB,
        help="Positive dB offset: ATSC pilot power below average data-shelf power.",
    )
    detect.add_argument("--bin-enbw-hz", type=float, default=EFFECTIVE_BIN_BW_HZ)
    detect.add_argument(
        "--pilot-capture-efficiency",
        type=float,
        default=PILOT_CAPTURE_EFFICIENCY,
    )
    detect.add_argument("--dtv-bandwidth-hz", type=float, default=DTV_BANDWIDTH_HZ)
    detect.add_argument("--frame-size-samples", type=int, default=None)
    detect.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=DEFAULT_NUM_INPUT_STREAMS,
    )
    detect.add_argument(
        "--pilot-frequency-tolerance-hz",
        type=float,
        default=DEFAULT_PILOT_FREQUENCY_TOLERANCE_HZ,
    )
    detect.add_argument("--lib-path", default=str(DEFAULT_LIB_PATH))
    detect.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    detect.set_defaults(func=_cmd_detect)

    channels = _add_command(
        "list-channels",
        "List the ATSC channels present in a weight bank (CSV on stdout).",
    )
    channels.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS_PATH))
    channels.set_defaults(func=_cmd_list_channels)

    check_profile = _add_command(
        "check-profile",
        "Validate a receiver-profile JSON (and optional stream map) against "
        "the detector-core contract.",
    )
    check_profile.add_argument("--receiver-profile", required=True,
                               help="Receiver-profile JSON to validate.")
    check_profile.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    check_profile.add_argument("--stream-map", default=None)
    check_profile.add_argument("--output-json", default=None)
    check_profile.set_defaults(func=_cmd_check_profile)

    check_layout = _add_command(
        "check-layout",
        "Report the detector row layout and the uint64 accumulator-bound "
        "check for a frame geometry.",
    )
    check_layout.add_argument("--receiver-profile", default=None)
    check_layout.add_argument("--stream-map", default=None)
    check_layout.add_argument(
        "--frame-size-samples",
        dest="frame_size_samples",
        type=int,
        default=None,
    )
    check_layout.add_argument(
        "--num-input-streams",
        dest="num_input_streams",
        type=int,
        default=None,
    )
    check_layout.add_argument(
        "--num-selected-channels",
        type=int,
        default=1,
    )
    check_layout.add_argument(
        "--detector-window-samples",
        type=int,
        default=None,
        help="Detector-window samples; defaults to the receiver profile's "
             "declared value.",
    )
    check_layout.add_argument("--output-json", default=None)
    check_layout.set_defaults(func=_cmd_check_layout)

    make_weights = _add_command(
        "make-weights",
        "Generate a packed int4 detector weight bank from a receiver profile.",
    )
    make_weights.add_argument("--receiver-profile", required=True,
                              help="Receiver-profile JSON describing the "
                                   "channelizer geometry.")
    make_weights.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    make_weights.add_argument(
        "--physical-channel",
        type=int,
        action="append",
        default=None,
        help="ATSC physical channel to include. Repeat as needed.",
    )
    make_weights.add_argument(
        "--physical-channel-range",
        default=None,
        help="Inclusive range like 14:36, comma list, or single channel.",
    )
    make_weights.add_argument(
        "--weight-coordinate-system",
        type=normalize_weight_coordinate_system,
        choices=[
            WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
        help=(
            "Coordinate convention used to generate the weights. Use "
            "post_nyquist_zone_normalization for detector-coordinate weights "
            "(input windows of a mirrored, even-zone channel time-reversed "
            "first), or raw_input_frequency_coordinate for native "
            "receiver-coordinate weights."
        ),
    )
    make_weights.add_argument(
        "--pilot-override",
        action="append",
        default=None,
        metavar="CH=HZ",
        help="Build a channel's template at a measured carrier instead of the nominal ATSC "
        "pilot, e.g. 33=584305600. The manifest keeps the nominal pilot as the key. Repeat as needed.",
    )
    make_weights.add_argument("--output", required=True,
                              help="Output path for the packed weight bank "
                                   "(.bin); a .manifest.json is written "
                                   "beside it.")
    make_weights.set_defaults(func=_cmd_make_weights)

    export_bundle = _add_command(
        "export-runtime-weight-bundle",
        "Export the compact runtime bundle (weights plus manifests) for "
        "deployment handoff.",
    )
    export_bundle.add_argument("--receiver-profile", required=True,
                               help="Receiver-profile JSON describing the "
                                    "channelizer geometry.")
    export_bundle.add_argument(
        "--detector-core-profile",
        default=str(DEFAULT_DETECTOR_CORE_PROFILE),
    )
    export_bundle.add_argument(
        "--weight-coordinate-system",
        type=normalize_weight_coordinate_system,
        choices=[
            WEIGHT_COORDINATE_POST_NYQUIST_ZONE,
            WEIGHT_COORDINATE_RAW_INPUT,
        ],
        required=True,
    )
    export_bundle.add_argument(
        "--physical-channel",
        type=int,
        action="append",
        default=None,
        help="ATSC physical channel to include. Repeat as needed.",
    )
    export_bundle.add_argument(
        "--physical-channel-range",
        default=None,
        help="Inclusive range like 14:36, comma list, or single channel.",
    )
    export_bundle.add_argument("--output-dir", required=True,
                               help="Directory to write the runtime bundle "
                                    "into.")
    export_bundle.set_defaults(func=_cmd_export_runtime_weight_bundle)

    validate_bundle = _add_command(
        "validate-runtime-weight-bundle",
        "Re-validate a bundle directory written by "
        "export-runtime-weight-bundle.",
    )
    validate_bundle.add_argument("--bundle-dir", type=Path, required=True,
                                 help="Bundle directory to validate.")
    validate_bundle.add_argument("--output-json", type=Path, default=None)
    validate_bundle.set_defaults(func=_cmd_validate_runtime_weight_bundle)

    calibrate = _add_command(
        "calibrate-bundle",
        "Build a detect-only commissioning bundle from captures.",
    )
    calibrate.add_argument("--plan", type=Path, required=True, help="Calibration plan JSON.")
    calibrate.add_argument("--output-dir", type=Path, required=True, help="New bundle directory.")
    calibrate.set_defaults(func=_cmd_calibrate_bundle)

    plan_run = _add_command(
        "plan-run",
        "Split a detect-only run into calibrate.json and test.json.",
    )
    plan_run.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    plan_run.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns, such as 'dtv/acq_*/dtv_fine_powers*.h5'.")
    plan_run.add_argument("--output-dir", type=Path, required=True)
    plan_run.add_argument("--epoch", required=True, help="When and where the run was recorded.")
    plan_run.add_argument("--receiver-profile", type=Path, default=None)
    plan_run.add_argument("--detector-core-profile", type=Path, default=Path(CORE_PROFILE))
    plan_run.add_argument("--null-quantile", type=float, default=0.99)
    plan_run.add_argument("--calibration-share", type=float, default=2.0 / 3.0,
                          help="Share of each channel's files, from the start, used to calibrate. 0.5 splits 48 h by day.")
    _add_threshold_arguments(plan_run)
    plan_run.set_defaults(func=_cmd_plan_run)

    process = _add_command(
        "process-run",
        "Split a detect-only run, calibrate, test, plot and summarize it in one step.",
    )
    process.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    process.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns, such as 'dtv/acq_*/dtv_fine_powers*.h5'.")
    process.add_argument("--epoch", required=True, help="When and where the run was recorded.")
    process.add_argument("--output-dir", type=Path, required=True, help="New or empty directory.")
    process.add_argument("--calibration-share", type=float, default=2.0 / 3.0,
                         help="Share of each channel's files, from the start, used to calibrate. 0.5 splits 48 h by day.")
    process.add_argument("--null-quantile", type=float, default=0.99)
    process.add_argument("--receiver-profile", type=Path, default=None)
    process.add_argument("--detector-core-profile", type=Path, default=Path(CORE_PROFILE))
    process.add_argument("--chunk-seconds", type=float, default=10.0)
    _add_threshold_arguments(process)
    process.set_defaults(func=_cmd_process_run)

    spectrogram = _add_command(
        "run-spectrogram",
        "Plot the fine powers of a detect-only run with and without the mask.",
    )
    spectrogram.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    spectrogram.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns, such as 'dtv/acq_*/dtv_fine_powers*.h5'.")
    spectrogram.add_argument("--channel", type=int, nargs="+", required=True, help="Physical channels.")
    spectrogram.add_argument("--chunk-seconds", type=float, default=10.0)
    spectrogram.add_argument("--output-dir", type=Path, required=True)
    spectrogram.set_defaults(func=_cmd_run_spectrogram)

    scores = _add_command(
        "run-scores",
        "Write the threshold rule's score cache from a detect-only run.",
    )
    scores.add_argument("--bundle-dir", type=Path, required=True, help="Bundle of the run.")
    scores.add_argument("--run", nargs="+", required=True, help="dtv_fine_powers files or glob patterns.")
    scores.add_argument("--channel", type=int, nargs="+", required=True, help="Physical channels.")
    scores.add_argument("--event-seconds", type=float, default=10.0, help="Chunk that clusters the error.")
    scores.add_argument("--jobs", type=int, default=1, help="Files scored in parallel.")
    scores.add_argument("--output-dir", type=Path, required=True)
    scores.set_defaults(func=_cmd_run_scores)

    test = _add_command(
        "test-bundle",
        "Check a bundle on captures it was not made from.",
    )
    test.add_argument("--bundle-dir", type=Path, required=True, help="Bundle directory to test.")
    test.add_argument("--plan", type=Path, required=True, help="Test plan JSON.")
    test.add_argument("--output-json", type=Path, default=None)
    test.set_defaults(func=_cmd_test_bundle)

    qualify = _add_command(
        "qualify-bundle",
        "Copy a tested bundle with its channels qualified for masking.",
    )
    qualify.add_argument("--bundle-dir", type=Path, required=True, help="Bundle that was tested.")
    qualify.add_argument("--test-report", type=Path, required=True,
                         help="Passing test-bundle --output-json report of that bundle.")
    qualify.add_argument("--reviewer", required=True, help="Person who read the report and qualifies the bundle.")
    qualify.add_argument("--output-dir", type=Path, required=True, help="New bundle directory.")
    qualify.set_defaults(func=_cmd_qualify_bundle)

    fine_capture = _add_command(
        "replay-fine-capture",
        "Replay declared native voltage captures and export exact fine scores with validity.",
    )
    fine_capture.add_argument("--plan", type=Path, required=True)
    fine_capture.add_argument("--output-dir", type=Path, required=True,
                              help="New evidence directory; existing outputs are never overwritten.")
    fine_capture.set_defaults(func=_cmd_replay_fine_capture)

    validate_capture = _add_command(
        "validate-fine-capture-replay",
        "Authenticate and recompute a fine capture replay.",
    )
    validate_capture.add_argument("--replay-dir", type=Path, required=True)
    validate_capture.set_defaults(func=_cmd_validate_fine_capture_replay)

    operating_policy = _add_command(
        "compile-operating-policy",
        "Compile the threshold rule's decisions into fine thresholds and the permanent mask for offline deployment review.",
    )
    operating_policy.add_argument("--plan", type=Path, required=True)
    operating_policy.add_argument("--output-dir", type=Path, required=True,
                                  help="New review directory; existing outputs are never overwritten.")
    operating_policy.add_argument("--require-complete", action="store_true",
                                  help="Refuse when a deployed channel's bundle profile does not carry its word yet.")
    operating_policy.set_defaults(func=_cmd_compile_operating_policy)

    validate_operating_policy = _add_command(
        "validate-operating-policy",
        "Authenticate and recompile an operating-policy review without activating it.",
    )
    validate_operating_policy.add_argument("--policy-dir", type=Path, required=True)
    validate_operating_policy.set_defaults(func=_cmd_validate_operating_policy)

    chime_plot = _add_command(
        "chime-plot",
        "Regenerate the diagnostic figures for a completed CHIME run "
        "directory.",
    )
    chime_plot.add_argument("--run-dir", type=Path, required=True,
                            help="CHIME run directory.")
    chime_plot.add_argument(
        "--clean-figures",
        action="store_true",
        help="Delete known CHIME figures before regenerating plots.",
    )
    chime_plot.set_defaults(func=_cmd_chime_plot)

    validate_products = _add_command(
        "validate-products",
        "Validate a CHIME run directory's products against the current product "
        "schema.",
    )
    validate_products.add_argument("--run-dir", type=Path, required=True,
                                   help="CHIME run directory.")
    validate_products.add_argument("--output-json", type=Path, default=None,
                                   help="Also write the validation report as "
                                        "JSON.")
    validate_products.set_defaults(func=_cmd_validate_products)

    characterize = _add_command(
        "characterize",
        "Archive characterization: the release the frame exports are keyed to (archive), "
        "the frame exports and the score cache.",
    )
    characterize.add_argument("step", choices=sorted(CHARACTERIZE_STEPS))
    characterize.add_argument("step_args", nargs=argparse.REMAINDER,
                              help="arguments of the step (see pilot-proxy characterize STEP --help)")
    characterize.set_defaults(func=_cmd_step(CHARACTERIZE_STEPS))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # A detached launcher (``nohup cmd &`` in a script) starts the process
    # with SIGINT ignored, and Python keeps that setting. An interrupt is then
    # lost and a long run cannot be stopped, so the default handler is
    # restored here.
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
