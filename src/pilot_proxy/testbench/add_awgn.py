#!/usr/bin/env python3
# coding=utf-8
"""Generate noisy complex IQ with GNU Radio's Gaussian noise source."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pilot_proxy.json_utils import write_json_strict

DB_LINEAR_BASE = 10.0
DB_POWER_FACTOR = 10.0
# GNU Radio vector sources are cheapest when padded to a small item multiple.
GNU_RADIO_VECTOR_ALIGNMENT_SAMPLES = 4


def _import_gnuradio_awgn():
    try:
        from gnuradio import analog, blocks, gr
    except ImportError as exc:
        raise SystemExit(
            "GNU Radio is required for GNU Radio AWGN generation. "
            "Run this command with the GNU Radio Python, typically "
            "/usr/bin/python3 on WSL, and set PYTHONNOUSERSITE=1 if "
            "user-site packages shadow the system GNU Radio NumPy."
        ) from exc
    return analog, blocks, gr


def _signal_and_noise_power_for_snr(
    clean: np.ndarray,
    *,
    snr_db: float,
    sample_rate_hz: float,
    snr_bandwidth_hz: float,
) -> tuple[float, float]:
    for name, value in (("snr_db", snr_db), ("sample_rate_hz", sample_rate_hz),
                        ("snr_bandwidth_hz", snr_bandwidth_hz)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    signal_power = float(np.mean(np.abs(clean.astype(np.complex64)) ** 2))
    if not np.isfinite(signal_power) or signal_power <= 0.0:
        raise ValueError("signal power must be positive and finite.")
    if sample_rate_hz <= 0.0 or snr_bandwidth_hz <= 0.0:
        raise ValueError("sample_rate_hz and snr_bandwidth_hz must be positive.")
    try:
        noise_power = signal_power / float(
            DB_LINEAR_BASE ** (float(snr_db) / DB_POWER_FACTOR)
        )
        noise_power *= float(sample_rate_hz) / float(snr_bandwidth_hz)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("requested noise power is outside the supported range.") from exc
    if not math.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("requested noise power must be positive and finite.")
    return signal_power, noise_power


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read complex64 IQ, add GNU Radio analog.noise_source_c Gaussian "
            "noise with blocks.add_cc, and write complex64 IQ."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-iq", type=Path, required=True)
    parser.add_argument("--output-iq", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--snr-bandwidth-hz", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive.")

    analog, blocks, gr = _import_gnuradio_awgn()
    clean = np.fromfile(args.input_iq, dtype=np.complex64, count=args.num_samples)
    if clean.size != args.num_samples:
        raise SystemExit(
            f"Input IQ is too short: need {args.num_samples}, got {clean.size}."
        )
    signal_power, noise_power = _signal_and_noise_power_for_snr(
        clean,
        snr_db=float(args.snr_db),
        sample_rate_hz=float(args.sample_rate_hz),
        snr_bandwidth_hz=float(args.snr_bandwidth_hz),
    )
    noise_amplitude = math.sqrt(noise_power)

    args.output_iq.parent.mkdir(parents=True, exist_ok=True)
    tb = gr.top_block()
    flowgraph_samples = int(
        math.ceil(float(args.num_samples) / GNU_RADIO_VECTOR_ALIGNMENT_SAMPLES)
        * GNU_RADIO_VECTOR_ALIGNMENT_SAMPLES
    )
    if flowgraph_samples != args.num_samples:
        padded_clean = np.pad(clean, (0, flowgraph_samples - args.num_samples))
    else:
        padded_clean = clean
    signal_src = blocks.vector_source_c(padded_clean.tolist(), False)
    head = blocks.head(gr.sizeof_gr_complex, flowgraph_samples)
    noise_src = analog.noise_source_c(
        analog.GR_GAUSSIAN,
        float(noise_amplitude),
        int(args.seed),
    )
    adder = blocks.add_cc()
    sink = blocks.vector_sink_c()
    tb.connect(signal_src, (adder, 0))
    tb.connect(noise_src, (adder, 1))
    tb.connect(adder, head)
    tb.connect(head, sink)
    tb.run()

    noisy = np.asarray(sink.data(), dtype=np.complex64)
    if noisy.size < args.num_samples:
        raise RuntimeError(
            "GNU Radio produced "
            f"{noisy.size} samples; expected {args.num_samples}."
        )
    noisy[: args.num_samples].tofile(args.output_iq)

    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "noise_source": "gnuradio",
            "gnuradio_block": "analog.noise_source_c",
            "add_block": "blocks.add_cc",
            "noise_type": "analog.GR_GAUSSIAN",
            "num_samples": int(args.num_samples),
            "flowgraph_samples": int(flowgraph_samples),
            "snr_db": float(args.snr_db),
            "sample_rate_hz": float(args.sample_rate_hz),
            "snr_bandwidth_hz": float(args.snr_bandwidth_hz),
            "seed": int(args.seed),
            "signal_power": float(signal_power),
            "requested_noise_power": float(noise_power),
            "noise_amplitude": float(noise_amplitude),
        }
        write_json_strict(args.metadata_json, metadata, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
