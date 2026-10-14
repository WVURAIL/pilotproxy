#!/usr/bin/env python3
# coding=utf-8
"""Generate a clean GNU Radio ATSC 8VSB complex64 waveform."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pilot_proxy.json_utils import write_json_strict

# GNU Radio's ATSC transmitter chain uses the A/53 8-VSB symbol-rate formula.
GNU_RADIO_ATSC_SYMBOL_RATE_HZ = 4_500_000.0 / 286.0 * 684.0
ATSC_CHANNEL_WIDTH_HZ = 6.0e6
HALF_SCALE = 2.0
ATSC_PILOT_OFFSET_HZ = (
    ATSC_CHANNEL_WIDTH_HZ - GNU_RADIO_ATSC_SYMBOL_RATE_HZ / HALF_SCALE
) / HALF_SCALE

DEFAULT_CLEAN_ATSC_IQ_SAMPLES = 600_000
DEFAULT_TS_PACKETS = 4096
DEFAULT_GENERATOR_SEED = 12345

# MPEG transport-stream packets are fixed at 188 bytes with a 4-byte header.
MPEG_TS_PACKET_BYTES = 188
MPEG_TS_HEADER_BYTES = 4
MPEG_TS_PAYLOAD_BYTES = MPEG_TS_PACKET_BYTES - MPEG_TS_HEADER_BYTES
# Header fields below follow ISO/IEC 13818-1 transport-stream packing.
MPEG_TS_SYNC_BYTE = 0x47
MPEG_TS_PAYLOAD_UNIT_START_FLAG = 0x40
MPEG_TS_PAYLOAD_ONLY_CONTROL_FLAG = 0x10
MPEG_TS_PID_HIGH_BITS_MASK = 0x1F
MPEG_TS_BYTE_MASK = 0xFF
MPEG_TS_CONTINUITY_COUNTER_MASK = 0x0F
MPEG_TS_PID_HIGH_SHIFT = 8
# Avoid PID 8191, the reserved null-packet PID, in deterministic test payloads.
MPEG_TS_PID_MODULUS = 8191

# GNU Radio ATSC blocks below require their historical vector sizes and RRC
# shaping parameters; these are not detector-tunable values.
ATSC_BASEBAND_LOWER_EDGE_HZ = -3.0e6
ATSC_FIELD_MUX_BLOCK_ITEMS = 1024
ATSC_PAYLOAD_ITEMS_PER_BLOCK = 832
ATSC_KEEP_PAYLOAD_OFFSET = 4
RRC_GAIN = 0.11
RRC_SAMPLE_RATE_DIVISOR = 2.0
RRC_ROLLOFF = 0.1152
RRC_NUM_TAPS = 200


def _import_gnuradio_atsc():
    try:
        from gnuradio import blocks, dtv, filter, gr
    except ImportError as exc:
        raise SystemExit(
            "GNU Radio with DTV blocks is required for ATSC generation. "
            "Run this command with the GNU Radio Python, typically "
            "/usr/bin/python3 on WSL, and set PYTHONNOUSERSITE=1 if "
            "user-site packages shadow the system GNU Radio NumPy."
        ) from exc
    return blocks, dtv, filter, gr


def make_transport_stream_packets(num_packets: int, seed: int) -> bytes:
    """Create deterministic 188-byte MPEG-TS-like packets for ATSC input."""
    rng = np.random.default_rng(int(seed))
    packets = bytearray()
    continuity = 0
    for packet_index in range(int(num_packets)):
        pid = int(packet_index % MPEG_TS_PID_MODULUS)
        payload = rng.integers(
            0,
            MPEG_TS_BYTE_MASK + 1,
            size=MPEG_TS_PAYLOAD_BYTES,
            dtype=np.uint8,
        )
        packet = bytearray(MPEG_TS_PACKET_BYTES)
        packet[0] = MPEG_TS_SYNC_BYTE
        packet[1] = MPEG_TS_PAYLOAD_UNIT_START_FLAG | (
            (pid >> MPEG_TS_PID_HIGH_SHIFT) & MPEG_TS_PID_HIGH_BITS_MASK
        )
        packet[2] = pid & MPEG_TS_BYTE_MASK
        packet[3] = (
            MPEG_TS_PAYLOAD_ONLY_CONTROL_FLAG
            | (continuity & MPEG_TS_CONTINUITY_COUNTER_MASK)
        )
        packet[MPEG_TS_HEADER_BYTES:] = payload.tobytes()
        packets.extend(packet)
        continuity = (continuity + 1) & MPEG_TS_CONTINUITY_COUNTER_MASK
    return bytes(packets)


def generate_atsc_iq(
    *,
    output_iq: Path,
    output_ts: Path | None,
    num_iq_samples: int,
    num_ts_packets: int,
    seed: int,
    symbol_rate_hz: float,
    pilot_offset_hz: float,
) -> dict[str, object]:
    """Run the GNU Radio ATSC transmit chain and write complex64 IQ."""
    if num_iq_samples <= 0:
        raise ValueError("num_iq_samples must be positive.")
    if num_ts_packets <= 0:
        raise ValueError("num_ts_packets must be positive.")

    blocks, dtv, filter, gr = _import_gnuradio_atsc()
    ts_bytes = make_transport_stream_packets(num_ts_packets, seed)
    if output_ts is not None:
        output_ts.parent.mkdir(parents=True, exist_ok=True)
        output_ts.write_bytes(ts_bytes)

    symbol_rate_hz = float(symbol_rate_hz)
    pilot_offset_hz = float(pilot_offset_hz)
    phase_inc = (
        (ATSC_BASEBAND_LOWER_EDGE_HZ + pilot_offset_hz) / symbol_rate_hz
    ) * (HALF_SCALE * math.pi)

    tb = gr.top_block()
    source = blocks.vector_source_b(list(ts_bytes), True)
    pad = dtv.atsc_pad()
    randomizer = dtv.atsc_randomizer()
    rs_encoder = dtv.atsc_rs_encoder()
    interleaver = dtv.atsc_interleaver()
    trellis = dtv.atsc_trellis_encoder()
    field_sync = dtv.atsc_field_sync_mux()
    vector_to_stream = blocks.vector_to_stream(
        gr.sizeof_char,
        ATSC_FIELD_MUX_BLOCK_ITEMS,
    )
    keep_payload = blocks.keep_m_in_n(
        gr.sizeof_char,
        ATSC_PAYLOAD_ITEMS_PER_BLOCK,
        ATSC_FIELD_MUX_BLOCK_ITEMS,
        ATSC_KEEP_PAYLOAD_OFFSET,
    )
    modulator = dtv.dvbs2_modulator_bc(
        dtv.FECFRAME_NORMAL,
        dtv.C1_4,
        dtv.MOD_8VSB,
        dtv.INTERPOLATION_OFF,
    )
    rotator = blocks.rotator_cc(phase_inc)
    taps = filter.firdes.root_raised_cosine(
        RRC_GAIN,
        symbol_rate_hz,
        symbol_rate_hz / RRC_SAMPLE_RATE_DIVISOR,
        RRC_ROLLOFF,
        RRC_NUM_TAPS,
    )
    pulse_shape = filter.fft_filter_ccc(1, taps)
    head = blocks.head(gr.sizeof_gr_complex, int(num_iq_samples))
    sink = blocks.vector_sink_c()

    tb.connect(
        source,
        pad,
        randomizer,
        rs_encoder,
        interleaver,
        trellis,
        field_sync,
        vector_to_stream,
        keep_payload,
        modulator,
        rotator,
        pulse_shape,
        head,
        sink,
    )
    tb.run()

    iq = np.asarray(sink.data(), dtype=np.complex64)
    if iq.size != int(num_iq_samples):
        raise RuntimeError(f"GNU Radio generated {iq.size}; expected {num_iq_samples}.")
    output_iq.parent.mkdir(parents=True, exist_ok=True)
    iq.tofile(output_iq)

    metadata = {
        "schema_version": "fstat_atsc_clean_iq_v1",
        "output_iq": str(output_iq),
        "output_ts": None if output_ts is None else str(output_ts),
        "num_iq_samples": int(num_iq_samples),
        "num_ts_packets": int(num_ts_packets),
        "seed": int(seed),
        "sample_rate_hz": float(symbol_rate_hz),
        "symbol_rate_hz": float(symbol_rate_hz),
        "atsc_channel_width_hz": float(ATSC_CHANNEL_WIDTH_HZ),
        "atsc_pilot_offset_hz": float(pilot_offset_hz),
        "complex_power": float(np.mean(np.abs(iq) ** 2)),
        "gnuradio_blocks": [
            "dtv.atsc_pad",
            "dtv.atsc_randomizer",
            "dtv.atsc_rs_encoder",
            "dtv.atsc_interleaver",
            "dtv.atsc_trellis_encoder",
            "dtv.atsc_field_sync_mux",
            "dtv.dvbs2_modulator_bc(MOD_8VSB)",
        ],
    }
    metadata_path = output_iq.with_suffix(output_iq.suffix + ".json")
    write_json_strict(metadata_path, metadata, indent=2)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a clean ATSC 8VSB complex64 .cfile with GNU Radio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-iq",
        type=Path,
        default=Path("generated/atsc/atsc_8vsb_complex64.cfile"),
    )
    parser.add_argument(
        "--output-ts",
        type=Path,
        default=Path("generated/atsc/atsc_test_pattern.ts"),
    )
    parser.add_argument(
        "--num-iq-samples",
        type=int,
        default=DEFAULT_CLEAN_ATSC_IQ_SAMPLES,
    )
    parser.add_argument("--num-ts-packets", type=int, default=DEFAULT_TS_PACKETS)
    parser.add_argument("--seed", type=int, default=DEFAULT_GENERATOR_SEED)
    parser.add_argument(
        "--symbol-rate-hz",
        type=float,
        default=GNU_RADIO_ATSC_SYMBOL_RATE_HZ,
    )
    parser.add_argument("--pilot-offset-hz", type=float, default=ATSC_PILOT_OFFSET_HZ)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = generate_atsc_iq(
        output_iq=args.output_iq,
        output_ts=args.output_ts,
        num_iq_samples=int(args.num_iq_samples),
        num_ts_packets=int(args.num_ts_packets),
        seed=int(args.seed),
        symbol_rate_hz=float(args.symbol_rate_hz),
        pilot_offset_hz=float(args.pilot_offset_hz),
    )
    print(f"Wrote {metadata['output_iq']}")
    print(f"Wrote {Path(str(metadata['output_iq'])).with_suffix('.cfile.json')}")
    if metadata["output_ts"] is not None:
        print(f"Wrote {metadata['output_ts']}")
    print(f"num_iq_samples={metadata['num_iq_samples']}")
    print(f"complex_power={metadata['complex_power']:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
