# coding=utf-8
"""Inject a known pilot tone into real CHIME baseband files.

Injection--recovery on real data is the sensitivity validation for the
detector: add a complex tone of known amplitude at the pilot frequency to a
copy of a real baseband file, run the untouched production pipeline over the
copy, and show recovered pilot excess tracks injected amplitude.

The harness works entirely in the file's own integer domain. On-disk baseband
is offset-binary 4+4-bit uint8 with components spanning exactly [-8, 7]. The
archive reader and injection path share one format module. Injection decodes
the integer components, adds the float tone, rounds the *sum*, and clips to
[-8, 7] --- so a zero-amplitude pass is byte-identical to the source
unconditionally (including components at -8, which a symmetric +/-7 quantizer
would corrupt), and every injected delta is exact apart from saturation,
which is counted and reported.

Files are copied wholesale and only the ``baseband`` dataset is rewritten in
place, so sibling datasets, attributes, and the ``*_<freq_id>.h5`` naming the
local source keys on are preserved; the injected tree drops straight into
``pilot-proxy chime-scan --source local``.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz

BASEBAND_DATASET = "baseband"
COMPONENT_MIN = -8
COMPONENT_MAX = 7
# Keep the tone comfortably inside the coarse channel; the detector's own
# capture-loss bounds (docs/METHOD_SPEC.md) are derived for in-band pilots.
MAX_ABS_NORMALIZED_FREQUENCY = 0.45
INJECTION_MANIFEST_FILENAME = "injection_manifest.json"


def _format_module():
    """Load the CHIME baseband format helpers when needed."""
    try:
        from pilot_proxy.chime import baseband_format as fmt
    except ImportError as exc:  # pragma: no cover - exercised via CLI error path
        raise SystemExit(
            "inject-pilot-tone could not load the CHIME baseband format. "
            "Reinstall Pilot Proxy with the chime extra."
        ) from exc
    return fmt


def resolve_baseband_frequency_hz(
    center_hz: float,
    *,
    baseband_frequency_hz: float | None = None,
    pilot_frequency_hz: float | None = None,
    physical_channel: int | None = None,
    sample_rate_hz: float,
) -> float:
    """Resolve the injected tone's baseband frequency from one specification.

    Exactly one of ``baseband_frequency_hz`` (relative to the file's channel
    center), ``pilot_frequency_hz`` (absolute RF), or ``physical_channel``
    (ATSC channel, nominal pilot) must be given.
    """
    provided = [
        value
        for value in (baseband_frequency_hz, pilot_frequency_hz, physical_channel)
        if value is not None
    ]
    if len(provided) != 1:
        raise ValueError(
            "specify exactly one of baseband_frequency_hz, pilot_frequency_hz, "
            "or physical_channel"
        )
    if physical_channel is not None:
        pilot_frequency_hz = float(physical_channel_to_pilot_hz(physical_channel))
    if pilot_frequency_hz is not None:
        baseband_frequency_hz = float(pilot_frequency_hz) - float(center_hz)
    if baseband_frequency_hz is None:
        raise RuntimeError("tone frequency resolution produced no baseband frequency")
    limit = MAX_ABS_NORMALIZED_FREQUENCY * float(sample_rate_hz)
    if abs(baseband_frequency_hz) > limit:
        raise ValueError(
            f"baseband tone frequency {baseband_frequency_hz:.1f} Hz is outside "
            f"+/-{limit:.1f} Hz of this file's channel center "
            f"({center_hz/1e6:.6f} MHz); wrong freq_id for this pilot?"
        )
    return float(baseband_frequency_hz)


def _decode_components(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(packed, dtype=np.uint8)
    real = (packed >> 4).astype(np.int32) - 8
    imag = (packed & 0x0F).astype(np.int32) - 8
    return real, imag


def _encode_components(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    return (((real + 8).astype(np.uint8) << 4) | (imag + 8).astype(np.uint8)).astype(
        np.uint8
    )


def inject_tone_into_baseband_file(
    source_path: Path,
    output_path: Path,
    *,
    baseband_frequency_hz: float,
    amplitude_lsb: float,
    phase_seed: int,
) -> dict[str, Any]:
    """Copy ``source_path`` to ``output_path`` with an injected pilot tone.

    The tone is ``amplitude_lsb * exp(j*(2*pi*f_bb*n/FS + phi_feed))`` with an
    independent seeded phase per feed (the detector combines feed powers
    incoherently, so inter-feed phase is irrelevant to recovered F; random
    phases avoid asserting a geometry the analysis never uses). Returns the
    per-file manifest entry, including the exact clip count.
    """
    import h5py

    fmt = _format_module()
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if float(amplitude_lsb) < 0:
        raise ValueError("amplitude_lsb must be >= 0")

    shutil.copy2(source_path, output_path)
    center_hz = fmt.channel_center_hz(str(source_path))

    if float(amplitude_lsb) == 0.0:
        identical = filecmp.cmp(str(source_path), str(output_path), shallow=False)
        if not identical:
            raise RuntimeError(
                f"zero-amplitude copy of {source_path} is not byte-identical; "
                "refusing to proceed (filesystem or format assumption broken)"
            )
        with h5py.File(str(output_path), "r") as handle:
            n_time, n_feeds = handle[BASEBAND_DATASET].shape
        return {
            "source": str(source_path),
            "output": str(output_path),
            "center_hz": float(center_hz),
            "sample_rate_hz": float(fmt.FS),
            "baseband_frequency_hz": float(baseband_frequency_hz),
            "amplitude_lsb": 0.0,
            "phase_seed": int(phase_seed),
            "n_time": int(n_time),
            "n_feeds": int(n_feeds),
            "clip_count": 0,
            "clip_fraction": 0.0,
            "rms_lsb_per_component": None,
            "byte_identical_to_source": True,
        }

    with h5py.File(str(output_path), "r+") as handle:
        packed = handle[BASEBAND_DATASET][...]
        n_time, n_feeds = packed.shape
        real, imag = _decode_components(packed)
        rms = float(
            np.sqrt(
                (real.astype(np.float64) ** 2 + imag.astype(np.float64) ** 2).mean()
                / 2.0
            )
        )
        rng = np.random.default_rng(int(phase_seed))
        phases = 2.0 * np.pi * rng.random(n_feeds)
        n = np.arange(n_time, dtype=np.float64)[:, np.newaxis]
        angle = (
            2.0 * np.pi * float(baseband_frequency_hz) / float(fmt.FS)
        ) * n + phases[np.newaxis, :]
        tone_re = float(amplitude_lsb) * np.cos(angle)
        tone_im = float(amplitude_lsb) * np.sin(angle)
        # Round the SUM (integer + tone), the physical quantizer model; for
        # a = 0 this reduces to the original integers exactly.
        new_re = np.round(real + tone_re).astype(np.int64)
        new_im = np.round(imag + tone_im).astype(np.int64)
        clipped = (
            (new_re < COMPONENT_MIN)
            | (new_re > COMPONENT_MAX)
            | (new_im < COMPONENT_MIN)
            | (new_im > COMPONENT_MAX)
        )
        clip_count = int(np.count_nonzero(clipped))
        new_re = np.clip(new_re, COMPONENT_MIN, COMPONENT_MAX).astype(np.int32)
        new_im = np.clip(new_im, COMPONENT_MIN, COMPONENT_MAX).astype(np.int32)
        handle[BASEBAND_DATASET][...] = _encode_components(new_re, new_im)

    verify_injection_preserves_siblings(source_path, output_path)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "center_hz": float(center_hz),
        "sample_rate_hz": float(fmt.FS),
        "baseband_frequency_hz": float(baseband_frequency_hz),
        "amplitude_lsb": float(amplitude_lsb),
        "phase_seed": int(phase_seed),
        "n_time": int(n_time),
        "n_feeds": int(n_feeds),
        "clip_count": clip_count,
        "clip_fraction": clip_count / float(n_time * n_feeds),
        "rms_lsb_per_component": rms,
        "byte_identical_to_source": False,
    }


def verify_injection_preserves_siblings(source_path: Path, output_path: Path) -> None:
    """Assert every dataset/attribute except ``baseband`` is unchanged."""
    import h5py

    def _tree(handle) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        datasets: dict[str, np.ndarray] = {}
        attrs: dict[str, dict[str, Any]] = {"/": dict(handle.attrs)}

        def _visit(name, item):
            if isinstance(item, h5py.Dataset):
                datasets[name] = item[...]
            attrs[name] = dict(item.attrs)

        handle.visititems(_visit)
        return datasets, attrs

    with h5py.File(str(source_path), "r") as src, h5py.File(
        str(output_path), "r"
    ) as dst:
        src_data, src_attrs = _tree(src)
        dst_data, dst_attrs = _tree(dst)
    if set(src_data) != set(dst_data):
        raise RuntimeError("injected file changed the set of datasets")
    for name, array in src_data.items():
        if name == BASEBAND_DATASET:
            continue
        if not np.array_equal(array, dst_data[name]):
            raise RuntimeError(f"injected file changed sibling dataset {name!r}")
    for name, mapping in src_attrs.items():
        theirs = dst_attrs.get(name, {})
        if set(mapping) != set(theirs) or any(
            not np.array_equal(np.asarray(mapping[key]), np.asarray(theirs[key]))
            for key in mapping
        ):
            raise RuntimeError(f"injected file changed attributes on {name!r}")


def inject_directory(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    amplitude_lsb: float,
    phase_seed: int,
    baseband_frequency_hz: float | None = None,
    pilot_frequency_hz: float | None = None,
    physical_channel: int | None = None,
) -> list[dict[str, Any]]:
    """Inject every file, preserving names; write the manifest sidecar."""
    fmt = _format_module()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, source in enumerate(sorted(Path(p) for p in input_paths)):
        center_hz = fmt.channel_center_hz(str(source))
        f_bb = resolve_baseband_frequency_hz(
            center_hz,
            baseband_frequency_hz=baseband_frequency_hz,
            pilot_frequency_hz=pilot_frequency_hz,
            physical_channel=physical_channel,
            sample_rate_hz=float(fmt.FS),
        )
        entries.append(
            inject_tone_into_baseband_file(
                source,
                output_dir / source.name,
                baseband_frequency_hz=f_bb,
                amplitude_lsb=amplitude_lsb,
                # Distinct per-file phases without per-file arguments.
                phase_seed=int(phase_seed) + index,
            )
        )
    manifest_path = output_dir / INJECTION_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps({"schema_version": "pilotproxy_injection_manifest_v1", "files": entries},
                   indent=2),
        encoding="utf-8",
    )
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy real CHIME baseband files with a known pilot tone injected "
            "(integer-domain; a zero-amplitude pass is byte-identical). The "
            "output directory drops into `pilot-proxy chime-scan --source "
            "local` unchanged."
        ),
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="A baseband .h5 file or a directory of them.")
    parser.add_argument("--glob", default="*.h5",
                        help="Filename glob when --input is a directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amplitude-lsb", type=float, required=True,
                        help=(
                            "Tone amplitude in raw 4-bit LSB units (per "
                            "complex sample). 0 performs the byte-identity "
                            "control copy. The manifest records each file's "
                            "measured per-component RMS for later dB "
                            "conversion."
                        ))
    frequency = parser.add_mutually_exclusive_group(required=True)
    frequency.add_argument("--baseband-frequency-hz", type=float,
                           help="Tone frequency relative to each file's channel center.")
    frequency.add_argument("--pilot-frequency-hz", type=float,
                           help="Absolute RF pilot frequency; converted per file.")
    frequency.add_argument("--physical-channel", type=int,
                           help="ATSC physical channel; nominal pilot, converted per file.")
    parser.add_argument("--phase-seed", type=int, default=20260701)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input.is_dir():
        paths = sorted(args.input.glob(args.glob))
        if not paths:
            raise SystemExit(f"no files matching {args.glob!r} under {args.input}")
    else:
        paths = [args.input]
    entries = inject_directory(
        paths,
        args.output_dir,
        amplitude_lsb=args.amplitude_lsb,
        phase_seed=args.phase_seed,
        baseband_frequency_hz=args.baseband_frequency_hz,
        pilot_frequency_hz=args.pilot_frequency_hz,
        physical_channel=args.physical_channel,
    )
    total_clip = sum(entry["clip_count"] for entry in entries)
    print(
        f"Injected {len(entries)} file(s) -> {args.output_dir} "
        f"(amplitude {args.amplitude_lsb} LSB, total clipped samples {total_clip}); "
        f"manifest: {args.output_dir / INJECTION_MANIFEST_FILENAME}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
