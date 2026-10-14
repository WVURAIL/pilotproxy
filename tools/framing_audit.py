#!/usr/bin/env python3
"""Baseband framing audit for pilot-proxy weight ROMs.

Answers one question for one CHIME baseband file: does the deployed weight
ROM look where the pilot actually is, or is there a frequency-frame mismatch
(e.g. a half-band / fs/2 shift between the ROM's assumed baseband framing
and the file's actual framing)?

Method: (1) incoherent full-channel fine spectrum (23.8 Hz bins, all feeds
summed); (2) exact response curves of the deployed target/reference filters
in the same raw-baseband frame (padded FFT of the conjugated,
window-flipped weights, matching the production spectral-sense path);
(3) line list above 5 sigma (MAD); (4) verdict comparing the strongest line
against the target lobe center and the nominal ATSC pilot.

Usage:
    python framing_audit.py FILE.h5 [--channel N] [--chunks 4]
        [--sense inverted] [--repo /path/to/pilot-proxy] [--png out.png]

Exit status: 0 = ALIGNED, 2 = MISMATCH, 3 = no line found (inconclusive).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

TS = 2.56e-6
K, W = 128, 128
NFFT = K * W
FS = 1.0 / TS
F_ENV = 1.0 / (K * TS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h5file", type=Path)
    ap.add_argument("--channel", type=int, default=None,
                    help="ATSC physical channel (default: infer from freq)")
    ap.add_argument("--chunks", type=int, default=4,
                    help="chunks of 16384 samples to average (default 4)")
    ap.add_argument("--sense", default="inverted",
                    choices=["inverted", "normal"])
    ap.add_argument("--repo", type=Path, default=Path("."),
                    help="pilot-proxy repo root (default: cwd)")
    ap.add_argument("--png", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    from pilot_proxy.chime.frame_adapter import (
        unpack_chime_offset_binary_i4_to_complex,
        unpack_twos_complement_i4_to_complex,
    )
    from pilot_proxy.detector_weights import DetectorWeightBank
    from pilot_proxy.atsc_channels import physical_channel_to_pilot_hz

    f = h5py.File(args.h5file, "r")
    ds = f["baseband"]
    cc = float(f.attrs["freq"]) * 1e6
    n_time, n_feed = ds.shape
    n_chunk = min(args.chunks, n_time // NFFT)

    chan = args.channel
    if chan is None:
        for c in range(14, 37):
            if abs(physical_channel_to_pilot_hz(c) - cc) < FS / 2:
                chan = c
                break
    if chan is None:
        print("could not infer physical channel; pass --channel")
        return 3
    pilot = physical_channel_to_pilot_hz(chan)
    print(f"file {args.h5file.name}: freq_id {f.attrs.get('freq_id', '?')}, "
          f"center {cc/1e6:.4f} MHz, physical channel {chan}, "
          f"nominal pilot {pilot/1e6:.6f} MHz")

    # deployed filter response in the raw-baseband frame
    bank = DetectorWeightBank(
        explicit_path=str(args.repo / "weights" / "chime_dtv_weights_k128.bin"))
    wp, valid = bank.get_weights_for_physical_channel(chan)
    if not valid or wp is None:
        print("no valid weights for this channel in the ROM")
        return 3
    from pilot_proxy.detector_geometry import (
        apply_spectral_sense_to_detector_matrix,
    )
    Wc = unpack_twos_complement_i4_to_complex(wp).astype(np.complex64)
    # empirical response sweep: real tones through the exact production path
    n = np.arange(NFFT)
    step = 250.0
    nu_grid = np.arange(-FS / 2, FS / 2, step)

    def _pipe(nu_p: float) -> np.ndarray:
        tone = np.exp(2j * np.pi * nu_p * n * TS).astype(np.complex64)
        x = apply_spectral_sense_to_detector_matrix(
            tone.reshape(1, W, K), spectral_sense=args.sense
        ).astype(np.complex64)
        z = np.einsum("fwk,mk->mfw", x, np.conj(Wc), optimize=True)
        return (np.abs(z) ** 2).sum(axis=(1, 2))

    R = np.stack([_pipe(nu_p) for nu_p in nu_grid]).T
    centers = np.empty(3)
    for m in range(3):
        j = int(np.argmax(R[m]))
        if 0 < j < nu_grid.size - 1:
            la, lb_, lc_ = np.log(R[m, j - 1:j + 2])
            centers[m] = nu_grid[j] + 0.5 * (la - lc_) / (
                la - 2 * lb_ + lc_) * step
        else:
            centers[m] = nu_grid[j]
    print(f"deployed lobes (raw frame): target {centers[0]:+.1f} Hz, "
          f"ref_lo {centers[1]:+.1f} Hz, ref_hi {centers[2]:+.1f} Hz")

    # full-channel incoherent spectrum
    S = np.zeros(NFFT)
    BLK = 256
    for c in range(n_chunk):
        raw = ds[c * NFFT:(c + 1) * NFFT, :]
        for b0 in range(0, n_feed, BLK):
            x = unpack_chime_offset_binary_i4_to_complex(
                raw[:, b0:b0 + BLK].T).astype(np.complex64)
            S += (np.abs(np.fft.fft(x, axis=1)) ** 2).sum(axis=0)
            del x
    S = np.fft.fftshift(S / n_chunk)
    nu = (np.arange(NFFT) - NFFT // 2) * (1.0 / (NFFT * TS))
    floor = np.median(S)
    mad = np.median(np.abs(S - floor))
    thr = floor + 5 * 1.4826 * mad

    hot = np.where(S > thr)[0]
    if hot.size == 0:
        print("no line above 5 sigma anywhere in the channel; inconclusive "
              "(quiet interval) - try a file from a strong-propagation epoch")
        return 3
    groups = np.split(hot, np.where(np.diff(hot) > 2)[0] + 1)
    lines = sorted(
        ((float(nu[int(g[np.argmax(S[g])])]),
          float((S[int(g[np.argmax(S[g])])] - floor) / floor)) for g in groups),
        key=lambda t: -t[1])
    print(f"lines above 5 sigma ({len(lines)}):")
    for nu_l, exc in lines[:8]:
        sky_inv = (cc - nu_l - pilot)
        sky_nor = (cc + nu_l - pilot)
        tag = ""
        if abs(nu_l) < 30:
            tag = "  [DC spur?]"
        elif abs(abs(nu_l) - FS / 5) < 60:
            tag = "  [fs/5 spur?]"
        print(f"  {nu_l:+10.1f} Hz  x{exc:9.1f}   sky-nominal: "
              f"{sky_inv:+9.0f} Hz (inv) / {sky_nor:+9.0f} Hz (nor){tag}")

    # verdict from the strongest non-spur line
    cand = [l for l in lines
            if abs(l[0]) > 60 and abs(abs(l[0]) - FS / 5) > 60]
    if not cand:
        print("only instrumental spurs found; inconclusive")
        return 3
    nu_star, exc = cand[0]
    miss = nu_star - centers[0]
    sup = 10 * np.log10(_pipe(nu_star)[0] / R[0].max())
    print(f"\nstrongest line: {nu_star:+.1f} Hz (x{exc:.0f} over floor); "
          f"target lobe at {centers[0]:+.1f} Hz")
    print(f"miss = {miss/1e3:+.2f} kHz; deployed target response at the "
          f"line: {sup:+.1f} dB")
    if abs(miss) < F_ENV / 4:
        print("VERDICT: ALIGNED - ROM frame matches this file")
        rc = 0
    else:
        shift = (centers[0] - nu_star) % FS
        if shift > FS / 2:
            shift -= FS
        halfband = abs(abs(shift) - FS / 2) < 8e3
        print("VERDICT: MISMATCH - ROM looks elsewhere"
              + ("  (consistent with a half-band fs/2 frame shift)"
                 if halfband else ""))
        rc = 2

    if args.png is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.semilogy(nu / 1e3, S / floor, lw=0.6, color="#444444",
                    label="measured spectrum")
        for m, (nm, col) in enumerate(zip(
                ["target", "ref_lo", "ref_hi"],
                ["#d62728", "#ff9896", "#ffbb78"])):
            ax.semilogy(nu_grid / 1e3,
                        1e3 * R[m] / R[m].max() + 1e-2,
                        lw=1.0, color=col, label=f"deployed {nm}")
        ax.axvline(nu_star / 1e3, color="#2ca02c", lw=0.8, ls=":",
                   label="strongest line")
        ax.set_xlabel("raw baseband frequency [kHz]")
        ax.set_ylabel("power / median bin")
        ax.set_title(f"framing audit: {args.h5file.name} "
                     f"(ch {chan}, miss {miss/1e3:+.1f} kHz)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150, bbox_inches="tight")
        print(f"wrote {args.png}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
