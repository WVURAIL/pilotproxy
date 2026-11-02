"""
Baseband HDF5 format support for the `chime-baseband` reader.

This module holds the on-disk format of CHIME and outrigger baseband files.
The format is kept in one small module so that the reader stays a thin wrapper
and the tests can write matching synthetic files. The module uses only NumPy
and h5py and holds no analysis logic.

On-disk layout:
    dataset  h["baseband"]    uint8 [n_time, n_feeds], offset-binary 4+4-bit
    attr     h.attrs["freq"]  channel-center frequency in MHz
    dataset  h["index_map/input"]  optional receiver input map
    attrs    git_version_tag / collection_server  optional receiver provenance
"""
from __future__ import annotations

import h5py
import numpy as np

from pilot_proxy.archive.instruments import load_instrument as _load_instrument

# The native CHIME baseband geometry is read from chime.yaml so that it is
# declared in one place. FS is bandwidth / n_channels, and NFFT is the configured
# frame length. The CHIME outriggers share this geometry through `extends: chime`,
# so these defaults cover every instrument the chime-baseband reader serves.
_NATIVE = _load_instrument("chime")
FS = _NATIVE.fs_hz     # Hz, per-channel (complex) sample rate = bandwidth / n_channels
NFFT = _NATIVE.nfft    # default frame/FFT length in time samples
HZ_PER_MHZ = 1_000_000.0


def unpack_4bit(packed: np.ndarray) -> np.ndarray:
    """Unpack offset-binary 4+4-bit samples to complex64.

    ``packed`` is uint8 [n_time, n_feeds], with the real part in the high
    nibble and the imaginary part in the low nibble.
    """
    packed = np.asarray(packed, dtype=np.uint8)
    real = (packed >> 4).astype(np.float32) - 8.0
    imag = (packed & 0x0F).astype(np.float32) - 8.0
    return (real + 1j * imag).astype(np.complex64)


def channel_center_hz(path: str) -> float:
    """Return the channel center frequency (Hz) from the file's `freq` attribute (MHz)."""
    with h5py.File(path, "r") as h:
        value = h.attrs["freq"]
        try:
            frequency_hz = float(value) * HZ_PER_MHZ
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid baseband 'freq' attribute in {path}: {value!r}") from exc
        if not np.isfinite(frequency_hz):
            raise ValueError(
                f"non-finite baseband 'freq' attribute in {path}: {value!r}")
        return frequency_hz


def iter_frames(path: str, nfft: int = NFFT):
    """Yield [nfft, n_feeds] complex64 frames from one file, dropping the final partial frame."""
    with h5py.File(path, "r") as h:
        bb = h["baseband"]
        if bb.ndim != 2:
            raise ValueError(
                f"expected 2-D 'baseband' dataset in {path}, got shape {bb.shape}")
        if bb.dtype != np.dtype("uint8"):
            raise ValueError(
                f"expected uint8 'baseband' dataset in {path}, got {bb.dtype}")
        if bb.shape[1] < 1:
            raise ValueError(
                f"expected at least one feed in 'baseband' dataset in {path}")
        if bb.shape[0] < nfft:
            raise ValueError(
                f"'baseband' dataset in {path} has {bb.shape[0]} time samples, "
                f"fewer than the effective nfft {nfft}")
        n_frames = bb.shape[0] // nfft
        for c in range(n_frames):
            yield unpack_4bit(bb[c * nfft:(c + 1) * nfft, :])


def make_synth_file(path, n_time, n_feeds, f_center_mhz, f_tone_bb,
                    tone_amp=3.0, noise_std=2.5, seed=0,
                    receiver_provenance=True) -> None:
    """Write a synthetic baseband .h5 file (uint8 4+4-bit) for tests.

    A complex sinusoid at baseband frequency `f_tone_bb` (Hz) is added to
    complex Gaussian noise. Each feed has a random tone phase, so the tone adds
    incoherently across feeds. `f_center_mhz` is written to the `freq`
    attribute, and a synthetic receiver identity is written unless
    `receiver_provenance` is False. A power spectrum of the result peaks at
    `f_tone_bb`.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(n_time)
    tone = np.exp(2j * np.pi * f_tone_bb * n / FS)
    phases = np.exp(2j * np.pi * rng.random(n_feeds))
    sig = tone_amp * tone[:, None] * phases[None, :]
    sig = sig + noise_std * (rng.standard_normal((n_time, n_feeds))
                             + 1j * rng.standard_normal((n_time, n_feeds))) / np.sqrt(2)
    re = (np.clip(np.round(sig.real), -8, 7).astype(np.int64) + 8)
    im = (np.clip(np.round(sig.imag), -8, 7).astype(np.int64) + 8)
    packed = ((re << 4) | im).astype(np.uint8)
    with h5py.File(path, "w") as h:
        h.create_dataset("baseband", data=packed)
        h.attrs["freq"] = f_center_mhz
        if receiver_provenance:
            h.attrs["git_version_tag"] = "synthetic"
            h.attrs["collection_server"] = "synthetic"
            input_map = np.empty(
                n_feeds,
                dtype=[("chan_id", "<u2"), ("correlator_input", "S32")],
            )
            input_map["chan_id"] = np.arange(n_feeds, dtype=np.uint16)
            input_map["correlator_input"] = [
                f"synthetic-{index}".encode("ascii") for index in range(n_feeds)
            ]
            h.create_dataset("index_map/input", data=input_map)
