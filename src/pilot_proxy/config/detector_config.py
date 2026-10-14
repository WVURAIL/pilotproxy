"""Detector geometry declared by a project: K, the fine bins, the frame length
and how inputs are summed. Quantities that also need the instrument (bin widths,
the summed term count P) are methods that take it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._files import ProfileError, check_keys, choice, integer, read_yaml

FEED_SUM_SCHEMES = ("all_inputs",)


@dataclass(frozen=True)
class DetectorConfig:
    detector_window: int   # K, samples per detector row
    fine_bins: int         # L_F = 2 nfft / K
    nfft: int              # samples per frame
    feed_sum: str          # how inputs are summed into one statistic

    def summed_terms(self, instrument) -> int:
        """P, the number of independent terms in the coarse power sum."""
        return int(instrument.n_inputs) * self.detector_window

    def coarse_null_dof(self, instrument) -> tuple[int, int]:
        """Degrees of freedom (2P, 4P) of the ideal coarse null law F(2P, 4P)."""
        p = self.summed_terms(instrument)
        return 2 * p, 4 * p

    def detector_bin_hz(self, instrument) -> float:
        """Width of one detector row bin, f_s / K."""
        return instrument.sample_rate_hz / self.detector_window

    def envelope_bin_hz(self, instrument) -> float:
        """Width of one fine bin, f_s / (K L_F)."""
        return self.detector_bin_hz(instrument) / self.fine_bins

    def psd_bin_hz(self, instrument) -> float:
        """Width of one frame-spectrum bin, f_s / nfft."""
        return instrument.sample_rate_hz / self.nfft

    def frame_seconds(self, instrument) -> float:
        """Frame length, nfft / f_s."""
        return self.nfft / instrument.sample_rate_hz


def detector_config_from_mapping(data: Mapping[str, Any], *, where: str) -> DetectorConfig:
    check_keys(data, required=("detector_window", "fine_bins", "nfft", "feed_sum"), where=where)
    window = integer(data["detector_window"], field="detector_window", where=where, minimum=1)
    fine = integer(data["fine_bins"], field="fine_bins", where=where, minimum=1)
    nfft = integer(data["nfft"], field="nfft", where=where, minimum=1)
    if fine * window != 2 * nfft:
        raise ProfileError(f"{where}: fine_bins must equal 2 nfft / detector_window")
    scheme = choice(data["feed_sum"], FEED_SUM_SCHEMES, field="feed_sum", where=where)
    return DetectorConfig(window, fine, nfft, scheme)


def load_detector_config(path: Path | str) -> DetectorConfig:
    path = Path(path)
    return detector_config_from_mapping(read_yaml(path), where=str(path))


__all__ = [
    "DetectorConfig",
    "FEED_SUM_SCHEMES",
    "detector_config_from_mapping",
    "load_detector_config",
]
