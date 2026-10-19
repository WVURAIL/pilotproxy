"""Synthetic archives for the release writer (``characterize archive``).

The populated product is two years of monthly acquisitions with four
repeating interference levels and, optionally, per-frame spectra with a
three-bin carrier at the nominal marker. Its eras come from the section 8.1
rule with the support gates relaxed to one frame, one acquisition and one
day, which the small product needs.
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "tests" / "products") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests" / "products"))

from test_pilotproxy_v5 import _replace, _write_product, fine_powers_with_sums  # noqa: E402

from pilot_proxy.characterization import eras, run  # noqa: E402
from pilot_proxy.config.project import default_project, default_project_dir  # noqa: E402
from pilot_proxy.products.npzio import load_npz  # noqa: E402
from pilot_proxy.products.reader import NFFT, Product  # noqa: E402

BANDS = (29, 34)                      # both keep the rule's eras (no author-dated list)
RELAXED_ERAS = eras.EraConfig(min_frames=1, min_units=1, min_days=1, min_peak_cohort=1,
                              units_sensitivity=(1,), threshold_sensitivity_db=((0.5, 1.0),))
GENERATED = "2026-09-27T00:00:00+00:00"


def populated_product(path: Path, channel: int, *, frames: int = 384, units: int = 24, spectra: bool = True) -> Path:
    """A v5 product whose frames fill one proxy-low era of ``units`` monthly acquisitions."""
    path = _write_product(path, channel, frames=frames, units=units)
    times = np.array([dt.datetime(2023 + m // 12, 1 + m % 12, 2, tzinfo=dt.timezone.utc).timestamp()
                      for m in range(units)])
    values = load_npz(path)
    with Product(path) as product:
        geometry = product.geometry
    fine = values["fine_power_u64"]
    fine[:, :, :] = 20
    fine[:, 0, geometry.nominal_fine_bin] = np.tile([40, 80, 160, 500], frames // 4)
    changes = dict(unit_time0_ctime=times, unit_input_map_sha256=np.full(units, "a" * 64), fine_power_u64=fine,
                   baseband_power_linear=np.tile([4.0, 4.1, 5.0, 10.0], frames // 4)[:, None])
    reference = values["p_ref_sum_u64"]
    norm = float(values["target_norm_sq"][0]) / float(values["reference_norm_sum_sq"][0])
    target = np.rint(reference * norm * np.tile([0.99, 1.01, 1.08, 20.0], frames // 4)[:, None]).astype(np.uint64)
    ratio = target / (reference * norm)
    excess = ratio - 1.0
    excess_db = np.full(excess.shape, np.nan)
    excess_db[excess > 0] = 10 * np.log10(excess[excess > 0])
    shelf_offset = (float(values["pilot_below_data_db"])
                    - 10 * np.log10(float(values["dtv_bandwidth_hz"]) / float(values["bin_enbw_hz"]))
                    - 10 * np.log10(float(values["pilot_capture_efficiency"])))
    # The frame powers are the all-bin sums of the fine powers (kernel core
    # 3.0.0): keep the fine shape and scale it to these levels exactly.
    changes["fine_power_u64"] = fine_powers_with_sums(
        fine, target, values["p_ref_lower_u64"], values["p_ref_upper_u64"])
    changes.update(p_target_u64=target, coarse_power_ratio=2.0 * target / reference,
                   normalized_coarse_power_ratio_db=10 * np.log10(ratio), normalized_pilot_excess=excess,
                   pilot_excess_db=excess_db, estimated_data_shelf_snr_db=excess_db + shelf_offset,
                   reject_mask=(ratio > 1).astype(np.uint8))
    if spectra:
        codes = np.zeros((frames, NFFT), dtype=np.int16)
        marker = int(round(geometry.nominal_psd_bin)) % NFFT
        codes[:, [(marker - 1) % NFFT, marker, (marker + 1) % NFFT]] = [1000, 2000, 1000]
        changes.update(psd_frame_db_i16=codes, psd_db_reference=np.ones((frames, 1)),
                       psd_db_step_per_code=np.float64(0.01), psd_db_invalid_code=np.int16(-32768))
    _replace(path, **changes)
    return path


def products_dir(tmp: Path, bands=BANDS) -> Path:
    project = default_project()
    directory = tmp / "products"
    directory.mkdir(parents=True, exist_ok=True)
    for band in bands:
        populated_product(directory / f"{project.target_freq_id(str(band))}.npz", band)
    return directory


def project_copy(tmp: Path) -> Path:
    """A writable copy of the repository's project profile."""
    target = tmp / "project"
    shutil.copytree(default_project_dir(), target)
    return target


def characterize(tmp: Path, *, products: Path | None = None, project_dir: Path | None = None,
                 out: str = "out") -> tuple[dict, Path]:
    """Write the release of the fixture products in-process; returns the summary and ``<out>/characterization``."""
    products = products or products_dir(tmp)
    result = run.characterize_archive(products, tmp / out, project_dir=project_dir or default_project_dir(),
                                      era_config=RELAXED_ERAS, generated=GENERATED)
    return result, tmp / out / "characterization"
