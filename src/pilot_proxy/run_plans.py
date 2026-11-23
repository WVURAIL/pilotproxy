"""Calibration and test plans from the fine-power files of a detect-only run."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from pilot_proxy.bundle_calibration import _expand, _rows
from pilot_proxy.paths import CONFIGS_DIR

# from the checkout or the installed package, so the commands run from any directory
CORE_PROFILE = str(CONFIGS_DIR / "detector_core" / "pilotproxy_cuda_local_reference_power_ratio.json")
CALIBRATION_SHARE = 2.0 / 3.0


@lru_cache(maxsize=None)
def _file_info(path: str) -> tuple[int, int | None, list[int]]:
    """Return the first FPGA sequence number, its time in ns (None if not recorded) and the coarse channels of one run file."""
    from pilot_proxy.run_files import read_attrs

    attrs = read_attrs(path)
    # the attributes are dropped when every row is unused (-1)
    start = int(attrs.get("fpga_seq_num", 0))
    time_ns = attrs.get("fpga_seq_time_nsec")
    return start, None if time_ns is None else int(time_ns), [int(c) for c in attrs.get("coarse_freq", [])]


def ordered_run_files(runs: list[str]) -> list[tuple[int | None, list[int], str]]:
    """Return (first frame time in ns or None, coarse channels, path) of each run file, in recording order.

    The order is by the first frame's time when every file with data records it, so a recording that an
    F-engine restart split, which resets fpga_seq_num, keeps its order. Otherwise it is by acquisition
    directory, whose name holds the start time of its Kotekan run, then by fpga_seq_num.
    """
    infos = [(*_file_info(path), path) for path in _expand(runs)]
    if all(time_ns is not None for _, time_ns, coarse, _ in infos if coarse):
        def key(item):
            return (-1 if item[1] is None else item[1], item[3])
    else:
        def key(item):
            return (str(Path(item[3]).parent), item[0], item[3])
    return [(time_ns, coarse, path) for _, time_ns, coarse, path in sorted(infos, key=key)]


def _limit_plan(path: Path | None) -> dict[str, Any]:
    """Return the power-limit inputs of a plan, with the paths of the table and the limits made absolute."""
    if path is None:
        raise ValueError("the power_limit threshold needs a JSON file that names the response table and the checked power "
                         "limits and gives the curvature and the margin")
    given = json.loads(Path(path).read_text())
    if not {"table", "limits", "curvature", "margin"} <= set(given):
        raise ValueError(f"{path} must name a table and limits and give the curvature and the margin")
    base = Path(path).resolve().parent
    return {**given, **{key: str((base / given[key]).resolve()) for key in ("table", "limits")}}


def write_run_plans(*, bundle_dir: Path, runs: list[str], output_dir: Path, epoch: str, receiver_profile: Path | None = None,
                    detector_core_profile: Path = Path(CORE_PROFILE), null_quantile: float = 0.99,
                    calibration_share: float = CALIBRATION_SHARE, threshold: str = "knee",
                    power_limit: Path | None = None, decision_version: str | None = None,
                    night_site: Sequence[float] | None = None) -> dict[str, Any]:
    """Split the files of each bound channel in time, and write calibrate.json and test.json.

    decision_version, when given, names the test the words are set for, in place of the calibration
    default: fine_decision_v1 for a Kotekan stage that runs only that test. night_site, when given as (latitude,
    east longitude) in degrees, takes the control quantile of the default word over the night frames there.
    """
    if not 0.0 < calibration_share < 1.0:
        raise ValueError("calibration_share must lie between 0 and 1")
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    receiver = receiver_profile or CONFIGS_DIR / "receiver_profiles" / f"{bundle['receiver_profile_id']}.json"
    files = ordered_run_files(runs)
    first: dict[str, list[str]] = {}
    last: dict[str, list[str]] = {}
    control = None
    for channel, row in _rows(bundle).items():
        mine = [path for _, coarse, path in files if int(row["receiver_channel_id"]) in coarse]
        if len(mine) < 2:
            raise ValueError(f"channel {channel} has {len(mine)} run files, and a split needs two")
        split = min(len(mine) - 1, max(1, round(len(mine) * calibration_share)))
        first[str(channel)], last[str(channel)] = mine[:split], mine[split:]
        if row["fine_calibration"]["status"] == "control":
            control = channel
    if control is None:
        raise ValueError("the bundle has no control channel, so the run gives no floor")
    calibrate = dict(receiver_profile=str(receiver.resolve()), detector_core_profile=str(detector_core_profile.resolve()),
                     capture_format="dtv_fine_powers", epoch=epoch, null_quantile=null_quantile, channels=first,
                     control=dict(channel=control, captures=first[str(control)]))
    if threshold != "knee":
        calibrate.update(threshold=threshold, power_limit=_limit_plan(power_limit))
    if decision_version is not None:
        calibrate["decision_version"] = decision_version
    if night_site is not None:
        latitude, longitude = (float(v) for v in night_site)
        calibrate["default_word"] = dict(night=dict(latitude_deg=latitude, longitude_deg_east=longitude))
    test = dict(capture_format="dtv_fine_powers", channels=last, control=last[str(control)])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibrate.json").write_text(json.dumps(calibrate, indent=1) + "\n")
    (output_dir / "test.json").write_text(json.dumps(test, indent=1) + "\n")
    return dict(calibrate=calibrate, test=test)
