"""Calibration and test plans from the fine-power files of a detect-only run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pilot_proxy.bundle_calibration import _expand, _rows

CORE_PROFILE = "configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"
CALIBRATION_SHARE = 2.0 / 3.0


def _file_info(path: str) -> tuple[int, list[int]]:
    """Return the first FPGA sequence number and the coarse channels of one run file."""
    import h5py

    with h5py.File(path, "r") as handle:
        data = handle[next(name for name in handle if name.startswith("dtv_fine_powers"))]
        start = int(data.attrs.get("fpga_seq_num", 0))
        # the attribute is dropped when every row is unused (-1)
        return start, [int(c) for c in data.attrs.get("coarse_freq", [])]


def write_run_plans(*, bundle_dir: Path, runs: list[str], output_dir: Path, epoch: str, receiver_profile: Path | None = None,
                    detector_core_profile: Path = Path(CORE_PROFILE), null_quantile: float = 0.99,
                    calibration_share: float = CALIBRATION_SHARE) -> dict[str, Any]:
    """Split the files of each bound channel in time, and write calibrate.json and test.json."""
    if not 0.0 < calibration_share < 1.0:
        raise ValueError("calibration_share must lie between 0 and 1")
    bundle = json.loads((Path(bundle_dir) / "pilot_profiles.json").read_text())
    receiver = receiver_profile or Path(f"configs/receiver_profiles/{bundle['receiver_profile_id']}.json")
    files = sorted(((*_file_info(path), path) for path in _expand(runs)), key=lambda item: (item[0], item[2]))
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
    test = dict(capture_format="dtv_fine_powers", channels=last, control=last[str(control)])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibrate.json").write_text(json.dumps(calibrate, indent=1) + "\n")
    (output_dir / "test.json").write_text(json.dumps(test, indent=1) + "\n")
    return dict(calibrate=calibrate, test=test)
