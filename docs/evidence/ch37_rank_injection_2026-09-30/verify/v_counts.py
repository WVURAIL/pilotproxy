"""Frame-screen sensitivity (verifier's own code): near-threshold counts per half and objective-set membership
(>= 500 per half) for the design cut and for cuts built the same way at other saturation levels, from cfar-choice's
original per-frame bulk arrays (w = dmax / median over the usable bulk)."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CF = Path("/home/djg/rail/output/cfar-choice-2026-09-29/data/bulk")
dc = json.loads((ROOT / "design/design_counts.json").read_text())
vi = json.loads((HERE / "v_ideal.json").read_text())["screen"]
cuts = {"null_0.999": vi["null_0.999"], "design_as_run": 1.2962289629059025, "Pd0.99": vi["alt"]["0.99"]["w_cut"], "design_Pd0.999": vi["w_cut"],
        "Pd0.9999": vi["alt"]["0.9999"]["w_cut"], "1.5": 1.5, "2.0": 2.0}
out = {"cuts": cuts, "channels": {}, "objective": {}}
for ch in range(14, 37):
    z = np.load(CF / f"ch{ch:02d}.npz")
    w = z["dmax"] / z["med"]
    cal = z["calibration"].astype(bool)
    row = {}
    for k, c in cuts.items():
        row[k] = [int(((w <= c) & cal).sum()), int(((w <= c) & ~cal).sum())]
    d = dc["channels"][str(ch)]
    row["design_match"] = row["design_as_run"] == [d["near_cal"], d["near_eval"]]
    out["channels"][ch] = row
for k in cuts:
    out["objective"][k] = [ch for ch in range(14, 37) if min(out["channels"][ch][k]) >= 500]
    out["objective_1000_" + k] = [ch for ch in range(14, 37) if min(out["channels"][ch][k]) >= 1000]
(HERE / "v_counts.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(cuts))
print("design counts reproduced on all channels:", all(v["design_match"] for v in out["channels"].values()))
for k in cuts:
    print(k, round(cuts[k], 4), out["objective"][k], "| >=1000:", out["objective_1000_" + k])
