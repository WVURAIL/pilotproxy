"""Cross-check of PilotProxy's worst-case power-limit test against the dissertation's corrected rule.

PilotProxy's calibrate-bundle path (bundle_calibration._limit_channel, power_limit_word) runs on the archive frames of
record (the v3 score cache and the archive products' fine powers) and on the channel-37 control product, with the
dissertation's inputs: power limits, response table, default-word scale lambda_c, curvature d_c and its error, and
the route term as the plan margin. Each word of PilotProxy's sweep is compared with the dissertation's sweep
(data/assembly/sweeps65 at REV), and the least ratio with corrected_rule.json. A second run uses the plan defaults
(no curvature, margin 0), as a plan without those fields would.

usage: crosscheck.py REV HALF [channels]   (dissertation commit, shelf half-width; writes out/hHALF_REV.json)
Reads the dissertation records with git show; changes nothing there.
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PP = Path.home()/"code/pilotproxy-wt-docs"
DISS = Path.home()/"code/dissertation-wt-bench"
SCORES = Path.home()/"work/v3-rerun-2026-11-21/scores"
CONTROL_PRODUCT = Path.home()/"rail/RFIM/Datasets/ch37-control-491-2026-09-24/ch37_control_491_20260924/_per_pilot/491.npz"
OUT = Path(__file__).resolve().parent/"out"
sys.path.insert(0, str(PP/"src"))

from pilot_proxy import bundle_calibration as bc                      # noqa: E402
from pilot_proxy.characterization import threshold_rule as T          # noqa: E402
from pilot_proxy.fine_decision import FINE_DECISION_V3_VERSION as V3  # noqa: E402
from pilot_proxy.power_limit_word import shelf_reference              # noqa: E402

REV, HALF = sys.argv[1], int(sys.argv[2])
CHANNELS = [int(c) for c in sys.argv[3:]] or list(range(14, 37))


def show(path, binary=False):
    raw = subprocess.check_output(["git", "-C", str(DISS), "show", f"{REV}:{path}"])
    return raw if binary else json.loads(raw)


corrected = {r["channel"]: r for r in show("data/assembly/corrected_rule.json")["channels"]}
thresholds = {r["channel"]: r for r in show("data/assembly/model_thresholds.json")["channels"]}
offset = show("data/assembly/estimate_offset.json")
offsets = {r["channel"]: r for r in offset["channels"]}
route = float(offset["route_disagreement"])
scale = {r["channel"]: r["scale"] for r in show("data/assembly/default_words.json")["channels"]}
tmp = Path(tempfile.mkdtemp())
(tmp/"channel_table.json").write_bytes(show("data/assembly/channel_table.json", binary=True))
table = T.load_table(tmp/"channel_table.json")

plan = dict(receiver_profile=str(PP/"configs/receiver_profiles/chime_dtv_fengine.json"),
            detector_core_profile=str(PP/"configs/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"))
k, cell_hz, descending = bc._receiver(plan, PP)
shelf = bc._shelf(plan, PP)
bundle, _ = bc._export(plan, PP, sorted({*CHANNELS, 37}), tmp/"bundle")
rows = bc._rows(bundle)
nominal = {c: bc.nominal_fine_bin(r["pilot_frequency_hz"], r["coarse_channel_center_hz"], cell_hz, descending)
           for c, r in rows.items()}
bulk, rank = bc._geometry(0, (), V3)

# the control, as the dissertation reads it: its valid frames, clustered by capture
import os
CONTROL_FRAMES = os.environ.get("CONTROL_FRAMES", "cache")
if CONTROL_FRAMES == "cache":
    # the control's frames of the v3 score cache, clustered by capture, as scripts/median/selection.py reads them
    with np.load(SCORES/"ch37.npz", allow_pickle=False) as z:
        crows, cunits = z["source_row_index"], z["frame_event_index"]
        cdecided = z["supported"] & ~z["always_masked"]
    with np.load(CONTROL_PRODUCT, allow_pickle=False) as z:
        cfines = z["fine_power_u64"][crows]
    if os.environ.get("CONTROL_DECIDED_ONLY"):
        cfines, cunits = cfines[cdecided], cunits[cdecided]
else:
    # every valid frame of the control product, clustered by capture
    with np.load(CONTROL_PRODUCT, allow_pickle=False) as z:
        ok = z["valid"][:, 0] > 0
        cfines, cunits = z["fine_power_u64"][ok], z["frame_unit_index"][ok]
control = (cfines, cunits)
control_reference = shelf_reference(nominal[37], HALF)

captured = {}
real_limit_test = bc.limit_test


def recording_limit_test(sweep, **kw):
    captured.update(sweep=sweep, **kw)
    return real_limit_test(sweep, **kw)


bc.limit_test = recording_limit_test


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale_ = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b)/scale_)) if a.size else 0.0


report = dict(rev=REV, half_width=HALF, control_frames=CONTROL_FRAMES, control_decided_only=bool(os.environ.get("CONTROL_DECIDED_ONLY")), control_frame_count=int(len(cfines)), pilotproxy=subprocess.check_output(["git", "-C", str(PP), "rev-parse", "--short", "HEAD"]).decode().strip(),
              shelf=shelf, shelf_dissertation=T.SHELF, route=route, control_nominal=nominal[37],
              control_reference_bins=int(control_reference[0].sum()), channels={})
for ch in CHANNELS:
    meta = json.loads((SCORES/f"ch{ch}.json").read_text())
    with np.load(SCORES/f"ch{ch}.npz", allow_pickle=False) as z:
        sc = {key: z[key] for key in z.files}
    with np.load(meta["original_product_identity"]["path"], allow_pickle=False) as p:
        fines = p["fine_power_u64"][sc["source_row_index"]]
    captures = sc["frame_event_index"]
    seen = bc.required_words(fines, anchor_bin=nominal[ch], bulk=bulk, rank=rank, decision_version=V3)
    sup = sc["supported"]
    words_match = bool(np.array_equal(seen >= 0, sup) and np.array_equal(seen[sup].astype(np.uint64), sc["required_q16"][sup]))
    weight_rel = rel(bc.capture_weights(captures), sc["probability_weight"])
    t = thresholds[ch]
    variants = {}
    for name, curvature, margin in (("plan", {ch: (offsets[ch]["curvature"], offsets[ch]["curvature_se"])}, route),
                                    ("defaults", {}, 0.0)):
        inputs = dict(table=table, rows={c: {} for c in CHANNELS}, limits={ch: t["power_limit"]}, coupling={},
                      curvature=curvature, margin=margin, min_captures=30, shelf=shelf)
        result = bc._limit_channel(ch, fines, captures, seen, control, anchor=nominal[ch], bulk=bulk, rank=rank,
                                   inputs=inputs, decision_version=V3, control_scale=scale[ch],
                                   reference=shelf_reference(nominal[ch], HALF), control_reference=control_reference)
        s, kw = captured["sweep"], dict(captured)
        x = np.maximum(kw["shelf"]*s["mean_excess"] - kw["shift"] - kw["offset"], 0.0)
        ratio = kw["coupling"]*kw["power_scale"]*(x + 2*s["se"] + kw["margin"])**2/kw["power_limit"]
        variants[name] = dict(result=result, ratio=ratio, sweep=s, kw=kw)
    # the dissertation's sweep of record and its ratio at every word
    d = np.load(io.BytesIO(show(f"data/assembly/sweeps65/ch{ch}.npz", binary=True)))
    dx = np.maximum(T.SHELF*d["mean_excess"] - offsets[ch]["predicted"], 0.0)
    dratio = t["power_scale"]*(dx + 2*d["se"])**2/t["power_limit"]
    deligible = (d["kept_events"] >= 30) & (d["kappa"] > 0)
    p = variants["plan"]
    s = p["sweep"]
    same_words = bool(np.array_equal(np.asarray(s["word"], np.uint64), np.asarray(d["word"], np.uint64)))
    row = dict(nominal_bin=nominal[ch], shelf_bins=int(shelf_reference(nominal[ch], HALF)[0].sum()),
               keep_words_match_cache=words_match, capture_weight_rel=weight_rel, same_sweep_words=same_words,
               power_scale_rel=rel(p["kw"]["power_scale"], t["power_scale"]),
               dissertation=dict(word=corrected[ch]["word"], ratio=corrected[ch]["ratio"],
                                 passing_words=corrected[ch]["passing_words"]))
    if same_words:
        e = deligible
        raw = d["mean_excess"] + d["selection_shift"]/T.SHELF      # the dissertation's e'(u) before its shift
        row.update(kappa_rel=rel(s["kappa"], d["kappa"]), kept_events_equal=bool(np.array_equal(s["kept_events"], d["kept_events"])),
                   mean_excess_rel=rel(s["mean_excess"][e], raw[e]),
                   shift_rel=rel(p["kw"]["shift"][e], d["selection_shift"][e]),
                   ratio_rel_plan=rel(p["ratio"][e], dratio[e]),
                   upper_shelf_diff_plan=float(np.max(np.abs(np.sqrt(p["ratio"][e]) - np.sqrt(dratio[e]))*np.sqrt(t["power_limit"]/t["power_scale"]))))
    for name, v in variants.items():
        r = v["result"]
        row[name] = dict(outcome=r["outcome"], reason=r.get("reason"), least_ratio=r.get("least_ratio"),
                         least_ratio_word=r.get("least_ratio_word"), passing_words=r["passing_words"],
                         word=r.get("word"), corrections=r.get("corrections"))
    report["channels"][ch] = row
    print(ch, "bins", row["shelf_bins"], "words", words_match, "w", f"{weight_rel:.1e}", "same", same_words,
          "| diss", corrected[ch]["word"], f"{corrected[ch]['ratio']:.4g}",
          "| plan", row["plan"]["least_ratio_word"], f"{row['plan']['least_ratio']:.4g}", row["plan"]["outcome"],
          "| defaults", row["defaults"]["least_ratio_word"], f"{row['defaults']['least_ratio']:.4g}",
          "| ratio_rel", f"{row.get('ratio_rel_plan', float('nan')):.2e}", flush=True)
OUT.mkdir(exist_ok=True)
(OUT/f"h{HALF}_{REV}_{CONTROL_FRAMES}{os.environ.get('CONTROL_DECIDED_ONLY', '') and '_decided'}.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
