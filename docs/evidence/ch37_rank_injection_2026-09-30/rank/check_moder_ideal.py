"""Code check of mode R against mode I on ideal data: i.i.d. F(4096, 8192) bulk and designated bins, c = 1. With a
pilot-free ideal D the two modes answer the same question (mode R by conditional injection on the observed D, mode
I exactly in D), so their Pd = 0.9 SNRs must agree within mode R's sampling error. Writes data/check_moder_ideal.json.
Usage (analysis interpreter, one thread, no bytecode): python -B check_moder_ideal.py"""
import json
import numpy as np
import core
from moder import ModeR

N, NBULK = 200_000, 125
rng = np.random.default_rng(20260930_37)
S = np.sort(rng.f(core.D1, core.D2, size=(N, NBULK)), axis=1)
D = rng.f(core.D1, core.D2, size=(N, 5))
month = np.repeat(np.arange(40), N // 40)
mi = core.ModeI(S, month, core.Tables(), offsets=("d0",))
pt = mi.run_point()
mr = ModeR(S, D, month, 1.0, seed=[core.SEED, 1])
pr = mr.run(np.ones(N))
out = {}
for p in (1e-3, 1e-2):
    a, b = pr[("s", p)], pt[("s", p, "d0")]
    ranks = [20, 41, 63, 94, 110, 120, 125]
    out[f"{p:g}"] = {r: dict(mode_R=float(a[r - 1]), mode_I=float(b[r - 1]), eta_R=float(pr[("eta", p)][r - 1]),
                             eta_I=float(pt[("eta", p)][r - 1])) for r in ranks}
    out[f"{p:g}"]["argmin"] = dict(mode_R=int(np.argmin(a)) + 1, mode_I=int(np.nanargmin(b)) + 1)
    print(p, {r: (round(v["mode_R"], 3), round(v["mode_I"], 3)) for r, v in out[f"{p:g}"].items() if r != "argmin"},
          out[f"{p:g}"]["argmin"], flush=True)
(core.ROOT / "data/check_moder_ideal.json").write_text(json.dumps(out, indent=1) + "\n")
