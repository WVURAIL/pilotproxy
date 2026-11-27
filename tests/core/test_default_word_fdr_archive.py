"""The false-discovery-rate default words, against the dissertation's words on the archive score cache.

The score cache of record (RFIM/Datasets/v3-rerun-2026-11-23-gate2/scores in the WVURAIL library, the dissertation's
scripts/paths.py SCORES and CONTROL65) holds each frame's v3 keep word (required_q16) and its time. The dissertation's
data/assembly/default_words.json (last
changed in dissertation f069322) holds the word and the scale lambda_c of each channel. Night is the Sun below the
horizon at DRAO. The test skips where either is absent: set PILOTPROXY_SCORE_CACHE and PILOTPROXY_DISSERTATION.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.detector_contract import null_power_ratio_from_weight_norms, weight_term_norms_sq
from pilot_proxy.detector_weights import DetectorWeightBank
from pilot_proxy.false_discovery import Q16, ControlNull, fdr_threshold
from pilot_proxy.paths import DEFAULT_WEIGHTS_PATH
from pilot_proxy.sun import is_night

SCORES = Path(os.environ.get("PILOTPROXY_SCORE_CACHE",
                             Path.home() / "rail/RFIM/Datasets/v3-rerun-2026-11-23-gate2/scores"))
RECORD = Path(os.environ.get("PILOTPROXY_DISSERTATION", Path.home() / "code/dissertation")) / \
    "data/assembly/default_words.json"
DRAO = (49.3208, -119.6236)
pytestmark = pytest.mark.skipif(not (SCORES / "ch37.npz").is_file() or not RECORD.is_file(),
                                reason="the archive score cache or the dissertation record is not here")


def _night_words(channel):
    z = np.load(SCORES / f"ch{channel}.npz")
    good = z["supported"] & ~z["always_masked"] & is_night(z["frame_time"], *DRAO)
    return z["required_q16"][good].astype(np.int64)


def test_fdr_words_are_the_dissertation_words():
    pytest.importorskip("scipy")
    record = json.loads(RECORD.read_text())
    null = ControlNull(_night_words(37).astype(np.float64) / Q16)
    assert (null.fit["u"], null.fit["k"]) == (record["tail"]["u"], record["tail"]["k"])
    assert (null.fit["xi"], null.fit["sigma"]) == pytest.approx((record["tail"]["xi"], record["tail"]["sigma"]), rel=1e-6)
    rows = [r for r in record["channels"] if r["channel"] != 37]
    assert [r["channel"] for r in rows] == list(range(14, 37))
    # lambda_c as calibrate-bundle forms it, mu0_c g_c / (mu0_37 g_37), with mu0_c from the bundled CHIME weights
    bank = DetectorWeightBank(explicit_path=DEFAULT_WEIGHTS_PATH)
    control = next(r for r in record["channels"] if r["channel"] == 37)
    reference = control["mu0"] * control["g"]
    scale = {}
    for r in rows:
        weights, _ = bank.get_weights_for_physical_channel(r["channel"])
        target, lower, upper = weight_term_norms_sq(np.asarray(weights).reshape(3, -1))
        assert null_power_ratio_from_weight_norms(target, lower + upper) == r["mu0"]
        scale[r["channel"]] = r["mu0"] * r["g"] / reference
    assert scale == {r["channel"]: r["scale"] for r in rows}
    found = {c: fdr_threshold(_night_words(c), s, null, record["fdr_level"]).word for c, s in scale.items()}
    assert found == {r["channel"]: r["word"] for r in rows}
