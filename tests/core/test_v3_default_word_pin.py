"""The v3 default words of calibrate-bundle, pinned against the dissertation's record.

tests/data/v3_control_ch37_keep_words.npz holds the exact v3 keep words of the control, channel 37, of the
dissertation's score cache (RFIM/Datasets/v3-rerun-2026-11-22/scores/ch37.npz, sha256 e44e4f05...). Each is
fine_keep_word_v3 of its [n_peak, d_rho]. They are sorted and split into night and day frames by
pilot_proxy.sun.is_night at DRAO (49.3208, -119.6236). The file holds no frame time, so the night split itself is
not retested here. The expected numbers are copied from data/assembly/default_words.json of the dissertation at
commit 03ca6bb.
"""
from pathlib import Path

import numpy as np
import pytest

from pilot_proxy.bundle_calibration import Q16, control_quantile_q16, scaled_word

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "v3_control_ch37_keep_words.npz"
pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="withheld from the code-only cut (CHIME_DERIVED.md)")

# dissertation 03ca6bb, data/assembly/default_words.json
QUANTILE = 0.99
CONTROL_FRAMES, CONTROL_FRAMES_ALL = 20751, 40477
CONTROL_ETA, CONTROL_ETA_ALL_FRAMES = 1.1325454711914062, 1.7162896728515578
# channel: (scale lambda_c = mu0_c g_c / (mu0_37 g_37), default word)
WORDS = {
    14: (0.999050750188293, 74153),
    15: (0.991334489036253, 73580),
    16: (0.9977553241427474, 74056),
    17: (0.9988036162459062, 74134),
    18: (1.0047220128492336, 74573),
    19: (1.0038858542415083, 74511),
    20: (0.9830965830935282, 72968),
    21: (1.0009370361824, 74293),
    22: (1.0009158256065478, 74291),
    23: (0.9923056611881246, 73652),
    24: (0.9990277555723106, 74151),
    25: (0.9955729791840807, 73894),
    26: (1.0071565885390275, 74754),
    27: (0.9901645894510833, 73493),
    28: (1.0001696594345364, 74236),
    29: (0.9975007834356026, 74038),
    30: (0.9945890952839284, 73821),
    31: (0.9991226582619457, 74158),
    32: (0.9950833335823396, 73858),
    33: (1.002475984708518, 74407),
    34: (1.001976238111173, 74370),
    35: (0.9908626068986391, 73545),
    36: (0.9863068805338696, 73207),
    37: (1.0, 74223),
}


def _keep_words():
    z = np.load(FIXTURE)
    return z["night_words"].astype(np.int64), z["day_words"].astype(np.int64)


def test_control_quantiles_are_the_dissertation_values():
    night, day = _keep_words()
    assert (night.size, night.size + day.size) == (CONTROL_FRAMES, CONTROL_FRAMES_ALL)
    assert control_quantile_q16(night, QUANTILE) / Q16 == CONTROL_ETA
    assert control_quantile_q16(np.concatenate([night, day]), QUANTILE) / Q16 == CONTROL_ETA_ALL_FRAMES


def test_default_words_are_the_dissertation_words():
    night, _ = _keep_words()
    quantile_q16 = control_quantile_q16(night, QUANTILE)
    assert {c: scaled_word(quantile_q16, scale) for c, (scale, _) in WORDS.items()} == {
        c: word for c, (_, word) in WORDS.items()}
