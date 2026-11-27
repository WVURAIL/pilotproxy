"""The false-discovery-rate default word of fine decision v3 on synthetic scores."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from pilot_proxy.false_discovery import (  # noqa: E402
    N_EMPIRICAL,
    Q16,
    ControlNull,
    fdr_threshold,
    gpd_fit,
    gpd_survival,
    pot,
)

SEED = 20261127


def _null_words(rng, n):
    # control scores near 1 with an exponential upper tail, as Q16 words
    return np.ceil((1.0 + 0.03 * rng.standard_normal(n) + 0.02 * rng.exponential(size=n)) * Q16).astype(np.int64)


@pytest.fixture(scope="module")
def null():
    rng = np.random.default_rng(SEED)
    return ControlNull(_null_words(rng, 20000) / Q16)


def test_the_gpd_fit_recovers_its_parameters():
    rng = np.random.default_rng(SEED)
    xi, sigma = 0.2, 0.5
    y = sigma / xi * (rng.uniform(size=50000) ** -xi - 1)
    fit = gpd_fit(y)
    assert fit[0] == pytest.approx(xi, abs=0.03) and fit[1] == pytest.approx(sigma, rel=0.03)


def test_a_clean_channel_masks_nothing(null):
    rng = np.random.default_rng(SEED + 1)
    found = fdr_threshold(_null_words(rng, 20000), 1.0, null)
    assert found.word is None and found.fdr is None and found.frames == 20000


def test_more_dtv_moves_the_threshold_down(null):
    rng = np.random.default_rng(SEED + 2)
    clean = _null_words(rng, 20000)
    dtv = np.ceil(rng.uniform(1.3, 3.0, size=20000) * Q16).astype(np.int64)
    light = fdr_threshold(np.concatenate([clean, dtv[:400]]), 1.0, null)
    heavy = fdr_threshold(np.concatenate([clean, dtv[:8000]]), 1.0, null)
    assert light.word is not None and heavy.word is not None
    assert heavy.word < light.word
    for found in (light, heavy):
        # the step-up rule: the estimate at the word meets the level
        assert found.fdr <= 0.01 and found.fdr == pytest.approx(found.control_share / found.channel_share)


def test_the_word_is_a_frame_of_the_channel_on_the_scaled_null(null):
    rng = np.random.default_rng(SEED + 3)
    words = np.concatenate([_null_words(rng, 20000), np.ceil(rng.uniform(1.3, 3.0, size=4000) * Q16).astype(np.int64)])
    found = fdr_threshold(words, 1.0, null)
    assert found.word in set(words.tolist())
    # a channel of twice the null level against the null scaled by 2 finds the same frame
    doubled = fdr_threshold(2 * words, 2.0, null)
    assert doubled.word == 2 * found.word


def test_beyond_the_empirical_range_the_null_is_the_gpd_tail(null):
    rng = np.random.default_rng(SEED + 4)
    # 1% strong frames: the rate needs a null share near 1e-4, below the N_EMPIRICAL / n = 1e-3 the empirical share
    # resolves
    words = np.concatenate([_null_words(rng, 20000), np.ceil(rng.uniform(1.5, 4.0, size=200) * Q16).astype(np.int64)])
    found = fdr_threshold(words, 1.0, null)
    assert found.word is not None and found.control_share_source == "tail"
    assert found.word / Q16 > null.tail_from
    f = null.fit
    assert found.control_share == pytest.approx(float(gpd_survival(found.word / Q16, f["u"], f["zeta"], f["xi"],
                                                                   f["sigma"])))
    # within the empirical range the share is the count of control frames above
    share, empirical = null(np.array([null.sorted[0]]))
    assert empirical[0] and share[0] == (null.n - 1) / null.n


def test_the_tail_fit_needs_enough_null_frames():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="control_quantile"):
        pot(_null_words(rng, N_EMPIRICAL * 40) / Q16)


def test_a_channel_without_frames_masks_nothing(null):
    assert fdr_threshold(np.empty(0, dtype=np.int64), 1.0, null).word is None
