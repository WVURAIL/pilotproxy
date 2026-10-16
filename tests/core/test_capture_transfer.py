import numpy as np
import pytest

from pilot_proxy.characterization.capture_transfer import (
    coherent_cross_moment, event_weights, exact_frame_join, fit_training_transfer,
    lag_moments, signed_proxy, transfer_residuals,
)


def test_signed_proxy_and_invalid_reference():
    p = np.array([[4, 4, 4], [8, 4, 4], [0, 0, 0]], dtype=np.uint64)
    got = signed_proxy(p, 1, 1, 2)
    np.testing.assert_allclose(got[:2], [-1, 0])
    assert np.isnan(got[2])
    with pytest.raises(ValueError):
        signed_proxy(p.astype(float), 1, 1, 2)


def test_exact_join_refuses_duplicates_and_does_not_round():
    i, j = exact_frame_join(np.array([10, 20, 30]), np.array([11, 20, 31]))
    assert i.tolist() == [1] and j.tolist() == [1]
    with pytest.raises(ValueError):
        exact_frame_join(np.array([10, 10]), np.array([10]))
    with pytest.raises(ValueError):
        exact_frame_join(np.array([-1]), np.array([10]))
    with pytest.raises(ValueError):
        exact_frame_join(np.array([2**64 - 1], dtype=np.uint64), np.array([10]))
    a, b = exact_frame_join(np.array([2**60], dtype=np.uint64), np.array([2**60 + 1], dtype=np.int64))
    assert not len(a) and not len(b)


def test_event_weights_preserve_original_measure_after_selection():
    w = event_weights([0, 0, 1, 1, 1, 1])
    np.testing.assert_allclose(w, [0.25, 0.25, 0.125, 0.125, 0.125, 0.125])
    assert w[[0, 2]].sum() == 0.375


def test_frozen_fit_predicts_new_data_without_refitting():
    x = np.array([-2., 0., 1., 3.])[:, None]
    ratio = 4 + 0.2 * x
    model = fit_training_transfer(x, ratio, [0, 0, 1, 1])
    new_x = np.array([[5.], [8.]])
    y, prediction, residual = transfer_residuals(model, new_x, 4 + 0.2 * new_x)
    np.testing.assert_allclose(y, prediction, atol=1e-14)
    np.testing.assert_allclose(residual, 0, atol=1e-14)
    old = model['slope'].copy()
    transfer_residuals(model, new_x, np.full_like(new_x, 900.))
    np.testing.assert_array_equal(old, model['slope'])


def test_training_live_mask_excludes_dead_inputs_without_holdout_selection():
    x = np.array([[1., 1.], [2., 2.]])
    model = fit_training_transfer(x, x + 4, [0, 1], auto=np.array([[1, 0], [1, 1]]))
    assert model['live'].tolist() == [True, False]
    with pytest.raises(ValueError):
        fit_training_transfer(x, x, [0])
    with pytest.raises(ValueError):
        fit_training_transfer(x, x, [0, 1], auto=np.ones(2))


def test_lag_sign_conjugation_and_gap_refusal():
    v = np.array([1 + 2j, 3 - 1j, -2 + 1j])[:, None]
    t = np.array([0, 16384, 3 * 16384])
    m, c, n = lag_moments(v, t, np.ones(3, bool), max_lag=2)
    assert n.tolist() == [3, 1, 1]
    np.testing.assert_allclose(m[1], np.conj(v[0]) * v[1])
    np.testing.assert_allclose(c[1], np.conj(v[0] - v.mean(0)) * (v[1] - v.mean(0)))
    _, _, n = lag_moments(v, t, np.array([True, False, True]), max_lag=2)
    assert n.tolist() == [2, 0, 0]


def test_coherent_signed_cross_moment_is_not_clipped():
    got = coherent_cross_moment(np.array([[1j], [-1j]]), [True, True])
    np.testing.assert_allclose(got, [-1])
    assert np.isnan(coherent_cross_moment(np.array([[1.]]), [True])).all()


def test_lag_origin_overflow_refused():
    with pytest.raises(ValueError):
        lag_moments(np.ones((1, 1)), np.array([2**63 - 1]), [True])
