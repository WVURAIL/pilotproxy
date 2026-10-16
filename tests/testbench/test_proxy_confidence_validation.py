from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = spec_from_file_location('confidence_validation', Path(__file__).parents[2] / 'tools' / 'validate_proxy_confidence_band.py')
validation = module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_analytic_targets_are_event_weighted_not_frame_weighted():
    words = np.arange(1, 9, dtype=np.uint64) * 65536
    np.testing.assert_allclose(validation.analytic_target('null_gaussian', words), 0)
    np.testing.assert_allclose(validation.analytic_target('independent_gaussian', words), .3)
    required, value, weight = [], [], []
    for q in range(8):
        r, m = validation.template('coupled_unequal', q)
        required.extend(r)
        value.extend(m)
        weight.extend([1 / len(r)] * len(r))
    required, value, weight = map(np.asarray, (required, value, weight))
    oracle = [np.average(value[required <= w], weights=weight[required <= w]) for w in words]
    np.testing.assert_allclose(validation.analytic_target('coupled_unequal', words), oracle)
    assert abs(oracle[-1] - value.mean()) > .01


def test_empirical_target_keeps_entire_score_ties():
    population = {'required': np.array([3, 1, 2, 2], np.uint64), 'proxy': np.array([10., -3., 2., 4.])}
    result = validation.empirical_target(population, np.array([0, 1, 2, 3], np.uint64))
    assert np.isnan(result[0])
    np.testing.assert_allclose(result[1:], [-3, 1, 13 / 4])


def test_independent_dataset_identity_and_overlap_preserve_event_mass():
    seed = validation.seed_for(20261016, 'correlated_blocks', 0, 'dataset')
    a, windows, clusters = validation.analytic_dataset('correlated_blocks', seed, 320)
    b, _, _ = validation.analytic_dataset('correlated_blocks', seed, 320)
    np.testing.assert_array_equal(a.excess, b.excess)
    np.testing.assert_allclose(np.bincount(a.event, weights=a.weight), 1)
    np.testing.assert_array_equal(clusters, a.event // 4)
    assert np.unique(clusters).size == 80
    np.testing.assert_array_equal(windows['last'], a.event >= 160)
    assert validation.seed_for(20261016, 'correlated_blocks', 1, 'dataset') != seed
    assert validation.seed_for(20261016, 'correlated_blocks', 0, 'bootstrap') != seed


def test_exact_miss_limits_match_closed_forms():
    alpha = .05 / 7
    none = validation.miss_interval(0, 400, alpha)
    all_missed = validation.miss_interval(400, 400, alpha)
    assert none['one_sided_lower'] == 0
    assert none['one_sided_upper'] == pytest.approx(1 - alpha ** (1 / 400))
    assert all_missed['one_sided_upper'] == 1
    assert all_missed['one_sided_lower'] == pytest.approx(alpha ** (1 / 400))


def test_rebuilds_observed_grid_and_joint_miss_oracle(monkeypatch):
    frames = validation.make_frames([65536, 131072, 131072], [0., 1., 2.], [0, 1, 2], [1., 1., 1.])
    windows = {'full': np.ones(3, bool), 'last': np.array([False, True, True])}
    captured = {}
    def fake_band(frames, words, **kwargs):
        captured['words'] = words.copy()
        result = {'eligible': np.ones(2, bool), 'signed_mean': np.array([0., .6]),
                  'se': np.array([.2, .2]), 'signed_upper': np.array([.4, 1.]),
                  'upper': np.array([.4, 1.]), 'kept_weight': np.array([1., 3.])}
        return {'status': 'available', 'critical': 2., 'windows': {'full': result, 'last': result}}
    monkeypatch.setattr(validation, 'simultaneous_upper_band', fake_band)
    out = validation.evaluate_dataset(frames, windows, None, lambda w: np.array([.5, .9]),
                                      SimpleNamespace(bootstrap_replicates=19), 5, .75)
    np.testing.assert_array_equal(captured['words'], [65536, 131072])
    assert out['miss'] and out['clipped_miss'] and not out['unavailable']
    assert out['eligible_points'] == 4
    assert out['selection_budget'] == .75
    assert out['required_multiplier'] == pytest.approx(1.25)
