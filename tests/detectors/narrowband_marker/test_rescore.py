"""Rescoring a score cache on its own fine geometry reproduces it, and a guard band changes the bulk."""
import numpy as np
import pytest


@pytest.mark.parametrize("reference_powers", [
    (1, 0),
    (0, 1),
    (1 << 62, 1 << 62),
    (1 << 63, 0),
    (0, 1 << 63),
    (1 << 63, 1 << 63),
    ((1 << 64) - 1, (1 << 64) - 1),
])
def test_score_rows_preserves_supported_equal_ratios_at_uint64_boundaries(reference_powers):
    from pilot_proxy.detectors.narrowband_marker import rescore as sc

    terms = np.ones((1, 3, 256), dtype=np.uint64)
    terms[:, 1, :] = reference_powers[0]
    terms[:, 2, :] = reference_powers[1]
    bulk = np.zeros(256, dtype=bool)
    bulk[:3] = True

    # Every designated ratio equals the positive background ratio, so the
    # first passing multiplier is exactly one, independent of its magnitude.
    words, supported, always_masked = sc.score_rows(terms, 128, bulk, 2, 2)
    np.testing.assert_array_equal(words, [65536])
    np.testing.assert_array_equal(supported, [True])
    np.testing.assert_array_equal(always_masked, [False])


def test_score_rows_requires_rank_support_from_nonzero_references():
    from pilot_proxy.detectors.narrowband_marker import rescore as sc

    terms = np.ones((3, 3, 256), dtype=np.uint64)
    terms[:, 1:, :] = 1 << 63
    bulk = np.zeros(256, dtype=bool)
    bulk[:3] = True
    # The frames have two, one, and zero supported background bins.
    terms[0, 1:, 2] = 0
    terms[1, 1:, 1:3] = 0
    terms[2, 1:, :] = 0

    words, supported, always_masked = sc.score_rows(terms, 128, bulk, 2, 2)
    np.testing.assert_array_equal(words, [65536, 0, 0])
    np.testing.assert_array_equal(supported, [True, False, False])
    np.testing.assert_array_equal(always_masked, [False, True, True])



def _cache_on_product(tmp_path, anchor):
    """A one-channel cache in the score-cache layout, built on a synthetic v5 product."""
    import json
    from pathlib import Path

    from pilot_proxy.characterization import score_cache
    from pilot_proxy.detectors.narrowband_marker import rescore as sc
    from pilot_proxy.detectors.narrowband_marker import anchors, scores
    from pilot_proxy.products.reader import Product
    from test_pilotproxy_v5 import _replace, _write_product

    product_path = _write_product(Path(tmp_path) / '506.npz', 36)
    with np.load(product_path, allow_pickle=False) as z:
        frames = z['reject_mask'].shape[0]
    # the health gate reads a per-frame baseband power the fixture lacks
    _replace(product_path, baseband_power_linear=np.full((frames, 1), 4.0))
    with Product(product_path) as product:
        rows = np.flatnonzero(product.selected)
        bulk = anchors.bulk_mask(anchor, pad_factor=product.fine_pad_factor, guard_fine_bins=product.fine_guard_bins,
                                 census_excluded_bins=product.fine_census_excluded_bins)
        rho = scores.candidate_rho_values(int(bulk.sum()))[0]
        words, rank, always = sc.score_rows(product.fine_terms(rows), anchor, bulk, 2, rho)
    inverse = np.arange(len(rows)) // 3
    word = 65536
    _, counts = np.unique(inverse, return_counts=True)
    arrays = dict(required_q16=words, supported=rank & ~always, frame_event_index=inverse,
                  probability_weight=1 / (len(counts) * counts[inverse]),
                  current_keep=(rank & ~always) & (words <= word))
    arrays['source_row_index'] = rows
    cache = Path(tmp_path) / 'cache'
    cache.mkdir()
    np.savez_compressed(cache / 'ch36.npz', **arrays)
    meta = {'channel': 36, 'npz_sha256': score_cache.sha(cache / 'ch36.npz'),
            'original_product_identity': {'path': str(product_path), 'sha256': score_cache.sha(product_path)},
            'current_threshold': {'kind': 'finite', 'eta_q16': str(word)},
            'score_geometry': {'anchor_bin': anchor, 'designated_half_width': 2,
                               'designated_bins': list(anchors.designated_set(anchor)),
                               'bulk_bins': [int(b) for b in np.flatnonzero(bulk)], 'rho_one_based': rho,
                               'cfar_rank_zero_based': rho - 1, 'intended_bulk_size': int(bulk.sum())}}
    (cache / 'ch36.json').write_text(json.dumps(meta))
    return cache / 'ch36.json', arrays


def test_rescore_reproduces_its_own_geometry_and_takes_a_guard_band(tmp_path):
    import json

    from pilot_proxy.detectors.narrowband_marker import rescore as sc

    cache, arrays = _cache_on_product(tmp_path, 128)
    same = sc.rescore(cache, tmp_path / 'same')
    with np.load(tmp_path / 'same/ch36.npz') as z:
        for key in ('required_q16', 'supported', 'current_keep', 'frame_event_index', 'source_row_index'):
            np.testing.assert_array_equal(z[key], arrays[key], err_msg=key)
    assert same['rho'] == same['previous_rho']
    guarded = sc.rescore(cache, tmp_path / 'guarded', marker_guard_bins=4)
    assert guarded['bulk_size'] < same['bulk_size']
    assert guarded['rho'] == (guarded['bulk_size'] + 1) // 2
    geometry = json.loads((tmp_path / 'guarded/ch36.json').read_text())['score_geometry']
    assert geometry['marker_guard_bins'] == 4 and geometry['rho_one_based'] == guarded['rho']
    assert not set(geometry['bulk_bins']) & {128 + k for k in range(-6, 7)}
    with pytest.raises(ValueError, match='already exists'):
        sc.rescore(cache, tmp_path / 'guarded', marker_guard_bins=4)


@pytest.mark.parametrize("guard_options", [{}, {"marker_guard_bins": 0}])
def test_rescore_clears_previous_guard_in_saved_geometry(tmp_path, guard_options):
    import json

    from pilot_proxy.detectors.narrowband_marker import rescore as sc

    cache, original = _cache_on_product(tmp_path, 128)
    sc.rescore(cache, tmp_path / 'guarded', marker_guard_bins=4)
    result = sc.rescore(tmp_path / 'guarded/ch36.json', tmp_path / 'restored', **guard_options)
    geometry = json.loads((tmp_path / 'restored/ch36.json').read_text())['score_geometry']
    original_geometry = json.loads(cache.read_text())['score_geometry']

    # Both the omitted argument and explicit zero restore the original search.
    # The saved description must agree with the actual background bins.
    assert result['marker_guard_bins'] == geometry['marker_guard_bins'] == 0
    assert geometry['bulk_bins'] == original_geometry['bulk_bins']
    assert geometry['rho_one_based'] == original_geometry['rho_one_based']
    with np.load(tmp_path / 'restored/ch36.npz') as restored:
        for key in ('required_q16', 'supported', 'current_keep'):
            np.testing.assert_array_equal(restored[key], original[key])
