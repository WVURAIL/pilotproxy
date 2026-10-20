"""Rescoring a score cache on a fine geometry: its own geometry reproduces it; a guard band changes the bulk."""
import numpy as np
import pytest

from pilot_proxy.characterization.score_cache import candidate_cache


def _cache_on_product(tmp_path, anchor):
    """A one-channel cache in the score-cache layout, built on a synthetic v5 product."""
    import json
    from pathlib import Path

    from pilot_proxy.characterization import knee
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
    knee_word = int(knee.propose(words, [str(e) for e in inverse], valid=list(rank), always_masked=list(always),
                                 rule=knee.RULE_ID_V1)['knee_word_q16'])
    _, candidates, arrays, _ = candidate_cache(words, rank, always, inverse, knee_word, knee_word)
    arrays['source_row_index'] = rows
    cache = Path(tmp_path) / 'cache'
    cache.mkdir()
    np.savez_compressed(cache / 'ch36.npz', **arrays)
    meta = {'channel': 36, 'npz_sha256': score_cache.sha(cache / 'ch36.npz'),
            'original_product_identity': {'path': str(product_path), 'sha256': score_cache.sha(product_path)},
            'knee_threshold': {'kind': 'finite', 'eta_q16': str(knee_word)},
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
        for key in ('required_q16', 'rank_supported', 'always_masked', 'supported', 'candidate_keep',
                    'candidate_ids', 'frame_event_index', 'source_row_index'):
            np.testing.assert_array_equal(z[key], arrays[key], err_msg=key)
    assert same['knee_word'] == same['previous_knee_word'] and same['rho'] == same['previous_rho']
    guarded = sc.rescore(cache, tmp_path / 'guarded', marker_guard_bins=4)
    assert guarded['bulk_size'] < same['bulk_size']
    assert guarded['rho'] == (guarded['bulk_size'] + 1) // 2
    geometry = json.loads((tmp_path / 'guarded/ch36.json').read_text())['score_geometry']
    assert geometry['marker_guard_bins'] == 4 and geometry['rho_one_based'] == guarded['rho']
    assert not set(geometry['bulk_bins']) & {128 + k for k in range(-6, 7)}
    with pytest.raises(ValueError, match='already exists'):
        sc.rescore(cache, tmp_path / 'guarded', marker_guard_bins=4)
