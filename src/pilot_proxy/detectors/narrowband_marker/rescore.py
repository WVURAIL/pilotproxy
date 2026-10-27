"""Rescore one channel's score cache with a given anchor, marker half-width h, and
marker guard band g, the fine geometry options of the detector v2 calibration.

The frames, events, and weights of the cache are kept, and the words, rank support,
knee, and candidates are recomputed from the cache's original product. The rank is
the median of the resulting bulk (``scores.candidate_rho_values``). At the cache's
own geometry, the recomputed words and support reproduce the cache. Every cache
records the knee of knee rule v1, and under that rule the recomputed knee and
candidates also reproduce the cache. The output is written to a new directory, and
nothing of record is changed.

    python -m pilot_proxy.detectors.narrowband_marker.rescore CACHE.json OUT_DIR
        [--anchor-bin A] [--designated-half-width H] [--marker-guard-bins G] [--knee-rule ID]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot_proxy.characterization import knee
from pilot_proxy.characterization.score_cache import (
    candidate_cache,
    require,
    sha,
    write_json,
)
from pilot_proxy.products.reader import Product

from . import anchors, scores


def score_rows(terms, anchor_bin, bulk, designated_half_width, rho):
    """Exact minimum keep words, rank support and always-masked flags of frames (rows, 3, 256)."""
    terms = np.asarray(terms)
    bulk_idx = np.flatnonzero(bulk)
    n = len(terms)
    words = np.zeros(n, np.uint64)
    always = np.zeros(n, bool)
    denominator = terms[:, 1, :].astype(np.int64) + terms[:, 2, :].astype(np.int64)
    support = (denominator[:, bulk_idx] > 0).sum(1)
    for i in range(n):
        if support[i] < rho:
            always[i] = True
            continue
        word = scores.required_eta_q16_by_rank(terms[i], anchor_bin=anchor_bin,
                                               designated_half_width=designated_half_width,
                                               bulk_mask=bulk)[rho - 1]
        always[i] = word == scores.ALWAYS_MASKED_Q16
        if not always[i]:
            words[i] = word
    return words, support >= rho, always


def rescore(cache_json, output, *, anchor_bin=None, designated_half_width=2, marker_guard_bins=0,
            knee_rule=knee.RULE_ID_V1):
    """Write ``ch<N>.json`` and ``ch<N>.npz`` for one cache rescored on a fine geometry.

    ``anchor_bin`` None keeps the cache's anchor. The source product is the cache's
    original product, whose frame powers are read and never changed. By default the
    recorded knee is that of rule v1, as in every cache. The threshold rule checks this
    knee and then decides with its own current rule. Returns the summary of the old and
    new geometry, knee, and supported frames.
    """
    cache_json = Path(cache_json)
    output = Path(output)
    meta = json.loads(cache_json.read_text())
    cache_npz = cache_json.with_suffix('.npz')
    require(sha(cache_npz) == meta['npz_sha256'], 'Cache npz differs from its metadata')
    channel = meta['channel']
    require(not (output / f'ch{channel}.json').exists(), 'Rescored cache already exists')
    with np.load(cache_npz, allow_pickle=False) as z:
        old = {key: z[key] for key in z.files}
    old_geometry = meta['score_geometry']
    anchor = int(old_geometry['anchor_bin'] if anchor_bin is None else anchor_bin)
    rows = old['source_row_index']
    product_identity = meta['original_product_identity']
    require(sha(product_identity['path']) == product_identity['sha256'], 'Source product differs from the cache')
    with Product(product_identity['path']) as product:
        bulk = anchors.bulk_mask(anchor, pad_factor=product.fine_pad_factor, guard_fine_bins=product.fine_guard_bins,
                                 census_excluded_bins=product.fine_census_excluded_bins,
                                 designated_half_width=designated_half_width, marker_guard_bins=marker_guard_bins)
        rho = scores.candidate_rho_values(int(bulk.sum()))[0]
        words, rank, always = score_rows(product.fine_terms(rows), anchor, bulk, designated_half_width, rho)
    inverse = old['frame_event_index']
    computed = np.ones(len(rows), bool)
    proposal = knee.propose(words, [str(int(event)) for event in inverse], valid=list(rank & computed),
                            always_masked=list(always), rule=knee_rule)
    knee_word = int(proposal['knee_word_q16'])
    reference, candidates, arrays, encoding = candidate_cache(words, rank, always, inverse, knee_word, knee_word)
    for key, value in old.items():
        arrays.setdefault(key, value)
    arrays['score_computed'] = computed
    arrays['current_keep'] = arrays['supported'] & (words <= np.uint64(knee_word))
    current = next(row for row in candidates if 'current_assignment' in row['roles'])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'ch{channel}.npz', **arrays)
    geometry = dict(old_geometry)
    geometry.update(anchor_bin=anchor, designated_half_width=designated_half_width,
                    designated_bins=list(anchors.designated_set(anchor, designated_half_width)),
                    bulk_bins=[int(b) for b in np.flatnonzero(bulk)], rho_one_based=int(rho),
                    cfar_rank_zero_based=int(rho) - 1, intended_bulk_size=int(bulk.sum()))
    if marker_guard_bins:
        geometry['marker_guard_bins'] = marker_guard_bins
    value = dict(meta)
    value.pop('accepted_exact_replay', None)
    value.update(score_geometry=geometry,
                 score_geometry_source=f'rescored on h={designated_half_width}, g={marker_guard_bins}, anchor {anchor}',
                 knee_threshold={'kind': 'finite', 'eta_q16': str(knee_word)},
                 current_threshold={'kind': 'finite', 'eta_q16': str(knee_word)}, current_candidate_id=current['id'],
                 weight_numerator_encoding=encoding,
                 weight_denominator_decimal=str(arrays['weight_denominator_decimal'].item()),
                 reference=reference, candidates=candidates, current_exact_mask_unchanged=False,
                 current_assignment_unchanged=False, npz_sha256=sha(output / f'ch{channel}.npz'),
                 rescore={'previous_cache': {'json': str(cache_json), 'json_sha256': sha(cache_json),
                                             'npz': str(cache_npz), 'npz_sha256': meta['npz_sha256']},
                          'knee_rule': knee_rule, 'source_sha256': sha(__file__),
                          'current_word': 'the knee of the rescored words'})
    write_json(output / f'ch{channel}.json', value)
    return {'channel': channel, 'anchor_bin': anchor, 'designated_half_width': designated_half_width,
            'marker_guard_bins': marker_guard_bins, 'bulk_size': int(bulk.sum()), 'rho': int(rho),
            'previous_bulk_size': len(old_geometry['bulk_bins']), 'previous_rho': old_geometry.get('rho_one_based'),
            'knee_word': knee_word, 'previous_knee_word': int(meta['knee_threshold']['eta_q16']),
            'supported_frames': int((rank & ~always).sum()), 'frames': len(rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rescore", description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path, help="ch<N>.json of a score cache (its .npz beside it)")
    parser.add_argument("output", type=Path, help="directory for the rescored ch<N>.json and .npz")
    parser.add_argument("--anchor-bin", type=int, default=None)
    parser.add_argument("--designated-half-width", type=int, default=2)
    parser.add_argument("--marker-guard-bins", type=int, default=0)
    parser.add_argument("--knee-rule", default=knee.RULE_ID_V1)
    args = parser.parse_args(argv)
    print(json.dumps(rescore(args.cache, args.output, anchor_bin=args.anchor_bin,
                             designated_half_width=args.designated_half_width,
                             marker_guard_bins=args.marker_guard_bins, knee_rule=args.knee_rule)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
