"""Recompute exact score words for another detector geometry, preserving the population and reference threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pilot_proxy.characterization.score_cache import (
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
    # Nonnegative reference powers have a positive sum when either is nonzero.
    # Test support without overflowing the uint64 powers or a signed cast.
    reference_present = (terms[:, 1, :] != 0) | (terms[:, 2, :] != 0)
    support = reference_present[:, bulk_idx].sum(1)
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


def rescore(cache_json, output, *, anchor_bin=None, designated_half_width=2, marker_guard_bins=0):
    """Write a new authenticated score cache without selecting an operating threshold."""
    cache_json = Path(cache_json)
    output = Path(output)
    meta = json.loads(cache_json.read_text())
    cache_npz = cache_json.with_suffix('.npz')
    require(sha(cache_npz) == meta['npz_sha256'], 'Cache npz differs from its metadata')
    channel = meta['channel']
    require(not any((output / f'ch{channel}{suffix}').exists() for suffix in ('.json', '.npz')),
            'Rescored cache already exists')
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
    computed = np.ones(len(rows), bool)
    current_word = int(meta["current_threshold"]["eta_q16"])
    arrays = dict(old)
    arrays.pop("rank_supported", None)
    arrays.pop("always_masked", None)
    arrays.update(required_q16=words, supported=rank & ~always, score_computed=computed,
                  current_keep=(rank & ~always) & (words <= current_word))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f'ch{channel}.npz', **arrays)
    geometry = dict(old_geometry)
    geometry.update(anchor_bin=anchor, designated_half_width=designated_half_width,
                    designated_bins=list(anchors.designated_set(anchor, designated_half_width)),
                    bulk_bins=[int(b) for b in np.flatnonzero(bulk)], rho_one_based=int(rho),
                    cfar_rank_zero_based=int(rho) - 1, intended_bulk_size=int(bulk.sum()))
    geometry['marker_guard_bins'] = marker_guard_bins
    value = dict(meta)
    value.pop('accepted_exact_replay', None)
    value.update(score_geometry=geometry,
                 score_geometry_source=f'rescored on h={designated_half_width}, g={marker_guard_bins}, anchor {anchor}',
                 current_exact_mask_unchanged=False, current_assignment_unchanged=True,
                 npz_sha256=sha(output / f'ch{channel}.npz'),
                 rescore={'previous_cache': {'json': str(cache_json), 'json_sha256': sha(cache_json),
                                             'npz': str(cache_npz), 'npz_sha256': meta['npz_sha256']},
                          'source_sha256': sha(__file__), 'current_word': current_word})
    write_json(output / f'ch{channel}.json', value)
    return {'channel': channel, 'anchor_bin': anchor, 'designated_half_width': designated_half_width,
            'marker_guard_bins': marker_guard_bins, 'bulk_size': int(bulk.sum()), 'rho': int(rho),
            'previous_bulk_size': len(old_geometry['bulk_bins']), 'previous_rho': old_geometry.get('rho_one_based'),
            'supported_frames': int((rank & ~always).sum()), 'frames': len(rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rescore", description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path, help="ch<N>.json of a score cache (its .npz beside it)")
    parser.add_argument("output", type=Path, help="directory for the rescored ch<N>.json and .npz")
    parser.add_argument("--anchor-bin", type=int, default=None)
    parser.add_argument("--designated-half-width", type=int, default=2)
    parser.add_argument("--marker-guard-bins", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(rescore(args.cache, args.output, anchor_bin=args.anchor_bin,
                             designated_half_width=args.designated_half_width,
                             marker_guard_bins=args.marker_guard_bins)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
