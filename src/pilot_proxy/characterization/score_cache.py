"""The current-era exact score cache: the input of the threshold rule.

For each channel of the working thresholds, the current-era frames of its fine
score export (or, for a channel rescored on a revised geometry, of its exact
replay) with their exact minimum keep words, the event weights (one unit per
source event, uniform over its frames) and a bounded grid of candidate words,
written as ``ch<N>.json`` and ``ch<N>.npz``. Every input is authenticated by
its sha256. The knee is recomputed from the frames
(:mod:`pilot_proxy.characterization.knee`) and must equal the recorded knee.

No assigned threshold, accepted mask, or source product is changed. The score
feature is the integer minimum deployable Q16 word, not measured RFI power.

    pilot-proxy characterize score-cache --working DIR --exports FILE --previous FILE --output DIR
        [--era-list projects/chime_atsc/eras/power_eras_2026-10-04.json]

``--working`` is the working-threshold delivery (``DELIVERY_MANIFEST.json`` and
``WORKING_THRESHOLDS.json``: each channel's current word and months, score
geometry and accepted replay), ``--exports`` the record naming each channel's
fine score export and its metadata, ``--previous`` the power-era threshold
record (the original score geometry and knee).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import knee

MAX_WORD = 2**64 - 1
# the 2026-10-04 inputs of record
LATEST_MANIFEST_SHA = '2d7eb52fd230353d4297901c6a5e966b8e6f2a35387ed56b22d6eb7e3452ad16'
BURST_PROPOSALS_SHA = 'fc1ee937b0439a0ad111329a6f8a778409b80278743a1d90534f2d344d389386'
QUANTILES = tuple(Fraction(i, 20) for i in range(21))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def exact(value):
    if value is None:
        return None
    value = Fraction(value)
    projection = float(value)
    require(math.isfinite(projection), 'Nonfinite fraction projection')
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'float': projection}


def integer_array(values, label):
    values = np.asarray(values)
    require(values.ndim == 1 and values.dtype.kind in 'iu', f'{label} must be an integer vector')
    return values


def event_weights(frame_event_index):
    """One unit of probability per event, uniform among its original frames."""
    inverse = integer_array(frame_event_index, 'frame_event_index')
    require(len(inverse) > 0 and np.all(inverse >= 0), 'Nonempty event support required')
    events = np.unique(inverse)
    require(np.array_equal(events, np.arange(len(events))), 'Event indices must be complete and contiguous')
    counts = np.bincount(inverse)
    lcm = math.lcm(*map(int, counts))
    denominator = len(counts) * lcm
    # Python integers avoid overflow even when many event lengths have a large LCM.
    weights = np.asarray([lcm // int(counts[event]) for event in inverse], dtype=object)
    require(sum(weights) == denominator, 'Event-balanced weights do not sum to one')
    return weights, denominator, counts


def candidate_cache(words, rank_supported, always_masked, frame_event_index, current_word, knee_word):
    words = np.asarray(words)
    rank, always = np.asarray(rank_supported), np.asarray(always_masked)
    require(words.ndim == 1 and words.dtype == np.uint64, 'Exact uint64 words required')
    require(rank.shape == words.shape and always.shape == words.shape and rank.dtype == bool and always.dtype == bool,
            'Aligned boolean support flags required')
    inverse = integer_array(frame_event_index, 'frame_event_index')
    require(inverse.shape == words.shape, 'Event and word rows disagree')
    for value in [current_word, knee_word]:
        require(type(value) is int and 0 <= value <= MAX_WORD, 'Exact finite current/knee word required')
    weights, denominator, counts = event_weights(inverse)
    supported = rank & ~always
    support_mass = int(sum(weights[supported]))
    sorted_rows = np.flatnonzero(supported)[np.argsort(words[supported], kind='stable')]
    unique, starts = np.unique(words[sorted_rows], return_index=True)
    masses = np.add.reduceat(weights[sorted_rows], starts) if len(unique) else np.empty(0, dtype=object)
    cumulative_mass = np.cumsum(masses, dtype=object)
    cumulative_word_mass = np.cumsum(np.asarray([int(w) * int(m) for w, m in zip(unique, masses, strict=True)], dtype=object))
    total_word_mass = int(cumulative_word_mass[-1]) if len(unique) else 0
    reference_mean = Fraction(total_word_mass, support_mass) if support_mass else None
    reference = {'supported_probability': exact(Fraction(support_mass, denominator)),
                 'weighted_word_sum': exact(Fraction(total_word_mass, denominator)),
                 'mean_word': exact(reference_mean), 'supported_frames': int(supported.sum()),
                 'unsupported_frames': int((~supported).sum())}
    roles = {}

    def add(word, role):
        roles.setdefault(int(word), []).append(role)

    if support_mass:
        index = 0
        for quantile in QUANTILES:
            # q=0 chooses the minimum supported word; all other boundaries are
            # nearest weighted-CDF values with exact integer comparisons.
            if quantile:
                while int(cumulative_mass[index]) * quantile.denominator < quantile.numerator * support_mass:
                    index += 1
            add(unique[index], f'quantile_{quantile.numerator}_over_{quantile.denominator}')
    add(current_word, 'current_assignment')
    add(knee_word, 'recorded_knee')
    add(0, 'numeric_zero')
    add(MAX_WORD, 'all_supported_endpoint')
    assignments = [({'kind': 'finite', 'eta_q16': str(word)}, labels) for word, labels in sorted(roles.items())]
    assignments.append(({'kind': 'reject_all'}, ['permanent_mask']))
    masks, candidates = [], []
    for index, (threshold, labels) in enumerate(assignments):
        permanent = threshold['kind'] == 'reject_all'
        if permanent:
            keep = np.zeros(len(words), bool)
            mass = word_mass = 0
            identity = 'reject_all'
        else:
            word = int(threshold['eta_q16'])
            keep = supported & (words <= np.uint64(word))
            position = int(np.searchsorted(unique, np.uint64(word), side='right')) - 1
            mass = int(cumulative_mass[position]) if position >= 0 else 0
            word_mass = int(cumulative_word_mass[position]) if position >= 0 else 0
            identity = f'q16_{word}'
        require(mass == int(sum(weights[keep])), 'Candidate prefix mass disagrees with exact mask')
        mean = Fraction(word_mass, mass) if mass else None
        g = mean / reference_mean if mean is not None and reference_mean else None
        if not support_mass:
            status = 'unavailable_no_supported_reference'
        elif not mass:
            status = 'unavailable_no_retained_support'
        elif not reference_mean:
            status = 'unavailable_zero_reference_mean'
        else:
            status = 'detector_summary_available'
        candidates.append({'id': identity, 'threshold': threshold, 'roles': labels,
            'mask_npz_row': index, 'retained_frames': int(keep.sum()),
            'retained_events': int(len(np.unique(inverse[keep]))),
            'retained_probability': exact(Fraction(mass, denominator)),
            'masked_probability': exact(Fraction(denominator - mass, denominator)),
            'unsupported_probability': exact(Fraction(denominator - support_mass, denominator)),
            'threshold_rejected_supported_probability': exact(Fraction(support_mass - mass, denominator)),
            'weighted_word_sum': exact(Fraction(word_mass, denominator)),
            'conditional_mean_word': exact(mean), 'g': exact(g), 'transfer_status': status,
            'permanent_mask': permanent, 'physical_science_qualified': False})
        masks.append(keep)
    require(len(candidates) <= 26, 'Bounded candidate count exceeded')
    arrays = {'required_q16': words, 'rank_supported': rank, 'always_masked': always,
              'supported': supported, 'frame_event_index': inverse,
              'event_frame_counts': counts, 'probability_weight': np.asarray([int(w) / denominator for w in weights]),
              'weight_denominator_decimal': np.asarray(str(denominator)),
              'candidate_keep': np.asarray(masks), 'candidate_ids': np.asarray([row['id'] for row in candidates])}
    if max(weights) <= MAX_WORD:
        arrays['weight_numerator'] = np.asarray(weights, np.uint64)
        encoding = 'uint64'
    else:
        arrays['weight_numerator_decimal'] = np.asarray(list(map(str, weights)))
        encoding = 'decimal_strings'
    return reference, candidates, arrays, encoding


class Authenticator:
    def __init__(self):
        self.files = {}

    def check(self, path, expected):
        path = Path(path)
        actual = sha(path)
        require(actual == expected, f'Input hash mismatch: {path}')
        self.files[str(path)] = actual

    def json(self, path, expected):
        self.check(path, expected)
        return json.loads(Path(path).read_text())

    def npz(self, path, expected):
        self.check(path, expected)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}


def event_number(key):
    payload = json.loads(str(key).split(':', 1)[1])
    require(len(payload) == 3, 'Unexpected source event key structure')
    return int(payload[1])


def source_identity(data, rows, months):
    n = len(data['selected'])
    require(np.array_equal(data['fine_source_row_index'], np.arange(n)), 'Source fine-row ordering changed')
    keys = data['source_event_keys']
    require(len(np.unique(keys)) == len(keys), 'Duplicate source event identity')
    require(np.array_equal([event_number(key) for key in keys], data['unit_event_id']), 'Source key/event number mismatch')
    units = data['frame_unit_index']
    require(units.dtype.kind in 'iu' and units.shape == (n,) and np.all(units >= 0) and np.all(units < len(keys)), 'Bad frame-to-unit mapping')
    require(np.array_equal(data['acquisition_id'], data['unit_event_id'][units]), 'Frame numeric/source-event identity mismatch')
    wanted = np.asarray([12 * int(month[:4]) + int(month[5:]) - 1 for month in months])
    timed = data['selected'] & data['finite_time']
    recomputed = np.flatnonzero(timed & np.isin(data['frame_month'], wanted))
    require(np.array_equal(rows, recomputed), 'Current cohort does not reproduce saved frame rows')
    before = np.bincount(units[timed], minlength=len(keys))
    after = np.bincount(units[rows], minlength=len(keys))
    require(not np.any((after > 0) & (after < before)), 'Current cohort splits a source event')
    frame_keys = keys[units[rows]]
    event_keys, inverse, counts = np.unique(frame_keys, return_inverse=True, return_counts=True)
    key_to_unit = {str(key): unit for unit, key in enumerate(keys)}
    event_source_units = np.asarray([key_to_unit[str(key)] for key in event_keys])
    require(len(set(zip(map(int, units[rows]), map(int, data['frame_in_unit'][rows])))) == len(rows), 'Duplicate frame identity')
    _, first = np.unique(inverse, return_index=True)
    event_months = data['frame_month'][rows[first]]
    require(np.array_equal(event_months[inverse], data['frame_month'][rows]), 'Source event has ambiguous calendar membership')
    legacy = data['unit_keys'][event_source_units]
    legacy_agreement = np.asarray([event_number(key) for key in legacy]) == data['unit_event_id'][event_source_units]
    arrays = {'source_row_index': rows, 'frame_index': data['frame_index'][rows],
        'frame_unit_index': units[rows], 'frame_in_unit': data['frame_in_unit'][rows],
        'frame_time': data['frame_time'][rows], 'frame_month': data['frame_month'][rows],
        'acquisition_id': data['acquisition_id'][rows], 'event_keys': event_keys,
        'event_source_unit_index': event_source_units,
        'event_acquisition_id': data['unit_event_id'][event_source_units],
        'event_month': np.asarray(event_months),
        'legacy_original_unit_key': legacy, 'legacy_unit_key_matches_event': legacy_agreement}
    return inverse, counts, arrays


def build(working_delivery, exports, previous, output, *, working_delivery_sha256=LATEST_MANIFEST_SHA,
          exports_sha256=BURST_PROPOSALS_SHA, era_list=None):
    """Write the score cache of every channel of the working thresholds into ``output``."""
    LATEST, HERE = Path(working_delivery), Path(output)
    HERE.mkdir(parents=True, exist_ok=True)
    require(not (HERE / 'INDEX.json').exists(), 'Completed score cache already exists')
    auth = Authenticator()
    manifest = auth.json(LATEST / 'DELIVERY_MANIFEST.json', working_delivery_sha256)
    working = auth.json(LATEST / 'WORKING_THRESHOLDS.json', manifest['WORKING_THRESHOLDS.json']['sha256'])
    require([row['channel'] for row in working['channels']] == list(range(14, 38)), 'All 24 channels required')
    legacy = auth.json(exports, exports_sha256)
    robust_path = Path(previous)
    robust_hash = working['channels'][0]['previous_geometry_preserved_in']['sha256']
    robust = auth.json(robust_path, robust_hash)
    old = {row['channel']: row for row in robust['channels']}
    if era_list is not None:
        cohorts = {row['channel']: row['current_months'] for row in json.loads(Path(era_list).read_text())['channels']}
        require(all(cohorts.get(row['channel']) == row['current_months'] for row in working['channels']),
                'Working-threshold months differ from the era list')
    plan = {'schema': 'current-era-exact-score-grid-plan-v1', 'source_sha256': sha(__file__),
        'working_thresholds_sha256': auth.files[str(LATEST / 'WORKING_THRESHOLDS.json')],
        'grid_fixed_before_generation': True,
        'quantiles': [exact(q) for q in QUANTILES], 'maximum_candidates_per_channel': 26,
        'other_roles': ['current_assignment', 'recorded_knee', 'numeric_zero', 'all_supported_endpoint', 'permanent_mask'],
        'weight_rule': 'Each original current-era event has equal mass; each included frame in that event has equal mass. Unsupported frames retain denominator mass.',
        'candidate_scope': 'Provisional bounded evaluation grid only; no global optimum, assignment, mask or geometry changes.',
        'score_feature': 'Exact minimum required uint64 Q16 word, including ceiling/minimum-word semantics; not unrounded score or physical RFI power.',
        'conditional_rule': 'Condition the same original event/frame measure on keep; do not re-equalize surviving events.',
        'transfer_f': 'retained_probability; masked_probability is also exported explicitly',
        'transfer_g': 'conditional_mean_word / supported_reference_mean_word; undefined for empty retention or zero/missing reference mean',
        'physical_response_derived': False, 'threshold_activated': False}
    write_json(HERE / 'PLAN.json', plan)
    index = []
    for record in sorted(working['channels'], key=lambda row: (row['channel'] != 19, row['channel'])):
        channel = record['channel']
        previous = old[channel]
        base = legacy[str(channel)]
        data = auth.npz(base['source_npz'], base['source_npz_sha256'])
        metadata = auth.json(base['source_metadata'], base['source_metadata_sha256'])
        require(previous['score_source']['sha256'] == base['source_npz_sha256'], 'Source lineage mismatch')
        require(metadata['geometry'] == previous['score_geometry'], 'Original exact score geometry mismatch')
        replay = auth.npz(record['exact_replay']['path'], record['exact_replay']['sha256'])
        revised = channel in (20, 34)
        rows = replay['source_row_index'] if revised else replay['cohort_source_row_index']
        inverse, counts, identities = source_identity(data, rows, record['current_months'])
        require(np.array_equal(identities['event_keys'], replay['event_keys']), 'Accepted event keys changed')
        require(np.array_equal(counts, replay['event_frames_before']), 'Accepted event counts changed')
        require(np.array_equal(identities['event_month'], replay['event_month']), 'Accepted event months changed')
        current_word = int(record['threshold']['eta_q16'])
        if revised:
            proposal = auth.json(record['candidate_source']['path'], record['candidate_source']['sha256'])
            require(np.array_equal(rows, np.flatnonzero(data['selected'] & data['finite_time'] & np.isin(data['frame_month'], identities['event_month']))), 'Revised geometry cohort mismatch')
            require(np.array_equal(replay['frame_event_inverse'], inverse), 'Revised geometry event mapping mismatch')
            require(np.array_equal(replay['original_required_q16'], data['fine_required_multiplier_q16'][rows]), 'Revised cache original-word identity mismatch')
            require(np.array_equal(np.flatnonzero(replay['new_bulk_mask']), record['score_geometry']['bulk_bins']), 'Revised bulk geometry mismatch')
            require(np.array_equal(replay['new_designated_bins'], record['score_geometry']['designated_bins']), 'Revised designated geometry mismatch')
            require(int(replay['new_rho']) == record['score_geometry']['rho_one_based'], 'Revised rank geometry mismatch')
            require(int(replay['new_word_q16']) == current_word, 'Revised working word mismatch')
            words, rank, always = replay['new_required_q16'], replay['new_rank_supported'], replay['new_always_masked']
            computed = np.ones(len(rows), bool)
            accepted_keep = replay['new_keep']
            knee_word = int(proposal['proposal']['knee_word_q16'])
        else:
            require(record['score_geometry'] == previous['score_geometry'], 'Working score geometry differs from original cache')
            require(int(replay['eta_q16']) == current_word, 'Working word differs from accepted replay')
            words = data['fine_required_multiplier_q16'][rows]
            rank, always = data['fine_rank_valid'][rows], data['fine_always_masked'][rows]
            computed = data['fine_score_computed'][rows]
            accepted_keep = np.isin(rows, replay['kept_source_row_index'])
            require(np.array_equal(rows[accepted_keep], replay['kept_source_row_index']), 'Accepted kept rows are not an exact ordered subset')
            knee_word = int(previous['proposal']['knee_word_q16'])
        require(not np.any(rank & ~always & ~computed), 'Unsupported computation marked as a supported score')
        # the recorded proposals were made with rule v1, so they are authenticated against it
        recomputed = knee.propose(words, [str(int(event)) for event in inverse], valid=list(rank & computed),
                                  always_masked=list(always), rule=knee.RULE_ID_V1)
        require(int(recomputed['knee_word_q16']) == knee_word, 'The knee does not reproduce the recorded knee')
        reference, candidates, arrays, encoding = candidate_cache(words, rank, always, inverse, current_word, knee_word)
        arrays.update(identities)
        arrays['score_computed'] = computed
        arrays['current_keep'] = accepted_keep
        current = next(row for row in candidates if 'current_assignment' in row['roles'])
        require(np.array_equal(arrays['candidate_keep'][current['mask_npz_row']], accepted_keep), 'Current candidate changes accepted mask')
        require(int(accepted_keep.sum()) == record['retention']['kept_frames'], 'Current retained-frame count mismatch')
        require(abs(current['retained_probability']['float'] - record['retention']['equal_event_retention']) <= 5e-14,
                'Current equal-event retention mismatch')
        value = {'schema': 'current-era-exact-score-candidates-v1', 'channel': channel,
            'current_threshold': record['threshold'], 'knee_threshold': {'kind': 'finite', 'eta_q16': str(knee_word)},
            'current_candidate_id': current['id'], 'current_months': record['current_months'],
            'score_geometry': record['score_geometry'], 'excluded_pilot_frequency_ids': record['excluded_pilot_frequency_ids_in_allocation'],
            'nominal_science_frequency_ids': record['nominal_science_frequency_ids'],
            'score_geometry_source': 'revised_exact_comparison' if revised else 'original_fine_export_with_accepted_current_cohort',
            'source_score_export': {'path': base['source_npz'], 'sha256': base['source_npz_sha256']},
            'accepted_exact_replay': record['exact_replay'], 'original_product_identity': record['original_product_identity'],
            'frames': len(rows), 'events': len(counts), 'weight_numerator_encoding': encoding,
            'weight_denominator_decimal': str(arrays['weight_denominator_decimal'].item()),
            'reference': reference, 'candidates': candidates,
            'legacy_unit_key_mismatch_events': int((~arrays['legacy_unit_key_matches_event']).sum()),
            'identity_rule': 'Authoritative source_event_keys, unit_event_id, acquisition_id and accepted replay event keys agree. legacy_original_unit_key is copied only as nonauthoritative metadata.',
            'conditional_weight_rule': plan['conditional_rule'], 'score_feature': plan['score_feature'],
            'transfer_f': plan['transfer_f'], 'transfer_g': plan['transfer_g'],
            'candidate_grid_global_optimality_claim': False, 'current_assignment_unchanged': True,
            'current_exact_mask_unchanged': True, 'physical_science_qualified': False, 'threshold_activated': False,
            'no_supported_score_policy': 'Unsupported probability remains in the original denominator and is not invented as a zero score; no supported reference means g is unavailable.'}
        np.savez_compressed(HERE / f'ch{channel}.npz', **arrays)
        value['npz_sha256'] = sha(HERE / f'ch{channel}.npz')
        write_json(HERE / f'ch{channel}.json', value)
        index.append({'channel': channel, 'json': f'ch{channel}.json', 'npz': f'ch{channel}.npz',
            'json_sha256': sha(HERE / f'ch{channel}.json'), 'npz_sha256': value['npz_sha256'],
            'frames': len(rows), 'events': len(counts), 'candidates': len(candidates),
            'current_candidate_id': current['id'], 'current_retained_probability': current['retained_probability'],
            'supported_probability': reference['supported_probability']})
        print(json.dumps({'channel': channel, 'frames': len(rows), 'candidates': len(candidates),
                          'current_keep': current['retained_probability']['float']}), flush=True)
    for path, expected in auth.files.items():
        require(sha(path) == expected, f'Final cached input changed: {path}')
    write_json(HERE / 'INPUT_VERIFICATION.json', {'schema': 'current-era-score-cache-input-verification-v1',
        'status': 'PASS', 'input_sha256': auth.files, 'source_sha256': sha(__file__),
        'original_full_psd_products_rehashed': False,
        'scope': 'Every consumed exact score/replay/metadata cache authenticated and all cohort/event/current-mask identities reproduced. Original full products remain linked by their accepted identities; no PSD or raw extraction.'})
    write_json(HERE / 'INDEX.json', {'schema': 'current-era-exact-score-candidate-index-v1', 'status': 'PASS',
        'channels': sorted(index, key=lambda row: row['channel']), 'plan_sha256': sha(HERE / 'PLAN.json'),
        'frames': sum(row['frames'] for row in index), 'events': sum(row['events'] for row in index),
        'candidates': sum(row['candidates'] for row in index), 'physical_science_qualified': False,
        'current_assignments_and_masks_unchanged': True, 'all_24_current_eras_present': True})
    write_json(HERE / 'MANIFEST.json', {'schema': 'current-era-score-cache-manifest-v1', 'files': {
        path.name: sha(path) for path in sorted(HERE.iterdir()) if path.is_file()}})
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pilot-proxy characterize score-cache", description=__doc__.split("\n\n")[0])
    parser.add_argument("--working", type=Path, required=True,
                        help="working-threshold delivery directory (DELIVERY_MANIFEST.json, WORKING_THRESHOLDS.json)")
    parser.add_argument("--working-sha256", default=LATEST_MANIFEST_SHA, help="sha256 of its DELIVERY_MANIFEST.json")
    parser.add_argument("--exports", type=Path, required=True, help="record naming each channel's fine score export")
    parser.add_argument("--exports-sha256", default=BURST_PROPOSALS_SHA, help="sha256 of that record")
    parser.add_argument("--previous", type=Path, required=True, help="power-era threshold record (original geometry and knee)")
    parser.add_argument("--era-list", type=Path, default=None, help="era list whose current months the working thresholds must use")
    parser.add_argument("--output", type=Path, required=True, help="new score cache directory")
    args = parser.parse_args(argv)
    index = build(args.working, args.exports, args.previous, args.output, working_delivery_sha256=args.working_sha256,
                  exports_sha256=args.exports_sha256, era_list=args.era_list)
    print(json.dumps({"channels": len(index), "frames": sum(row["frames"] for row in index), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
