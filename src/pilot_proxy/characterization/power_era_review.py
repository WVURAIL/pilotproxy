"""The power-era inputs and the era list: monthly power features, the power-era proposals and the working cohorts.

Two steps, run in order by ``pilot-proxy characterize power-eras``:

1. :func:`prepare_features` turns the era-review monthly reduction of each
   channel's fine export into monthly power features (the frame maximum in the
   pilot neighborhood, event median, monthly median across events), checked
   against the export's own frames, months and events;
2. :func:`review` runs the power-era rule (:mod:`.power_eras`) on the features
   and forms each channel's working cohort: the saved current era, refined by a
   confirmed sustained power change inside it, kept as reviewed on the
   user-reviewed channels, and held with a stated reason otherwise.

The cohorts (``threshold_cohorts.json``) are the era list: each channel's
current months, the cohort the score cache is built on. The list of record is
``projects/chime_atsc/eras/power_eras_2026-10-04.json``.

    pilot-proxy characterize power-eras --review DIR --plates DIR --exports DIR --control-export DIR --output DIR

``--review`` is the era-review delivery (``FINALIZATION.json``,
``DELIVERY_MANIFEST.json``, ``data/ch<N>.npz``, ``USER_REVIEW.json``),
``--plates`` the channel-plate delivery (``data/ch<N>.json``: each export's
digest and saved eras), ``--exports`` and ``--control-export`` the fine score
exports of the screened channels and of the channel 37 control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import power_eras
from .power_eras import PowerEraConfig, input_from_dict, propose_power_eras

REVIEWED = {14, 19, 33, 36}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def month_name(value):
    return f'{int(value) // 12:04d}-{int(value) % 12 + 1:02d}'


def checked(path, digest):
    path = Path(path)
    if sha(path) != digest:
        raise ValueError(f'Input hash differs: {path}')
    return path


def prepare_features(review, plates, exports, control_export, output):
    """Write ``<output>/features/ch<N>.json`` and ``<output>/FEATURE_INPUTS.json`` for channels 14 to 37."""
    REVIEW, PLATES, HERE = Path(review), Path(plates), Path(output)
    HERE.mkdir(parents=True, exist_ok=True)
    final = json.loads((REVIEW / 'FINALIZATION.json').read_text())
    delivery_path = checked(REVIEW / 'DELIVERY_MANIFEST.json', final['delivery_manifest_sha256'])
    files = json.loads(delivery_path.read_text())['files']
    old_manifest = json.loads((PLATES / 'DELIVERY_MANIFEST.json').read_text())['files']
    destination = HERE / 'features'
    destination.mkdir(exist_ok=False)
    receipts = []
    for channel in range(14, 38):
        reduced_path = checked(REVIEW / 'data' / f'ch{channel}.npz', files[f'data/ch{channel}.npz']['sha256'])
        old_meta_path = checked(PLATES / 'data' / f'ch{channel}.json', old_manifest[f'data/ch{channel}.json']['sha256'])
        meta = json.loads(old_meta_path.read_text())['metadata']
        source_folder = Path(control_export) if channel == 37 else Path(exports)
        source_path = checked(source_folder / f'ch{channel}.npz', meta['export_sha256'])
        with np.load(reduced_path, allow_pickle=False) as source:
            reduced = {key: source[key] for key in source.files}
        with np.load(source_path, allow_pickle=False) as source:
            selected = source['selected'] & source['finite_time']
            times = source['frame_time'][selected]
            months = source['frame_month'][selected]
            event_ids = source['acquisition_id'][selected]
            event_keys = source['source_event_keys'][source['frame_unit_index'][selected]]
        assert np.isfinite(times).all()
        assert len(set(zip(event_ids.tolist(), event_keys.tolist()))) == len(np.unique(event_ids)) == len(np.unique(event_keys))
        days = np.floor(times / 86400).astype(np.int64)
        # Verify that UTC timestamp and the stored calendar month agree.
        inferred = np.array([datetime.fromtimestamp(float(t), timezone.utc).year * 12
                             + datetime.fromtimestamp(float(t), timezone.utc).month - 1 for t in times])
        np.testing.assert_array_equal(months, inferred)
        rows = []
        for i, month in enumerate(reduced['months']):
            cohort = months == month
            events = len(np.unique(event_ids[cohort]))
            frames = int(cohort.sum())
            assert events == reduced['event_counts'][i] and frames == reduced['frame_counts'][i]
            level = reduced['peak_db_quantiles'][i, 1]
            median_spectrum = reduced['monthly_event_median_psd_db'][i]
            peak_index = int(np.nanargmax(median_spectrum)) if frames else None
            peak_offset = float(reduced['sky_offset_hz'][peak_index]) if frames else None
            rows.append({
                'month': month_name(month), 'level_db': float(level) if frames else None,
                'event_count': events, 'frame_count': frames,
                'day_count': len(np.unique(days[cohort])),
                'peak_offset_hz': peak_offset, 'tracking_usable': True,
                'metadata': {
                    'q25_db': float(reduced['peak_db_quantiles'][i, 0]) if frames else None,
                    'q75_db': float(reduced['peak_db_quantiles'][i, 2]) if frames else None,
                    'peak_frequency_definition': 'Argmax of monthly per-bin median of event mean spectra; diagnostic only.',
                    'peak_near_window_edge': bool(peak_index is not None and (peak_index < 3 or peak_index >= len(median_spectrum) - 3)),
                    'carrier_identity': 'Unverified; a secondary tone or noise maximum may dominate.',
                    'tracking_usable_meaning': 'Fixed-window maximum is defined; this is not verified tracking of a particular transmitter.',
                },
            })
        output = {
            'schema_version': 'monthly-power-features-v1', 'channel': channel,
            'provenance': {
                'feature': 'frame maximum in 501-bin pilot neighborhood -> event median -> monthly median across events',
                'relative_power': True, 'smoothing_before_detection': False,
                'source_files': {str(path): sha(path) for path in [reduced_path, old_meta_path, source_path]},
                'support_population': 'Same authenticated selected finite-time frames as era-review reduction.',
                'tone_exclusions_applied': [], 'reference_geometry_changed': False,
                'study_scope': 'Exploratory power-distribution eras; not causal transmitter states or physical residual RFI.',
            },
            'months': rows,
        }
        dump(destination / f'ch{channel}.json', output)
        receipts.append({'channel': channel, 'frames': len(times), 'events': len(np.unique(event_ids)),
                         'feature_sha256': sha(destination / f'ch{channel}.json')})
    dump(HERE / 'FEATURE_INPUTS.json', {'status': 'PASS', 'channels': receipts,
         'source': str(REVIEW), 'source_delivery_manifest_sha256': sha(delivery_path),
         'source_hashes_verified': True, 'UTC_days_recomputed_from_exact_frame_times': True,
         'frames': sum(r['frames'] for r in receipts), 'events': sum(r['events'] for r in receipts),
         'script_sha256': sha(__file__)})
    return receipts


def months(first, last):
    def index(text):
        return int(text[:4]) * 12 + int(text[5:]) - 1
    return [f'{i // 12:04d}-{i % 12 + 1:02d}' for i in range(index(first), index(last) + 1)]


def working_cohort(channel, proposal, saved):
    current = next(era for era in saved['era_definitions'] if era['era'] == saved['current_era'])
    first, last = current['first_month'], current['last_month']
    old_span = months(first, last)
    boundaries = proposal['boundaries']
    usable = [b for b in boundaries if b['proposed_cohort_first_month'] is not None]
    usable.sort(key=lambda b: b['proposed_cohort_first_month'])
    row = {'channel': channel, 'current_months': old_span,
           'status': 'unchanged_current_era', 'era_basis': 'preserved_saved_current_era',
           'hold_reason': None, 'user_reviewed': channel in REVIEWED,
           'previous_first_month': first, 'previous_last_month': last,
           'automatic_boundary_count': len(boundaries), 'selected_candidate_id': None,
           'automatic_candidate_first_month': usable[-1]['proposed_cohort_first_month'] if usable else None,
           'pending_change_count': len(proposal['pending_changes']),
           'historical_timing_ambiguity_candidate_ids': [],
           'interpretation': 'Working threshold cohort only; no accepted historical boundary is overwritten.'}
    if channel in REVIEWED:
        row['era_basis'] = 'user_reviewed_reference'
        row['review_note'] = ('Reviewed span retained. Automatic proposal and timing uncertainty are shown separately; '
                              'agreement was not used to tune the common detector.')
        return row

    def fallback(reason):
        row.update(status='previous_cohort_fallback', era_basis='saved_current_era_with_explicit_hold', hold_reason=reason)
        return row

    if not proposal['supported_ranges']:
        return fallback('No month satisfies the declared event/frame/day support requirements.')
    unresolved = [b for b in boundaries if b['proposed_cohort_first_month'] is None]
    if unresolved:
        # An unrelated historical ambiguity cannot invalidate an independently
        # confirmed recent change. Require complete timing strictly before both
        # the saved current span and the chosen confirmation evidence; unknown,
        # overlapping, or later timing still holds the prior cohort.
        chosen_confirmation = usable[-1].get('confirmation_months', []) if usable else []
        historical_limit = min([first, *chosen_confirmation]) if chosen_confirmation else None
        for candidate in unresolved:
            evidence = candidate.get('confirmation_months', [])
            interval = candidate.get('candidate_edge_interval', {})
            timing = [*evidence, interval.get('old_last_month'), interval.get('new_first_month')]
            complete = evidence and all(isinstance(month, str) and len(month) == 7 for month in timing)
            if not (historical_limit and complete and max(timing) < historical_limit):
                return fallback('An automatic candidate has unresolved ordering or timing relevant to the current cohort; retain the prior explicit cohort.')
            row['historical_timing_ambiguity_candidate_ids'].append(candidate['candidate_id'])
    if not usable:
        if proposal['pending_changes']:
            return fallback('Power differences are visible, but none has the required consecutive confirmation on both sides.')
        if len(saved['era_definitions']) > 1:
            return fallback('The common rule did not confirm a new boundary. Nondetection is insufficient to merge previously defined eras.')
        row['era_basis'] = 'saved_single_era_no_confirmed_new_power_change'
        return row
    chosen = usable[-1]
    candidate_first = chosen['proposed_cohort_first_month']
    if chosen['overlapping_confirmation_windows']:
        return fallback('Latest candidate overlaps another confirmation window; a unique new current era is not established.')
    if not first <= candidate_first <= last:
        return fallback('The latest automatic boundary falls outside the saved current era; this refinement does not merge old eras or expand their endpoint.')
    later_pending = [p for p in proposal['pending_changes'] if p['new_first_month'] > candidate_first]
    if later_pending:
        return fallback('A later power change remains pending after the candidate; preserve the prior threshold cohort until the latest state is confirmed.')
    later_bad_gap = [g for g in proposal['gaps'] if g['first_supported_month'] > candidate_first
                     and g['status'] in {'difference_unconfirmed', 'insufficient_confirmation'}]
    if later_bad_gap:
        return fallback('A later gap lacks a confirmed comparison of the observed levels; the latest state is uncertain.')
    confirmation = set(chosen['confirmation_months'])
    edge_months = [m['month'] for m in proposal['months'] if m['month'] in confirmation
                   and m['metadata'].get('peak_near_window_edge')]
    if edge_months:
        return fallback('A dominant spectral feature approaches the search-window edge near the change; a tracking exception needs review.')
    row.update(current_months=months(candidate_first, last),
               status='unchanged_current_era' if candidate_first == first else 'revised_current_era',
               era_basis='automatic_sustained_power_refinement',
               selected_candidate_id=chosen['candidate_id'],
               selected_boundary_kind=chosen['kind'],
               observation_bounds=chosen['observation_bounds'],
               candidate_edge_interval=chosen['candidate_edge_interval'])
    return row


def review(output, plates, review_dir, *, design=None):
    """Propose each channel's power eras from ``<output>/features`` and write the working cohorts (the era list)."""
    HERE, PLATES, REVIEW = Path(output), Path(plates), Path(review_dir)
    features = load(HERE / 'FEATURE_INPUTS.json')
    assert features['status'] == 'PASS'
    config = PowerEraConfig()
    source = Path(power_eras.__file__)
    lock = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'config': asdict(config), 'config_digest': config.digest,
            'module_sha256': sha(source), 'design_sha256': sha(design) if design else None,
            'real_data_scope': 'All real channel displays were previously seen; this is exploratory, not held-out validation.',
            'parameters_changed_after_first_24_channel_run': False,
            'cohort_refinement_policy': 'Refine within an existing current-era span only. Do not silently merge historical eras or extend endpoints. Explicit reviewed overrides and prior-cohort holds remain identified.'}
    dump(HERE / 'CONFIG_LOCK.json', lock)
    destination = HERE / 'proposals'
    destination.mkdir(exist_ok=False)
    results = []
    for source_row in features['channels']:
        channel = source_row['channel']
        input_path = HERE / 'features' / f'ch{channel}.json'
        assert sha(input_path) == source_row['feature_sha256']
        document = load(input_path)
        records, identity, provenance, supplied_config = input_from_dict(document)
        assert supplied_config.digest == config.digest
        result = propose_power_eras(records, channel=identity, provenance=provenance, config=config)
        result['input_file'] = {'path': str(input_path), 'sha256': sha(input_path)}
        dump(destination / f'ch{channel}.json', result)
        old_path = PLATES / 'data' / f'ch{channel}.json'
        assert sha(old_path) == document['provenance']['source_files'][str(old_path)]
        saved = load(old_path)['metadata']
        row = working_cohort(channel, result, saved)
        row['proposal_path'] = str(destination / f'ch{channel}.json')
        row['proposal_sha256'] = sha(destination / f'ch{channel}.json')
        row['saved_era_definitions'] = saved['era_definitions']
        results.append(row)
        print(f"CH{channel}: {len(result['boundaries'])} candidates; {row['status']}; {row['current_months'][0]}", flush=True)
    summary = {'schema': 'working-power-era-review-v1', 'channels': results,
               'config_lock_sha256': sha(HERE / 'CONFIG_LOCK.json'),
               'user_review_source': {'path': str(REVIEW / 'USER_REVIEW.json'), 'sha256': sha(REVIEW / 'USER_REVIEW.json')},
               'accepted_historical_eras_changed': False, 'source_sha256': sha(__file__)}
    dump(HERE / 'working_eras.json', summary)
    cohorts = {'schema': 'robust-power-era-threshold-cohorts-v1',
               'provenance': {'path': 'working_eras.json', 'sha256': sha(HERE / 'working_eras.json')},
               'channels': [{key: row[key] for key in ['channel', 'current_months', 'status', 'era_basis', 'hold_reason']}
                            for row in results]}
    dump(HERE / 'threshold_cohorts.json', cohorts)
    dump(destination / 'manifest.json', {p.name: sha(p) for p in destination.iterdir() if p.is_file()})
    return cohorts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pilot-proxy characterize power-eras", description=__doc__.split("\n\n")[0])
    parser.add_argument("--review", type=Path, required=True, help="era-review delivery (the monthly reduction)")
    parser.add_argument("--plates", type=Path, required=True, help="channel-plate delivery (export digests, saved eras)")
    parser.add_argument("--exports", type=Path, required=True, help="fine score exports of the screened channels")
    parser.add_argument("--control-export", type=Path, required=True, help="fine score export of the channel 37 control")
    parser.add_argument("--design", type=Path, default=None, help="the rule's design record, hashed into the config lock")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args(argv)
    receipts = prepare_features(args.review, args.plates, args.exports, args.control_export, args.output)
    cohorts = review(args.output, args.plates, args.review, design=args.design)
    print(json.dumps({"channels": len(receipts), "frames": sum(r["frames"] for r in receipts),
                      "events": sum(r["events"] for r in receipts), "era_list": str(args.output / "threshold_cohorts.json"),
                      "revised": [row["channel"] for row in cohorts["channels"] if row["status"] == "revised_current_era"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
