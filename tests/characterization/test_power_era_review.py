"""The working-cohort policy that forms the era list; no archive or accepted input is changed."""
from __future__ import annotations

import copy

import pytest

from pilot_proxy.characterization import power_era_review as analysis
from pilot_proxy.characterization.power_eras import MonthlyPower, propose_power_eras


def saved(*, first='2021-01', last='2022-12', prior=True):
    eras = ([{'era': 1, 'first_month': '2018-01', 'last_month': '2020-12'}]
            if prior else [])
    eras.append({'era': 2, 'first_month': first, 'last_month': last})
    return {'current_era': 2, 'era_definitions': eras}


def boundary(first='2022-01', *, identity='recent', confirmation=None,
             overlaps=(), kind='observed_power_change'):
    confirmation = confirmation or analysis.months('2021-08', '2022-05')
    return {'candidate_id': identity, 'proposed_cohort_first_month': first,
            'confirmation_months': confirmation,
            'overlapping_confirmation_windows': list(overlaps), 'kind': kind,
            'observation_bounds': {'first_new_consistent_month': first},
            'candidate_edge_interval': {'old_last_month': '2021-11', 'new_first_month': '2022-02'}}


def proposal(*boundaries, pending=(), gaps=(), edges=(), supported=True):
    return {'boundaries': list(boundaries), 'pending_changes': list(pending),
            'gaps': list(gaps), 'supported_ranges': [{'first_month': '2021-01', 'last_month': '2022-12'}] if supported else [],
            'months': [{'month': month, 'metadata': {'peak_near_window_edge': True}} for month in edges]}


def assert_prior(row, expected, *, fallback):
    current = next(e for e in expected['era_definitions'] if e['era'] == expected['current_era'])
    assert row['current_months'] == analysis.months(current['first_month'], current['last_month'])
    assert row['status'] == ('previous_cohort_fallback' if fallback else 'unchanged_current_era')
    assert bool(row['hold_reason']) is fallback
    assert row['selected_candidate_id'] is None


@pytest.mark.parametrize('channel', [14, 19, 33, 36])
def test_user_reviewed_spans_override_automatic_candidates_and_holds(channel):
    original = saved()
    result = analysis.working_cohort(channel, proposal(boundary(), supported=False), original)
    assert_prior(result, original, fallback=False)
    assert result['era_basis'] == 'user_reviewed_reference'
    assert result['user_reviewed']
    assert result['automatic_candidate_first_month'] == '2022-01'


def test_confirmed_refinement_is_subset_and_never_extends_saved_end():
    original = saved(last='2022-06')
    observed = proposal(boundary())
    before = copy.deepcopy((original, observed))
    result = analysis.working_cohort(20, observed, original)
    assert result['current_months'] == analysis.months('2022-01', '2022-06')
    assert result['status'] == 'revised_current_era'
    assert result['selected_candidate_id'] == 'recent'
    assert result['hold_reason'] is None
    assert (original, observed) == before


def test_matching_onset_has_unchanged_calendar_span():
    original = saved(first='2022-01')
    result = analysis.working_cohort(20, proposal(boundary()), original)
    assert result['current_months'] == analysis.months('2022-01', '2022-12')
    assert result['status'] == 'unchanged_current_era'
    assert result['selected_candidate_id'] == 'recent'


def test_no_detection_cannot_merge_existing_eras():
    original = saved()
    result = analysis.working_cohort(20, proposal(), original)
    assert_prior(result, original, fallback=True)
    assert 'merge' in result['hold_reason']


def test_single_era_no_detection_is_explicitly_preserved():
    original = saved(prior=False)
    result = analysis.working_cohort(20, proposal(), original)
    assert_prior(result, original, fallback=False)
    assert result['era_basis'] == 'saved_single_era_no_confirmed_new_power_change'


@pytest.mark.parametrize('candidate', ['2020-12', '2023-01'])
def test_outside_candidate_never_merges_or_extends(candidate):
    original = saved()
    result = analysis.working_cohort(20, proposal(boundary(candidate)), original)
    assert_prior(result, original, fallback=True)
    assert 'outside' in result['hold_reason']


def test_latest_confirmed_candidate_chosen_independent_of_input_order():
    earlier = boundary('2021-06', identity='old', confirmation=analysis.months('2021-01','2021-10'))
    recent = boundary()
    a = analysis.working_cohort(20, proposal(recent, earlier), saved())
    b = analysis.working_cohort(20, proposal(earlier, recent), saved())
    assert a == b
    assert a['selected_candidate_id'] == 'recent'


@pytest.mark.parametrize('observed, phrase', [
    (proposal(boundary(), supported=False), 'No month'),
    (proposal(boundary(None)), 'unresolved'),
    (proposal(boundary(overlaps=['other'])), 'overlaps'),
    (proposal(boundary(), pending=[{'new_first_month':'2022-08'}]), 'later power change'),
    (proposal(boundary(), gaps=[{'first_supported_month':'2022-08', 'status':'difference_unconfirmed'}]), 'later gap'),
    (proposal(boundary(), gaps=[{'first_supported_month':'2022-08', 'status':'insufficient_confirmation'}]), 'later gap'),
    (proposal(boundary(), edges=['2022-02']), 'window edge'),
    (proposal(pending=[{'new_first_month':'2022-08'}]), 'none has'),
])
def test_relevant_ambiguity_has_explicit_prior_cohort_fallback(observed, phrase):
    original = saved()
    result = analysis.working_cohort(20, observed, original)
    assert_prior(result, original, fallback=True)
    assert phrase in result['hold_reason']


def test_compatible_later_gap_does_not_claim_continuity_or_suppress_refinement():
    result = analysis.working_cohort(20, proposal(boundary(), gaps=[
        {'first_supported_month':'2022-08', 'status':'compatible_observed_levels'}]), saved())
    assert result['status'] == 'revised_current_era'
    assert 'continuity_established' not in result


def test_old_pending_and_tracking_diagnostics_do_not_suppress_recent_refinement():
    result = analysis.working_cohort(20, proposal(boundary(),
        pending=[{'new_first_month':'2019-01'}], edges=['2019-01']), saved())
    assert result['status'] == 'revised_current_era'


def test_resolved_later_boundary_is_not_blocked_by_older_disjoint_timing_ambiguity():
    old = boundary(None, identity='ambiguous-history', confirmation=analysis.months('2019-01','2019-10'))
    old['candidate_edge_interval'] = {'old_last_month':'2019-04', 'new_first_month':'2019-07'}
    result = analysis.working_cohort(20, proposal(old, boundary()), saved())
    assert result['status'] == 'revised_current_era'
    assert result['selected_candidate_id'] == 'recent'
    assert result['historical_timing_ambiguity_candidate_ids'] == ['ambiguous-history']


@pytest.mark.parametrize('evidence, interval', [
    ([], {'old_last_month':'2019-04', 'new_first_month':'2019-07'}),
    (analysis.months('2019-01','2019-10'), {}),
    (analysis.months('2020-08','2021-05'), {'old_last_month':'2020-12', 'new_first_month':'2021-01'}),
    (analysis.months('2022-01','2022-10'), {'old_last_month':'2022-04', 'new_first_month':'2022-07'}),
])
def test_unknown_overlapping_or_later_ambiguity_still_holds(evidence, interval):
    ambiguous = boundary(None, identity='ambiguous')
    ambiguous['confirmation_months'] = evidence
    ambiguous['candidate_edge_interval'] = interval
    original = saved()
    result = analysis.working_cohort(20, proposal(ambiguous, boundary()), original)
    assert_prior(result, original, fallback=True)


def test_ambiguity_before_saved_start_but_overlapping_confirmation_still_holds():
    old = boundary(None, identity='ambiguous-history', confirmation=analysis.months('2020-03','2020-12'))
    old['candidate_edge_interval'] = {'old_last_month':'2020-07', 'new_first_month':'2020-08'}
    recent = boundary('2021-02', confirmation=analysis.months('2020-09','2021-06'))
    original = saved()
    result = analysis.working_cohort(20, proposal(old, recent), original)
    assert_prior(result, original, fallback=True)


@pytest.mark.parametrize('gap_status', ['difference_unconfirmed', 'insufficient_confirmation'])
def test_unconfirmed_post_onset_gap_inside_confirmation_union_still_holds(gap_status):
    original = saved()
    result = analysis.working_cohort(20, proposal(boundary(), gaps=[
        {'first_supported_month':'2022-04', 'status':gap_status}]), original)
    assert_prior(result, original, fallback=True)
    assert 'later gap' in result['hold_reason']


def synthetic(values, *, first='2021-01'):
    labels = analysis.months(first, '2030-12')[:len(values)]
    rows = [MonthlyPower(month=month, level_db=float(value), event_count=8,
                         frame_count=40, day_count=4) for month, value in zip(labels, values)]
    return propose_power_eras(rows, channel=20, provenance={'population':'independent synthetic monthly levels'})


def test_real_detector_step_drives_only_current_span_refinement():
    detected = synthetic([0]*12+[6]*12)
    assert len(detected['boundaries']) == 1
    result = analysis.working_cohort(20, detected, saved())
    assert result['status'] == 'revised_current_era'
    assert result['current_months'] == analysis.months('2022-01','2022-12')


def test_pending_latest_state_inside_confirmation_union_is_not_hidden():
    # The first new level is confirmed. A further level at the archive edge
    # still needs confirmation, even when a tied passing edge includes it as
    # the one allowed outlier in the preceding candidate's confirmation union.
    detected = synthetic([0]*12+[6]*5+[12])
    assert len(detected['boundaries']) == 1
    assert any(p['new_first_month'] == '2022-06' for p in detected['pending_changes'])
    assert max(detected['boundaries'][0]['confirmation_months']) == '2022-06'
    original = saved(last='2022-06')
    result = analysis.working_cohort(20, detected, original)
    assert_prior(result, original, fallback=True)
