#!/usr/bin/env python3
"""Historical stack-to-proxy amplitude-ratio derivation (October 2026).

This reproduces the original capture comparison and its old duty scenario.
The ratio is not an intrinsic coherent fraction; use tools/threshold_rule.py
for current conditional threshold assignments. The archived record is
``src/pilot_proxy/characterization/data/coherent_fraction.json``.

Definition that reproduces the five hard-coded values of
simple-threshold-rule-2026-10-11/channel6mhz/coherent_rule.json to machine precision:

    c = (A_bao[ch] - A_bao[37]) / x_all[ch]

A_bao   capture class amplitude on the three east-west (inter-cylinder, 22/44/66 m) classes,
        amendment 7 of the frame analysis (the table of record's column A_bao_keep_all):
        per dump, for each class the larger of the XX and YY readings of excess_inband_median
        (Eq. tolerance:classamp on the raw frame, normalized to single-sample noise, in-band
        per-bin median less the sky reference), then the median over the three classes;
        then the median over the four dumps (pilot 2026-09-16, D1-D3 2026-09-17; keep_all).
        RAW scale (no gain solution applied).
A_bao[37]  the same quantity on the channel 37 control = capture_floor 7.934e-05.
x_all   archive per-input DTV shelf power over noise, all frames of the current era, no mask
        (last element of threshold-rule-latest-2026-10-12/sweeps/chNN.npz).

The original values were computed from the 4-significant-figure table-of-record CSV; this
script recomputes A_bao from the frame-residual products as well and reports both.

Also: calibrated-scale c, the sensitivity of the per-channel rule to c, and the capture
cross-check a_cap = A_net^2 * 16384 * G against a_tol = 1/|R|, with G the admissible minimum of the
33-frame dumps that measured c, the same for every band
(pilot_proxy.characterization.threshold_rule.temporal_gain), as the rule prices it.
Read-only on every input; writes c.json (the full report) in --out and, with --data-file, the
coherent-fraction record the threshold rule reads.

    python tools/archive/derive_stack_amplitude_ratio.py --record <dissertation figure_src/data/record> \\
        --rule <threshold rule output> --scores <score cache> --channel-table <RFIsher channel_table.json> \\
        --hardcoded <simple-threshold-rule coherent_rule.json> --out DIR [--data-file FILE]
"""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pilot_proxy.characterization.threshold_rule import temporal_gain

# the inputs, set from the command line (see main)
REC = TOR = PRODUCTS = KAPPA = ANET_CAL = RULE = SCORES = TABLE = OLD = OUT = None

DUMPS = {'pilot': '20260916162300', 'D1': '20260917040230', 'D2': '20260917090230', 'D3': '20260917140230'}
EW = ((1, 0), (2, 0), (3, 0))
N = 16384
CAL_PER_ONSKY = 6.574        # calendar years per retained on-sky year (duty 0.152)
LIMIT = 30.0                 # calendar-year limit
CONTROL = 37
NOT_DEPLOYED_ALWAYS = {33: 'pilot off the detector frequency'}


def a_bao_products():
    """A_bao per channel per dump, recomputed from frame_residual_<event>.csv (amendment 7)."""
    per_dump = {}
    for label, ev in DUMPS.items():
        cls = {}
        with gzip.open(PRODUCTS / f'frame_residual_{ev}.csv.gz', 'rt') as fh:
            for r in csv.DictReader(fh):
                if r['frames'] != 'all' or r['pol'] not in ('0', '1') or not r['excess_inband_median']:
                    continue
                k = (int(r['ew']), int(r['ns']))
                v = float(r['excess_inband_median'])
                if k in EW and math.isfinite(v):
                    cls.setdefault(int(r['channel']), {}).setdefault(k, []).append(v)
        per_dump[label] = {ch: float(np.median([max(v) for v in d.values()])) for ch, d in cls.items()}
    chans = sorted({c for d in per_dump.values() for c in d})
    return {ch: dict(per_dump={l: per_dump[l].get(ch) for l in DUMPS},
                     A=float(np.median([per_dump[l][ch] for l in DUMPS if ch in per_dump[l]])))
            for ch in chans}


def table_of_record():
    rows = {}
    with open(TOR) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(int(r['channel']), r)
    return rows


def kappa_ew():
    """A_cal/A_raw on identical live pairs, median over ew1-ew3 of each channel's in-band median."""
    k = {}
    with gzip.open(KAPPA, 'rt') as fh:
        for r in csv.DictReader(fh):
            if r['cls'] in ('ew1', 'ew2', 'ew3'):
                k.setdefault(int(r['channel']), {})[r['cls']] = float(r['kappa_median'])
    return {ch: dict(per_class=d, median=float(np.median(list(d.values())))) for ch, d in k.items()}


def ch35_direct_cal():
    """Channel 35 calibrated and raw A_net (keep_all, own calibrated floor), median over the EW classes."""
    raw, cal = [], []
    with gzip.open(ANET_CAL, 'rt') as fh:
        for r in csv.DictReader(fh):
            if r['channel'] == '35' and r['policy'] == 'keep_all' and r['cls'] in ('ew1', 'ew2', 'ew3'):
                raw.append(float(r['A_raw_record_A_net'])); cal.append(float(r['A_cal_stack_A_net']))
    return dict(A_net_raw=float(np.median(raw)), A_net_cal=float(np.median(cal)),
                ratio=float(np.median(cal) / np.median(raw)))


def era(ch):
    j = json.loads((SCORES / f'ch{ch}.json').read_text())
    with np.load(SCORES / f'ch{ch}.npz') as z:
        t = z['frame_time'][z['current_keep'].astype(bool)]
    d = lambda s: dt.datetime.fromtimestamp(float(s), dt.timezone.utc).date().isoformat()
    m = j['current_months']
    return dict(months=f'{m[0]}..{m[-1]}', n_months=len(m), first_frame=d(t.min()), last_frame=d(t.max()),
                current_frames=int(t.size))


def sweep(ch):
    with np.load(RULE / f'sweeps/ch{ch}.npz') as z:
        return {k: z[k] for k in ('word', 'kappa', 'x', 'se', 'kept_frames')}


def rule(sw, knee_word, R, c, gain, years_per=CAL_PER_ONSKY):
    """Per-channel rule at coherent fraction c and gain G. Returns (outcome, where, kappa, years, r_hi)."""
    r_hi = (c * (sw['x'] + 2 * sw['se'])) ** 2 * N * gain * abs(R)
    i = int(np.flatnonzero(sw['word'] == knee_word)[0])
    yk = years_per / sw['kappa'][i]
    if r_hi[i] <= 1 and yk <= LIMIT:
        return dict(outcome='deployed', at='knee', kappa=float(sw['kappa'][i]), years=float(yk), r_hi=float(r_hi[i]),
                    knee_r_hi=float(r_hi[i]))
    ok = np.flatnonzero((sw['word'] < knee_word) & (r_hi <= 1))
    if ok.size:
        j = int(ok[np.argmax(sw['word'][ok])])
        yj = years_per / sw['kappa'][j]
        if yj <= LIMIT:
            return dict(outcome='deployed', at='below knee', kappa=float(sw['kappa'][j]), years=float(yj),
                        r_hi=float(r_hi[j]), knee_r_hi=float(r_hi[i]))
        return dict(outcome='not deployed', at='out of time', kappa=float(sw['kappa'][j]), years=float(yj),
                    r_hi=float(r_hi[j]), knee_r_hi=float(r_hi[i]))
    return dict(outcome='not deployed', at='above tolerance', kappa=None, years=None, r_hi=None, knee_r_hi=float(r_hi[i]))


def derive():
    old = json.loads(Path(OLD).read_text())
    c_old = {int(k): v for k, v in old['coherent_fraction'].items()}
    tor = table_of_record()
    prod = a_bao_products()
    kap = kappa_ew()
    d35 = ch35_direct_cal()
    dec = {c['channel']: c for c in json.loads((RULE / 'decisions.json').read_text())['channels']}
    R = {c['channel']: c['response_dv'] for c in json.loads(Path(TABLE).read_text())['channels']}
    floor_rec = float(tor[CONTROL]['A_bao_keep_all'])
    floor_prod = prod[CONTROL]['A']
    assert abs(floor_rec - old['capture_floor']) < 1e-12

    chans = {}
    for ch in range(14, 38):
        sw = sweep(ch)
        x, se = float(sw['x'][-1]), float(sw['se'][-1])
        t = tor.get(ch, {})
        A_rec = float(t['A_bao_keep_all']) if t.get('A_bao_keep_all') else None
        A_prod = prod[ch]['A'] if ch in prod else None
        e = dict(channel=ch, archive=dict(x_all=x, se_all=se, kept_frames_all=int(sw['kept_frames'][-1]), era=era(ch)))
        if A_rec is None:
            e.update(capture=None, c_raw=None, in_set=False,
                     reason='no 2026 capture class amplitude (channel 24 is read on its 2020 in-band basis)')
            chans[ch] = e
            continue
        A_net_rec, A_net_prod = A_rec - floor_rec, A_prod - floor_prod
        e['capture'] = dict(dumps={l: f'{ev[:4]}-{ev[4:6]}-{ev[6:8]}T{ev[8:10]}:{ev[10:12]}:{ev[12:]}Z' for l, ev in DUMPS.items()},
                            A_bao_record=A_rec, A_bao_products=A_prod, A_bao_per_dump=prod[ch]['per_dump'],
                            A_net_raw_record=A_net_rec, A_net_raw_products=A_net_prod,
                            long_bao_measured=t.get('bao_long_measured') == 'True',
                            long_bao_gate=float(t['bao_long_floor_level']) if t.get('bao_long_floor_level') else None,
                            short_bao_A_net=float(t['bao_short_A_net']) if t.get('bao_short_A_net') else None,
                            short_bao_measured=t.get('bao_short_measured') == 'True',
                            short_bao_gate=float(t['bao_short_floor_level']) if t.get('bao_short_floor_level') else None,
                            record_disposition=t['disposition'])
        e['c_raw'] = A_net_rec / x if ch != CONTROL else None
        e['c_raw_products'] = A_net_prod / x if ch != CONTROL else None
        if ch in c_old:
            e['c_hardcoded'] = c_old[ch]
            e['c_raw_minus_hardcoded'] = e['c_raw'] - c_old[ch]
        if ch in kap and ch != CONTROL:
            e['kappa_cal_over_raw_ew'] = kap[ch]
            e['A_net_cal_est'] = kap[ch]['median'] * A_net_rec
            e['c_cal'] = kap[ch]['median'] * e['c_raw']
        if ch == 35:
            e['ch35_direct_calibrated'] = d35
            e['c_cal_direct'] = e['c_raw'] * d35['ratio']
        # membership
        if ch == CONTROL:
            e.update(in_set=False, reason='control channel (defines the floor; no DTV, c = 0 by construction)')
        elif ch == 33:
            e.update(in_set=False, reason=f"long BAO range measured, but the archive under-reads it: x = {x:.3g} gives c = {e['c_raw']:.2f} > 1 (pilot off the detector frequency)")
        elif not e['capture']['long_bao_measured']:
            e.update(in_set=False, reason='long BAO range not measured above the control gate: A_net is at the floor, c is not a measurement (noise over a small x)')
        else:
            e.update(in_set=True, reason='long BAO range measured above the control gate and archive x physical (c < 1)')
        chans[ch] = e

    inset = [ch for ch, e in chans.items() if e['in_set']]
    cr = np.array([chans[ch]['c_raw'] for ch in inset])
    cc = np.array([chans[ch]['c_cal'] for ch in inset if 'c_cal' in chans[ch]])
    allc = {ch: e['c_raw'] for ch, e in chans.items() if e.get('c_raw') is not None and ch != CONTROL}
    summary = dict(
        definition='c = (A_bao[ch] - A_bao[37]) / x_all[ch]; A_bao = table-of-record A_bao_keep_all (raw frame, east-west 22/44/66 m classes, larger polarization, in-band median, median over 4 dumps); x_all = archive all-frame shelf power (current era, no mask)',
        reproduced=[ch for ch in c_old if abs(chans[ch]['c_raw'] - c_old[ch]) < 1e-12],
        max_abs_diff_vs_hardcoded=max(abs(chans[ch]['c_raw'] - c_old[ch]) for ch in c_old),
        max_rel_diff_products_vs_record=max(abs(chans[ch]['c_raw_products'] / chans[ch]['c_raw'] - 1) for ch in inset),
        channel_set=inset,
        c_raw=dict(median=float(np.median(cr)), min=float(cr.min()), max=float(cr.max()),
                   max_over_min=float(cr.max() / cr.min()), log10_std=float(np.std(np.log10(cr))),
                   q25=float(np.quantile(cr, .25)), q75=float(np.quantile(cr, .75))),
        c_cal=dict(channels=[ch for ch in inset if 'c_cal' in chans[ch]], median=float(np.median(cc)),
                   min=float(cc.min()), max=float(cc.max()),
                   note='kappa = A_cal/A_raw on the EW classes; channel 22 has no calibrated product'),
        kappa_ew_median_over_set=float(np.median([kap[ch]['median'] for ch in inset if ch in kap])),
        c_raw_all_channels=allc,
        excluded={ch: chans[ch]['reason'] for ch in chans if not chans[ch]['in_set']},
    )

    # sensitivity
    c0 = old['c']
    cgrid = {'0.011': 0.011, '0.0156': 0.0156, '0.0192': 0.0192, '0.026': 0.026, '0.032': 0.032,
             '2x (0.0384)': 2 * c0, '5x (0.096)': 5 * c0,
             f"calibrated median ({summary['c_cal']['median']:.3f})": summary['c_cal']['median']}
    sens, conv_check = {}, {}
    for ch in range(14, 38):
        sw = sweep(ch)
        d = dec[ch]
        row = {}
        for lab, c in cgrid.items():
            if ch in NOT_DEPLOYED_ALWAYS:
                r = dict(outcome='not deployed', at=NOT_DEPLOYED_ALWAYS[ch], knee_r_hi=rule(sw, d['knee_word'], d['response'], c, temporal_gain(ch))['knee_r_hi'])
            else:
                r = rule(sw, d['knee_word'], d['response'], c, temporal_gain(ch))
                r_onsky = rule(sw, d['knee_word'], d['response'], c, temporal_gain(ch), years_per=1.0)
                if r_onsky['outcome'] != r['outcome']:
                    conv_check.setdefault(ch, []).append(lab)
            row[lab] = r
        # baseline reproduction against decisions.json (which used c0)
        b = rule(sw, d['knee_word'], d['response'], c0, temporal_gain(ch), years_per=1.0)
        row['_baseline_matches_decisions'] = (('deployed' if b['outcome'] == 'deployed' else 'not useful') ==
                                              ('deployed' if d['outcome'] == 'useful' else 'not useful')) or ch in NOT_DEPLOYED_ALWAYS
        kr0 = rule(sw, d['knee_word'], d['response'], c0, temporal_gain(ch))['knee_r_hi']
        row['_c_knee_limit'] = c0 / math.sqrt(kr0) if kr0 > 0 else math.inf   # c at which the knee's r_hi reaches 1
        row['_knee_r_hi_check'] = abs(rule(sw, d['knee_word'], d['response'], c0, temporal_gain(ch))['knee_r_hi'] / d['knee_r_hi'] - 1) if d['knee_r_hi'] else None
        sens[ch] = row
    changes = {}
    for ch, row in sens.items():
        outs = [(row[l]['outcome'], row[l]['at']) for l in cgrid]
        if len(set(outs)) > 1:
            changes[ch] = {l: f"{row[l]['outcome']} ({row[l]['at']})" for l in cgrid}

    # capture cross-check in the rule's own units (each band's G)
    xc = {}
    floor_a1 = floor_rec ** 2 * N
    for ch, e in chans.items():
        if e.get('capture') is None or ch == CONTROL:
            continue
        a_tol = 1 / abs(R[ch])
        cap = e['capture']
        G = temporal_gain(ch)
        floor_a = floor_a1 * G
        a_ew = max(cap['A_net_raw_record'], 0) ** 2 * N * G
        a_short = cap['short_bao_A_net'] ** 2 * N * G if cap['short_bao_A_net'] else None
        gate_ew = cap['long_bao_gate'] ** 2 * N * G if cap['long_bao_gate'] else None
        d = dec[ch]
        a_arch_knee = d['knee_r_hi'] / abs(R[ch]) if d['knee_r_hi'] is not None else None
        xc[ch] = dict(gain=G, a_tol=a_tol, a_cap_ew=a_ew, ratio_ew=a_ew / a_tol,
                      a_cap_short=a_short, ratio_short=a_short / a_tol if a_short else None,
                      short_measured=cap['short_bao_measured'], long_measured=cap['long_bao_measured'],
                      a_floor=floor_a, floor_over_tol=floor_a / a_tol,
                      a_gate_long=gate_ew, gate_over_tol=gate_ew / a_tol if gate_ew else None,
                      archive_knee_r_hi=d['knee_r_hi'], archive_a_knee_hi=a_arch_knee,
                      archive_outcome='deployed' if d['outcome'] == 'useful' else f"not deployed ({d['reason']})",
                      capture_verdict_ew=('over a_tol' if a_ew > a_tol else 'under a_tol') + ('' if cap['long_bao_measured'] else ' (at floor: not measured)'),
                      era=e['archive']['era']['months'])

    out = dict(generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'), script=str(Path(__file__).resolve()),
               inputs=dict(table_of_record=str(TOR), frame_residual=[str(PRODUCTS / f'frame_residual_{ev}.csv.gz') for ev in DUMPS.values()],
                           kappa_summary=str(KAPPA), anet_cal_class=str(ANET_CAL), sweeps=str(RULE / 'sweeps'),
                           decisions=str(RULE / 'decisions.json'), scores=str(SCORES), channel_table=str(TABLE), hardcoded=str(OLD)),
               summary=summary, channels=chans,
               sensitivity=dict(c_grid=cgrid, rule='knee if r_hi<=1 and 6.574/kappa<=30 y; else loosest word below knee with r_hi<=1 if within 30 y; else not deployed; 33 never',
                                outcomes=sens, changes=changes,
                                calendar_convention_differences=conv_check),
               crosscheck=dict(units='a = A_net^2 * 16384 * G, G the band\'s admissible minimum (A_net normalized to single-sample noise, per baseline)',
                               floor=floor_rec, a_floor_per_unit_gain=floor_a1, channels=xc))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'c.json').write_text(json.dumps(out, indent=1, default=float))

    # console summary
    print('reproduced:', summary['reproduced'], 'max |diff|', f"{summary['max_abs_diff_vs_hardcoded']:.1e}",
          'products vs record', f"{summary['max_rel_diff_products_vs_record']:.2%}")
    print(f"{'ch':>3} {'A_net':>9} {'x_all':>9} {'c_raw':>8} {'kapEW':>6} {'c_cal':>7}  in  era")
    for ch, e in chans.items():
        if e.get('capture') is None:
            print(f"{ch:>3} {'-':>9} {e['archive']['x_all']:9.3g} {'-':>8}  -- {e['reason'][:40]}"); continue
        k = e.get('kappa_cal_over_raw_ew', {}).get('median')
        print(f"{ch:>3} {e['capture']['A_net_raw_record']:9.3e} {e['archive']['x_all']:9.3g} {e['c_raw'] or 0:8.4f} "
              f"{k or float('nan'):6.2f} {e.get('c_cal', float('nan')):7.4f}  {'Y' if e['in_set'] else '.'}  {e['archive']['era']['months']}")
    s = summary
    print('c_raw', {k: round(v, 5) for k, v in s['c_raw'].items()}, 'c_cal', s['c_cal']['median'], s['c_cal']['min'], s['c_cal']['max'])
    print('baseline matches decisions:', all(r['_baseline_matches_decisions'] for r in sens.values()),
          'max knee r_hi rel err', max(r['_knee_r_hi_check'] or 0 for r in sens.values()))
    print('outcome or placement changes:'); [print(' ', ch, v) for ch, v in changes.items()]
    print('c at which the knee r_hi reaches 1:', {ch: round(r['_c_knee_limit'], 4) for ch, r in sens.items()})
    for ch in changes:
        for l in cgrid:
            r = sens[ch][l]
            print(f"   {ch} {l:>24}: {r['outcome']:>12} {r['at']:>16} kappa={r.get('kappa')} years={r.get('years')} r_hi={r.get('r_hi')}")
    for ch in (14, 15, 22, 26, 27, 36, 25):
        print('per-dump A_bao', ch, {k: (round(v, 7) if v else v) for k, v in chans[ch]['capture']['A_bao_per_dump'].items()},
              'lowest-epoch A_bao', tor[ch]['A_bao_lowest_epoch'], tor[ch]['lowest_epoch'])
    print('calendar-convention differences:', conv_check)
    print(f"{'ch':>3} {'a_tol':>9} {'a_ew/tol':>9} {'a_sh/tol':>9} {'floor/tol':>9} {'gate/tol':>9} {'arch_knee_rhi':>13}  archive / capture")
    for ch, r in xc.items():
        print(f"{ch:>3} {r['a_tol']:9.3e} {r['ratio_ew']:9.3g} {r['ratio_short'] if r['ratio_short'] is not None else float('nan'):9.3g} "
              f"{r['floor_over_tol']:9.3g} {r['gate_over_tol'] or float('nan'):9.3g} {r['archive_knee_r_hi'] or float('nan'):13.3g}  {r['archive_outcome']} / {r['capture_verdict_ew']}")
    return out


DEFINITION = "c = (A[band] - A[control]) / x_all[band] on the raw frame, times the calibrated-to-raw amplitude ratio of the same east-west baseline classes. A is the capture's in-band visibility amplitude over its auto power (median over 3 classes and 4 dumps of 2026-09-16/17); x_all is the archive's current-era shelf power over all frames, with no mask."
WHY_CALIBRATED = "Science data are gain-calibrated before redundant baselines are stacked. The raw stack adds per-input phases at chance efficiency (0.035-0.065); calibrated, the fixed-direction signal stacks at 0.14-0.19. The noise normalization is the same on both scales. Band 35's direct calibrated reading gives 0.078."


def data_record(out):
    """The coherent-fraction record the threshold rule reads: the calibrated median over the measured bands."""
    s, chans = out['summary'], out['channels']
    return {'schema': 'coherent-fraction-v1', 'value': s['c_cal']['median'], 'scale': 'calibrated',
            'statistic': 'median over the bands below',
            'by_band': {str(ch): chans[ch]['c_cal'] for ch in s['c_cal']['channels']},
            'by_band_raw': {str(ch): chans[ch]['c_raw'] for ch in s['channel_set']},
            'definition': DEFINITION, 'why_calibrated': WHY_CALIBRATED,
            'provenance': {'script': Path(__file__).name,
                           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'generated': out['generated']}}


def main(argv=None):
    global REC, TOR, PRODUCTS, KAPPA, ANET_CAL, RULE, SCORES, TABLE, OLD, OUT
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--record', type=Path, required=True, help='the dissertation figure_src/data/record directory')
    parser.add_argument('--rule', type=Path, required=True, help='threshold rule output (sweeps/, decisions.json)')
    parser.add_argument('--scores', type=Path, required=True, help='the score cache')
    parser.add_argument('--channel-table', type=Path, required=True, help="RFIsher's channel_table.json")
    parser.add_argument('--hardcoded', type=Path, required=True, help='the earlier rule\'s coherent_rule.json (values reproduced)')
    parser.add_argument('--out', type=Path, required=True, help='directory for c.json')
    parser.add_argument('--data-file', type=Path, default=None, help='write the coherent-fraction record here')
    args = parser.parse_args(argv)
    REC = args.record
    TOR = REC / 'table_of_record.csv'
    PRODUCTS = REC / 'products'
    KAPPA = REC / 'provenance/inband/excised/tables/kappa_summary.csv.gz'
    ANET_CAL = REC / 'provenance/inband/tables/anet_cal_class.csv.gz'
    RULE, SCORES, TABLE, OLD, OUT = args.rule, args.scores, args.channel_table, args.hardcoded, args.out
    out = derive()
    if args.data_file is not None:
        args.data_file.write_text(json.dumps(json.loads(json.dumps(data_record(out), default=float)), indent=1) + '\n')


if __name__ == '__main__':
    main()
