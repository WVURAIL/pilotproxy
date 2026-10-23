#!/usr/bin/env python3
"""Freeze and replay a descriptive same-frame capture transfer study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import uuid

import numpy as np

from pilot_proxy.characterization.capture_transfer import (
    coherent_cross_moment, event_weights, exact_frame_join, fit_training_transfer,
    lag_moments, signed_proxy, transfer_residuals,
)
from pilot_proxy.characterization.threshold_rule import SHELF

EPOCHS = ['20260916162300', '20260917040230', '20260917090230', '20260917140230']
FINITE_CHANNELS = {14, 16, 18, 21, 25, 26, 29, 32}
CHANNELS = {14: 844, 16: 813, 17: 798, 18: 783, 21: 736, 25: 675, 26: 660,
            29: 614, 32: 568, 35: 521, 37: 491}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def digest_arrays(arrays):
    h = hashlib.sha256()
    for name, a in sorted(arrays.items()):
        a = np.ascontiguousarray(a)
        h.update(json.dumps([name, a.dtype.str, a.shape]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def require_frozen_input_map(protocol, raw_path, sorted_ids):
    """Refuse changed input serial/column coordinates even when IDs still sort."""
    expected = protocol['raw_coordinates'][str(raw_path)]['input_map_sha256']
    if digest_arrays({'input_ids': sorted_ids}) != expected:
        raise ValueError('raw input map differs from frozen protocol')


def require_saved_control_identity(control, exact_receipt):
    """Bind the reread control autos to the arrays verified against native data."""
    actual = digest_arrays({'origins': control['frame_fpga0'], 'autos': control['autos']})
    if actual != exact_receipt['saved_used_arrays_sha256']:
        raise ValueError('saved control arrays differ from exact reduction receipt')


def read_checkpoint(path, protocol_hash, raw_hash, origin, source_hash):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if 'payload_sha256' not in arrays:
        raise ValueError('frame checkpoint lacks payload digest')
    recorded_hash = str(arrays.pop('payload_sha256'))
    if digest_arrays(arrays) != recorded_hash:
        raise ValueError('frame checkpoint payload mismatch')
    if (str(arrays['protocol_sha256']) != protocol_hash or str(arrays['raw_sha256']) != raw_hash
            or int(arrays['origin']) != int(origin) or str(arrays['execution_source_sha256']) != source_hash):
        raise ValueError('frame checkpoint identity mismatch')
    return arrays


def write_summary_checkpoint(path, record, input_ids, provenance, protocol_hash, source_hash, channel, epoch):
    """Preserve verified observations so interrupted summaries need not reread raw data."""
    arrays = {**record, 'input_ids': input_ids,
              'metadata': np.array(json.dumps(dict(protocol_sha256=protocol_hash,
                  source_sha256=source_hash, channel=channel, epoch=epoch,
                  provenance=provenance), sort_keys=True))}
    arrays['payload_sha256'] = np.array(digest_arrays(arrays))
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_summary_checkpoint(path, protocol_hash, source_hash, channel, epoch):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    recorded_hash = str(arrays.pop('payload_sha256'))
    if digest_arrays(arrays) != recorded_hash:
        raise ValueError('summary checkpoint payload mismatch')
    metadata = json.loads(str(arrays.pop('metadata')))
    if (metadata['protocol_sha256'] != protocol_hash or metadata['source_sha256'] not in ([source_hash] if isinstance(source_hash, str) else source_hash)
            or metadata['channel'] != channel or metadata['epoch'] != epoch):
        raise ValueError('summary checkpoint identity mismatch')
    return arrays, metadata['provenance']

def write_json(path, data):
    path = Path(path)
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def load_protocol(path, amendment_path=None):
    path = Path(path)
    d = json.loads(path.read_text())
    for key in ['bundle_profiles', 'bundle_weights', 'weight_bank', 'library']:
        if sha(d[key]['path']) != d[key]['sha256']:
            raise ValueError(f"changed frozen input: {key}")
    revisions = {}
    if amendment_path is not None:
        amendment = json.loads(Path(amendment_path).read_text())
        if amendment['protocol_sha256'] != sha(path):
            raise ValueError('source amendment targets a different protocol')
        revisions = {v['path']: v for v in amendment['sources']}
        d['_summary_cache_source_hashes'] = amendment.get('compatible_summary_sources', [])
        d['_execution_amendment'] = {'path': str(Path(amendment_path).resolve()),
                                     'sha256': sha(amendment_path)}
    for source in d['sources']:
        expected = source['sha256']
        if source['path'] in revisions:
            revision = revisions[source['path']]
            if revision['original_sha256'] != expected:
                raise ValueError('source amendment predecessor mismatch')
            expected = revision['amended_sha256']
        if sha(source['path']) != expected:
            raise ValueError(f"changed analysis source: {source['path']}")
    return d, sha(path)


def freeze(args):
    """Inspect identifiers and geometry only; no measured power/score values."""
    from pilot_proxy.detector_weights import DetectorWeightBank
    import h5py

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    profiles_path = args.bundle / 'pilot_profiles.json'
    profiles = json.loads(profiles_path.read_text())['profiles']
    candidate = {int(p['physical_channel']): p for p in profiles
                 if p['fine_calibration']['status'] == 'calibrated'}
    if set(candidate) != FINITE_CHANNELS:
        raise ValueError('expected exactly the eight frozen finite profiles')
    bank = DetectorWeightBank(explicit_path=args.weight_bank)
    payload = (args.bundle / 'weights.bin').read_bytes()
    for ch, p in candidate.items():
        w, valid = bank.get_weights_for_physical_channel(ch)
        offset, size = p['weight_bank_offset_bytes'], p['weight_bank_nbytes']
        if not valid or w.tobytes() != payload[offset:offset + size]:
            raise ValueError(f'original bank differs from candidate channel {ch}')
    common = None
    metadata = {}
    for epoch in EPOCHS:
        rows = {}
        for p in sorted((args.datasets / f'pilot_reduce_{epoch}').glob('*.npz')):
            with np.load(p, allow_pickle=False) as z:
                m = json.loads(str(z['meta']))
            if int(m['channel']) in CHANNELS:
                rows[int(p.stem)] = m
        common = set(rows) if common is None else common & set(rows)
        metadata[epoch] = rows
    inband = {}
    raw_coordinates = {}
    canonical_ids = None
    for ch, pilot in CHANNELS.items():
        cells = sorted(fid for fid in common if all(
            metadata[e][fid]['channel'] == ch and not metadata[e][fid]['is_pilot']
            and (args.datasets / f'inband_dump_{e}' / f'baseband_{e}_{fid}.h5').is_file()
            for e in EPOCHS))
        inband[ch] = cells
        for epoch in EPOCHS:
            for fid in [pilot, *cells]:
                prefix = 'pilot' if fid == pilot else 'inband'
                raw = args.datasets / f'{prefix}_dump_{epoch}' / f'baseband_{epoch}_{fid}.h5'
                with h5py.File(raw, 'r') as f:
                    ids = f['index_map/input'][:]
                    ids = ids[np.argsort(ids['chan_id'])]
                    if not np.array_equal(ids['chan_id'], np.arange(2048)):
                        raise ValueError('complete unique input coordinate required')
                    if canonical_ids is None:
                        canonical_ids = ids
                    elif not np.array_equal(canonical_ids, ids):
                        raise ValueError('input serial identity differs across raw captures')
                    m = metadata[epoch][fid]
                    if (int(f.attrs['time0_fpga_count']) != m['time0_fpga']
                            or float(f.attrs['delta_time']) != m['delta_time']
                            or int(f.attrs['freq_id']) != fid):
                        raise ValueError('raw/saved metadata mismatch')
                    raw_coordinates[str(raw)] = {
                        'input_map_sha256': digest_arrays({'input_ids': ids}),
                        'time0_fpga': int(f.attrs['time0_fpga_count']),
                        'sample_seconds': float(f.attrs['delta_time']),
                        'freq_id': fid, 'baseband_shape': list(f['baseband'].shape)}
    record = {
        'schema': 'capture_transfer_protocol_v1',
        'scope': 'descriptive differential transfer; no isolated DTV truth or physical coverage certification; heldout epochs were characterized historically and are only held out from this fit',
        'epochs': EPOCHS, 'training_epochs': EPOCHS[:2], 'holdout_epochs': EPOCHS[2:],
        'channels': CHANNELS, 'inband_frequency_ids': inband, 'raw_coordinates': raw_coordinates,
        'permanent_reject_diagnostic_channels': [17, 35],
        'cell_selection': 'all common saved nonpilot cells with native inband HDF5 available in all4 epochs; determined from metadata only',
        'datasets': str(args.datasets.resolve()),
        'frame_origin': 'exact saved frame_fpga0; capture-aligned, not global FPGA origin zero',
        'frame_samples': 16384, 'window_samples': 128, 'sample_seconds': 2.56e-6,
        'input_coordinate': 'chan_id sorted 0..2047; verify same correlator_input identifiers across files',
        'weights': 'equal original epoch mass; equal original frame mass within epoch; never reset after masking',
        'response': 'mean available nonpilot-cell raw autos per input / same-frame CH37 fid491 raw auto; divide by training weighted mean and subtract1',
        'predictor': 'signed per-input coarse pilot proxy; exact candidate fine selection uses full array',
        'fit': 'unregularized weighted intercept+slope per input on all training frames; no threshold tuning',
        'live_inputs': 'all training target/control autos >0.5 and finite ratios/proxies; fixed for holdout',
        'comparison': 'heldout weighted mean squared residual vs frozen training mean baseline; report each epoch and both all/current-word selections',
        'descriptive_acceptance': 'model MSE <= baseline MSE in each heldout epoch; unavailable if no retained frames or nonfinite fixed inputs',
        'uncertainty': 'no frame/input replication; block uncertainty unavailable below8 epochs per half; sample count does not resolve physical identifiability',
        'physical_qualification': 'not evaluated; cross-frequency sky/noise/leveling and gain transfer are not identified',
        'covariance': 'lags0..8 within capture, both endpoints kept, exact lag spacing; signed raw moment and finite-record demeaned covariance; no inferred long-time G',
        'visibility': 'saved raw redundant stacks / (16384 * frame mean auto on fixed training live inputs); primary EW classes(1,0),(2,0),(3,0) copolar; separate NS>=40m; representative cell nearest channel midpoint; no post-stack gain correction',
        'gains': 'daily gain solutions exist but are not applied; this raw power differential comparison is conditional on instrumental/sky stability',
        'shelf_coefficient': float(SHELF), 'profiles': {str(k): v for k, v in candidate.items()},
        'bundle_profiles': {'path': str(profiles_path.resolve()), 'sha256': sha(profiles_path)},
        'bundle_weights': {'path': str((args.bundle / 'weights.bin').resolve()), 'sha256': sha(args.bundle / 'weights.bin')},
        'weight_bank': {'path': str(args.weight_bank.resolve()), 'sha256': sha(args.weight_bank)},
        'library': {'path': str(args.library.resolve()), 'sha256': sha(args.library)},
        'source_sha256': sha(Path(__file__)),
    }
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *[root / 'src/pilot_proxy' / name for name in [
        'characterization/capture_transfer.py', 'characterization/threshold_rule.py',
        'fine_decision.py', 'fine_reduction.py', 'detector_weights.py', 'detect.py',
        'kernel.py', 'chime/frame_adapter.py', 'chime/hdf5_input.py']]]
    reducer = args.datasets.parent / 'output/channel-ruling-execution-2026-09-14/rebuild/author_actions/capture-runbook/reduce/reduce_dump.py'
    sources.append(reducer)
    record['sources'] = [{'path': str(f.resolve()), 'sha256': sha(f)} for f in sources]
    record['payload_auto_verification'] = 'independent exact raw auto check for first analyzed frame of every payload file; selected saved arrays hashed; pilot/control autos checked on all analyzed frames'
    write_json(out / 'protocol.json', record)
    print(json.dumps({'protocol': str(out / 'protocol.json'), 'sha256': sha(out / 'protocol.json'),
                      'cells_per_channel': {k: len(v) for k, v in inband.items()}}))


def reduce_one(args):
    import cupy as cp
    import h5py

    from pilot_proxy.chime.frame_adapter import pack_chime_block_for_detector
    from pilot_proxy.chime.hdf5_input import CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4
    from pilot_proxy.detect import detect_packed_detector_input
    from pilot_proxy.detector_weights import DetectorWeightBank
    from pilot_proxy.fine_decision import fine_mask_decision
    from pilot_proxy.fxfft import fxfft256
    from pilot_proxy.kernel import PilotProxyKernel

    p, protocol_hash = load_protocol(args.protocol, args.source_amendment)
    ch, epoch = args.channel, args.epoch
    if epoch not in p['epochs'] or str(ch) not in p['channels']:
        raise ValueError('epoch/channel outside frozen protocol')
    fid = p['channels'][str(ch)]
    out = args.protocol.parent / 'exact'
    out.mkdir(exist_ok=True)
    dest = out / f'{epoch}_ch{ch}.npz'
    receipt = dest.with_suffix('.json')
    if dest.exists() or receipt.exists():
        raise FileExistsError(dest)
    data = Path(p['datasets'])
    raw = data / f'pilot_dump_{epoch}' / f'baseband_{epoch}_{fid}.h5'
    saved = data / f'pilot_reduce_{epoch}' / f'{fid}.npz'
    with np.load(saved, allow_pickle=False) as z:
        origins, want_autos = z['frame_fpga0'], z['autos']
        saved_meta = json.loads(str(z['meta']))
    kernel = PilotProxyKernel(p['library']['path'])
    if kernel.specs.K != 128:
        raise ValueError('K128 library required')
    bank = DetectorWeightBank(explicit_path=p['weight_bank']['path'], expected_kernel=kernel.specs)
    weights, valid = bank.get_weights_for_physical_channel(ch)
    if not valid:
        raise ValueError('invalid weights')
    profile = p['profiles'].get(str(ch))
    calibration = profile['fine_calibration'] if profile else None
    fine, per_input, global_power, keeps = [], [], [], []
    selected_raw_hash = hashlib.sha256()
    checkpoint_dir = out / 'frames' / f'{epoch}_ch{ch}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    with h5py.File(raw, 'r') as f:
        ids = f['index_map/input'][:]
        order = np.argsort(ids['chan_id'])
        ids = ids[order]
        require_frozen_input_map(p, raw, ids)
        if not np.array_equal(ids['chan_id'], np.arange(2048)):
            raise ValueError('complete unique input coordinate required')
        time0 = int(f.attrs['time0_fpga_count'])
        dt = float(f.attrs['delta_time'])
        if dt != p['sample_seconds'] or int(f.attrs['freq_id']) != fid:
            raise ValueError('sample clock/frequency mismatch')
        if time0 != saved_meta['time0_fpga']:
            raise ValueError('saved reduction origin mismatch')
        for row, origin in enumerate(origins):
            start = int(origin) - time0
            block = f['baseband'][start:start + 16384, :][:, order]
            if block.shape != (16384, 2048):
                raise ValueError('incomplete requested frame')
            selected_raw_hash.update(np.ascontiguousarray(block).tobytes())
            re = (block >> 4).astype(np.int16) - 8
            im = (block & 15).astype(np.int16) - 8
            autos = (re * re + im * im).mean(0).astype(np.float32)
            if not np.array_equal(autos, want_autos[row]):
                raise ValueError(f'raw/saved per-input auto mismatch at frame {row}')
            raw_hash = hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()
            cache = checkpoint_dir / f'{row:04d}.npz'
            reused = cache.exists()
            if reused:
                z = read_checkpoint(cache, protocol_hash, raw_hash, origin, sha(Path(__file__)))
                per, glob, fp = z['per'], z['glob'], z['fine']
                keep = bool(z['keep'])
                target_norm, reference_norm = int(z['target_norm']), int(z['reference_norm'])
                cache_hits += 1
            else:
                packed = pack_chime_block_for_detector(
                    np.ascontiguousarray(block.T)[:, None, :], frame_size_samples=16384,
                    detector_window_samples=128, spectral_sense='inverted', frames_in_chunk=1,
                    sample_encoding=CHIME_NATIVE_OFFSET_BINARY_COMPLEX_INT4,
                    selected_coarse_channel=0, physical_channel=ch).packed
                result = detect_packed_detector_input(packed=packed, weights=weights,
                                                     kernel=kernel, emit_row_projections=True)
                rows = cp.asnumpy(result['matched_filter_row_projections'])[0].astype(np.int64)
                rows = rows.reshape(3, 2048, 128, 2)
                # Per-input frame powers: each input's all-bin sum of its own
                # fixed-point fine powers (kernel core 3.0.0), so the inputs
                # sum exactly to the global frame powers.
                spectra = fxfft256(rows).astype(np.int64)
                per = np.sum(spectra * spectra, axis=(2, 3), dtype=np.uint64)
                glob = np.array([result['results'][0][name] for name in
                                 ['p_target_u64', 'p_ref_lower_u64', 'p_ref_upper_u64']], dtype=np.uint64)
                fp = result['fine_powers_u64'][0]
                target_norm, reference_norm = result['target_norm_sq'], result['reference_norm_sum_sq']
                if calibration:
                    decision = fine_mask_decision(fp, anchor_bin=calibration['anchor_bin'],
                        designated_half_width=calibration['designated_half_width'],
                        bulk_mask=[int(v, 16) for v in calibration['bulk_mask_words_hex']],
                        cfar_rank=calibration['cfar_rank'], multiplier_q16=calibration['cfar_multiplier_q16'])
                    keep = decision.valid and decision.mask == 0
                else:
                    keep = ch == 37
                temporary = checkpoint_dir / (cache.name + '.' + uuid.uuid4().hex + '.tmp')
                with temporary.open('xb') as handle:
                    checkpoint_arrays = dict(per=per, glob=glob, fine=fp, keep=keep,
                                             target_norm=target_norm, reference_norm=reference_norm,
                                             origin=origin, raw_sha256=raw_hash, protocol_sha256=protocol_hash,
                                             execution_source_sha256=sha(Path(__file__)))
                    np.savez_compressed(handle, **checkpoint_arrays,
                                        payload_sha256=digest_arrays(checkpoint_arrays))
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.rename(cache)
            if not np.array_equal(per.sum(1, dtype=np.uint64), glob):
                raise ValueError('per-input/global frame power sum identity failed')
            fine.append(fp); per_input.append(per); global_power.append(glob); keeps.append(keep)
            print(json.dumps({'checkpoint': str(cache), 'frame': row, 'reused': reused}), flush=True)
    arrays = dict(frame_fpga0=origins, input_ids=ids, powers=np.array(global_power),
                  per_input_powers=np.array(per_input), fine_powers=np.array(fine), keep=np.array(keeps),
                  target_norm=np.array(target_norm), reference_norm=np.array(reference_norm))
    with dest.open('xb') as f:
        np.savez_compressed(f, **arrays)
    write_json(receipt, {'protocol_sha256': protocol_hash, 'product_sha256': sha(dest),
        'raw_path': str(raw), 'raw_bytes': raw.stat().st_size,
        'selected_raw_sorted_input_bytes_sha256': selected_raw_hash.hexdigest(),
        'selected_raw_hash_scope': 'ordered complete analyzed frames after chan_id permutation; not whole HDF5 bytes',
        'saved_reduction_path': str(saved),
        'saved_used_arrays_sha256': digest_arrays({'origins': origins, 'autos': want_autos}),
        'raw_auto_exact_match': True, 'per_input_sum_exact_match': True,
        'frames': len(origins), 'kept': int(sum(keeps)), 'source_sha256': sha(Path(__file__)),
        'control_has_no_policy': ch == 37, 'permanent_reject': ch in [17, 35],
        'frame_checkpoints_reused': cache_hits, 'source_amendment': p.get('_execution_amendment')})
    print(json.dumps({'product': str(dest), 'frames': len(origins), 'kept': int(sum(keeps))}))


def read_saved(path, names):
    with np.load(path, allow_pickle=False) as z:
        a = {name: z[name] for name in names}
        meta = json.loads(str(z['meta']))
    return a, {'path': str(path), 'meta': meta, 'selected_arrays_sha256': digest_arrays(a)}


def finite_summary(v):
    a = np.asarray(v)
    good = np.isfinite(a)
    return {'count': int(a.size), 'finite': int(good.sum()),
            'median': float(np.median(a[good])) if good.any() else None,
            'q10': float(np.quantile(a[good], .1)) if good.any() else None,
            'q90': float(np.quantile(a[good], .9)) if good.any() else None}


def summarize(args):
    import h5py
    p, protocol_hash = load_protocol(args.protocol, args.source_amendment)
    out = args.protocol.parent / 'summary'
    out.mkdir(exist_ok=True)
    ch = args.channel
    if str(ch) not in p['channels']:
        raise ValueError('channel outside protocol')
    epochs = p['epochs']
    datasets = Path(p['datasets'])
    records, provenance = [], []
    identity = None
    cells = p['inband_frequency_ids'][str(ch)]
    if not cells:
        report = {'schema': 'capture_transfer_summary_v1', 'protocol_sha256': protocol_hash,
                  'channel': ch, 'status': 'unavailable:no native-backed common nonpilot payload cells',
                  'physical_qualification': 'not evaluated'}
        write_json(out / f'ch{ch}.json', report)
        print(json.dumps(report))
        return
    midpoint = 470 + (ch - 14) * 6 + 3
    representative = min(cells, key=lambda fid: (abs((800 - fid * .390625) - midpoint), fid))
    checkpoint_dir = args.protocol.parent / 'summary-inputs'
    checkpoint_dir.mkdir(exist_ok=True)
    execution_source_hash = sha(Path(__file__))
    cache_sources = [execution_source_hash, *p.get('_summary_cache_source_hashes', [])]
    cell_checkpoint_dir = args.protocol.parent / 'summary-cells'
    cell_checkpoint_dir.mkdir(exist_ok=True)
    for event, epoch in enumerate(epochs):
        checkpoint = checkpoint_dir / f'{epoch}_ch{ch}.npz'
        if checkpoint.exists():
            record, receipts = read_summary_checkpoint(
                checkpoint, protocol_hash, cache_sources, ch, epoch)
            ids = record.pop('input_ids')
            if identity is None:
                identity = ids
            elif not np.array_equal(identity, ids):
                raise ValueError('checkpoint input serial identity changed across captures')
            records.append(record)
            provenance.extend(receipts)
            print(json.dumps({'summary_checkpoint': str(checkpoint), 'reused': True}), flush=True)
            continue
        receipt_start = len(provenance)
        exact_path = args.protocol.parent / 'exact' / f'{epoch}_ch{ch}.npz'
        receipt = json.loads(exact_path.with_suffix('.json').read_text())
        if receipt['protocol_sha256'] != protocol_hash or sha(exact_path) != receipt['product_sha256']:
            raise ValueError('exact product provenance mismatch')
        provenance.append(receipt)
        with np.load(exact_path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        if identity is None:
            identity = a['input_ids']
        elif not np.array_equal(identity, a['input_ids']):
            raise ValueError('input serial identity changed across captures')
        t = a['frame_fpga0']
        control_path = datasets / f'pilot_reduce_{epoch}' / '491.npz'
        control, receipt = read_saved(control_path, ['frame_fpga0', 'autos'])
        provenance.append(receipt)
        i, j = exact_frame_join(t, control['frame_fpga0'])
        if len(i) != len(t):
            raise ValueError('control lacks exact origins')
        control_auto = control['autos'][j]
        control_exact_path = args.protocol.parent / 'exact' / f'{epoch}_ch37.npz'
        control_receipt = json.loads(control_exact_path.with_suffix('.json').read_text())
        if (control_receipt['protocol_sha256'] != protocol_hash
                or sha(control_exact_path) != control_receipt['product_sha256']):
            raise ValueError('control exact product provenance mismatch')
        provenance.append(control_receipt)
        require_saved_control_identity(control, control_receipt)
        with np.load(control_exact_path, allow_pickle=False) as z:
            if not np.array_equal(z['input_ids'], identity):
                raise ValueError('control input serials differ')
        powers = []
        for fid in cells:
            names = ['frame_fpga0', 'autos'] + (['stacks', 'keys', 'count'] if fid == representative else [])
            cell_checkpoint = cell_checkpoint_dir / f'{epoch}_ch{ch}_fid{fid}.npz'
            cell_key = f'{epoch}:fid{fid}'
            if cell_checkpoint.exists():
                saved, receipts = read_summary_checkpoint(
                    cell_checkpoint, protocol_hash, cache_sources, ch, cell_key)
                ids = saved.pop('input_ids')
                if not np.array_equal(ids, identity):
                    raise ValueError('payload checkpoint input serial identity mismatch')
                receipt, = receipts
                provenance.append(receipt)
                i, j = exact_frame_join(t, saved['frame_fpga0'])
                if len(i) != len(t):
                    raise ValueError('payload checkpoint lacks exact origins')
                reused = True
            else:
                saved, receipt = read_saved(datasets / f'pilot_reduce_{epoch}' / f'{fid}.npz', names)
                provenance.append(receipt)
                if receipt['meta']['delta_time'] != p['sample_seconds']:
                    raise ValueError('saved sample clock mismatch')
                i, j = exact_frame_join(t, saved['frame_fpga0'])
                if len(i) != len(t):
                    raise ValueError('payload cell lacks exact origins')
                raw_path = datasets / f'inband_dump_{epoch}' / f'baseband_{epoch}_{fid}.h5'
                with h5py.File(raw_path) as raw:
                    ids = raw['index_map/input'][:]
                    order = np.argsort(ids['chan_id'])
                    if not np.array_equal(ids[order], identity):
                        raise ValueError('payload input serial identity mismatch')
                    require_frozen_input_map(p, raw_path, ids[order])
                    start = int(t[0]) - int(raw.attrs['time0_fpga_count'])
                    block = raw['baseband'][start:start + 16384, :][:, order]
                    re, im = (block >> 4).astype(np.int16) - 8, (block & 15).astype(np.int16) - 8
                    auto_check = (re * re + im * im).mean(0).astype(np.float32)
                    if not np.array_equal(auto_check, saved['autos'][j[0]]):
                        raise ValueError('payload first-frame raw/saved auto mismatch')
                    receipt['first_frame_raw_auto_exact_match'] = True
                    receipt['first_frame_fpga0'] = int(t[0])
                    receipt['first_frame_sorted_raw_sha256'] = digest_arrays({'packed': block})
                write_summary_checkpoint(cell_checkpoint, saved, identity, [receipt],
                                         protocol_hash, execution_source_hash, ch, cell_key)
                reused = False
            print(json.dumps({'payload_checkpoint': str(cell_checkpoint), 'reused': reused}), flush=True)
            powers.append(saved['autos'][j])
            if fid == representative:
                stacks = saved['stacks'][j]
                keys = saved['keys']
                rep_auto = saved['autos'][j]
        auto = np.mean(powers, axis=0, dtype=float)
        ratio = np.divide(auto, control_auto, out=np.full(auto.shape, np.nan), where=control_auto > 0)
        x = signed_proxy(a['per_input_powers'], int(a['target_norm']), int(a['reference_norm']), p['shelf_coefficient'])
        global_x = signed_proxy(a['powers'], int(a['target_norm']), int(a['reference_norm']), p['shelf_coefficient'])
        records.append(dict(t=t, x=x, global_x=global_x, ratio=ratio, auto=auto,
                            control=control_auto, keep=a['keep'], event=np.full(len(t), event),
                            stacks=stacks, rep_auto=rep_auto, keys=keys))
        write_summary_checkpoint(checkpoint, records[-1], identity, provenance[receipt_start:],
                                 protocol_hash, execution_source_hash, ch, epoch)
        print(json.dumps({'summary_checkpoint': str(checkpoint), 'reused': False}), flush=True)
    train = records[:2]
    cat = lambda name: np.concatenate([r[name] for r in train])
    model = fit_training_transfer(cat('x'), cat('ratio'), cat('event'), auto=cat('auto'), control=cat('control'))
    live = model['live']
    report = {'schema': 'capture_transfer_summary_v1', 'protocol_sha256': protocol_hash,
        'channel': ch, 'control': ch == 37, 'live_training_inputs': int(live.sum()),
        'input_count_is_not_replication': True, 'training_epochs': epochs[:2], 'holdout_epochs': epochs[2:],
        'inband_cells': cells, 'visibility_cell': representative,
        'slope': finite_summary(model['slope'][live]), 'epochs': [],
        'uncertainty_status': 'unavailable:2 training and2 heldout acquisitions; descriptive support only',
        'physical_qualification': 'not evaluated: neither CH37 normalization nor prediction identifies isolated DTV',
        'source_sha256': sha(Path(__file__)), 'source_amendment': p.get('_execution_amendment'),
        'provenance': provenance}
    compact = {f'model_{k}': v for k, v in model.items()}
    all_event = np.concatenate([r['event'] for r in records])
    all_weight = event_weights(all_event)
    start = 0
    for event, r in enumerate(records):
        y, prediction, residual = transfer_residuals(model, r['x'], r['ratio'])
        w = all_weight[start:start + len(r['t'])]
        start += len(r['t'])
        entry = {'epoch': epochs[event], 'split': 'training' if event < 2 else 'holdout',
                 'frames': len(r['t']), 'kept': int(r['keep'].sum()), 'views': {}}
        valid_holdout = bool(live.any() and np.isfinite(y[:, live]).all()
                             and np.isfinite(residual[:, live]).all())
        # Preserve the science-used east-west classes separately from long north-south classes.
        key = r['keys']
        copol = key[:, 2] == key[:, 3]
        east_west = np.isin(key[:, 0], [1, 2, 3]) & (key[:, 1] == 0) & copol
        north_south = (key[:, 0] == 0) & (np.abs(key[:, 1]) * .3048 >= 40) & copol
        normalizer = 16384 * r['rep_auto'][:, live].mean(1) if live.any() else np.full(len(r['t']), np.nan)
        visibility_ew = r['stacks'][:, east_west] / normalizer[:, None]
        visibility_ns = r['stacks'][:, north_south] / normalizer[:, None]
        for label, sel in [('all', np.ones(len(r['t']), bool)), ('current_word', r['keep'])]:
            measured = {'frames': int(sel.sum()), 'original_mass': float(w[sel].sum()),
                        'signed_global_proxy_mean': float(r['global_x'][sel].mean()) if sel.any() else None}
            if sel.any() and valid_holdout:
                err = residual[sel][:, live]
                baseline = y[sel][:, live] - model['y0'][live]
                mse, base_mse = float(np.mean(err ** 2)), float(np.mean(baseline ** 2))
                measured.update(model_mse=mse, baseline_mse=base_mse,
                    mse_ratio=mse / base_mse if base_mse else None,
                    mean_signed_residual=float(err.mean()),
                    prediction_better_than_baseline=bool(mse <= base_mse),
                    per_input_signed_residual=finite_summary(err.mean(0)))
                for name, values in [('proxy', r['x'][:, live]), ('power_change', y[:, live]),
                                      ('visibility_ew', visibility_ew), ('visibility_ns', visibility_ns)]:
                    moment, covariance, counts = lag_moments(values, r['t'], sel)
                    prefix = f'{event}_{label}_{name}'
                    compact[prefix + '_moment'] = moment
                    compact[prefix + '_covariance'] = covariance
                    compact[prefix + '_pairs'] = counts
                    measured[name + '_lag_pairs'] = counts.tolist()
                    if name.startswith('visibility'):
                        cross = coherent_cross_moment(values, sel)
                        compact[prefix + '_cross'] = cross
                        measured[name + '_signed_cross_moment'] = finite_summary(cross)
            else:
                measured['status'] = 'unavailable:no selected frames or nonfinite fixed training inputs'
            entry['views'][label] = measured
        report['epochs'].append(entry)
        for name, values in [('origins', r['t']), ('keep', r['keep']), ('signed_proxy', r['x']),
                              ('ratio', r['ratio']), ('prediction', prediction), ('response', y)]:
            compact[f'{event}_{name}'] = values
    heldout = [r['views']['current_word'] for r in report['epochs'][2:]]
    report['descriptive_holdout_acceptance'] = (
        'unavailable' if any('prediction_better_than_baseline' not in r for r in heldout)
        else 'pass' if all(r['prediction_better_than_baseline'] for r in heldout) else 'fail')
    array_path = out / f'ch{ch}.npz'
    with array_path.open('xb') as f:
        np.savez_compressed(f, **compact)
    report['arrays_sha256'] = sha(array_path)
    write_json(out / f'ch{ch}.json', report)
    print(json.dumps({k: report[k] for k in ['channel', 'live_training_inputs', 'descriptive_holdout_acceptance']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    for option in ['datasets', 'bundle', 'weight-bank', 'library', 'output']:
        f.add_argument('--' + option, type=Path, required=True)
    for name in ['reduce-one', 'summarize']:
        q = sub.add_parser(name)
        q.add_argument('--protocol', type=Path, required=True)
        q.add_argument('--channel', type=int, required=True)
        q.add_argument('--source-amendment', type=Path,
                       help='Explicit source-only revision receipt; the frozen analysis protocol is unchanged.')
        if name == 'reduce-one':
            q.add_argument('--epoch', required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'reduce-one': reduce_one, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
