"""Mutation tests for capture identity guards; no measured-value adjustment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / 'tools/characterize_capture_transfer.py'
    spec = importlib.util.spec_from_file_location('capture_transfer_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_serial_mutation_and_order_refused(tool):
    ids = np.array([(0, b'a'), (1, b'b')], dtype=[('chan_id', '<u2'), ('serial', 'S4')])
    p = {'raw_coordinates': {'raw.h5': {'input_map_sha256': tool.digest_arrays({'input_ids': ids})}}}
    tool.require_frozen_input_map(p, Path('raw.h5'), ids)
    changed = ids.copy()
    changed['serial'][1] = b'c'
    with pytest.raises(ValueError, match='input map'):
        tool.require_frozen_input_map(p, Path('raw.h5'), changed)
    with pytest.raises(ValueError, match='input map'):
        tool.require_frozen_input_map(p, Path('raw.h5'), ids[::-1])


def test_control_auto_and_origin_mutation_refused(tool):
    control = {'frame_fpga0': np.array([10, 16394]), 'autos': np.ones((2, 3), dtype=np.float32)}
    receipt = {'saved_used_arrays_sha256': tool.digest_arrays({
        'origins': control['frame_fpga0'], 'autos': control['autos']})}
    tool.require_saved_control_identity(control, receipt)
    changed = {k: v.copy() for k, v in control.items()}
    changed['autos'][1, 2] += 1
    with pytest.raises(ValueError, match='control arrays'):
        tool.require_saved_control_identity(changed, receipt)
    changed = {k: v.copy() for k, v in control.items()}
    changed['frame_fpga0'][1] += 1
    with pytest.raises(ValueError, match='control arrays'):
        tool.require_saved_control_identity(changed, receipt)


def test_checkpoint_binds_raw_frame_protocol_origin_and_source(tool, tmp_path):
    path = tmp_path / 'frame.npz'
    arrays = dict(raw_sha256='raw', protocol_sha256='protocol', origin=123,
                  execution_source_sha256='source', per=np.zeros((3, 2048), dtype=np.uint64))
    np.savez(path, **arrays, payload_sha256=tool.digest_arrays(arrays))
    got = tool.read_checkpoint(path, 'protocol', 'raw', 123, 'source')
    np.testing.assert_array_equal(got['per'], 0)
    for params in [('wrong', 'raw', 123, 'source'), ('protocol', 'wrong', 123, 'source'),
                   ('protocol', 'raw', 124, 'source'), ('protocol', 'raw', 123, 'wrong')]:
        with pytest.raises(ValueError, match='checkpoint identity'):
            tool.read_checkpoint(path, *params)


def test_summary_checkpoint_preserves_arrays_and_refuses_mutation(tool, tmp_path):
    path = tmp_path / 'summary.npz'
    record = {'t': np.array([12, 16396], dtype=np.uint64),
              'stacks': np.array([[1 + 2j], [-3 + 4j]], dtype=np.complex64),
              'x': np.array([[-.1], [.2]])}
    ids = np.array([(0, b'a')], dtype=[('chan_id', '<u2'), ('serial', 'S4')])
    receipts = [{'raw_hash': 'abc', 'saved_hash': 'def'}]
    tool.write_summary_checkpoint(path, record, ids, receipts, 'protocol', 'source', 29, 'epoch')
    restored, got_receipts = tool.read_summary_checkpoint(path, 'protocol', 'source', 29, 'epoch')
    assert got_receipts == receipts
    for key, value in record.items():
        np.testing.assert_array_equal(restored[key], value)
        assert restored[key].dtype == value.dtype
    np.testing.assert_array_equal(restored['input_ids'], ids)
    for args in [('wrong', 'source', 29, 'epoch'), ('protocol', 'wrong', 29, 'epoch'),
                 ('protocol', 'source', 37, 'epoch'), ('protocol', 'source', 29, 'other')]:
        with pytest.raises(ValueError, match='checkpoint identity'):
            tool.read_summary_checkpoint(path, *args)
    with np.load(path) as z:
        altered = {key: z[key] for key in z.files}
    altered['x'][0, 0] += 1
    np.savez_compressed(path, **altered)
    with pytest.raises(ValueError, match='checkpoint payload'):
        tool.read_summary_checkpoint(path, 'protocol', 'source', 29, 'epoch')

@pytest.mark.parametrize('field', ['fine', 'keep', 'target_norm', 'reference_norm', 'per', 'glob'])
def test_checkpoint_payload_mutation_refused(tool, tmp_path, field):
    path = tmp_path / 'frame.npz'
    arrays = dict(raw_sha256=np.array('raw'), protocol_sha256=np.array('protocol'),
                  origin=np.array(123), execution_source_sha256=np.array('source'),
                  fine=np.zeros(256, dtype=np.uint64), keep=np.array(False),
                  target_norm=np.array(123), reference_norm=np.array(246),
                  per=np.zeros((3, 2048), dtype=np.uint64), glob=np.zeros(3, dtype=np.uint64))
    digest = tool.digest_arrays(arrays)
    np.savez(path, **arrays, payload_sha256=digest)
    tool.read_checkpoint(path, 'protocol', 'raw', 123, 'source')
    arrays[field].flat[0] = arrays[field].flat[0] + 1
    np.savez(path, **arrays, payload_sha256=digest)
    with pytest.raises(ValueError, match='checkpoint payload'):
        tool.read_checkpoint(path, 'protocol', 'raw', 123, 'source')


def test_legacy_checkpoint_without_payload_digest_refused(tool, tmp_path):
    path = tmp_path / 'legacy.npz'
    np.savez(path, raw_sha256='raw', protocol_sha256='protocol', origin=123,
             execution_source_sha256='source')
    with pytest.raises(ValueError, match='lacks payload digest'):
        tool.read_checkpoint(path, 'protocol', 'raw', 123, 'source')
