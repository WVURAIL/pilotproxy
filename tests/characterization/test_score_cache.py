import json
import numpy as np
import pytest
from pilot_proxy.characterization import score_cache as cache

def export(tmp_path):
    product = tmp_path / "product.npz"
    np.savez(product, dummy=np.arange(5))
    source = tmp_path / "ch19.npz"
    np.savez(source, selected=np.ones(5, bool), finite_time=np.ones(5, bool),
             is_current=np.ones(5, bool), fine_source_row_index=np.arange(5),
             frame_unit_index=np.array([10, 10, 10, 20, 20]),
             fine_required_multiplier_q16=np.array([1, 2, 3, 4, 5], dtype=np.uint64),
             fine_score_computed=np.ones(5, bool), fine_rank_valid=np.array([1,1,0,1,1], bool),
             fine_always_masked=np.zeros(5, bool), frame_month=np.full(5, 2026 * 12),
             frame_time=np.arange(5, dtype=float))
    meta = dict(channel=19, export_sha256=cache.sha(source),
                product=dict(path=str(product), sha256=cache.sha(product)), geometry={"anchor_bin": 128})
    path = source.with_suffix(".json")
    path.write_text(json.dumps(meta))
    return path

def test_build_preserves_event_weights_and_unsupported_mass(tmp_path):
    source = export(tmp_path)
    result = cache.build(source, tmp_path / "out", current_word=4)
    with np.load(tmp_path / "out/ch19.npz") as z:
        np.testing.assert_allclose(z["probability_weight"], [1/6,1/6,1/6,1/4,1/4])
        np.testing.assert_array_equal(z["current_keep"], [1,1,0,1,0])
        assert z["probability_weight"].sum() == pytest.approx(1)
    assert result["events"] == 2 and result["current_months"] == ["2026-01"]
    with pytest.raises(ValueError, match="already exists"):
        cache.build(source, tmp_path / "out")

def test_modified_export_refused(tmp_path):
    source = export(tmp_path)
    source.with_suffix(".npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        cache.build(source, tmp_path / "out")


def _replace_export(source, **changes):
    with np.load(source.with_suffix('.npz'), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays.update(changes)
    np.savez(source.with_suffix('.npz'), **arrays)
    metadata = json.loads(source.read_text())
    metadata['export_sha256'] = cache.sha(source.with_suffix('.npz'))
    source.write_text(json.dumps(metadata))


def test_population_filters_precede_event_weights_and_preserve_unsupported_mass(tmp_path):
    source = export(tmp_path)
    january = 2026 * 12
    _replace_export(source,
        selected=np.array([1, 1, 0, 1, 1, 1, 1, 1], dtype=bool),
        finite_time=np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool),
        is_current=np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool),
        fine_source_row_index=np.arange(8),
        frame_unit_index=np.array([10, 10, 10, 20, 20, 30, 30, 30]),
        fine_required_multiplier_q16=np.arange(1, 9, dtype=np.uint64),
        fine_score_computed=np.ones(8, bool),
        fine_rank_valid=np.array([1, 0, 1, 1, 1, 1, 1, 1], dtype=bool),
        fine_always_masked=np.zeros(8, bool),
        frame_month=np.array([january] * 3 + [january + 6] * 2 + [january + 7] * 3),
        frame_time=np.arange(8, dtype=float))

    result = cache.build(source, tmp_path / 'out', current_word=4)
    with np.load(tmp_path / 'out/ch19.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['source_row_index'], [0, 1, 3])
        np.testing.assert_array_equal(saved['frame_event_index'], [0, 0, 1])
        # Two selected events have equal initial mass. The unsupported frame
        # keeps its quarter of the denominator after the detector mask.
        np.testing.assert_array_equal(saved['probability_weight'], [0.25, 0.25, 0.5])
        np.testing.assert_array_equal(saved['current_keep'], [True, False, True])
        assert saved['probability_weight'][saved['current_keep']].sum() == 0.75
    assert result['frames'] == 3 and result['events'] == 2
    assert result['current_months'] == ['2026-01', '2026-07']


@pytest.mark.parametrize('changes, message', [
    ({'fine_source_row_index': np.array([1, 0, 2, 3, 4])}, 'row order'),
    ({'fine_required_multiplier_q16': np.arange(1, 6, dtype=float)}, 'uint64'),
    ({'is_current': np.zeros(5, bool)}, 'No selected'),
])
def test_authenticated_invalid_export_is_refused_before_writing(tmp_path, changes, message):
    source = export(tmp_path)
    _replace_export(source, **changes)
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match=message):
        cache.build(source, output)
    assert not output.exists()


def test_changed_source_product_is_refused(tmp_path):
    source = export(tmp_path)
    (tmp_path / 'product.npz').write_bytes(b'changed product')
    with pytest.raises(ValueError, match='Product hash'):
        cache.build(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_cli_preserves_the_requested_exact_current_word(tmp_path, capsys):
    source = export(tmp_path)
    output = tmp_path / 'out'
    word = str((1 << 64) - 1)
    assert cache.main(['--metadata', str(source), '--output', str(output), '--current-word', word]) == 0
    assert json.loads(capsys.readouterr().out) == {'channel': 19, 'frames': 5}
    metadata = json.loads((output / 'ch19.json').read_text())
    assert metadata['current_threshold']['eta_q16'] == word
    with np.load(output / 'ch19.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['current_keep'], [True, True, False, True, True])
