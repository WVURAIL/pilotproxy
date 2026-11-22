import json
import runpy
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture(scope="module")
def cache_tool():
    return runpy.run_path(str(TOOLS / "prepare_fine_validation_cache.py"))


@pytest.fixture(scope="module")
def supplement_tool():
    return runpy.run_path(str(TOOLS / "supplement_fine_campaign_plan.py"))


def test_case_index_continues_the_record_order(cache_tool):
    case_index = cache_tool["case_index"]
    record = [(ch, c, p) for ch in range(14, 37) for c in range(7) for p in range(4)]
    assert [case_index(*key) for key in record] == list(range(644))
    assert case_index(37, 0, 0) == 644
    assert case_index(37, 6, 3) == 671


def test_mapped_replaces_only_a_matching_prefix(cache_tool):
    mapped = cache_tool["mapped"]
    assert mapped("/old/root/a/b.cfile", [["/old/root", "/new"]]) == Path("/new/a/b.cfile")
    assert mapped("/other/b.cfile", [["/old/root", "/new"]]) == Path("/other/b.cfile")


def test_levels_are_the_whole_grid_lattice(supplement_tool):
    levels = supplement_tool["lattice"]()
    assert len(levels) == 61
    assert levels[0] == -66.0 and levels[-1] == -21.0
    assert all(abs(b - a - 0.75) < 1e-12 for a, b in zip(levels, levels[1:]))


def _sealed(path, data):
    path.write_text(json.dumps(data))
    import hashlib

    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")


def test_a_record_channel_is_refused(tmp_path, supplement_tool):
    record = tmp_path / "plan.json"
    _sealed(record, {"source_sha256": {}})
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "plan.json").write_text(json.dumps({"profiles": [{"physical_channel": 32}], "weights_sha256": "0"}))
    with pytest.raises(ValueError, match="record channel"):
        supplement_tool["derive"](record, cache, tmp_path / "w.bin")


def test_an_unsealed_record_plan_is_refused(tmp_path, supplement_tool):
    record = tmp_path / "plan.json"
    record.write_text("{}")
    record.with_suffix(".sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ValueError, match="digest differs"):
        supplement_tool["derive"](record, tmp_path, tmp_path / "w.bin")
