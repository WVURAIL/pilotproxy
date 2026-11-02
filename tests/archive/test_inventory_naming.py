"""Managed inventory names preserve the selected frequency channels."""

from __future__ import annotations

import pytest

from pilot_proxy.archive.commands import survey_chime
from pilot_proxy.archive.inventory import derive_inventory_name
from pilot_proxy.archive.names import validate_identifier


def test_frequency_list_and_range_have_distinct_names():
    assert derive_inventory_name("chime", "614,706") == "chime-fid614_706"
    assert derive_inventory_name("chime", "614-706") == "chime-fid614-706"


@pytest.mark.parametrize("selection", [0, "0", [0], (0,), {0}])
def test_channel_zero_is_not_named_as_all_channels(selection):
    assert derive_inventory_name("chime", selection) == "chime-fid0"


@pytest.mark.parametrize("selection", [None, "all", " ALL ", "*"])
def test_all_channels_keep_the_existing_name(selection):
    assert derive_inventory_name("chime", selection) == "chime"


@pytest.mark.parametrize("selection", [
    "614,706-708", "708,706,614,707,614", [708, 614, 706, 707],
    {614, 706, 707, 708},
])
def test_equivalent_selections_have_one_stable_name(selection):
    assert derive_inventory_name("chime", selection) == "chime-fid614_706-708"


@pytest.mark.parametrize("selection", ["614/706", "614..706", "", [], False])
def test_invalid_selections_are_not_sanitized_into_inventory_names(selection):
    with pytest.raises(SystemExit, match="malformed freq_id selection"):
        derive_inventory_name("chime", selection)


def test_long_names_preserve_list_and_range_identity_before_hashing():
    singles = ",".join(str(value) for value in range(10, 42, 2))
    ranges = ",".join(f"{value}-{value + 2}" for value in range(10, 42, 4))
    single_name = derive_inventory_name("chime", singles)
    range_name = derive_inventory_name("chime", ranges)
    assert single_name != range_name
    assert single_name == derive_inventory_name("chime", list(range(10, 42, 2)))
    for name in (single_name, range_name):
        assert len(name) <= len("chime-") + 40
        assert validate_identifier(name) == name


def test_survey_default_managed_paths_separate_lists_ranges_and_zero(tmp_path):
    paths = {
        selection: survey_chime(root=tmp_path, freq_ids=selection, dry_run=True)
        for selection in ("614,706", "614-706", 0, None)
    }
    assert paths == {
        "614,706": tmp_path / "data/chime-fid614_706/inventory.jsonl",
        "614-706": tmp_path / "data/chime-fid614-706/inventory.jsonl",
        0: tmp_path / "data/chime-fid0/inventory.jsonl",
        None: tmp_path / "data/chime/inventory.jsonl",
    }
    assert not (tmp_path / "data").exists()


def test_explicit_survey_name_is_preserved(tmp_path):
    assert survey_chime(
        root=tmp_path, name="saved-pilots", freq_ids="614,706", dry_run=True,
    ) == tmp_path / "data/saved-pilots/inventory.jsonl"
