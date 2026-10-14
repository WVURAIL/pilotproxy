# coding=utf-8
"""The health gate names measurements once, and migrates archived files on read.

5fb0e32 renamed the measurements in the writer while the 2020--2026 archive kept
the old spellings. Rather than let every reader accept either vintage, the gate
names each measurement in the current vocabulary and resolves the archived
spelling through one migration map at read time. That map is deletable once the
archive has been reprocessed; until then these properties keep it honest.
"""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy import archive_health
from pilot_proxy.archive_health import ArchiveHealthError, _array
from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_TO_CURRENT,
)


def test_current_products_read_without_the_migration() -> None:
    product = {"coarse_power_ratio": np.asarray([[1.5]])}
    assert _array(product, "coarse_power_ratio")[0, 0] == 1.5


def test_archived_products_resolve_through_the_migration() -> None:
    product = {ARCHIVED_COARSE_POWER_RATIO: np.asarray([[1.5]])}
    assert _array(product, "coarse_power_ratio")[0, 0] == 1.5


def test_current_spelling_wins_when_a_product_somehow_has_both() -> None:
    product = {
        "coarse_power_ratio": np.asarray([[1.5]]),
        ARCHIVED_COARSE_POWER_RATIO: np.asarray([[9.9]]),
    }
    assert _array(product, "coarse_power_ratio")[0, 0] == 1.5


@pytest.mark.parametrize("current", sorted(set(ARCHIVED_TO_CURRENT.values())))
def test_every_renamed_measurement_has_a_route(current: str) -> None:
    archived = {v: k for k, v in ARCHIVED_TO_CURRENT.items()}[current]
    assert _array({archived: np.asarray([[0.25]])}, current)[0, 0] == 0.25


def test_a_genuinely_absent_field_still_fails_closed() -> None:
    with pytest.raises(ArchiveHealthError):
        _array({}, "coarse_power_ratio")
    with pytest.raises(ArchiveHealthError):
        _array({"something_else": np.zeros(1)}, "frame_index")


def test_the_migration_is_a_bijection() -> None:
    """One archived spelling per current name, so the inverse is well defined."""
    currents = list(ARCHIVED_TO_CURRENT.values())
    assert len(currents) == len(set(currents))


def test_no_retired_spelling_is_also_a_current_name() -> None:
    retired = set(ARCHIVED_TO_CURRENT)
    for current in ARCHIVED_TO_CURRENT.values():
        assert current not in retired, current


def test_the_gate_body_does_not_name_retired_spellings() -> None:
    """Retired names must live only in the migration map, not in gate logic."""
    import inspect

    source = inspect.getsource(archive_health)
    for retired in ARCHIVED_TO_CURRENT:
        # The inverted map and its import are the only permitted mentions.
        assert f'"{retired}"' not in source, retired
