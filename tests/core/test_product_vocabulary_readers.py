# coding=utf-8
"""Reading measurements from current and archived products through one vocabulary."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.archived_product_keys import (
    ARCHIVED_COARSE_POWER_RATIO,
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_REFERENCE_NORM_SUM_SQ,
    ARCHIVED_TO_CURRENT,
    CURRENT_TO_ARCHIVED,
    measurement,
)
from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)
from pilot_proxy.product_contract import (
    CurrentProductContractError,
    fine_power_ratio_of,
    null_power_ratio_of,
)

COARSE = np.asarray([[1.1], [0.9], [4.0]])


def test_the_migration_map_inverts_cleanly() -> None:
    assert len(CURRENT_TO_ARCHIVED) == len(ARCHIVED_TO_CURRENT)
    assert CURRENT_TO_ARCHIVED["coarse_power_ratio"] == ARCHIVED_COARSE_POWER_RATIO
    assert "frame_index" not in CURRENT_TO_ARCHIVED


def test_measurement_prefers_the_current_key() -> None:
    product = {"coarse_power_ratio": COARSE, ARCHIVED_COARSE_POWER_RATIO: COARSE + 1.0}
    np.testing.assert_array_equal(measurement(product, "coarse_power_ratio"), COARSE)


def test_measurement_resolves_the_archived_spelling() -> None:
    product = {ARCHIVED_COARSE_POWER_RATIO: COARSE}
    np.testing.assert_array_equal(measurement(product, "coarse_power_ratio"), COARSE)


def test_measurement_fails_closed_on_a_missing_field() -> None:
    with pytest.raises(KeyError, match="coarse_power_ratio"):
        measurement({"valid": np.ones((3, 1), dtype=np.uint8)}, "coarse_power_ratio")


def test_measurement_reads_an_npz_of_either_vocabulary(tmp_path) -> None:
    current = tmp_path / "current.npz"
    archived = tmp_path / "archived.npz"
    np.savez(current, coarse_power_ratio=COARSE)
    np.savez(archived, **{ARCHIVED_COARSE_POWER_RATIO: COARSE})
    for path in (current, archived):
        with np.load(path, allow_pickle=False) as product:
            np.testing.assert_array_equal(measurement(product, "coarse_power_ratio"), COARSE)


def test_null_power_ratio_is_derived_under_both_spellings() -> None:
    current = {"target_norm_sq": np.asarray([6326]), "reference_norm_sum_sq": np.asarray([12791])}
    archived = {"target_norm_sq": np.asarray([6326]), ARCHIVED_REFERENCE_NORM_SUM_SQ: np.asarray([12791])}
    assert null_power_ratio_of(current) == null_power_ratio_of(archived) == 2.0 * 6326 / 12791


def _exact_terms(n_frames: int = 2, bins: int = 4) -> np.ndarray:
    terms = np.zeros((n_frames, 3, bins), dtype=np.uint64)
    terms[:, WEIGHT_TERM_TARGET] = np.asarray([[10, 20, 30, 0], [7, 7, 7, 7]], dtype=np.uint64)
    terms[:, WEIGHT_TERM_REF_LOWER] = np.asarray([[5, 5, 5, 0], [3, 4, 0, 1]], dtype=np.uint64)
    terms[:, WEIGHT_TERM_REF_UPPER] = np.asarray([[5, 15, 25, 0], [4, 3, 0, 6]], dtype=np.uint64)
    return terms


def test_fine_power_ratio_is_formed_from_the_exact_terms() -> None:
    ratio = fine_power_ratio_of({"fine_power_u64": _exact_terms()})
    assert ratio.dtype == np.float64 and ratio.shape == (2, 4)
    # ratio is 2 S_t / (S_l + S_u), and a zero denominator gives 0.0 as in fine_reduce
    np.testing.assert_allclose(ratio[0], [2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(ratio[1], [2.0, 2.0, 0.0, 2.0])


def test_fine_power_ratio_reads_the_stored_archived_ratio() -> None:
    stored = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(fine_power_ratio_of({ARCHIVED_FINE_POWER_RATIO: stored}), stored)
    np.testing.assert_array_equal(fine_power_ratio_of({"fine_power_ratio": stored}), stored)


def test_fine_power_ratio_prefers_the_exact_terms_over_a_stored_ratio() -> None:
    product = {"fine_power_u64": _exact_terms(), ARCHIVED_FINE_POWER_RATIO: np.ones((2, 4), dtype=np.float32)}
    np.testing.assert_allclose(fine_power_ratio_of(product)[0], [2.0, 2.0, 2.0, 0.0])


def test_fine_power_ratio_refuses_a_product_without_fine_terms() -> None:
    with pytest.raises(CurrentProductContractError, match="no fine terms"):
        fine_power_ratio_of({"fine_power_u64": np.zeros((3, 0, 0), dtype=np.uint64)})
    with pytest.raises(CurrentProductContractError, match="shape"):
        fine_power_ratio_of({"fine_power_u64": np.zeros((3, 2, 4), dtype=np.uint64)})
    with pytest.raises(CurrentProductContractError, match="neither"):
        fine_power_ratio_of({"coarse_power_ratio": COARSE})
