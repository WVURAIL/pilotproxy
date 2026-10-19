"""The product view: one coordinate system for v6 products and the v5 products
of record, read through ``open_product``.

Kernel core 3.0.0 defines the frame powers as the all-bin sums of the fine
powers. A v6 product stores them so, and the reader verifies it; a v5 product
stored row-domain powers, and the reader replaces them by the sums of the fine
powers every v5 product retains.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_pilotproxy_v5 import _write_product
from pilot_proxy.product_contract import (
    FINE_DERIVED_FRAME_COLUMNS,
    coarse_columns_from_fine,
    frame_powers_from_fine,
    pilot_excess_from_fine_powers,
)
from pilot_proxy.products.npzio import load_npz
from pilot_proxy.products.reader import (
    FRAME_POWER_DEFINITION_V5,
    FRAME_POWER_DEFINITION_V6,
    PRODUCT_SCHEMA_TOKEN,
    V5_SCHEMA_TOKEN,
    Product,
    ProductContractError,
    ProductView,
    coarse_reject_mask,
    is_current_product,
    open_product,
)


def test_a_current_product_opens_with_its_frames_and_decisions(tmp_path):
    path = _write_product(tmp_path / "552.npz", 33, revision=6)
    values = load_npz(path)
    view = open_product(values)
    assert isinstance(view, ProductView) and view.is_current and view.schema == PRODUCT_SCHEMA_TOKEN
    assert view.frame_power_definition == FRAME_POWER_DEFINITION_V6
    assert is_current_product(values)
    n = view.valid.size
    for column in (view.rejected, view.shelf_db, view.statistic, view.frame_unit_index):
        assert column.shape == (n,)
    # at eta = 1 the exact integer decision is the stored survey flag
    np.testing.assert_array_equal(view.rejected_at_multiplier(1.0), view.rejected)
    np.testing.assert_array_equal(view.rejected, values["reject_mask"][:, 0].astype(bool))
    # a larger multiplier never rejects more
    assert not (view.rejected_at_multiplier(1.5) & ~view.rejected_at_multiplier(1.0)).any()
    with Product(path) as product:
        assert product.geometry.physical_channel == view.physical_channel == 33
        np.testing.assert_array_equal(product.rejected, view.rejected)


def test_a_v5_product_is_read_through_its_fine_sums(tmp_path):
    path = _write_product(tmp_path / "552.npz", 33, revision=5)
    values = load_npz(path)
    fine = np.asarray(values["fine_power_u64"])
    view = open_product(values)
    assert view.schema == V5_SCHEMA_TOKEN and not view.is_current
    assert not is_current_product(values)
    assert view.frame_power_definition == FRAME_POWER_DEFINITION_V5 == (
        "recomputed_from_fine_power_u64 (v5 stored row-domain powers ignored)"
    )

    # The frame powers are the exact all-bin sums, term by term (Python ints),
    # and they are not the stored v5 row-domain powers.
    for term, field in enumerate(("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64")):
        expected = [sum(int(v) for v in fine[f, term]) for f in range(fine.shape[0])]
        assert [int(v) for v in view.columns[field][:, 0]] == expected
        assert [int(v) for v in frame_powers_from_fine(values)[term]] == expected
        assert not np.array_equal(view.columns[field], values[field])
    assert tuple(view.columns) == FINE_DERIVED_FRAME_COLUMNS

    # Every coordinate follows the fine sums: the excess, the survey flag, Q.
    target = view.columns["p_target_u64"][:, 0].astype(np.float64)
    reference = view.columns["p_ref_sum_u64"][:, 0].astype(np.float64)
    nt = int(values["target_norm_sq"][0])
    nrs = int(values["reference_norm_sum_sq"][0])
    mu0 = 2.0 * nt / nrs
    np.testing.assert_allclose(view.statistic, 2.0 * target / reference / mu0, rtol=1e-14)
    np.testing.assert_allclose(view.normalized_excess, 2.0 * target / reference / mu0 - 1.0,
                               rtol=0, atol=1e-14)
    np.testing.assert_array_equal(view.normalized_excess, pilot_excess_from_fine_powers(values))
    exact_reject = [
        int(t) * nrs > nt * int(r)
        for t, r in zip(view.columns["p_target_u64"][:, 0], view.columns["p_ref_sum_u64"][:, 0])
    ]
    assert view.rejected.tolist() == exact_reject
    np.testing.assert_array_equal(coarse_reject_mask(values), view.rejected)

    # The lazy Product serves the same replacement for the coarse columns.
    with Product(path) as product:
        np.testing.assert_array_equal(
            product.frame_column("normalized_pilot_excess"), view.normalized_excess
        )
        np.testing.assert_array_equal(product.frame_column("p_target_u64"),
                                      view.columns["p_target_u64"][:, 0])
        assert not np.array_equal(product.frame_column("normalized_pilot_excess"),
                                  values["normalized_pilot_excess"][:, 0])


def test_a_v5_product_without_fine_powers_cannot_be_read(tmp_path):
    values = dict(load_npz(_write_product(tmp_path / "552.npz", 33, revision=5)))
    del values["fine_power_u64"]
    with pytest.raises(ProductContractError, match="fine_power_u64"):
        open_product(values)


def test_a_v6_product_must_store_the_fine_sums(tmp_path):
    values = dict(load_npz(_write_product(tmp_path / "552.npz", 33, revision=6)))
    values["p_target_u64"] = values["p_target_u64"] + np.uint64(1)
    with pytest.raises(ProductContractError, match="fine-power sums"):
        open_product(values)

    values = dict(load_npz(_write_product(tmp_path / "553.npz", 33, revision=6)))
    values["normalized_pilot_excess"] = values["normalized_pilot_excess"] + 1e-9
    with pytest.raises(ProductContractError, match="normalized_pilot_excess"):
        open_product(values)


def test_coarse_columns_from_fine_on_v6_equal_the_stored_columns(tmp_path):
    values = load_npz(_write_product(tmp_path / "552.npz", 33, revision=6))
    derived = coarse_columns_from_fine(values)
    for name in FINE_DERIVED_FRAME_COLUMNS:
        np.testing.assert_array_equal(derived[name], values[name])
    rows = np.asarray([3, 0, 5])
    selected = coarse_columns_from_fine(values, rows)
    for name in FINE_DERIVED_FRAME_COLUMNS:
        np.testing.assert_array_equal(selected[name], derived[name][rows])
    np.testing.assert_array_equal(
        pilot_excess_from_fine_powers(values, rows),
        derived["normalized_pilot_excess"][rows, 0],
    )


def test_the_view_refuses_what_the_contract_does_not_admit(tmp_path):
    values = dict(load_npz(_write_product(tmp_path / "552.npz", 33, revision=6)))
    with pytest.raises(TypeError):
        open_product(values).rejected_at_multiplier(True)
    with pytest.raises(ValueError):
        open_product(values).rejected_at_multiplier(0.0)
    del values["p_target_u64"]
    with pytest.raises(ProductContractError):
        open_product(values)
