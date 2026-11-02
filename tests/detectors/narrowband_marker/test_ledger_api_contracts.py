"""Public ledger helpers preserve rank conventions and typed anchor records."""

import json
import math

import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import anchors, scores
from pilot_proxy.fine_decision import fine_required_multiplier_q16
from pilot_proxy.json_utils import json_dumps_strict
from pilot_proxy.products.reader import Product
from test_results_archive_anchors import _product


def test_zero_based_detector_rank_selects_the_expected_one_based_boundary():
    powers = np.full((3, 256), 100, dtype=np.uint64)
    powers[0, [0, 2, 4]] = [300, 100, 200]
    powers[0, 10] = 400
    bulk = np.zeros(256, dtype=bool)
    bulk[[0, 2, 4]] = True
    boundaries = scores.required_eta_q16_by_rank(
        powers, anchor_bin=10, designated_half_width=0, bulk_mask=bulk,
    )
    # Sorted background ratios are 1, 2, 3; the designated ratio is 4.
    # Ceil(65536 * 4 / background) gives these exact keep boundaries.
    assert boundaries == (262144, 131072, 87382)
    for index, expected in enumerate((262144, 131072, 87382)):
        rho = scores.rank_from_index(np.int64(index))
        assert rho == index + 1
        assert boundaries[rho - 1] == expected
        result = fine_required_multiplier_q16(
            powers, anchor_bin=10, designated_half_width=0,
            bulk_mask=bulk, cfar_rank=index,
        )
        assert result.valid is True
        assert result.multiplier_q16 == expected


@pytest.mark.parametrize("value,error", [(True, TypeError), (1.5, TypeError),
                                        ("1", TypeError), (-1, ValueError)])
def test_rank_conversion_refuses_values_that_would_change_rank(value, error):
    with pytest.raises(error, match="cfar_rank"):
        scores.rank_from_index(value)


@pytest.mark.parametrize("empty", [False, True])
def test_anchor_record_keeps_scalar_types_and_missing_estimates(tmp_path, empty):
    path = _product(tmp_path, on_lines=[(65, 5.0)], static_lines=[(200, 50.0)])
    with Product(path) as product:
        mask = np.zeros(product.n_frames, dtype=bool) if empty else product.selected
        result = anchors.anchor(product, mask, "ledger", replicates=0)
    record = anchors.anchor_record(result)
    assert tuple(record) == anchors.ANCHOR_COLUMNS
    assert record["channel"] == 36 and type(record["channel"]) is int
    assert record["freq_id"] == 506 and type(record["freq_id"]) is int
    assert record["aliased_out_of_window"] is False
    assert record["label"] == "ledger"
    assert record["frames_masked"] == (0 if empty else 120)
    assert record["frames_on"] == (0 if empty else 60)
    assert record["frames_quiet"] == (0 if empty else 60)
    assert record["anchor_bin"] == (-1 if empty else 65)
    assert record["designated_bins"] == ("" if empty else "63;64;65;66;67")
    assert record["status"] == ("empty" if empty else "ok")
    assert math.isnan(record["boot_mode_mass"])
    if empty:
        assert math.isnan(record["anchor_contrast"])
    else:
        assert record["anchor_contrast"] == 4.0
        assert type(record["anchor_contrast"]) is float
        assert record["bulk_size"] == 126
    assert not any(isinstance(value, (np.ndarray, np.generic)) for value in record.values())
    # The in-memory ledger keeps NaN; public strict JSON represents it as null.
    exported = json.loads(json_dumps_strict(record))
    assert exported["boot_mode_mass"] is None
    assert exported["designated_bins"] == record["designated_bins"]
    assert exported["anchor_contrast"] == (None if empty else 4.0)
