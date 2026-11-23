"""Public ledger helpers preserve rank conventions."""

import numpy as np
import pytest

from pilot_proxy.detectors.narrowband_marker import scores
from pilot_proxy.fine_decision import fine_required_multiplier_q16


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
