"""Review M6 and T13 review H1: a product that does not declare the v5 schema is refused by the reader."""
from __future__ import annotations

import numpy as np
import pytest

from pilot_proxy.products.reader import ProductContractError, coarse_reject_mask, open_product

REFUSED = "pre-v5 products are not read"


def _pre_v5(n=40):
    return {"valid": np.ones((n, 1), dtype=np.uint8), "reject_mask": np.zeros((n, 1), dtype=np.uint8),
            "physical_channel": np.array([35], dtype=np.int32), "freq_id": np.array([521], dtype=np.int64)}


def test_a_pre_v5_product_is_refused():
    with pytest.raises(ProductContractError, match=REFUSED):
        open_product(_pre_v5())
    with pytest.raises(ProductContractError, match=REFUSED):
        coarse_reject_mask(_pre_v5())
