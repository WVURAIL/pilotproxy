# coding=utf-8
"""Key names inside the archived per-pilot survey products.

The 2020--2026 survey archive was written under the detector's original
measurement vocabulary; those npz keys are immutable historical data. The
current vocabulary renamed the measurements everywhere else, and
``scripts/check_current_measurement_vocabulary.py`` keeps the retired
spellings off every current surface -- except this module, its single
designated home. Read archived products through these constants; current
products use the current vocabulary directly and never need this module.

``measurement`` is the one reader both kinds of product go through: it takes
the current name and falls back to the archived spelling, so analysis code
written against the current vocabulary runs unchanged on the 2020--2026
archive and on the 2026-09 rebuild.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ARCHIVED_COARSE_POWER_RATIO = "fstat_raw"
ARCHIVED_FINE_POWER_RATIO = "fstat_fine"
ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB = "fstat_level_db"
ARCHIVED_PILOT_EXCESS_DB = "pnr_bin_db"
ARCHIVED_DATA_SHELF_SNR_DB = "snr_shelf_db"
ARCHIVED_FINE_NULL_BULK_EXCEEDANCE_FRACTION = "fine_nd_flag_rate"
ARCHIVED_NORMALIZED_PILOT_EXCESS = "pilot_excess_corrected"
ARCHIVED_REFERENCE_NORM_SUM_SQ = "ref_norm_sum_sq"


# Migration map, not an alias table. 5fb0e32 ("Use exact power-ratio
# measurement coordinates") renamed these measurements in the writer while the
# 2020--2026 archive kept the left-hand spellings. Code names measurements in
# the current vocabulary throughout; this map resolves the archived spelling
# when an archived product is read, and is deletable once that archive has
# been reprocessed. Nothing else should reference the retired names.
ARCHIVED_TO_CURRENT = {
    ARCHIVED_COARSE_POWER_RATIO: "coarse_power_ratio",
    ARCHIVED_FINE_POWER_RATIO: "fine_power_ratio",
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB: "normalized_coarse_power_ratio_db",
    ARCHIVED_PILOT_EXCESS_DB: "pilot_excess_db",
    ARCHIVED_DATA_SHELF_SNR_DB: "estimated_data_shelf_snr_db",
    # The weight-norm correction: "mu0" in the archive, and the exact rational
    # 2*target_norm_sq/reference_norm_sum_sq everywhere since.
    "mu0": "null_power_ratio",
    ARCHIVED_REFERENCE_NORM_SUM_SQ: "reference_norm_sum_sq",
}

# current spelling -> the spelling the 2020--2026 archive used.
CURRENT_TO_ARCHIVED = {current: archived for archived, current in ARCHIVED_TO_CURRENT.items()}


def measurement(product: Mapping[str, Any], name: str) -> np.ndarray:
    """Read one measurement by its current name from a current or archived product.

    The current key is preferred; a product that predates the rename is read
    through the migration map. Anything still missing raises ``KeyError`` so
    a caller cannot mistake an absent measurement for a zero one. ``mu0`` and
    the fine power ratio are not stored by current products at all; use
    ``product_contract.null_power_ratio_of`` and ``fine_power_ratio_of``.
    """
    if name in product:
        return np.asarray(product[name])
    archived = CURRENT_TO_ARCHIVED.get(name)
    if archived is not None and archived in product:
        return np.asarray(product[archived])
    raise KeyError(f"product carries neither {name!r} nor its archived spelling")
