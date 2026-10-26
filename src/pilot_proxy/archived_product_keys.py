# coding=utf-8
"""Key names in the archived per-pilot survey products.

The 2020-2026 survey archive was written with the detector's original
measurement names, and these npz keys are fixed historical data. The
measurements were renamed everywhere else, and
``scripts/check_current_measurement_vocabulary.py`` rejects the retired
spellings on every current surface except this module. Archived products are
read through these constants. Current products use the current names directly
and do not need this module.

Both kinds of product are read through ``measurement``. It takes the current
name and falls back to the archived spelling, so analysis code written with
the current names runs unchanged on the 2020-2026 archive and on the 2026-09
rebuild.
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


# Commit 5fb0e32 ("Use exact power-ratio measurement coordinates") renamed
# these measurements in the writer, and the 2020-2026 archive kept the
# left-hand spellings. Code uses the current names throughout. This migration
# map is used only to read an archived product, and it can be deleted once that
# archive has been reprocessed. No other module should reference the retired names.
ARCHIVED_TO_CURRENT = {
    ARCHIVED_COARSE_POWER_RATIO: "coarse_power_ratio",
    ARCHIVED_FINE_POWER_RATIO: "fine_power_ratio",
    ARCHIVED_NORMALIZED_COARSE_POWER_RATIO_DB: "normalized_coarse_power_ratio_db",
    ARCHIVED_PILOT_EXCESS_DB: "pilot_excess_db",
    ARCHIVED_DATA_SHELF_SNR_DB: "estimated_data_shelf_snr_db",
    # The weight-norm correction is "mu0" in the archive. Current code computes
    # it as the exact rational 2*target_norm_sq/reference_norm_sum_sq.
    "mu0": "null_power_ratio",
    ARCHIVED_REFERENCE_NORM_SUM_SQ: "reference_norm_sum_sq",
}

# current spelling to the spelling of the 2020-2026 archive
CURRENT_TO_ARCHIVED = {current: archived for archived, current in ARCHIVED_TO_CURRENT.items()}


def measurement(product: Mapping[str, Any], name: str) -> np.ndarray:
    """Read one measurement by its current name from a current or archived product.

    The current key is read when present. A product written before the rename
    is read through the migration map. A measurement that is still missing
    raises ``KeyError``, so a caller cannot mistake an absent measurement for a
    zero one. Current products do not store ``mu0`` or the fine power ratio,
    and these are read with ``product_contract.null_power_ratio_of`` and
    ``fine_power_ratio_of``.
    """
    if name in product:
        return np.asarray(product[name])
    archived = CURRENT_TO_ARCHIVED.get(name)
    if archived is not None and archived in product:
        return np.asarray(product[archived])
    raise KeyError(f"product carries neither {name!r} nor its archived spelling")
