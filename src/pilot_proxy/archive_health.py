# coding=utf-8
"""Frame-health gate for per-pilot CHIME products.

:func:`evaluate_frame_health` marks the frames that cannot be used and records
every reason that excludes each one. The product reader and the
characterization frame exports apply it.

The two byte conventions are kept distinct. CHIME stores (-8, -8) as native
offset-binary 0x00, and lossless detector repacking writes the same sample as
two's-complement 0x88.

The archive health audit that also used this module (summaries, exclusion
ledgers, corrected spectra and diagnostic plots) was removed with the v6
archive pipeline; the tag pre-purge-2026-11-23 holds it.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pilot_proxy.archived_product_keys import measurement
from pilot_proxy.product_contract import (
    FINE_DERIVED_FRAME_COLUMNS,
    CurrentProductContractError,
    coarse_columns_from_fine,
)

FRAME_HEALTH_GATE_SCHEMA_VERSION = "pilotproxy_archive_frame_health_gate_v2"

REASON_DETECTOR_INVALID = "detector_invalid"
REASON_DETECTOR_POWERS_ALL_ZERO = "detector_powers_all_zero"
REASON_BASEBAND_NONFINITE = "baseband_power_nonfinite"
REASON_BASEBAND_OUT_OF_BOUNDS = "baseband_power_out_of_complex_int4_bounds"
REASON_BASEBAND_CEILING = "baseband_power_at_negative_full_scale_ceiling"
REASON_FILL_ABOVE_ONE_PERCENT = "fill_samples_above_one_percent"
REASON_COARSE_NONFINITE = "coarse_power_ratio_nonfinite"
REASON_FINE_NONFINITE = "fine_power_ratio_nonfinite"
REASON_FINE_NEGATIVE = "fine_power_ratio_negative"

REASON_DEFINITIONS: dict[str, str] = {
    REASON_DETECTOR_INVALID: (
        "The stored detector valid bit is false; the analyzer omitted this "
        "frame from both integrated spectra."
    ),
    REASON_DETECTOR_POWERS_ALL_ZERO: (
        "Both stored detector power terms are zero."
    ),
    REASON_BASEBAND_NONFINITE: (
        "The stored frame-mean baseband power is not finite."
    ),
    REASON_BASEBAND_OUT_OF_BOUNDS: (
        "The stored mean power lies outside the closed [0, 128] range of "
        "decoded complex int4 samples."
    ),
    REASON_BASEBAND_CEILING: (
        "The mean power is exactly 128, its mathematical maximum.  Every "
        "decoded sample is therefore (-8, -8): native CHIME offset-binary "
        "raw byte 0x00, equivalently detector-input/post-repack "
        "two's-complement byte 0x88."
    ),
    REASON_FILL_ABOVE_ONE_PERCENT: (
        "More than 1% of the frame's samples are the archive's 0x00 "
        "missing-data fill (fill_sample_count > railed_sample_total / 100).  "
        "Only products that store the fill count can trigger it."
    ),
    REASON_COARSE_NONFINITE: (
        "A detector-valid row has a non-finite coarse power ratio."
    ),
    REASON_FINE_NONFINITE: (
        "A detector-valid row contains a non-finite fine power ratio."
    ),
    REASON_FINE_NEGATIVE: (
        "A detector-valid row contains a negative fine power ratio."
    ),
}

ENCODING_INTERPRETATION = (
    "baseband_power_linear is evaluated after decoding the native CHIME "
    "offset-binary nibbles.  A frame mean of exactly 128 reaches the maximum "
    "possible sample power (-8)^2 + (-8)^2, so every native raw byte is 0x00. "
    "The losslessly repacked detector-input representation of that same "
    "sample is two's-complement 0x88."
)

class ArchiveHealthError(ValueError):
    """The archive cannot be interpreted safely under the health gate."""


@dataclass(frozen=True)
class FrameHealthResult:
    """Frame inclusion mask and every independently triggered reason mask."""

    include: np.ndarray
    reasons: Mapping[str, np.ndarray]

    @property
    def excluded(self) -> np.ndarray:
        return ~self.include

    @property
    def reason_counts(self) -> dict[str, int]:
        return {
            code: int(np.count_nonzero(mask))
            for code, mask in self.reasons.items()
            if np.any(mask)
        }


def _array(product: Mapping[str, Any], name: str) -> np.ndarray:
    """Read one field, naming it in the current vocabulary.

    Archived products predate the measurement rename, so a current name that is
    absent is resolved once through the migration map. Anything still missing
    fails closed.
    """
    try:
        return measurement(product, name)
    except KeyError:
        raise ArchiveHealthError(
            f"archive product is missing required field {name!r}"
        ) from None


# The detector columns (frame powers, ratio floats, reject_mask, valid) of a
# product that stores its exact fine powers are read through
# product_contract.coarse_columns_from_fine. Since kernel core 3.0.0, the frame
# powers are the all-bin sums of fine_power_u64. In a v6 product these sums
# equal the stored columns. In a v5 product of record they replace the stored
# row-domain powers, so the gate describes the products by the current
# definition. Archived products without fine powers
# are read from their stored columns. The columns are computed once per open
# product and cached through a weak reference, so a closed product releases
# them.
_FINE_DERIVED_COLUMNS = frozenset(FINE_DERIVED_FRAME_COLUMNS)
_FINE_COLUMN_CACHE: dict[int, tuple[Any, dict[str, np.ndarray]]] = {}


def _fine_derived_columns(product: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    if "fine_power_u64" not in product:
        return None
    key = id(product)
    cached = _FINE_COLUMN_CACHE.get(key)
    if cached is not None and cached[0]() is product:
        return cached[1]
    try:
        columns = {
            name: values[:, 0]
            for name, values in coarse_columns_from_fine(product).items()
        }
    except (CurrentProductContractError, KeyError) as exc:
        raise ArchiveHealthError(
            f"cannot form the frame powers from fine_power_u64: {exc}"
        ) from exc
    try:
        ref = weakref.ref(product, lambda _ref, k=key: _FINE_COLUMN_CACHE.pop(k, None))
    except TypeError:  # a plain dict cannot be weakly referenced, so it is not cached
        return columns
    _FINE_COLUMN_CACHE[key] = (ref, columns)
    return columns


def _frame_vector(
    product: Mapping[str, Any], name: str, n_frames: int
) -> np.ndarray:
    if name in _FINE_DERIVED_COLUMNS:
        derived = _fine_derived_columns(product)
        if derived is not None:
            values = derived[name]
            if values.shape != (n_frames,):
                raise ArchiveHealthError(
                    f"{name} from fine_power_u64 has {values.shape[0]} frames; "
                    f"expected {n_frames}"
                )
            return values
    return _stored_frame_vector(product, name, n_frames)


def _stored_frame_vector(
    product: Mapping[str, Any], name: str, n_frames: int
) -> np.ndarray:
    """Return the column as stored, without re-deriving it from the fine powers."""
    arr = _array(product, name)
    if arr.shape == (n_frames, 1):
        return arr[:, 0]
    if arr.shape == (n_frames,):
        return arr
    raise ArchiveHealthError(
        f"{name} must have shape ({n_frames},) or ({n_frames}, 1); got {arr.shape}"
    )


def _binary_frame_vector(
    product: Mapping[str, Any], name: str, n_frames: int, *, stored: bool = False
) -> np.ndarray:
    reader = _stored_frame_vector if stored else _frame_vector
    values = reader(product, name, n_frames)
    if not np.all((values == 0) | (values == 1)):
        raise ArchiveHealthError(f"{name} must contain only exact 0/1 values")
    return values.astype(bool, copy=False)


def _validate_frame_identity(product: Mapping[str, Any], n_frames: int) -> None:
    frame_index = _frame_vector(product, "frame_index", n_frames)
    if not np.array_equal(frame_index, np.arange(n_frames, dtype=frame_index.dtype)):
        raise ArchiveHealthError("frame_index must be the exact positional sequence 0..N-1")
    unit_index = _frame_vector(product, "frame_unit_index", n_frames)
    frame_in_unit = _frame_vector(product, "frame_in_unit", n_frames)
    if unit_index.dtype.kind not in "iu" or frame_in_unit.dtype.kind not in "iu":
        raise ArchiveHealthError("frame identity arrays must use integer dtypes")
    source_keys = _array(product, "source_event_keys").reshape(-1)
    unit_order = _array(product, "unit_order").reshape(-1)
    if source_keys.size != unit_order.size or source_keys.size == 0:
        raise ArchiveHealthError(
            "source_event_keys and unit_order must be aligned non-empty unit axes"
        )
    if np.any(unit_index < 0) or np.any(unit_index >= source_keys.size):
        raise ArchiveHealthError("frame_unit_index contains an out-of-range unit row")
    if np.any(frame_in_unit < 0):
        raise ArchiveHealthError("frame_in_unit must be non-negative")


def evaluate_frame_health(product: Mapping[str, Any]) -> FrameHealthResult:
    """Evaluate the versioned health gate, excluding any unsafe row.

    Required fields and frame identities are validated before a mask is
    returned.  A malformed or unclassifiable product raises instead of
    silently treating the row as healthy.
    """

    frame_index = _array(product, "frame_index")
    if frame_index.ndim != 1:
        raise ArchiveHealthError("frame_index must be one-dimensional")
    n_frames = int(frame_index.size)
    if n_frames == 0:
        raise ArchiveHealthError("archive products with zero frames are not auditable")
    _validate_frame_identity(product, n_frames)

    valid = _binary_frame_vector(product, "valid", n_frames)
    _binary_frame_vector(product, "reject_mask", n_frames)
    p_target = _frame_vector(product, "p_target_u64", n_frames)
    p_reference = _frame_vector(product, "p_ref_sum_u64", n_frames)
    if p_target.dtype.kind not in "iu" or p_reference.dtype.kind not in "iu":
        raise ArchiveHealthError("stored detector powers must use integer dtypes")
    baseband_power = np.asarray(
        _frame_vector(product, "baseband_power_linear", n_frames), dtype=np.float64
    )
    coarse = np.asarray(
        _frame_vector(product, "coarse_power_ratio", n_frames),
        dtype=np.float64,
    )
    # The fine measurement is exact uint64 terms in current products and a
    # float32 ratio in archived ones. Integers admit neither a non-finite nor a
    # negative value, so those two checks apply only to the archived form.
    if "fine_power_u64" in product:
        fine_terms = _array(product, "fine_power_u64")
        if (
            fine_terms.ndim != 3
            or fine_terms.shape[0] != n_frames
            or fine_terms.shape[2] <= 0
        ):
            raise ArchiveHealthError(
                "fine_power_u64 must have shape (N, terms, B), B > 0; "
                f"got {fine_terms.shape}"
            )
        if fine_terms.dtype.kind != "u":
            raise ArchiveHealthError(
                "fine_power_u64 must use an unsigned integer dtype; got "
                f"{fine_terms.dtype}"
            )
        fine_nonfinite = np.zeros(n_frames, dtype=bool)
        fine_negative = np.zeros(n_frames, dtype=bool)
        # Current products count the archive's 0x00 missing-data fill per
        # frame. Archived products have no fill count, so the rule is empty
        # for them. fill > total // 100 is fill > 1% of total in integers.
        fill = _frame_vector(product, "fill_sample_count", n_frames)
        total = _frame_vector(product, "railed_sample_total", n_frames)
        if fill.dtype.kind != "u" or total.dtype.kind != "u":
            raise ArchiveHealthError(
                "fill_sample_count and railed_sample_total must use unsigned "
                "integer dtypes"
            )
        fill_above_one_percent = fill > total // np.uint64(100)
    else:
        fine = _array(product, "fine_power_ratio")
        if fine.ndim != 2 or fine.shape[0] != n_frames or fine.shape[1] <= 0:
            raise ArchiveHealthError(
                f"fine_power_ratio must have shape (N, B), B > 0; got {fine.shape}"
            )
        fine_nonfinite = ~np.all(np.isfinite(fine), axis=1)
        fine_negative = np.any(fine < 0.0, axis=1)
        fill_above_one_percent = np.zeros(n_frames, dtype=bool)

    reasons: dict[str, np.ndarray] = {
        REASON_DETECTOR_INVALID: ~valid,
        REASON_DETECTOR_POWERS_ALL_ZERO: (p_target == 0) & (p_reference == 0),
        REASON_BASEBAND_NONFINITE: ~np.isfinite(baseband_power),
        REASON_BASEBAND_OUT_OF_BOUNDS: (
            np.isfinite(baseband_power)
            & ((baseband_power < 0.0) | (baseband_power > 128.0))
        ),
        REASON_BASEBAND_CEILING: baseband_power == 128.0,
        REASON_FILL_ABOVE_ONE_PERCENT: fill_above_one_percent,
        REASON_COARSE_NONFINITE: valid & ~np.isfinite(coarse),
        REASON_FINE_NONFINITE: valid & fine_nonfinite,
        REASON_FINE_NEGATIVE: valid & fine_negative,
    }
    excluded = np.zeros(n_frames, dtype=bool)
    for mask in reasons.values():
        excluded |= mask
    return FrameHealthResult(include=~excluded, reasons=reasons)
