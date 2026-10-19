# coding=utf-8
"""Current per-pilot product and decision-method identities.

Schema identity, scientific measurement, and rejection policy are separate
coordinates.  Keeping them explicit prevents a diagnostic fine spectrum or an
implemented kernel entry point from being mistaken for the active mask used by
an archive product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from pilot_proxy.archived_product_keys import (
    ARCHIVED_FINE_POWER_RATIO,
    ARCHIVED_REFERENCE_NORM_SUM_SQ,
)
from pilot_proxy.atsc_channels import (
    ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    ATSC_UHF_MIN_PHYSICAL_CHANNEL,
)
from pilot_proxy.detector_contract import (
    COARSE_POWER_DEFINITION,
    NORMALIZED_POSITIVE_EXCESS_MASK_RULE,
    normalized_positive_excess,
)
from pilot_proxy.detector_reference import REFERENCE_WEIGHT_TERMS
from pilot_proxy.dtv_units import (
    normalized_coarse_power_ratio_to_db,
    normalized_pilot_excess_to_db,
    pilot_excess_db_to_data_shelf_snr_db,
    power_terms_to_coarse_power_ratio,
    power_terms_to_normalized_coarse_power_ratio,
)
from pilot_proxy.fine_reduction import (
    WEIGHT_TERM_REF_LOWER,
    WEIGHT_TERM_REF_UPPER,
    WEIGHT_TERM_TARGET,
)

PER_PILOT_PRODUCT_SCHEMA_NAME = "pilotproxy_per_pilot_product"
# v2 retains the exact uint64 fine-power terms and the lower/upper
# reference split. v1 products collapsed both, which made an exact Q16
# fine replay impossible after the fact; they are rejected, not migrated.
# v3 adds the per-frame PSD the analyzer already computed and discarded,
# so a later pass can apply a new frame mask, window, or threshold to the
# spectra instead of being limited to the two archived accumulators.
# v4 added the per-frame railed-sample counts; saturation cannot be
# recovered later because no raw samples are stored.
# v5 splits the archive's 0x00 missing-data fill signature out of the
# railed count (fill_sample_count), so incomplete baseband is no longer
# recorded as clipping. No v4 cohort was ever produced; v4 snapshots are
# rejected like every other development snapshot.
# v6 (kernel core 3.0.0) keeps every field name and redefines the frame
# powers: p_target_u64, p_ref_lower_u64 and p_ref_upper_u64 are the exact
# all-bin sums of fine_power_u64 (P_r = sum_f S_r[f]), and every derived
# column (the ratio floats, reject_mask, valid) is formed from those sums as
# before. Fine powers are mandatory for every in-band pilot. The v5 products of
# record stored the row-domain sums of |z|^2 instead; they stay on disk
# unedited, and readers recompute their frame powers from their stored
# fine_power_u64 (coarse_columns_from_fine), never from the stored v5 powers.
PER_PILOT_PRODUCT_SCHEMA_REVISION = 6
PER_PILOT_PRODUCT_SCHEMA_TOKEN = (
    f"{PER_PILOT_PRODUCT_SCHEMA_NAME}_v{PER_PILOT_PRODUCT_SCHEMA_REVISION}"
)
PER_PILOT_PRODUCT_V5_SCHEMA_REVISION = 5
PER_PILOT_PRODUCT_V5_SCHEMA_TOKEN = (
    f"{PER_PILOT_PRODUCT_SCHEMA_NAME}_v{PER_PILOT_PRODUCT_V5_SCHEMA_REVISION}"
)

# Every per-frame array the analyzer writes (length n_frames along axis 0),
# in one authoritative place: event-keyed alignment in combine, the
# resume/compare gate, and the deep audit all treat per-frame data by this
# list, so a schema bump that adds a per-frame field must extend it here or
# the field silently escapes alignment and comparison.
PER_FRAME_PRODUCT_KEYS = (
    "frame_index",
    "p_target_u64",
    "p_ref_sum_u64",
    "p_ref_lower_u64",
    "p_ref_upper_u64",
    "coarse_power_ratio",
    "normalized_coarse_power_ratio_db",
    "pilot_excess_db",
    "estimated_data_shelf_snr_db",
    "normalized_pilot_excess",
    "reject_mask",
    "valid",
    "baseband_power_linear",
    "railed_sample_count",
    "fill_sample_count",
    "railed_sample_total",
    "fine_power_u64",
    "psd_frame_db_i16",
    "psd_db_reference",
    "frame_unit_index",
    "frame_in_unit",
)
# Per-frame PSD encoding: int16 codes at 0.01 dB per step about a
# per-frame reference. The sentinel marks a frame that never reached the
# transform and sits outside the clipped range, so it cannot be read back
# as a measured level.
PSD_DB_INVALID = -32768
PSD_DB_MIN = -32767
PSD_DB_MAX = 32767
PSD_DB_STEP = 0.01

SOURCE_EVENT_KEY_SCHEMA_VERSION = "pilotproxy_namespaced_source_event_key_v1"

COARSE_MEASUREMENT_METHOD = "coarse_local_reference_power_ratio"
FINE_MEASUREMENT_METHOD = "fine_local_reference_power_ratio"
ACTIVE_DECISION_METHOD = "coarse_normalized_positive_excess"
ACTIVE_DECISION_IMPLEMENTATION = "host_exact_integer_comparison"
FINE_TERMS_METHOD = "exact_fine_power_terms"
FINE_TERMS_ROLE = "measurement_only_no_scan_time_decision"
FINE_CANDIDATE_DECISION_METHOD = "fine_order_statistic_cfar"
FINE_CANDIDATE_CALIBRATION_STATUS = "pending_campaign"

FINE_STATUS_ENABLED = "enabled"
FINE_STATUS_PENDING = "pending"
FINE_STATUS_OUT_OF_BAND = "not_applicable_pilot_out_of_band"
# The frame powers are defined from the fine powers, so an in-band pilot
# without them has no frame powers at all: the only status without terms is
# a pilot outside the measured band (zero powers, NaN ratios, valid = 0).
FINE_STATUS_WITHOUT_TERMS = frozenset({FINE_STATUS_OUT_OF_BAND})
# The statuses whose fine_power_u64 are the measured terms (or, out of band,
# the empty terms whose sums are the zero frame powers). A v5 product with any
# other status (disabled_by_option, *_lacks_matched_filter_row_projections)
# stores an all-zero placeholder, so frame powers formed from it would be
# zeros rather than measurements.
_FINE_STATUSES_WITH_FRAME_POWERS = frozenset(
    {FINE_STATUS_ENABLED, FINE_STATUS_PENDING, FINE_STATUS_OUT_OF_BAND}
)

# The per-frame columns coarse_columns_from_fine derives from fine_power_u64,
# in the order the writer stores them. In a v6 product each equals the stored
# column (integers and flags exactly, floats to the reader's tolerance).
FINE_DERIVED_FRAME_COLUMNS = (
    "p_target_u64",
    "p_ref_lower_u64",
    "p_ref_upper_u64",
    "p_ref_sum_u64",
    "coarse_power_ratio",
    "normalized_coarse_power_ratio_db",
    "normalized_pilot_excess",
    "pilot_excess_db",
    "estimated_data_shelf_snr_db",
    "reject_mask",
    "valid",
)
FINE_DERIVED_INTEGER_COLUMNS = frozenset(
    {
        "p_target_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "p_ref_sum_u64",
        "reject_mask",
        "valid",
    }
)
# Stored and recomputed float columns agree to this tolerance; the products
# are formed by the same float64 operations, and the tolerance only absorbs
# platform differences in log10.
FINE_DERIVED_FLOAT_RTOL = 4e-13
FINE_DERIVED_FLOAT_ATOL = 1e-12
_UINT64_LIMIT = float(1 << 64)


def current_decision_contract() -> dict[str, Any]:
    """Return a fresh JSON-safe description of product decision semantics."""
    return {
        "measurements": [COARSE_MEASUREMENT_METHOD, FINE_MEASUREMENT_METHOD],
        "active_decision": {
            "method": ACTIVE_DECISION_METHOD,
            "implementation": ACTIVE_DECISION_IMPLEMENTATION,
            "output_field": "reject_mask",
        },
        "fine_measurement": {
            "method": FINE_TERMS_METHOD,
            "role": FINE_TERMS_ROLE,
            "terms_field": "fine_power_u64",
            "terms_note": (
                "exact uint64 [3, num_bins] from the frozen fxfft256: the "
                "deployed fine statistic's numerators and denominators. "
                "Besides their exact all-bin sums (the frame powers), the "
                "scan stores nothing derived from them, so the ratio, the "
                "CFAR baseline, the designated-set decision, and any "
                "threshold are recomputed in post-processing at whatever "
                "operating point is calibrated."
            ),
            "no_scan_time_decision": (
                "The scan applies no fine decision. A float ratio formed with "
                "numpy.fft was previously stored and was never bit-equal to "
                "the deployed statistic; it is no longer written."
            ),
        },
        "fine_candidate_decision": {
            "method": FINE_CANDIDATE_DECISION_METHOD,
            "calibration_status": FINE_CANDIDATE_CALIBRATION_STATUS,
            "active": False,
        },
        "frame_powers": {
            "definition": COARSE_POWER_DEFINITION,
            "fields": ["p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64"],
            "terms_field": "fine_power_u64",
            "invariant": (
                "for fine_status enabled, p_r[:, 0] == "
                "fine_power_u64[:, r, :].sum(axis=1) exactly (r = target, "
                "lower, upper); p_ref_sum = lower + upper"
            ),
        },
    }


# The fine-terms note the v5 products of record carry (their decision
# contract predates the frame_powers block).
_V5_FINE_TERMS_NOTE = (
    "exact uint64 [3, num_bins] from the frozen fxfft256: the "
    "deployed fine statistic's numerators and denominators. The "
    "scan stores these and nothing derived from them, so the "
    "ratio, the CFAR baseline, the designated-set decision, and "
    "any threshold are recomputed in post-processing at whatever "
    "operating point is calibrated."
)


def v5_decision_contract() -> dict[str, Any]:
    """The decision contract exactly as the v5 products of record store it.

    It is :func:`current_decision_contract` without the ``frame_powers``
    block (v6 added it with the kernel core 3.0.0 frame-power definition) and
    with that release's fine-terms note. Audits compare v5 products with it
    exactly; nothing writes it.
    """
    contract = current_decision_contract()
    del contract["frame_powers"]
    contract["fine_measurement"]["terms_note"] = _V5_FINE_TERMS_NOTE
    return contract


class CurrentProductContractError(ValueError):
    """A per-pilot product does not satisfy the only supported contract."""


def null_power_ratio_of(product: Mapping[str, Any]) -> float:
    """The weight-norm correction, derived rather than read.

    ``mu_0 = 2 * target_norm_sq / reference_norm_sum_sq`` is an exact rational
    of two integers every product carries, so it is never stored. Archived
    products spell the denominator differently; the migration map resolves it,
    and the derivation reproduces their stored ``mu0`` exactly on all 23
    archived channels.
    """
    target = int(np.asarray(product["target_norm_sq"]).reshape(-1)[0])
    # NpzFile is not a full Mapping, so test membership rather than use .get.
    name = (
        "reference_norm_sum_sq"
        if "reference_norm_sum_sq" in product
        else ARCHIVED_REFERENCE_NORM_SUM_SQ
    )
    denominator = int(np.asarray(product[name]).reshape(-1)[0])
    if denominator <= 0:
        raise CurrentProductContractError(
            "reference_norm_sum_sq must be positive to form the weight-norm "
            "correction"
        )
    return 2.0 * target / denominator


def fine_power_ratio_of(product: Mapping[str, Any]) -> np.ndarray:
    """The per-frame fine statistic, ``F2[b] = 2 S_t[b] / (S_l[b] + S_u[b])``.

    Current products retain the exact ``uint64`` fine-power terms
    (``fine_power_u64``, shape ``(N, 3, B)``) and store nothing derived from
    them, so the ratio is formed here, the way ``fine_reduction.fine_reduce``
    forms it (a zero denominator gives 0.0). Archived products stored a
    ``float32`` ratio formed with ``numpy.fft`` instead; it is returned as
    stored, under either spelling. The two are the same statistic but not the
    same numbers: the archived ratio was never bit-equal to the deployed
    fixed-point transform.

    Returns ``float64`` of shape ``(N, B)``.
    """
    if "fine_power_u64" in product:
        terms = np.asarray(product["fine_power_u64"])
        if terms.ndim != 3:
            raise CurrentProductContractError(
                "fine_power_u64 must have shape (N, 3, B); got "
                f"{terms.shape}"
            )
        if terms.shape[1] == 0 or terms.shape[2] == 0:
            # The (N, 0, 0) placeholder of a product whose fine stage was off.
            raise CurrentProductContractError(
                "product retains no fine terms (fine_num_bins = 0), so no fine "
                "power ratio can be formed"
            )
        if terms.shape[1] != 3:
            raise CurrentProductContractError(
                "fine_power_u64 must have shape (N, 3, B); got "
                f"{terms.shape}"
            )
        s_t = terms[:, WEIGHT_TERM_TARGET].astype(np.float64)
        s_l = terms[:, WEIGHT_TERM_REF_LOWER].astype(np.float64)
        s_u = terms[:, WEIGHT_TERM_REF_UPPER].astype(np.float64)
        den = s_l + s_u
        positive = den > 0.0
        return np.where(positive, 2.0 * s_t / np.where(positive, den, 1.0), 0.0)
    for name in ("fine_power_ratio", ARCHIVED_FINE_POWER_RATIO):
        if name in product:
            return np.asarray(product[name], dtype=np.float64)
    raise CurrentProductContractError(
        "product carries neither fine_power_u64 nor a stored fine power ratio"
    )


def _selected_fine_terms(product: Mapping[str, Any], rows: Any) -> np.ndarray:
    """``fine_power_u64`` (optionally a frame selection) as exact uint64 (n, 3, B).

    An ``(N, 0, 0)`` member (a pilot outside the band) is returned as is; its
    frame powers are the empty sums, zero.
    """
    if "fine_power_u64" not in product:
        raise CurrentProductContractError(
            "product carries no fine_power_u64, so its frame powers cannot be "
            "formed"
        )
    if "fine_status" in product:
        status = str(np.asarray(product["fine_status"]).reshape(()).item())
        if status not in _FINE_STATUSES_WITH_FRAME_POWERS:
            raise CurrentProductContractError(
                f"fine_status {status!r}: the product's fine_power_u64 is a "
                "placeholder, not measured fine powers, so its frame powers "
                "cannot be formed from them"
            )
    terms = np.asarray(product["fine_power_u64"])
    if terms.dtype != np.dtype(np.uint64) or terms.ndim != 3:
        raise CurrentProductContractError(
            "fine_power_u64 must be uint64 of shape (N, 3, B); got "
            f"{terms.dtype} {terms.shape}"
        )
    if terms.shape[1:] != (0, 0) and terms.shape[1] != REFERENCE_WEIGHT_TERMS:
        raise CurrentProductContractError(
            "fine_power_u64 must have shape (N, 3, B) or (N, 0, 0); got "
            f"{terms.shape}"
        )
    if any(stride == 0 and size > 1 for stride, size in zip(terms.strides, terms.shape)):
        # A broadcast view carries a shape but not the measured values.
        raise CurrentProductContractError(
            "fine_power_u64 is a metadata-only (broadcast) view; the frame "
            "powers need its values"
        )
    if rows is not None:
        terms = terms[np.asarray(rows) if not isinstance(rows, slice) else rows]
        if terms.ndim != 3:
            raise CurrentProductContractError(
                "rows must select frames along axis 0 (an index vector, a "
                "boolean mask or a slice)"
            )
    return terms


def frame_powers_from_fine(
    product: Mapping[str, Any], rows: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frame powers ``(p_target, p_ref_lower, p_ref_upper)`` from fine powers.

    ``P_r = sum_f fine_power_u64[:, r, f]``, the kernel core 3.0.0 definition,
    as exact ``uint64`` vectors of shape ``(n,)`` (``n`` frames, or the
    ``rows`` selection). This is what a v6 product stores in
    ``p_*_u64[:, 0]``; on a v5 product of record it is the frame power the
    stored row-domain ``p_*_u64`` are replaced by. The deployed bound keeps
    every sum far below ``2**64`` (about ``2**56`` at 2048 inputs); numpy
    would wrap silently, so a float64 shadow sum refuses any sum that could
    have reached ``2**64`` instead of returning a wrapped value.
    """
    terms = _selected_fine_terms(product, rows)
    count = int(terms.shape[0])
    if terms.shape[1] == 0:
        zeros = np.zeros(count, dtype=np.uint64)
        return zeros, zeros.copy(), zeros.copy()
    sums = terms.sum(axis=2, dtype=np.uint64)
    shadow = terms.sum(axis=2, dtype=np.float64)
    if shadow.size and float(shadow.max()) >= _UINT64_LIMIT * (1.0 - 1e-9):
        raise CurrentProductContractError(
            "a frame power sum of fine_power_u64 reaches 2**64; the fine "
            "powers are outside the detector's bound"
        )
    return (
        np.ascontiguousarray(sums[:, WEIGHT_TERM_TARGET]),
        np.ascontiguousarray(sums[:, WEIGHT_TERM_REF_LOWER]),
        np.ascontiguousarray(sums[:, WEIGHT_TERM_REF_UPPER]),
    )


def _exact_norms_of(product: Mapping[str, Any]) -> tuple[int, int]:
    target = int(np.asarray(product["target_norm_sq"]).reshape(-1)[0])
    name = (
        "reference_norm_sum_sq"
        if "reference_norm_sum_sq" in product
        else ARCHIVED_REFERENCE_NORM_SUM_SQ
    )
    reference = int(np.asarray(product[name]).reshape(-1)[0])
    if target <= 0 or reference <= 0:
        raise CurrentProductContractError(
            "target_norm_sq and reference_norm_sum_sq must be positive to form "
            "the weight-norm correction"
        )
    return target, reference


def _reference_sum(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    total = lower + upper
    if np.any(total < lower):
        raise CurrentProductContractError(
            "the reference frame power p_ref_lower + p_ref_upper exceeds uint64"
        )
    return total


def pilot_excess_from_fine_powers(
    product: Mapping[str, Any], rows: Any = None
) -> np.ndarray:
    """The normalized pilot excess ``e = 2 P_0 / (P_1 + P_2) / mu_0 - 1``.

    ``P_r`` are :func:`frame_powers_from_fine` and ``mu_0`` is
    :func:`null_power_ratio_of` (``2 target_norm_sq / reference_norm_sum_sq``),
    so ``e = P_0 reference_norm_sum_sq / ((P_1 + P_2) target_norm_sq) - 1``,
    evaluated in float64 in that order, exactly as the v6 writer stores
    ``normalized_pilot_excess``. NaN where ``P_1 + P_2 == 0``. Returns float64
    of shape ``(n,)``. This is the excess every analysis reads, for v5 and v6
    products alike.
    """
    null_power_ratio_of(product)  # validates the norms
    target_norm, reference_norm = _exact_norms_of(product)
    p_target, p_lower, p_upper = frame_powers_from_fine(product, rows)
    p_reference = _reference_sum(p_lower, p_upper)
    ratio = np.asarray(
        power_terms_to_normalized_coarse_power_ratio(
            p_target,
            p_reference,
            target_norm_sq=target_norm,
            reference_norm_sum_sq=reference_norm,
        ),
        dtype=np.float64,
    )
    return ratio - 1.0


def coarse_columns_from_fine(
    product: Mapping[str, Any], rows: Any = None
) -> dict[str, np.ndarray]:
    """Every per-frame column a v6 product derives from ``fine_power_u64``.

    Returns the columns of :data:`FINE_DERIVED_FRAME_COLUMNS`, each of shape
    ``(n, 1)`` as the product stores them, computed exactly as the v6 writer
    computes them: the frame powers ``P_r = sum_f S_r[f]`` (uint64),
    ``p_ref_sum_u64 = lower + upper``, ``coarse_power_ratio = 2 P_0 / p_ref_sum``,
    ``normalized_coarse_power_ratio_db = 10 log10(Q)``,
    ``normalized_pilot_excess = Q - 1`` (:func:`pilot_excess_from_fine_powers`),
    ``pilot_excess_db``, ``estimated_data_shelf_snr_db`` from the product's
    calibration scalars, ``reject_mask`` (the exact integer normalized
    positive-excess decision) and ``valid = p_ref_sum != 0``. Every float is
    NaN where the reference power is zero. For a v5 product of record these
    replace the stored row-domain columns; for a v6 product they equal the
    stored ones.
    """
    target_norm, reference_norm = _exact_norms_of(product)
    p_target, p_lower, p_upper = frame_powers_from_fine(product, rows)
    p_reference = _reference_sum(p_lower, p_upper)
    coarse = np.asarray(
        power_terms_to_coarse_power_ratio(p_target, p_reference), dtype=np.float64
    )
    ratio = np.asarray(
        power_terms_to_normalized_coarse_power_ratio(
            p_target,
            p_reference,
            target_norm_sq=target_norm,
            reference_norm_sum_sq=reference_norm,
        ),
        dtype=np.float64,
    )
    ratio_db = np.asarray(normalized_coarse_power_ratio_to_db(ratio), dtype=np.float64)
    excess = ratio - 1.0
    excess_db = np.asarray(normalized_pilot_excess_to_db(excess), dtype=np.float64)

    def _scalar(name: str) -> float:
        return float(np.asarray(product[name]).reshape(-1)[0])

    shelf = np.asarray(
        pilot_excess_db_to_data_shelf_snr_db(
            excess_db,
            pilot_below_data_db=_scalar("pilot_below_data_db"),
            bin_enbw_hz=_scalar("bin_enbw_hz"),
            dtv_bandwidth_hz=_scalar("dtv_bandwidth_hz"),
            pilot_capture_efficiency=_scalar("pilot_capture_efficiency"),
        ),
        dtype=np.float64,
    )
    reject = np.fromiter(
        (
            normalized_positive_excess(
                int(num),
                int(den),
                target_norm_sq=target_norm,
                reference_norm_sum_sq=reference_norm,
            )
            for num, den in zip(p_target.tolist(), p_reference.tolist())
        ),
        dtype=np.uint8,
        count=int(p_target.size),
    )
    valid = (p_reference != 0).astype(np.uint8)
    columns = {
        "p_target_u64": p_target,
        "p_ref_lower_u64": p_lower,
        "p_ref_upper_u64": p_upper,
        "p_ref_sum_u64": p_reference,
        "coarse_power_ratio": coarse,
        "normalized_coarse_power_ratio_db": ratio_db,
        "normalized_pilot_excess": excess,
        "pilot_excess_db": excess_db,
        "estimated_data_shelf_snr_db": shelf,
        "reject_mask": reject,
        "valid": valid,
    }
    return {name: columns[name].reshape(-1, 1) for name in FINE_DERIVED_FRAME_COLUMNS}


def fine_derived_column_mismatches(
    product: Mapping[str, Any], derived: Mapping[str, np.ndarray] | None = None
) -> list[str]:
    """Names of stored columns that disagree with :func:`coarse_columns_from_fine`.

    Integer columns and flags must match exactly; floats to
    ``FINE_DERIVED_FLOAT_RTOL``/``ATOL`` with NaN in the same places. An empty
    list means the product satisfies the v6 frame-power invariant.
    """
    if derived is None:
        derived = coarse_columns_from_fine(product)
    mismatches: list[str] = []
    for name in FINE_DERIVED_FRAME_COLUMNS:
        stored = np.asarray(product[name])
        expected = derived[name]
        if stored.shape != expected.shape:
            mismatches.append(name)
            continue
        if name in FINE_DERIVED_INTEGER_COLUMNS:
            if stored.dtype.kind not in {"u", "i"} or not np.array_equal(
                stored.astype(expected.dtype, copy=False), expected
            ) or np.any(stored < 0):
                mismatches.append(name)
        elif not np.allclose(
            np.asarray(stored, dtype=np.float64),
            expected,
            rtol=FINE_DERIVED_FLOAT_RTOL,
            atol=FINE_DERIVED_FLOAT_ATOL,
            equal_nan=True,
        ) or not np.array_equal(np.isnan(stored), np.isnan(expected)):
            mismatches.append(name)
    return mismatches


def exact_integer_array(
    value: Any,
    *,
    field: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
    ndim: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> np.ndarray:
    """Return an integer array without accepting bool/float/string coercion."""
    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"} or array.dtype.kind == "b":
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have an integer dtype"
        )
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if minimum is not None and array.size and np.any(array < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum}"
        )
    if maximum is not None and array.size and np.any(array > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum}"
        )
    return array


def exact_integer_scalar(
    product: Mapping[str, Any],
    field: str,
    *,
    dtype: np.dtype[Any] | type[np.generic],
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    array = exact_integer_array(
        product.get(field, np.asarray([], dtype=dtype)),
        field=field,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
    )
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    if array.size != 1:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be scalar"
        )
    return int(array.reshape(()).item())


def _exact_string_scalar(product: Mapping[str, Any], field: str) -> str:
    if field not in product:
        raise CurrentProductContractError(
            f"current per-pilot product is missing required field {field!r}"
        )
    value = np.asarray(product[field])
    if value.size != 1 or value.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string scalar"
        )
    return str(value.reshape(()).item())


def _exact_float_array(
    product: Mapping[str, Any],
    field: str,
    *,
    shape: tuple[int, ...] | None = None,
    ndim: int | None = None,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    allow_nan: bool = True,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    array = np.asarray(product[field])
    if array.dtype != np.dtype(dtype):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have dtype "
            f"{np.dtype(dtype)}, got {array.dtype}"
        )
    if shape is not None and array.shape != shape:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must have shape {shape}, "
            f"got {array.shape}"
        )
    if ndim is not None and array.ndim != ndim:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be {ndim}D"
        )
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} contains non-finite values"
        )
    finite = array[np.isfinite(array)]
    if minimum is not None and finite.size and np.any(finite < minimum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be >= {minimum} when finite"
        )
    if maximum is not None and finite.size and np.any(finite > maximum):
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be <= {maximum} when finite"
        )
    return array


def _exact_string_vector(
    product: Mapping[str, Any], field: str, *, length: int
) -> list[str]:
    array = np.asarray(product[field])
    if array.ndim != 1 or array.size != length or array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            f"current per-pilot field {field!r} must be a string vector of "
            f"length {length}"
        )
    return [str(value) for value in array.tolist()]


def validate_current_product_identity(
    product: Mapping[str, Any], *, allow_empty_checkpoint: bool = False
) -> None:
    """Require the only supported schema and decision semantics (v6).

    Development snapshots must be deleted and regenerated. This function
    intentionally contains no aliases, migration, or best-effort fallback;
    the v5 products of record are read through
    ``pilot_proxy.products.reader`` (which recomputes their frame powers
    from their fine powers), never validated here.

    Beyond the field contract it enforces the v6 frame-power invariant: every
    frame power is the exact all-bin sum of ``fine_power_u64``
    (``p_r[:, 0] == fine_power_u64[:, r, :].sum(axis=1)``, r = target, lower,
    upper) and every derived column equals :func:`coarse_columns_from_fine`.
    """
    schema_name = _exact_string_scalar(product, "schema_name")
    schema_revision = exact_integer_scalar(
        product,
        "schema_revision",
        dtype=np.int64,
        minimum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
        maximum=PER_PILOT_PRODUCT_SCHEMA_REVISION,
    )
    schema_token = _exact_string_scalar(product, "schema_version")
    if (
        schema_name != PER_PILOT_PRODUCT_SCHEMA_NAME
        or schema_revision != PER_PILOT_PRODUCT_SCHEMA_REVISION
        or schema_token != PER_PILOT_PRODUCT_SCHEMA_TOKEN
    ):
        raise CurrentProductContractError(
            "unsupported per-pilot product identity: "
            f"schema_name={schema_name!r}, schema_revision={schema_revision!r}, "
            f"schema_version={schema_token!r}; delete the product and regenerate "
            "it with the current PilotProxy release. A checkpoint from an older "
            "schema cannot be resumed here: the per-frame railed/fill counts "
            "are taken from raw samples the product does not retain, so they "
            "cannot be backfilled -- finish that cohort on the release that "
            "started it, or rebuild from the archive"
        )

    event_key_schema = _exact_string_scalar(
        product, "source_event_key_schema_version"
    )
    if event_key_schema != SOURCE_EVENT_KEY_SCHEMA_VERSION:
        raise CurrentProductContractError(
            "unsupported source-event identity schema: "
            f"{event_key_schema!r}; delete the product and regenerate it so "
            "event keys retain their archive/campaign namespace"
        )

    raw_contract = _exact_string_scalar(product, "decision_contract_json")
    try:
        decision_contract = json.loads(raw_contract)
    except json.JSONDecodeError as exc:
        raise CurrentProductContractError(
            "current per-pilot decision_contract_json is invalid JSON"
        ) from exc
    if decision_contract != current_decision_contract():
        raise CurrentProductContractError(
            "per-pilot decision contract does not match the current release; "
            "delete the product and regenerate it"
        )

    required = {
        "physical_channel",
        "freq_id",
        "pilot_in_band",
        "pilot_frequency_hz",
        "chime_frequency_hz",
        "nfft",
        "sample_rate_hz",
        "detector_window_samples",
        "num_input_streams",
        "sense",
        "frame_index",
        "p_target_u64",
        "p_ref_sum_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "fine_power_u64",
        "psd_frame_db_i16",
        "psd_db_reference",
        "coarse_power_ratio",
        "normalized_coarse_power_ratio_db",
        "pilot_excess_db",
        "estimated_data_shelf_snr_db",
        "reject_mask",
        "valid",
        "target_norm_sq",
        "reference_norm_sum_sq",
        "normalized_pilot_excess",
        "baseband_power_linear",
        "railed_sample_count",
        "fill_sample_count",
        "railed_sample_total",
        "integrated_spectrum_before_mask",
        "integrated_spectrum_after_mask",
        "fine_pad_factor",
        "fine_num_bins",
        "fine_p_fa",
        "fine_guard_fine_bins",
        "fine_designated_bins",
        "fine_census_excluded_bins",
        "fine_status",
        "source_event_keys",
        "unit_keys",
        "unit_order",
        "unit_time0_ctime",
        "unit_time0_fpga",
        "unit_event_id",
        "unit_delta_time",
        "archive_version",
        "unit_git_version_tag",
        "unit_input_map_sha256",
        "unit_collection_server",
        "unit_scope",
        "frame_unit_index",
        "frame_in_unit",
        "detector_contract_json",
        "max_chunks_per_file",
        "rational_overflow_count",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
        "pilot_below_data_db",
        "bin_enbw_hz",
        "dtv_bandwidth_hz",
        "pilot_capture_efficiency",
    }
    missing = sorted(required.difference(product))
    if missing:
        raise CurrentProductContractError(
            "current per-pilot product is missing required arrays: "
            + ", ".join(missing)
        )

    if "mask" in product:
        raise CurrentProductContractError(
            "per-pilot product contains the ambiguous field 'mask'; the current "
            "decision field is 'reject_mask'"
        )

    exact_integer_scalar(
        product,
        "physical_channel",
        dtype=np.int32,
        minimum=ATSC_UHF_MIN_PHYSICAL_CHANNEL,
        maximum=ATSC_UHF_MAX_PHYSICAL_CHANNEL,
    )
    exact_integer_scalar(
        product, "freq_id", dtype=np.int64, minimum=0, maximum=1023
    )
    nfft = exact_integer_scalar(
        product, "nfft", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "detector_window_samples", dtype=np.int64, minimum=1
    )
    num_input_streams = exact_integer_scalar(
        product, "num_input_streams", dtype=np.int64, minimum=1
    )
    sense = exact_integer_scalar(
        product, "sense", dtype=np.int64, minimum=-1, maximum=1
    )
    if sense not in {-1, 1}:
        raise CurrentProductContractError(
            "current per-pilot field 'sense' must be exactly -1 or 1"
        )
    exact_integer_scalar(
        product, "max_chunks_per_file", dtype=np.int64, minimum=-1
    )
    exact_integer_scalar(
        product, "rational_overflow_count", dtype=np.uint64, minimum=0
    )
    pilot_in_band = exact_integer_array(
        product["pilot_in_band"],
        field="pilot_in_band",
        dtype=np.uint8,
        ndim=1,
        minimum=0,
        maximum=1,
    )
    if pilot_in_band.shape != (1,):
        raise CurrentProductContractError(
            "current per-pilot field 'pilot_in_band' must have shape (1,)"
        )

    _exact_float_array(
        product, "pilot_frequency_hz", shape=(1,), allow_nan=False
    )
    _exact_float_array(
        product, "chime_frequency_hz", shape=(1,), allow_nan=False
    )
    sample_rate = _exact_float_array(
        product, "sample_rate_hz", shape=(), allow_nan=False, minimum=0.0
    )
    if float(sample_rate.reshape(()).item()) <= 0.0:
        raise CurrentProductContractError(
            "current per-pilot field 'sample_rate_hz' must be positive"
        )
    for field, minimum, maximum in (
        ("pilot_below_data_db", 0.0, None),
        ("bin_enbw_hz", 0.0, None),
        ("dtv_bandwidth_hz", 0.0, None),
        ("pilot_capture_efficiency", 0.0, 1.0),
    ):
        values = _exact_float_array(
            product,
            field,
            shape=(),
            allow_nan=False,
            minimum=minimum,
            maximum=maximum,
        )
        if field != "pilot_below_data_db" and float(values.reshape(()).item()) <= 0.0:
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must be positive"
            )
    for field in (
        "detector_contract_json",
        "weights_hash",
        "weight_bank_sha256",
        "weight_manifest_sha256",
        "detector_version",
        "mask_rule",
        "reference_placement_json",
    ):
        _exact_string_scalar(product, field)
    if _exact_string_scalar(product, "mask_rule") != NORMALIZED_POSITIVE_EXCESS_MASK_RULE:
        raise CurrentProductContractError(
            "current per-pilot mask_rule does not declare the normalized "
            "positive-excess policy"
        )

    frame_index = exact_integer_array(
        product["frame_index"],
        field="frame_index",
        dtype=np.int64,
        ndim=1,
        minimum=0,
    )
    frame_count = int(frame_index.size)
    if (frame_count == 0 and not allow_empty_checkpoint) or not np.array_equal(
        frame_index, np.arange(frame_count, dtype=np.int64)
    ):
        raise CurrentProductContractError(
            "current per-pilot frame_index must be a non-empty contiguous "
            "zero-based int64 vector"
        )

    unit_order_array = np.asarray(product["unit_order"])
    if unit_order_array.ndim != 1 or unit_order_array.dtype.kind not in {"U", "S"}:
        raise CurrentProductContractError(
            "current per-pilot field 'unit_order' must be a 1D string vector"
        )
    unit_count = int(unit_order_array.size)
    if unit_count == 0 and not allow_empty_checkpoint:
        raise CurrentProductContractError(
            "current per-pilot unit_order must not be empty"
        )
    unit_order = _exact_string_vector(
        product, "unit_order", length=unit_count
    )
    unit_keys = _exact_string_vector(product, "unit_keys", length=unit_count)
    source_event_keys = _exact_string_vector(
        product, "source_event_keys", length=unit_count
    )
    _exact_string_vector(product, "archive_version", length=unit_count)
    git_version_tags = _exact_string_vector(
        product, "unit_git_version_tag", length=unit_count
    )
    input_map_sha256 = _exact_string_vector(
        product, "unit_input_map_sha256", length=unit_count
    )
    _exact_string_vector(product, "unit_collection_server", length=unit_count)
    unit_scope = _exact_string_vector(product, "unit_scope", length=unit_count)
    if any(not value.strip() for value in unit_scope):
        raise CurrentProductContractError(
            "current per-pilot unit_scope values must not be empty"
        )
    lowercase_hex = set("0123456789abcdef")
    if any(
        value and (len(value) != 64 or not set(value) <= lowercase_hex)
        for value in input_map_sha256
    ):
        raise CurrentProductContractError(
            "current per-pilot unit_input_map_sha256 values must be empty or "
            "64-character lowercase SHA-256 digests"
        )
    if any(
        scope != "local" and (not tag or not input_hash)
        for scope, tag, input_hash in zip(
            unit_scope, git_version_tags, input_map_sha256
        )
    ):
        raise CurrentProductContractError(
            "current per-pilot archive units require nonempty "
            "unit_git_version_tag and unit_input_map_sha256 values"
        )
    if len(set(unit_order)) != unit_count or len(set(unit_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must each be unique"
        )
    if set(unit_order) != set(unit_keys):
        raise CurrentProductContractError(
            "current per-pilot unit_order and unit_keys must contain the exact "
            "same unit identities"
        )
    if any(not value for value in unit_order + unit_keys + source_event_keys):
        raise CurrentProductContractError(
            "current per-pilot unit and source-event identities must not be empty"
        )
    if len(set(source_event_keys)) != unit_count:
        raise CurrentProductContractError(
            "current per-pilot source_event_keys must be unique"
        )

    unit_event_id = exact_integer_array(
        product["unit_event_id"],
        field="unit_event_id",
        dtype=np.int64,
        ndim=1,
        minimum=-1,
    )
    unit_time0_fpga = exact_integer_array(
        product["unit_time0_fpga"],
        field="unit_time0_fpga",
        dtype=np.uint64,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("unit_event_id", unit_event_id),
        ("unit_time0_fpga", unit_time0_fpga),
    ):
        if values.shape != (unit_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({unit_count},)"
            )
    _exact_float_array(
        product, "unit_time0_ctime", shape=(unit_count,), allow_nan=True
    )
    unit_delta_time = _exact_float_array(
        product,
        "unit_delta_time",
        shape=(unit_count,),
        allow_nan=True,
        minimum=0.0,
    )
    if np.any(np.isfinite(unit_delta_time) & (unit_delta_time <= 0.0)):
        raise CurrentProductContractError(
            "current per-pilot finite unit_delta_time values must be positive"
        )
    finite_delta_time = unit_delta_time[np.isfinite(unit_delta_time)]
    if finite_delta_time.size and not np.allclose(
        finite_delta_time,
        1.0 / float(sample_rate.reshape(()).item()),
        rtol=1e-12,
        atol=0.0,
    ):
        raise CurrentProductContractError(
            "current per-pilot unit_delta_time disagrees with sample_rate_hz"
        )

    frame_unit_index = exact_integer_array(
        product["frame_unit_index"],
        field="frame_unit_index",
        dtype=np.int32,
        ndim=1,
        minimum=0,
        maximum=unit_count - 1,
    )
    frame_in_unit = exact_integer_array(
        product["frame_in_unit"],
        field="frame_in_unit",
        dtype=np.int32,
        ndim=1,
        minimum=0,
    )
    for field, values in (
        ("frame_unit_index", frame_unit_index),
        ("frame_in_unit", frame_in_unit),
    ):
        if values.shape != (frame_count,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count},)"
            )
    if frame_count == 0 and unit_count != 0:
        raise CurrentProductContractError(
            "an empty current per-pilot checkpoint must not claim consumed units"
        )
    if frame_count and set(frame_unit_index.tolist()) != set(range(unit_count)):
        raise CurrentProductContractError(
            "current per-pilot frame_unit_index must reference every consumed "
            "unit at least once"
        )

    exact_powers: dict[str, np.ndarray] = {}
    for field in (
        "p_target_u64",
        "p_ref_lower_u64",
        "p_ref_upper_u64",
        "p_ref_sum_u64",
    ):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint64,
            minimum=0,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_powers[field] = values
    for frame, (lower, upper, total) in enumerate(
        zip(
            exact_powers["p_ref_lower_u64"].reshape(-1),
            exact_powers["p_ref_upper_u64"].reshape(-1),
            exact_powers["p_ref_sum_u64"].reshape(-1),
        )
    ):
        combined = int(lower) + int(upper)
        if combined >= (1 << 64):
            raise CurrentProductContractError(
                "current per-pilot reference power sum exceeds uint64 at "
                f"frame {frame}"
            )
        if combined != int(total):
            raise CurrentProductContractError(
                "current per-pilot lower and upper reference powers do not "
                f"match p_ref_sum_u64 at frame {frame}"
            )
    exact_flags: dict[str, np.ndarray] = {}
    for field in ("reject_mask", "valid"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint8,
            minimum=0,
            maximum=1,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        exact_flags[field] = values
    # ADC/F-engine saturation and 0x00 fill, counted from the raw int4 nibbles
    # while the frame was in hand. The product stores no raw samples, so an
    # absent or wrong count cannot be repaired later; validate it here rather
    # than trusting the writer.
    railed: dict[str, np.ndarray] = {}
    for field in ("railed_sample_count", "fill_sample_count", "railed_sample_total"):
        values = exact_integer_array(
            product[field],
            field=field,
            dtype=np.uint64,
            minimum=0,
        )
        if values.shape != (frame_count, 1):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape "
                f"({frame_count}, 1)"
            )
        railed[field] = values
    sample_counts_fit = all(
        int(railed_count) + int(fill_count) <= int(total)
        for railed_count, fill_count, total in zip(
            railed["railed_sample_count"].reshape(-1),
            railed["fill_sample_count"].reshape(-1),
            railed["railed_sample_total"].reshape(-1),
        )
    )
    if not sample_counts_fit:
        raise CurrentProductContractError(
            "current per-pilot railed_sample_count + fill_sample_count exceeds "
            "railed_sample_total; the saturation counters and their denominator "
            "disagree"
        )
    expected_total = np.uint64(2 * nfft * num_input_streams)
    if np.any(railed["railed_sample_total"] != expected_total):
        raise CurrentProductContractError(
            "current per-pilot railed_sample_total does not equal "
            "2 * nfft * num_input_streams "
            f"({int(expected_total)}); the denominator no longer describes the "
            "frame it was counted from"
        )
    for field in PER_FRAME_PRODUCT_KEYS:
        if field not in product:
            raise CurrentProductContractError(
                f"current per-pilot product is missing per-frame field {field!r}"
            )
        values = np.asarray(product[field])
        if values.ndim == 0 or int(values.shape[0]) != frame_count:
            raise CurrentProductContractError(
                f"current per-pilot per-frame field {field!r} does not have "
                f"{frame_count} frames along axis 0"
            )
    for field, minimum in (
        ("coarse_power_ratio", 0.0),
        ("normalized_coarse_power_ratio_db", None),
        ("pilot_excess_db", None),
        ("estimated_data_shelf_snr_db", None),
        ("normalized_pilot_excess", -1.0),
        ("baseband_power_linear", 0.0),
    ):
        _exact_float_array(
            product,
            field,
            shape=(frame_count, 1),
            allow_nan=True,
            minimum=minimum,
        )

    fine_num_bins = exact_integer_scalar(
        product, "fine_num_bins", dtype=np.int64, minimum=0
    )
    # The exact deployed terms: target, lower reference, upper reference, per
    # fine bin. Every fine diagnostic and decision is recomputed from these in
    # post-processing, so the scan stores nothing derived from them.
    fine_terms = exact_integer_array(
        product["fine_power_u64"],
        field="fine_power_u64",
        dtype=np.uint64,
        ndim=3,
        minimum=0,
    )
    if fine_terms.shape[0] != frame_count:
        raise CurrentProductContractError(
            "current per-pilot fine_power_u64 frame axis must match frame_index"
        )
    if fine_terms.shape[2] != fine_num_bins:
        raise CurrentProductContractError(
            "current per-pilot fine_power_u64 width must match fine_num_bins"
        )
    fine_status = _exact_string_scalar(product, "fine_status")
    if fine_status == FINE_STATUS_ENABLED:
        if fine_num_bins <= 0:
            raise CurrentProductContractError(
                "current per-pilot enabled fine measurement must have bins"
            )
        expected_fine_shape = (
            frame_count,
            REFERENCE_WEIGHT_TERMS,
            fine_num_bins,
        )
        if fine_terms.shape != expected_fine_shape:
            raise CurrentProductContractError(
                "current per-pilot enabled fine_power_u64 must have shape "
                f"{expected_fine_shape}"
            )
    elif fine_status == FINE_STATUS_PENDING:
        if frame_count != 0:
            raise CurrentProductContractError(
                "current per-pilot fine status may be pending only in an empty "
                "checkpoint"
            )
        if fine_num_bins != 0 or fine_terms.shape != (0, 0, 0):
            raise CurrentProductContractError(
                "current per-pilot pending fine measurement must not carry terms"
            )
    elif fine_status in FINE_STATUS_WITHOUT_TERMS:
        if fine_num_bins != 0 or fine_terms.shape != (frame_count, 0, 0):
            raise CurrentProductContractError(
                "current per-pilot fine status without measurements must use "
                f"shape ({frame_count}, 0, 0) and fine_num_bins 0"
            )
    else:
        allowed = sorted(
            {FINE_STATUS_ENABLED, FINE_STATUS_PENDING, *FINE_STATUS_WITHOUT_TERMS}
        )
        raise CurrentProductContractError(
            f"current per-pilot fine_status must be one of {allowed}"
        )

    # The frame powers are the fine powers' sums, so an in-band pilot without
    # fine powers has no frame powers: fine products are mandatory in band.
    if int(pilot_in_band[0]) == 1 and fine_status not in {
        FINE_STATUS_ENABLED,
        FINE_STATUS_PENDING,
    }:
        raise CurrentProductContractError(
            "current per-pilot product of an in-band pilot must retain fine "
            f"powers (fine_status 'enabled'); got {fine_status!r}"
        )
    if int(pilot_in_band[0]) == 0 and fine_status != FINE_STATUS_OUT_OF_BAND:
        raise CurrentProductContractError(
            "current per-pilot product of an out-of-band pilot must declare "
            f"fine_status {FINE_STATUS_OUT_OF_BAND!r}"
        )

    exact_integer_scalar(
        product, "fine_pad_factor", dtype=np.int64, minimum=1
    )
    exact_integer_scalar(
        product, "fine_guard_fine_bins", dtype=np.int64, minimum=0
    )
    fine_p_fa = _exact_float_array(
        product, "fine_p_fa", shape=(), allow_nan=False, minimum=0.0, maximum=1.0
    )
    if not 0.0 < float(fine_p_fa.reshape(()).item()) < 1.0:
        raise CurrentProductContractError(
            "current per-pilot fine_p_fa must be strictly between zero and one"
        )
    for field in ("fine_designated_bins", "fine_census_excluded_bins"):
        exact_integer_array(product[field], field=field, dtype=np.int64, ndim=1)
    exact_norms: dict[str, int] = {}
    for field in ("target_norm_sq", "reference_norm_sum_sq"):
        values = exact_integer_array(
            product[field], field=field, dtype=np.int64, minimum=1
        )
        if values.shape != (1,):
            raise CurrentProductContractError(
                f"current per-pilot field {field!r} must have shape (1,)"
            )
        exact_norms[field] = int(values[0])
    # The weight-norm correction is not stored. It is the exact rational
    # 2 * target_norm_sq / reference_norm_sum_sq of two integers validated
    # above, so a float copy would be its lossy form and would need its own
    # name; the contract used to check the two agreed, which is the same
    # statement as not needing both.
    if exact_norms["reference_norm_sum_sq"] <= 0:
        raise CurrentProductContractError(
            "current per-pilot reference_norm_sum_sq must be positive to form "
            "the weight-norm correction"
        )

    p_target = exact_powers["p_target_u64"].reshape(-1)
    p_ref_sum = exact_powers["p_ref_sum_u64"].reshape(-1)
    valid = exact_flags["valid"].reshape(-1)
    reject = exact_flags["reject_mask"].reshape(-1)
    expected_valid = (p_ref_sum != 0).astype(np.uint8)
    if not np.array_equal(valid, expected_valid):
        raise CurrentProductContractError(
            "current per-pilot valid flags disagree with p_ref_sum != 0"
        )
    expected_reject = np.asarray(
        [
            normalized_positive_excess(
                int(num),
                int(den),
                target_norm_sq=exact_norms["target_norm_sq"],
                reference_norm_sum_sq=exact_norms["reference_norm_sum_sq"],
            )
            for num, den in zip(p_target, p_ref_sum)
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(reject, expected_reject):
        raise CurrentProductContractError(
            "current per-pilot reject_mask disagrees with the exact normalized "
            "positive-excess decision over stored powers and norms"
        )
    # The v6 frame-power invariant, in exact integers: each stored frame power
    # is the all-bin sum of that term's stored fine powers (zero for a pilot
    # outside the band, whose fine member is (N, 0, 0)).
    for field, from_fine in zip(
        ("p_target_u64", "p_ref_lower_u64", "p_ref_upper_u64"),
        frame_powers_from_fine(product),
    ):
        stored = exact_powers[field].reshape(-1)
        if not np.array_equal(stored, from_fine):
            frame = int(np.flatnonzero(stored != from_fine)[0])
            raise CurrentProductContractError(
                f"current per-pilot {field} is not the all-bin sum of "
                f"fine_power_u64 at frame {frame} (stored {int(stored[frame])}, "
                f"sum {int(from_fine[frame])}); the v6 frame powers are "
                "P_r = sum_f fine_power_u64[:, r, f]"
            )
    mismatches = fine_derived_column_mismatches(product)
    if mismatches:
        raise CurrentProductContractError(
            "current per-pilot columns disagree with their values from the "
            f"fine-power sums: {', '.join(mismatches)}"
        )
    spectrum_before = _exact_float_array(
        product,
        "integrated_spectrum_before_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    spectrum_after = _exact_float_array(
        product,
        "integrated_spectrum_after_mask",
        shape=(nfft,),
        allow_nan=False,
        minimum=0.0,
    )
    if np.any(spectrum_after > spectrum_before):
        raise CurrentProductContractError(
            "current per-pilot integrated spectrum after mask exceeds before mask"
        )


def current_decision_contract_json() -> str:
    """Return the canonical stable serialization stored in products."""
    return json.dumps(current_decision_contract(), sort_keys=True, separators=(",", ":"))
