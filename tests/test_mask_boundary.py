"""Equality-boundary regression for the norm-corrected mask (strict `>`).

Random vectors essentially never hit exact rational equality, so `>` and
`>=` implementations pass identical parity tests; this pins the one input
class where they differ. Signature-adaptive on purpose: it asserts the
POLICY (equality is NOT excess) against whatever the reference exposes.
"""
import inspect
from pilot_proxy import detector_contract as dc


def test_exact_equality_is_not_positive_excess():
    fn = dc.normalized_positive_excess
    params = list(inspect.signature(fn).parameters)
    # 3 * 4 == 2 * 6: an exact-equality case for any argument naming that
    # pairs (p_target, reference_norm_sum_sq) against (target_norm_sq, p_ref_sum)
    vals = {"p_target": 3, "reference_norm_sum_sq": 4,
            "target_norm_sq": 2, "p_ref_sum": 6}
    kwargs = {p: vals[p] for p in params if p in vals}
    for p in params:
        if p not in kwargs:  # e.g. a `valid` flag
            kwargs[p] = True
    result = fn(**kwargs)
    assert not bool(result), "equality must NOT mask under strict positive excess"
