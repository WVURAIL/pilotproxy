"""A frame exactly on the norm-corrected mask boundary is not masked.

Random inputs rarely make the two products of the rule equal, so a test on
random inputs can miss a ``>=`` written in place of ``>``. This test builds an
exact equality and requires it to give no excess. The keyword arguments are
built from the parameter names that ``inspect.signature`` reports for
``normalized_positive_excess``, so the test adapts to a change in its signature.
"""
import inspect
from pilot_proxy import detector_contract as dc


def test_exact_equality_is_not_positive_excess():
    fn = dc.normalized_positive_excess
    params = list(inspect.signature(fn).parameters)
    # 3 * 4 == 2 * 6 is an exact equality for any argument naming that pairs
    # (p_target, reference_norm_sum_sq) against (target_norm_sq, p_ref_sum)
    vals = {"p_target": 3, "reference_norm_sum_sq": 4,
            "target_norm_sq": 2, "p_ref_sum": 6}
    kwargs = {p: vals[p] for p in params if p in vals}
    for p in params:
        if p not in kwargs:  # any other parameter, such as a `valid` flag
            kwargs[p] = True
    result = fn(**kwargs)
    assert not bool(result), "equality must NOT mask under strict positive excess"
