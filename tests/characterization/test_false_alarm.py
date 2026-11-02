"""The order-statistic index of an empirical null threshold, and the project's support rule."""
from __future__ import annotations

from fractions import Fraction

from pilot_proxy.testbench import fine_validation_stats as fa


def test_the_higher_index_is_the_exact_ceiling():
    for n in (1, 2, 999, 1000, 1001, 10_000, 123_457):
        for pfa in (Fraction(1, 1000), 0.001, Fraction(1, 3), 0.25):
            q = 1 - Fraction(str(pfa)) if not isinstance(pfa, Fraction) else 1 - pfa
            expected = -((-q.numerator * (n - 1)) // q.denominator)          # ceil((1 - pfa)(n - 1)) exactly
            assert fa.higher_index(n, pfa) == expected
    # the float 0.001 is read as the decimal 1/1000
    assert fa.higher_index(1001, 0.001) == 999


def test_the_support_rule_is_ten_false_alarms():
    assert fa.MIN_FRAMES_PER_FALSE_ALARM == 10.0
