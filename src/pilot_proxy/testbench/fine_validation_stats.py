"""Statistical bookkeeping for separately frozen fine-detector experiments.

These helpers do not certify a detector, physical residual or telescope policy.
Trials must be independent across IDs; stages/SNRs may deliberately be paired.
The single-look ideal fine F(2,4) has infinite variance: a finite sample standard
deviation at M=1 is descriptive, not a finite theoretical scaling benchmark.
SciPy is required for exact binomial intervals (available in the test extra).

The order-statistic calibration and the fixed-sample binomial acceptance
diagnostics (:func:`higher_index` to :func:`family_power_lower_bound`) are
design and reference calculations, not permission to extend or retune an
experiment after evaluation; policies may share trials, and no independence
between policies is assumed by the family lower bound. A fixed-rank one-sided
tolerance limit's rank depends only on n, the false-alarm target and the
confidence, through the integer binomial count whose upper bound is at most
the target; strict exceedances keep ties.
"""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta, betabinom, binom

from pilot_proxy.config.project import default_project

MAX_Q16 = (1 << 64) - 1
ALWAYS_MASKED_Q16 = 1 << 64
MIN_FRAMES_PER_FALSE_ALARM = float(default_project().register.value("detection.minimum_frames_per_false_alarm"))


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a probability")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        result = Fraction(str(value))
    else:
        raise TypeError(f"{name} must be a real probability or Fraction")
    if not 0 < result < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _real_vector(values, name, *, positive=False):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all() or np.any(result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must contain finite {'positive' if positive else 'nonnegative'} values; no rows are dropped")
    return result


def _requirements(values):
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("logical Q16 requirements must be a nonempty vector")
    result = [_integer(v, "requirement", 1) for v in raw]
    if any(v > ALWAYS_MASKED_Q16 for v in result):
        raise ValueError("logical Q16 requirements exceed the sentinel")
    return result


def higher_index(n, pfa):
    q = 1-_probability(pfa, "pfa")
    return (q.numerator*(n-1)+q.denominator-1)//q.denominator


def empirical_threshold(responses, *, pfa):
    """Observed higher (1-pfa) quantile; no interpolation or finite-row trimming."""
    values = _real_vector(responses, "null responses")
    index = higher_index(len(values), pfa)
    threshold = float(np.partition(values, index)[index])
    exceeded = int(np.count_nonzero(values > threshold))
    return {"status": "available", "threshold": threshold,
            "index": index, "trials": len(values), "exceedances": exceeded,
            "empirical_pfa": exceeded/len(values), "nominal_pfa": float(pfa),
            "method": "observed higher quantile; strict response > threshold",
            "independent_validation": False}


def exact_q16_threshold(requirements, *, pfa):
    """Higher quantile of decoded exact keep boundaries, with ties kept.

    Sentinel 2**64 remains in every denominator and in sorting. If selected,
    there is no legal threshold; it is not replaced with the uint64 maximum.
    Serialized zero+bitset encodings must be decoded before calling.
    """
    values = _requirements(requirements)
    index = higher_index(len(values), pfa)
    threshold = sorted(values)[index]
    available = threshold <= MAX_Q16
    exceeded = sum(v > threshold for v in values) if available else None
    return {"status": "available" if available else "unavailable",
            "threshold": threshold if available else None,
            "selected_requirement": threshold, "index": index,
            "trials": len(values), "sentinel_trials": values.count(ALWAYS_MASKED_Q16),
            "exceedances": exceeded,
            "empirical_pfa": exceeded/len(values) if available else None,
            "nominal_pfa": float(pfa), "independent_validation": False,
            "method": "exact higher keep-boundary quantile; strict requirement > threshold"}


def higher_rank(trials: int, nominal_pfa: float) -> int:
    """One-based rank for ceil((1-p)*(n-1)), calculated without float rounding."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    p = Fraction(str(nominal_pfa))
    if not 0 < p < 1:
        raise ValueError("nominal_pfa must lie strictly between zero and one")
    v = (1-p)*(trials-1)
    return (v.numerator + v.denominator - 1)//v.denominator + 1


def calibration_reference(trials: int, nominal_pfa: float) -> dict:
    """Continuous-IID repeated-calibration distribution of strict exceedance p.

    If T is order statistic r from n draws, 1-F(T) ~ Beta(n-r+1,r).
    This is a sampling distribution, not a posterior or a confidence interval
    conditional on one observed threshold. For atoms and a strict comparison,
    it provides an upper stochastic reference under the same IID null law.
    """
    rank = higher_rank(trials, nominal_pfa)
    a, b = trials-rank+1, rank
    return {"trials": trials, "rank": rank, "beta_a": a, "beta_b": b,
            "mean_pfa": float(beta.mean(a, b)),
            "sd_pfa": float(beta.std(a, b)),
            "central_95_pfa": beta.ppf([.025, .975], a, b).tolist(),
            "assumption": "continuous IID scores; calibration and validation share the same null law"}


def accepted_count_limit(trials: int, cap: float, *, family_tests: int = 1334,
                         confidence: float = .95) -> int:
    """Largest k whose one-sided Bonferroni Clopper--Pearson upper <= cap.

    CP inversion is equivalent to BinomialCDF(k; n, cap) <= alpha/family.
    Returns -1 if even zero observed events cannot demonstrate the cap.
    """
    if trials < 1 or int(trials) != trials or family_tests < 1 or int(family_tests) != family_tests:
        raise ValueError("positive integer trial and family counts required")
    if not 0 < cap < 1 or not 0 < confidence < 1:
        raise ValueError("cap and confidence must lie strictly between zero and one")
    alpha = (1-confidence)/family_tests
    lo, hi = -1, trials
    while hi-lo > 1:
        k = (lo+hi)//2
        if binom.cdf(k, trials, cap) <= alpha:
            lo = k
        else:
            hi = k
    return lo


def confidence_count_limit(trials, *, pfa, confidence=0.95):
    """Fixed design count for one population's one-sided tolerance limit."""
    n = _integer(trials, "trials", 1)
    a = _probability(pfa, "pfa")
    c = _probability(confidence, "confidence")
    return accepted_count_limit(n, float(a), family_tests=1, confidence=float(c))


def _confidence_threshold(values, *, pfa, confidence, independent_null, independence_source, q16):
    a = _probability(pfa, "pfa")
    c = _probability(confidence, "confidence")
    if not isinstance(independent_null, bool):
        raise TypeError("independent_null must be a boolean")
    if not isinstance(independence_source, str):
        raise TypeError("independence_source must be a string")
    if independent_null and not independence_source.strip():
        raise ValueError("independently sampled nulls require a sampling source")
    n = len(values)
    k = confidence_count_limit(n, pfa=a, confidence=c)
    minimum = math.ceil(MIN_FRAMES_PER_FALSE_ALARM / a)
    out = {"status": "unavailable", "threshold": None, "trials": n,
           "nominal_pfa": float(a), "confidence": float(c), "minimum_trials": minimum,
           "allowed_exceedances": k, "rank": None, "index": None,
           "exceedances": None, "empirical_pfa": None, "pfa_upper": None,
           "independent_null": independent_null, "independence_source": independence_source,
           "independent_validation": False, "statistic": "requirement_q16" if q16 else "score",
           "method": "fixed-rank iid tolerance limit; strict response > threshold",
           "assumption": "iid null scores representative of the deployment null; fixed sampling design"}
    if q16:
        out["sentinel_trials"] = values.count(ALWAYS_MASKED_Q16)
    if k < 0:
        out["status"] = "unsupported: no finite sample-derived confidence threshold"
        return out
    index = n - k - 1
    threshold = sorted(values)[index]
    bound = float(beta.ppf(float(c), k + 1, n - k))
    out.update({"index": index, "rank": index + 1, "pfa_upper": bound,
                "selected_requirement" if q16 else "selected_score": threshold})
    if q16 and threshold == ALWAYS_MASKED_Q16:
        out["status"] = "unavailable: confidence threshold selects always-masked sentinel"
        return out
    exceeded = sum(v > threshold for v in values)
    out.update({"exceedances": int(exceeded), "empirical_pfa": exceeded / n})
    if n < minimum:
        out["status"] = f"unsupported: independent trials {n} < {minimum}"
    elif not independent_null:
        out["status"] = "unavailable: independent null sampling not established"
    elif bound > float(a):
        out["status"] = "unavailable: confidence bound exceeds target"
    else:
        out.update({"status": "available", "threshold": threshold})
    return out


def confidence_threshold(responses, *, pfa, confidence=0.95, independent_null=False, independence_source=""):
    """Smallest fixed-rank score threshold with the declared iid coverage.

    Same-sample selection is valid because the rank is fixed by n, pfa and
    confidence. Ties keep the design-count bound, not a tighter random bound.
    Unknown dependence is diagnostic only; no input observations are dropped.
    """
    values = _real_vector(responses, "null responses").tolist()
    return _confidence_threshold(values, pfa=pfa, confidence=confidence, independent_null=independent_null,
                                 independence_source=independence_source, q16=False)


def exact_q16_confidence_threshold(requirements, *, pfa, confidence=0.95,
                                   independent_null=False, independence_source=""):
    """Smallest integer keep boundary with fixed-rank iid coverage.

    Sentinel requirements remain trials. Selecting the sentinel refuses a
    finite threshold. Decode serialized zero+bitset requirements first.
    """
    return _confidence_threshold(_requirements(requirements), pfa=pfa, confidence=confidence,
                                 independent_null=independent_null, independence_source=independence_source, q16=True)


def gate_count_limit(trials: int, nominal_pfa: float, *, family_tests: int = 1334,
                     confidence: float = .95, cap_multiple: float = 2.) -> int:
    """Full low-tail acceptance region, including pointwise CP width.

    Refuses a noncontiguous region instead of presuming cap-only acceptance.
    """
    end = accepted_count_limit(trials, cap_multiple*nominal_pfa,
                               family_tests=family_tests, confidence=confidence)
    if end < 0:
        return -1
    k = np.arange(end+1)
    alpha = (1-confidence)/2
    lower = np.zeros(end+1)
    lower[1:] = beta.ppf(alpha, k[1:], trials-k[1:]+1)
    upper = beta.isf(alpha, k+1, trials-k)
    accepted = upper-lower <= nominal_pfa
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return -1
    last = int(indices[-1])
    if not np.all(accepted[:last+1]):
        raise ValueError("acceptance region is not a contiguous lower tail")
    return last


def prospective_power(calibration_trials: int, validation_trials: int,
                      nominal_pfa: float, *, family_tests: int = 1334) -> dict:
    """Reference power for fresh independent calibration and validation."""
    ref = calibration_reference(calibration_trials, nominal_pfa)
    limit = gate_count_limit(validation_trials, nominal_pfa, family_tests=family_tests)
    # Sum the rejection tail directly: 1-CDF loses precision for large designs.
    failure = float(np.exp(logsumexp(betabinom.logpmf(
        np.arange(limit+1, validation_trials+1), validation_trials,
        ref['beta_a'], ref['beta_b'])))) if limit >= 0 else 1.
    failure = min(1., max(0., failure))
    power = 1-failure
    return {"calibration_trials": calibration_trials, "validation_trials": validation_trials,
            "nominal_pfa": nominal_pfa, "accepted_exceedances_max": limit,
            "single_policy_power": power, "single_policy_failure_probability": failure,
            "reference": "continuous same-null IID order-statistic calibration and fresh fixed-size validation"}


def family_power_lower_bound(policy_failure_probabilities: list[float]) -> float:
    """Union-bound lower bound; valid without independence among policies."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in policy_failure_probabilities):
        raise ValueError("failure probabilities must be finite and within [0,1]")
    return max(0., 1-math.fsum(policy_failure_probabilities))


# the names this module has always exported (the dissertation's null replay imports the first two)
empirical_null_threshold = empirical_threshold
_higher_index = higher_index
exact_q16_null_threshold = exact_q16_threshold


def binomial_interval(successes, trials, *, confidence=0.95, side="two-sided"):
    """Exact Clopper-Pearson bounds; k counts events, which may be false alarms."""
    from scipy.stats import beta

    n = _integer(trials, "trials", 1)
    k = _integer(successes, "successes")
    if k > n:
        raise ValueError("successes cannot exceed trials")
    level = float(_probability(confidence, "confidence"))
    if side not in {"two-sided", "upper", "lower"}:
        raise ValueError("side must be two-sided, upper or lower")
    alpha = 1.0-level
    tail = alpha/2 if side == "two-sided" else alpha
    lo = 0.0 if k == 0 or side == "upper" else float(beta.ppf(tail, k, n-k+1))
    hi = 1.0 if k == n or side == "lower" else float(beta.isf(tail, k+1, n-k))
    return {"successes": k, "trials": n, "estimate": k/n, "lower": lo,
            "upper": hi, "confidence": level, "side": side,
            "method": "Clopper-Pearson exact binomial", "independent_trial_assumption": True}


def false_alarm_validation(exceedances, trials, *, nominal_pfa, cap_pfa,
                           family_tests=1, confidence=0.95):
    """Pointwise precision plus a separate Bonferroni family upper-cap check.

    The cap is an explicit engineering tolerance, not the nominal calibration
    target. Dependence between policies is allowed; trials within a policy must
    be independent. Call with the complete predeclared family size.
    """
    nominal = float(_probability(nominal_pfa, "nominal_pfa"))
    cap = float(_probability(cap_pfa, "cap_pfa"))
    family = _integer(family_tests, "family_tests", 1)
    level = float(_probability(confidence, "confidence"))
    pointwise = binomial_interval(exceedances, trials, confidence=level)
    per_test = 1-(1-level)/family
    if per_test == 1.0:
        raise ValueError("family size exceeds floating confidence resolution")
    simultaneous = binomial_interval(exceedances, trials, confidence=per_test, side="upper")
    return {"pointwise": pointwise, "simultaneous_upper": simultaneous,
            "family_tests": family, "family_confidence": level,
            "nominal_pfa": nominal, "cap_pfa": cap,
            "cap_demonstrated": simultaneous["upper"] <= cap,
            "pointwise_interval_width": pointwise["upper"]-pointwise["lower"],
            "pointwise_width_at_most_nominal": pointwise["upper"]-pointwise["lower"] <= nominal,
            "physical_acceptance": False}


def raw_crossing_brackets(snr_db, rates, *, target):
    """Report every raw adjacent crossing and target plateau, without smoothing.

    A unique result needs one upward crossing, no downward crossing and no
    multi-point target plateau. Endpoints outside the sampled range are never
    extrapolated. Linear interpolation is labeled as a display estimate.
    """
    x = np.asarray(snr_db, dtype=float)
    y = np.asarray(rates, dtype=float)
    t = float(_probability(target, "target"))
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape:
        raise ValueError("at least two aligned SNR/rate points are required")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("SNR grid must be finite and strictly increasing")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("rates must be finite probabilities; no rows are dropped")
    upward, downward, plateau = [], [], []
    for i in range(len(x)-1):
        if y[i] == t == y[i+1]:
            plateau.append([i, i+1])
        direction = None
        if y[i] < t <= y[i+1]:
            direction = upward
        elif y[i] > t >= y[i+1]:
            direction = downward
        if direction is not None:
            estimate = x[i]+(x[i+1]-x[i])*(t-y[i])/(y[i+1]-y[i])
            direction.append({"indices": [i, i+1], "snr_lo_db": float(x[i]),
                              "snr_hi_db": float(x[i+1]), "rate_lo": float(y[i]),
                              "rate_hi": float(y[i+1]), "estimate_db": float(estimate)})
    # A curve that touches the target and turns back is not an identified crossing.
    touches = [i for i in range(1, len(y)-1) if y[i] == t and
               ((y[i-1] < t and y[i+1] < t) or (y[i-1] > t and y[i+1] > t))]
    ambiguous = len(upward) > 1 or bool(downward or plateau or touches)
    unique = len(upward) == 1 and not ambiguous
    return {"target": t, "upward_brackets": upward, "downward_brackets": downward,
            "plateau_segments": plateau, "touch_indices": touches,
            "exact_target_indices": np.flatnonzero(y == t).tolist(),
            "ambiguous": ambiguous, "status": "unique" if unique else "ambiguous" if ambiguous else "unbracketed",
            "estimate_db": upward[0]["estimate_db"] if unique else None,
            "raw_rates": y.tolist(), "snr_db": x.tolist(),
            "method": "raw adjacent sampled crossings; linear display interpolation; no extrapolation"}
