"""The knee: each channel's fine threshold word on its current era.

The knee is an exact Q16 keep word read from the equal-event distribution of a
channel's current-era fine keep boundaries: the word that maximizes the
cumulative curve minus the straight line between the smallest and largest
words, plotted against log word. Every source event carries equal weight and
its frames share that weight equally. The curve can bend two ways: above the
line, where a population of low-scoring frames ends (the usual case), or below
it, where nearly every frame carries the signal and a population of
high-scoring frames begins. The knee is the stronger of the two bends, the word
where the curve stands farthest from the line on either side (rule v2). Rule
v1, the bend above the line only, reproduces the knees recorded before
2026-10-19. The rule describes the data it is read
from. It is not a false-alarm calibration, a residual estimate or an
observing-time optimum, and it never chooses reject-all; whether a channel is
deployed at its knee is the threshold rule's decision
(:mod:`pilot_proxy.characterization.threshold_rule`).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

MAX_WORD = (1 << 64) - 1
ALWAYS_MASKED = 1 << 64
Q16 = 65536
RULE_ID = "equal-event-log-ecdf-knee-v2"
RULE_ID_V1 = "equal-event-log-ecdf-knee-v1"
RULE = {
    "id": RULE_ID,
    "population": "Current-era selected timed frames, channels separate.",
    "weight": "Each source event has equal total weight; frames share their event weight equally.",
    "cdf": "Conditional CDF of finite deployable requirements using exact integer event weights; "
           "zero scores map to word 1.",
    "knee": "D(word) = F(word) - (log(word)-log(min_word))/(log(max_word)-log(min_word)); take the word "
            "of largest |D|: the largest D (the bend above the line) unless the most negative D is larger in "
            "magnitude (the bend below it). Larger word breaks exact ties within a side; the bend above wins an "
            "exact tie between sides. Rule v1 takes the largest D only.",
    "constant_distribution": "Use its single exact deployable word, retaining all finite supported scores.",
    "unsupported": "Invalid scores and always-masked sentinels are counted separately. No finite "
                   "supported scores means no numeric proposal.",
}


class CohortSupportError(ValueError):
    """The requested calendar cohort cannot be formed from whole source events."""


def _integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an exact integer")
    return int(value)


def propose(requirements: Sequence, event_ids: Sequence[str], *, valid=None, always_masked=None,
            rule: str = RULE_ID) -> dict:
    """The knee of one channel's current-era frames.

    ``requirements`` are the exact Q16 keep boundaries (2**64 means always
    masked), ``event_ids`` the source event of each frame. ``valid`` marks
    frames whose score exists; ``always_masked`` marks sentinel frames.
    ``rule`` is ``RULE_ID`` (v2, the stronger bend) or ``RULE_ID_V1`` (the
    bend above the line, the knees recorded before 2026-10-19).
    """
    if rule not in (RULE_ID, RULE_ID_V1):
        raise ValueError(f"unknown knee rule {rule!r}")
    words = [_integer(value, "requirement") for value in requirements]
    events = list(event_ids)
    n = len(words)
    if len(events) != n or any(not isinstance(event, str) or not event for event in events):
        raise ValueError("requirements and nonempty source event keys must align")
    if any(value < 0 or value > ALWAYS_MASKED for value in words):
        raise ValueError("requirement is outside the score encoding")
    valid = [True] * n if valid is None else list(valid)
    always = [False] * n if always_masked is None else list(always_masked)
    if len(valid) != n or len(always) != n:
        raise ValueError("support masks must align with requirements")
    if any(not isinstance(value, (bool, np.bool_)) for value in valid + always):
        raise ValueError("support masks must be boolean")
    always = [bool(flag) or word == ALWAYS_MASKED for flag, word in zip(always, words)]
    counts = Counter(events)
    common = math.lcm(*counts.values()) if counts else 1
    weights = [common // counts[event] for event in events]
    total_weight = common * len(counts)
    supported = [bool(ok) and not flag for ok, flag in zip(valid, always)]
    finite_words = [max(1, word) for word, ok in zip(words, supported) if ok]
    finite_weights = [weight for weight, ok in zip(weights, supported) if ok]
    by_word = Counter()
    for word, weight in zip(finite_words, finite_weights):
        by_word[word] += weight
    result = {
        "rule_id": rule, "current_frames": n, "current_events": len(counts),
        "invalid_frames": sum(not bool(value) for value in valid),
        "always_masked_frames": sum(flag and bool(ok) for flag, ok in zip(always, valid)),
        "zero_score_frames": sum(word == 0 and ok for word, ok in zip(words, supported)),
        "finite_supported_frames": len(finite_words),
    }
    if not by_word:
        return dict(result, status="unavailable_no_finite_supported_scores", threshold_q16=None)
    support_weight = sum(finite_weights)
    grid = sorted(by_word)
    cumulative = 0
    best_word, best_objective = grid[-1], -math.inf
    low_word, low_objective = grid[-1], math.inf
    for word in grid:
        cumulative += by_word[word]
        if grid[-1] == grid[0]:
            objective = 0.0
        else:
            # The threshold stays an exact input word; logarithms only shape
            # the knee objective.
            span = math.log1p((grid[-1] - grid[0]) / grid[0])
            location = math.log1p((word - grid[0]) / grid[0])
            objective = cumulative / support_weight - location / span
        if objective >= best_objective:
            best_word, best_objective = word, objective
        if objective <= low_objective:
            low_word, low_objective = word, objective
    bend = "above"
    if rule == RULE_ID and -low_objective > best_objective:
        best_word, best_objective, bend = low_word, low_objective, "below"
    kept = [ok and max(1, word) <= best_word for word, ok in zip(words, supported)]
    kept_by_event = Counter(event for event, keep in zip(events, kept) if keep)
    kept_weight = sum(weight for weight, keep in zip(weights, kept) if keep)
    result.update({
        "status": "knee", "threshold_q16": str(best_word),
        "threshold_eta": best_word / Q16, "threshold_exact_eta": f"{best_word}/{Q16}",
        "knee_word_q16": str(best_word), "knee_objective": best_objective, "knee_bend": bend,
        "minimum_finite_word_q16": str(grid[0]), "maximum_finite_word_q16": str(grid[-1]),
        "kept_frames": sum(kept), "surviving_events": len(kept_by_event),
        "frame_retention": sum(kept) / n,
        "equal_event_retention": kept_weight / total_weight,
    })
    return result


def _month_index(label: str) -> int:
    year, month = map(int, label.split("-"))
    return 12 * year + month - 1


def cohort_rows(data: Mapping, months: Sequence[str]) -> np.ndarray:
    """Rows of a fine histogram export that fall in the given calendar months.

    A cohort must hold whole source events; a month list that splits one is
    refused rather than silently reweighted.
    """
    wanted = np.asarray([_month_index(month) for month in months], dtype=np.int64)
    timed = data["selected"] & data["finite_time"]
    selected = timed & np.isin(data["frame_month"], wanted)
    units = data["frame_unit_index"]
    total = np.bincount(units[timed], minlength=len(data["source_event_keys"]))
    retained = np.bincount(units[selected], minlength=len(total))
    if np.any((retained > 0) & (retained < total)):
        raise CohortSupportError("calendar cohort splits a source event")
    return np.flatnonzero(selected)


def propose_from_export(data: Mapping, months: Sequence[str], *, rule: str = RULE_ID) -> dict:
    """The knee of a fine histogram export restricted to ``months``."""
    rows = cohort_rows(data, months)
    requirements = data["fine_required_multiplier_q16"][rows]
    always = data["fine_always_masked"][rows]
    valid = data["fine_score_computed"][rows] & data["fine_rank_valid"][rows]
    # A zero requirement is a supported zero score only when the export says
    # so; it must not hide an unknown score.
    if np.any(valid & ~always & (requirements == 0) & ~data["fine_score_zero"][rows]):
        raise ValueError("zero encoded requirement lacks explicit zero-score support")
    events = data["source_event_keys"][data["frame_unit_index"][rows]]
    return propose(requirements, list(map(str, events)), valid=list(valid), always_masked=list(always), rule=rule)
