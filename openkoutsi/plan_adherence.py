"""Deterministic plan-adherence scoring math.

Two related plain-arithmetic scores (no LLM involved), in the spirit of the
Fitness/Fatigue/Form engine in :mod:`openkoutsi.fatigue_metrics`:

1. **Per-workout match score (0–100)** — how well the linked activity/activities
   fulfilled a single planned workout, penalising both under- and over-performing
   relative to the target Load and duration.
2. **Plan adherence score (0–100)** — a Load-weighted roll-up of the per-workout
   scores over the elapsed portion of a plan, with missed sessions counting as
   zero and skips graded by reason.

This module is pure (operates on primitive numbers); the DB orchestration and
persistence live in ``backend.app.services.plan_adherence``.

All constants below are finalised design decisions and are intentionally
hard-coded — there is no configuration surface.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

# ── Cycling Load/duration blend ──────────────────────────────────────────────
# Fixed weighting when both a target Load and a target duration are present.
LOAD_BLEND = 0.70
DURATION_BLEND = 0.30

# ── Auto-match gate ──────────────────────────────────────────────────────────
# A single activity must reach this fraction of the target Load and duration to
# auto-link to an otherwise-empty planned workout. Shared with
# ``activity_workout_matcher`` so the matcher gate and the score cannot drift
# apart: at 60% under target the per-dimension deviation score is exactly 0.60.
MATCH_THRESHOLD = 0.60

# ── Skip-reason → forgiveness factor ─────────────────────────────────────────
# A skipped workout contributes score 0 at weight ``(1 − f) × weight``: a fully
# excused skip barely dents the score, a discretionary one approaches a full
# miss. Plain lookup over the known reason set — no free-text parsing.
SKIP_FORGIVENESS: dict[str, float] = {
    "illness": 0.90,
    "injury": 0.90,
    "fatigue": 0.60,
    "travel": 0.50,
    "weather": 0.40,
}
# Unrecognised / free-form / no reason → near-full miss.
DISCRETIONARY_FORGIVENESS = 0.10

# ── Supplemental (non-cycling) weight ────────────────────────────────────────
# Supplemental workouts carry a flat weight, set a little below a typical
# cycling session: 0.75 × mean(cycling target_load in the plan), or a constant
# fallback when the plan has no cycling target loads.
SUPPLEMENTAL_WEIGHT_FACTOR = 0.75
SUPPLEMENTAL_WEIGHT_FALLBACK = 30.0


def deviation_score(actual: float, target: float) -> float:
    """Symmetric per-dimension score in ``[0, 1]``.

    On target → 1.0; 20% off either way → 0.8; ≥100% over, or 0, → 0.0. Over-
    and under-shooting are penalised equally.
    """
    if target is None or target <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - abs(actual - target) / target))


def meets_threshold(
    actual: Optional[float],
    target: Optional[float],
    threshold: float = MATCH_THRESHOLD,
) -> bool:
    """Whether *actual* reaches at least *threshold* of *target* (matcher gate).

    A missing/zero target is not a constraint, so it passes. Shared by
    ``activity_workout_matcher._matches`` and the scoring so the auto-match gate
    and the per-workout score stay defined against the same comparison.
    """
    if target is None or target <= 0:
        return True
    return (actual or 0.0) >= target * threshold


def cycling_match_score(
    actual_load: float,
    actual_duration_s: float,
    target_load: Optional[float],
    target_duration_min: Optional[float],
) -> float:
    """Per-workout match score (0–100) for a cycling session.

    Graded on Load + duration with the fixed 0.70/0.30 blend when both targets
    are present, otherwise on whichever target exists. With neither target set,
    falls back to completion-only (100 — the caller only scores completed
    workouts).
    """
    load_s: Optional[float] = None
    dur_s: Optional[float] = None
    if target_load is not None and target_load > 0:
        load_s = deviation_score(actual_load, target_load)
    if target_duration_min is not None and target_duration_min > 0:
        dur_s = deviation_score(actual_duration_s, target_duration_min * 60)

    if load_s is not None and dur_s is not None:
        return 100.0 * (LOAD_BLEND * load_s + DURATION_BLEND * dur_s)
    if load_s is not None:
        return 100.0 * load_s
    if dur_s is not None:
        return 100.0 * dur_s
    return 100.0


def supplemental_match_score(has_activity: bool) -> float:
    """Per-workout score for a supplemental (non-cycling) session: done/missed."""
    return 100.0 if has_activity else 0.0


def forgiveness_factor(skip_reason: Optional[str]) -> float:
    """Map a skip reason to its forgiveness factor ``f ∈ [0, 1]``."""
    if not skip_reason:
        return DISCRETIONARY_FORGIVENESS
    return SKIP_FORGIVENESS.get(skip_reason.strip().lower(), DISCRETIONARY_FORGIVENESS)


def supplemental_weight(cycling_target_loads: Iterable[float]) -> float:
    """Flat weight for supplemental workouts, derived from the plan's cycling load."""
    loads = [x for x in cycling_target_loads if x]
    if not loads:
        return SUPPLEMENTAL_WEIGHT_FALLBACK
    return SUPPLEMENTAL_WEIGHT_FACTOR * (sum(loads) / len(loads))


def plan_adherence(contributions: Iterable[Tuple[float, float]]) -> Optional[float]:
    """Load-weighted roll-up of ``(weight, score)`` pairs into a 0–100 score.

    ``adherence = 100 × Σ(weight_i × score_i / 100) / Σ(weight_i)``. Returns
    None when there is nothing contributing yet (empty / just-started plan).
    """
    total_weight = 0.0
    weighted = 0.0
    for weight, score in contributions:
        if weight <= 0:
            continue
        total_weight += weight
        weighted += weight * (score / 100.0)
    if total_weight <= 0:
        return None
    return 100.0 * weighted / total_weight
