"""
Rule-based training plan week generation — pure business logic, no database.

Given a :class:`~openkoutsi.plan_schema.PlanConfig` this module produces, for
every week of a plan:

* a build-vs-recovery **week role** with a controlled week-over-week progression
  (a configurable 5–10% ramp across ``build_weeks`` build weeks, then a recovery
  week — a "2/1" or "3/1" cadence),
* per-day durations and target Loads that stay **consistent** with each other
  and with the day's textual description (the description is rendered from the
  actual scaled interval structure, not a fixed string), and
* optional mapping of each week's total ride time into the athlete's available
  **weekly-hours band** (recovery weeks near the low end, peak build weeks near
  the high end).

It also enforces some deterministic realism: threshold/VO2max ("hard") days are
capped per week and never scheduled back-to-back — excess hard days are eased to
tempo — so a plan never prescribes, say, hard threshold work every day.
"""

from __future__ import annotations

from typing import Optional

from openkoutsi.plan_schema import PlanConfig


# day_of_week: 1=Mon ... 7=Sun
_BASE_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 2, "workout_type": "threshold", "duration_min": 60,   "target_load": 80,   "description": "2×20 min at threshold power"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_load": 40,   "description": "Zone 2 aerobic"},
    {"day_of_week": 4, "workout_type": "endurance", "duration_min": 75,   "target_load": 55,   "description": "Steady endurance with some tempo efforts"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 120,  "target_load": 90,   "description": "Long easy endurance ride"},
    {"day_of_week": 7, "workout_type": "recovery",  "duration_min": 45,   "target_load": 25,   "description": "Active recovery spin"},
]

_PEAK_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 2, "workout_type": "vo2max",    "duration_min": 60,   "target_load": 90,   "description": "5×5 min VO2max intervals"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_load": 40,   "description": "Zone 2 aerobic"},
    {"day_of_week": 4, "workout_type": "threshold", "duration_min": 90,   "target_load": 100,  "description": "3×20 min threshold"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 150,  "target_load": 120,  "description": "Long endurance with race-pace effort"},
    {"day_of_week": 7, "workout_type": "recovery",  "duration_min": 45,   "target_load": 25,   "description": "Active recovery"},
]

_RECOVERY_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 2, "workout_type": "recovery",  "duration_min": 45,   "target_load": 25,   "description": "Easy spin"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_load": 35,   "description": "Zone 2"},
    {"day_of_week": 4, "workout_type": "tempo",     "duration_min": 60,   "target_load": 55,   "description": "Moderate tempo"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 90,   "target_load": 65,   "description": "Shorter long ride"},
    {"day_of_week": 7, "workout_type": "rest",      "duration_min": None, "target_load": None, "description": None},
]

# Base parameters per workout type: (duration_min, target_load, description).
# The description here is only a fallback; :func:`describe_workout` renders text
# from the actual scaled numbers for the structured types.
_BASE_PARAMS: dict[str, tuple[int, int, str]] = {
    "easy":           (60,  40,  "Zone 2 aerobic endurance"),
    "recovery":       (50,  30,  "Easy active-recovery spin"),
    "tempo":          (60,  65,  "Tempo effort at ~75-85% FTP"),
    "threshold":      (60,  80,  "2×20 min at threshold power"),
    "vo2max":         (60,  90,  "5×5 min VO2max intervals"),
    "endurance":      (90,  70,  "Steady aerobic endurance ride"),
    "long":           (120, 90,  "Long steady endurance ride"),
    "strength":       (45,  20,  "Off-bike strength session"),
    "yoga":           (30,  10,  "Flexibility and recovery"),
    "cross-training": (60,  40,  "Cross-training session"),
    "rest":           (0,   0,   "Rest day"),
}

# Off-bike sessions keep a fixed prescription: they are not part of the ride
# volume the weekly-hours band governs, and they are not scaled by progression.
_OFF_BIKE_TYPES = frozenset({"strength", "yoga"})

# High-intensity ride types, for the "not too often / never back-to-back" rule.
_HARD_TYPES = frozenset({"threshold", "vo2max"})

# Maximum hard (threshold/VO2max) days per week, keyed by intensity preference
# (the dialog seeds intensity from the athlete's experience level).
_MAX_HARD_DAYS: dict[str, int] = {"low": 1, "moderate": 2, "high": 3}


def week_template(week_num: int, total_weeks: int, goal: Optional[str]) -> list[dict]:
    """Choose the template for a given week number (legacy / no-config path)."""
    if week_num == total_weeks:
        return _RECOVERY_WEEK
    if week_num % 4 == 0:
        return _RECOVERY_WEEK
    if goal == "peak_fitness" and week_num >= total_weeks - 3:
        return _PEAK_WEEK
    return _BASE_WEEK


def progression_factor(week_num: int, total_weeks: int, periodization: str) -> float:
    """
    Legacy progression multiplier (0.6 – 1.3) for the no-config path.

    Retained for backward compatibility; the config-driven builder uses
    :func:`progression_scale` instead.
    """
    if week_num % 4 == 0 or week_num == total_weeks:
        return 0.7

    if periodization == "race_prep":
        if week_num >= total_weeks - 1:
            return 0.75
        progress = week_num / max(total_weeks - 2, 1)
        return 0.85 + progress * 0.45
    elif periodization == "maintenance":
        return 1.0
    else:  # base_building (default)
        progress = week_num / total_weeks
        return 0.85 + progress * 0.25


def intensity_multiplier(intensity_preference: str) -> float:
    return {"low": 0.85, "moderate": 1.0, "high": 1.15}.get(intensity_preference, 1.0)


# ---------------------------------------------------------------------------
# Week role & progression (config-driven)
# ---------------------------------------------------------------------------

def _cfg_int(config: PlanConfig, name: str, default: int) -> int:
    value = getattr(config, name, None)
    return int(value) if value is not None else default


def _cfg_float(config: PlanConfig, name: str, default: float) -> float:
    value = getattr(config, name, None)
    return float(value) if value is not None else default


def is_recovery_week(week_num: int, total_weeks: int, build_weeks: int) -> bool:
    """Whether ``week_num`` is a recovery week under a ``build_weeks``:1 cadence.

    The final week of a plan is always a recovery/taper week.
    """
    cycle_len = max(build_weeks, 1) + 1
    return week_num % cycle_len == 0 or week_num == total_weeks


def week_role(week_num: int, total_weeks: int, build_weeks: int) -> str:
    """``"recovery"`` or ``"build"`` for a given week."""
    return "recovery" if is_recovery_week(week_num, total_weeks, build_weeks) else "build"


def block_position(week_num: int, build_weeks: int) -> int:
    """1-indexed position of ``week_num`` within its build/recovery cycle.

    Returns 1..``build_weeks`` for build weeks and ``build_weeks + 1`` for the
    recovery week that closes the cycle.
    """
    cycle_len = max(build_weeks, 1) + 1
    return ((week_num - 1) % cycle_len) + 1


def progression_scale(week_num: int, total_weeks: int, config: PlanConfig) -> float:
    """Raw week multiplier combining intensity, ramp, block overload and recovery.

    * within a build block, each week ramps by ``weekly_progression_pct`` %,
    * each new block starts ~half a step above the previous block's baseline
      (progressive overload),
    * recovery weeks drop to ``recovery_week_factor`` of the block baseline,
    * ``race_prep`` tapers the final two weeks, ``maintenance`` stays flat.

    The result is capped to keep long plans from running away. The same curve
    is reused to place each week within the weekly-hours band.
    """
    periodization = getattr(config, "periodization", "base_building")
    intensity = intensity_multiplier(getattr(config, "intensity_preference", "moderate"))
    build_weeks = _cfg_int(config, "build_weeks", 3)
    pct = _cfg_float(config, "weekly_progression_pct", 7.0) / 100.0
    recovery_factor = _cfg_float(config, "recovery_week_factor", 0.6)

    cycle_len = max(build_weeks, 1) + 1
    block_index = (week_num - 1) // cycle_len
    recovery = is_recovery_week(week_num, total_weeks, build_weeks)

    # race_prep taper: shed load over the final two weeks regardless of cadence.
    if periodization == "race_prep" and week_num >= total_weeks - 1:
        return intensity * (0.55 if week_num == total_weeks else 0.75)

    if periodization == "maintenance":
        # Flat load; recovery weeks still dip.
        return intensity * (recovery_factor if recovery else 1.0)

    step = 1.0 + pct
    block_bump = 1.0 + pct * 0.5 * block_index  # gentle block-over-block overload
    if recovery:
        raw = intensity * recovery_factor * block_bump
    else:
        position = block_position(week_num, build_weeks)  # 1..build_weeks
        raw = intensity * (step ** (position - 1)) * block_bump

    # Ceiling so many blocks can't compound into an unreasonable load.
    return min(raw, intensity * 1.6)


def _week_scales(total_weeks: int, config: PlanConfig) -> list[float]:
    """Progression scale for weeks 1..total_weeks (index 0 == week 1)."""
    return [progression_scale(w, total_weeks, config) for w in range(1, total_weeks + 1)]


def _hours_band(config: PlanConfig) -> Optional[tuple[float, float]]:
    lo = getattr(config, "weekly_hours_min", None)
    hi = getattr(config, "weekly_hours_max", None)
    if lo is None and hi is None:
        return None
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    lo, hi = float(lo), float(hi)
    return (min(lo, hi), max(lo, hi))


def _week_hours_target(week_num: int, total_weeks: int, config: PlanConfig) -> Optional[float]:
    """Target total ride hours for a week, mapping the progression curve into the band."""
    band = _hours_band(config)
    if band is None:
        return None
    lo, hi = band
    scales = _week_scales(total_weeks, config)
    s = scales[week_num - 1]
    s_min, s_max = min(scales), max(scales)
    frac = 0.5 if s_max <= s_min else (s - s_min) / (s_max - s_min)
    return lo + frac * (hi - lo)


# ---------------------------------------------------------------------------
# Day descriptions (rendered from the actual scaled numbers)
# ---------------------------------------------------------------------------

# Threshold interval shape keyed by intensity preference. Tuple is
# (rep_min, rest_min, low_pct, high_pct, max_total_work_min).
_THRESHOLD_SHAPE: dict[str, tuple[int, int, int, int, int]] = {
    "low":      (10, 5, 95, 100, 30),
    "moderate": (12, 4, 98, 102, 45),
    "high":     (15, 4, 100, 105, 60),
}

_VO2_SHAPE: dict[str, tuple[int, int, int, int, int]] = {
    "low":      (3, 3, 108, 115, 20),
    "moderate": (4, 4, 110, 118, 25),
    "high":     (5, 4, 112, 120, 30),
}


def _interval_text(
    label: str,
    duration_min: int,
    shape: dict[str, tuple[int, int, int, int, int]],
    intensity_preference: str,
    block_index: int,
) -> str:
    rep, rest, low, high, cap = shape.get(intensity_preference, shape["moderate"])
    # Early blocks start a touch easier at the top end.
    if block_index == 0:
        high = max(low, high - 3)
    # Scale the warm-up/cool-down down for short sessions so the described
    # structure never exceeds the prescribed duration (a 20-minute threshold day
    # can't hold a 15-minute rep plus a 10+10 warm-up/cool-down).
    warmup = cooldown = max(5, min(10, round(duration_min * 0.15)))
    avail = max(duration_min - warmup - cooldown, 5)
    if avail < rep + rest:
        # Not enough room for a full rep plus recovery: one shortened rep.
        reps = 1
        rep = max(5, min(rep, avail))
    else:
        # N reps need N·rep + (N−1)·rest of work time (last rep has no trailing
        # recovery). Cap by the type's total at-intensity budget.
        reps = max(1, min((avail + rest) // (rep + rest), cap // rep))
    return (
        f"{reps}×{rep} min {label} ({low}–{high}% FTP), {rest} min easy between; "
        f"{warmup} min warm-up and {cooldown} min cool-down."
    )


_PROSE: dict[str, str] = {
    "recovery": "Easy Zone 1–2 recovery spin — low power, high cadence; leave it feeling fresher.",
    "easy": "Zone 2 aerobic endurance — conversational pace to build base fitness.",
    "tempo": "Tempo — sustained ~76–90% FTP; comfortably hard with controlled breathing.",
    "endurance": "Steady Zone 2 endurance — smooth aerobic effort, optional short tempo surges.",
    "long": "Long aerobic ride — mostly Zone 2; fuel well and keep the effort steady.",
    "strength": "Off-bike strength — core, glutes and legs to support on-bike power.",
    "yoga": "Mobility and flexibility work to aid recovery.",
    "cross-training": "Cross-training — steady aerobic effort in another discipline.",
    "rest": "Rest day — recover and let the training adapt.",
}


def describe_workout(
    workout_type: str,
    duration_min: Optional[int],
    *,
    intensity_preference: str = "moderate",
    block_index: int = 0,
) -> str:
    """Render a focus-oriented description consistent with ``duration_min``.

    Structured types (threshold/VO2max) get an interval breakdown derived from
    the duration so the text always matches the prescribed volume; other types
    get focus prose.
    """
    wtype = (workout_type or "").lower()
    if wtype == "threshold" and duration_min:
        return _interval_text("at threshold", duration_min, _THRESHOLD_SHAPE,
                              intensity_preference, block_index)
    if wtype == "vo2max" and duration_min:
        return _interval_text("VO2max", duration_min, _VO2_SHAPE,
                              intensity_preference, block_index)
    return _PROSE.get(wtype, _BASE_PARAMS.get(wtype, (0, 0, "Training session"))[2])


# ---------------------------------------------------------------------------
# Hard-day guardrail
# ---------------------------------------------------------------------------

def _apply_hard_day_guardrail(
    configured_days: dict[int, "object"], intensity_preference: str
) -> dict[int, tuple[str, bool]]:
    """Resolve the effective workout type per configured day.

    Enforces two rules on high-intensity (threshold/VO2max) days:

    * at most ``_MAX_HARD_DAYS`` of them per week, and
    * never on back-to-back calendar days.

    Excess hard days are eased to ``tempo``. Returns ``{day: (type, eased)}``
    where ``eased`` is True when the day was downgraded.
    """
    cap = _MAX_HARD_DAYS.get(intensity_preference, 2)
    resolved: dict[int, tuple[str, bool]] = {}
    hard_count = 0
    last_hard_day: Optional[int] = None
    for day in range(1, 8):
        dc = configured_days.get(day)
        if dc is None:
            continue
        wtype = dc.workout_type
        if wtype in _HARD_TYPES:
            back_to_back = last_hard_day is not None and day == last_hard_day + 1
            if hard_count >= cap or back_to_back:
                resolved[day] = ("tempo", True)
                continue
            hard_count += 1
            last_hard_day = day
        resolved[day] = (wtype, False)
    return resolved


# ---------------------------------------------------------------------------
# Week building
# ---------------------------------------------------------------------------

def _rest_day(day_num: int) -> dict:
    return {
        "day_of_week": day_num,
        "workout_type": "rest",
        "duration_min": None,
        "target_load": None,
        "description": None,
    }


def build_week_from_config(config: PlanConfig, week_num: int, total_weeks: int) -> list[dict]:
    """Build a week's workout list (7 day dicts) from the user's PlanConfig."""
    intensity_pref = getattr(config, "intensity_preference", "moderate")
    build_weeks = _cfg_int(config, "build_weeks", 3)
    block_index = (week_num - 1) // (max(build_weeks, 1) + 1)
    scale = progression_scale(week_num, total_weeks, config)

    configured_days = {dc.day_of_week: dc for dc in config.day_configs}
    resolved = _apply_hard_day_guardrail(configured_days, intensity_pref)

    # First pass: base ride durations for the volume distribution (on-bike only).
    hours_target = _week_hours_target(week_num, total_weeks, config)
    ride_base_total = 0
    for day, (wtype, _eased) in resolved.items():
        if wtype in _OFF_BIKE_TYPES:
            continue
        base_dur = _BASE_PARAMS.get(wtype, _BASE_PARAMS["easy"])[0]
        ride_base_total += base_dur
    # Factor that maps the summed base ride minutes onto the hours-band target.
    if hours_target is not None and ride_base_total > 0:
        ride_factor = (hours_target * 60.0) / ride_base_total
    else:
        ride_factor = scale  # no band → scale base durations by progression

    week: list[dict] = []
    for day_num in range(1, 8):
        if day_num not in resolved:
            week.append(_rest_day(day_num))
            continue

        dc = configured_days[day_num]
        wtype, eased = resolved[day_num]
        base_dur, base_tss, _base_desc = _BASE_PARAMS.get(wtype, _BASE_PARAMS["easy"])

        if wtype in _OFF_BIKE_TYPES:
            # Fixed prescription, unaffected by progression or the hours band.
            duration = base_dur or None
            load = base_tss or None
        elif base_dur:
            duration = max(round(base_dur * ride_factor), 20)
            # Keep Load consistent with the (possibly renormalised) duration by
            # holding the type's Load-per-minute intensity.
            load_per_min = base_tss / base_dur
            load = round(duration * load_per_min)
        else:
            duration = None
            load = None

        # Description is rendered from the final duration so text ↔ numbers agree.
        description = dc.notes or describe_workout(
            wtype, duration,
            intensity_preference=intensity_pref,
            block_index=block_index,
        )
        if eased and not dc.notes:
            description += " (eased from a hard session — too many hard days this week)."

        week.append({
            "day_of_week": day_num,
            "workout_type": wtype,
            "duration_min": duration,
            "target_load": load,
            "description": description,
        })
    return week


def build_week_meta(
    config: PlanConfig,
    week_num: int,
    total_weeks: int,
    days: Optional[list[dict]] = None,
) -> dict:
    """Return week-level metadata for the plan header/summary.

    ``{week_number, week_type, focus, target_load, target_hours, base_load}``.

    ``days`` lets a caller (e.g. the LLM path) pass the week's actual generated
    day dicts so the reported target Load/hours match what was produced; when
    omitted the rule-based builder's output is summarised instead.
    """
    build_weeks = _cfg_int(config, "build_weeks", 3)
    base_load = _cfg_int(config, "weekly_base_load", 0)
    periodization = getattr(config, "periodization", "base_building")
    if days is None:
        days = build_week_from_config(config, week_num, total_weeks)

    total_load = sum(d["target_load"] or 0 for d in days)
    total_min = sum(d["duration_min"] or 0 for d in days)
    target_hours = round(total_min / 60.0, 1)

    recovery = is_recovery_week(week_num, total_weeks, build_weeks)
    if periodization == "race_prep" and week_num >= total_weeks - 1:
        week_type = "taper"
        focus = "Taper — shed fatigue and freshen up for your event; keep intensity, cut volume."
    elif recovery:
        week_type = "recovery"
        focus = "Recovery week — reduced load to absorb the block's training and rebuild form."
    else:
        week_type = "build"
        pos = block_position(week_num, build_weeks)
        pct = _cfg_float(config, "weekly_progression_pct", 7.0)
        focus = (
            f"Build week {pos} of {build_weeks} — progressive overload "
            f"(~{pct:.0f}% week-over-week)."
        )

    return {
        "week_number": week_num,
        "week_type": week_type,
        "focus": focus,
        "target_load": total_load,
        "target_hours": target_hours,
        "base_load": base_load,
    }


def build_all_week_meta(config: PlanConfig, total_weeks: int) -> list[dict]:
    """Week metadata for every week of a plan (rule-based)."""
    return [build_week_meta(config, w, total_weeks) for w in range(1, total_weeks + 1)]


def week_meta_from_weeks(config: PlanConfig, weeks_data: list[list[dict]]) -> list[dict]:
    """Week metadata summarising already-generated weeks (e.g. from the LLM).

    ``weeks_data`` is a list of weeks, each a list of day dicts with
    ``duration_min``/``target_load``. Week roles/focus come from the cadence in
    ``config``; the target Load/hours are summed from the actual days so the
    header matches the generated plan.
    """
    total_weeks = len(weeks_data)
    return [
        build_week_meta(config, w, total_weeks, days=weeks_data[w - 1])
        for w in range(1, total_weeks + 1)
    ]
