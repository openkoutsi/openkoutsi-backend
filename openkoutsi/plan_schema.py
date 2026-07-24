from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DayConfig(BaseModel):
    day_of_week: int  # 1=Mon … 7=Sun
    workout_type: str  # "threshold", "easy", "long", "strength", "yoga", …
    notes: Optional[str] = None


class PlanConfig(BaseModel):
    days_per_week: int
    day_configs: list[DayConfig]
    periodization: str = "base_building"  # "base_building" | "race_prep" | "maintenance"
    intensity_preference: str = "moderate"  # "low" | "moderate" | "high"
    long_description: Optional[str] = None  # free-text for LLM

    # --- Structure/progression parameters (issue #29) -----------------------
    # All optional with sensible defaults so older stored configs (which lack
    # these keys) keep deserializing unchanged.

    # Week-over-week ramp within a build block, as a percentage (5–10% typical;
    # novices lower, experienced higher). Clamped by ``clamp_plan_params``.
    weekly_progression_pct: float = 7.0
    # Number of build weeks before a recovery week (a "2/1" or "3/1" cadence).
    build_weeks: int = 3
    # How far a recovery week scales down relative to the block baseline.
    recovery_week_factor: float = 0.6
    # Weekly Load coming from non-workout riding (commuting, etc.). Additive
    # context only — surfaced in week summaries and the LLM prompt; it does not
    # reduce the prescribed workout loads.
    weekly_base_load: int = 0
    # Weekly training-time the athlete has available, as a range in hours. When
    # both are set the builder maps each week's total ride time into this band
    # (recovery weeks near the low end, peak build weeks near the high end).
    # ``None`` leaves weekly volume unconstrained (legacy behaviour).
    weekly_hours_min: Optional[float] = None
    weekly_hours_max: Optional[float] = None


# --- Per-experience suggested defaults --------------------------------------
#
# Single source of truth for the structure parameters suggested for each
# self-reported experience level (see ``services.athlete_experience`` for the
# canonical level names). The plan create/regenerate dialog prefills from these;
# the values remain user-overridable per plan.
EXPERIENCE_PLAN_DEFAULTS: dict[str, dict] = {
    "novice":       {"weekly_progression_pct": 5.0,  "build_weeks": 2, "intensity_preference": "low"},
    "intermediate": {"weekly_progression_pct": 7.0,  "build_weeks": 3, "intensity_preference": "moderate"},
    "experienced":  {"weekly_progression_pct": 9.0,  "build_weeks": 3, "intensity_preference": "high"},
    "semi-pro":     {"weekly_progression_pct": 10.0, "build_weeks": 3, "intensity_preference": "high"},
    "elite":        {"weekly_progression_pct": 10.0, "build_weeks": 3, "intensity_preference": "high"},
}

# Used when the level is unset/unknown.
_FALLBACK_DEFAULTS = EXPERIENCE_PLAN_DEFAULTS["intermediate"]


def plan_defaults_for(level: Optional[str]) -> dict:
    """Return the suggested structure defaults for an experience ``level``.

    Falls back to the intermediate defaults for an unset or unrecognised level.
    """
    if level and level in EXPERIENCE_PLAN_DEFAULTS:
        return dict(EXPERIENCE_PLAN_DEFAULTS[level])
    return dict(_FALLBACK_DEFAULTS)


# Bounds for the numeric structure parameters, enforced server-side.
PROGRESSION_PCT_BOUNDS = (3.0, 12.0)
BUILD_WEEKS_BOUNDS = (2, 4)
RECOVERY_FACTOR_BOUNDS = (0.4, 0.9)
HOURS_BOUNDS = (0.0, 40.0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_plan_params(config: PlanConfig) -> PlanConfig:
    """Return a copy of ``config`` with structure parameters clamped to sane bounds.

    Keeps the builder and the LLM prompt from ever seeing runaway values (a 50%
    weekly ramp, a negative base load, an inverted hours range, …). Called by the
    API layer before generation and persistence.
    """
    data = config.model_dump()

    data["weekly_progression_pct"] = _clamp(
        float(data.get("weekly_progression_pct") or 0.0), *PROGRESSION_PCT_BOUNDS
    )
    data["build_weeks"] = int(
        _clamp(int(data.get("build_weeks") or 3), *BUILD_WEEKS_BOUNDS)
    )
    data["recovery_week_factor"] = _clamp(
        float(data.get("recovery_week_factor") or 0.6), *RECOVERY_FACTOR_BOUNDS
    )
    data["weekly_base_load"] = max(0, int(data.get("weekly_base_load") or 0))

    lo = data.get("weekly_hours_min")
    hi = data.get("weekly_hours_max")
    lo = _clamp(float(lo), *HOURS_BOUNDS) if lo is not None else None
    hi = _clamp(float(hi), *HOURS_BOUNDS) if hi is not None else None
    # If only one endpoint is given, treat it as a point value.
    if lo is not None and hi is None:
        hi = lo
    if hi is not None and lo is None:
        lo = hi
    # Keep the range ordered.
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    data["weekly_hours_min"] = lo
    data["weekly_hours_max"] = hi

    return PlanConfig(**data)
