"""
Rule-based training plan week generation — pure business logic, no database.
"""

from __future__ import annotations

from typing import Optional

from openkoutsi.plan_schema import PlanConfig


# day_of_week: 1=Mon ... 7=Sun
_BASE_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 2, "workout_type": "threshold", "duration_min": 60,   "target_tss": 80,   "description": "2×20 min at threshold power"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_tss": 40,   "description": "Zone 2 aerobic"},
    {"day_of_week": 4, "workout_type": "endurance", "duration_min": 75,   "target_tss": 55,   "description": "Steady endurance with some tempo efforts"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 120,  "target_tss": 90,   "description": "Long easy endurance ride"},
    {"day_of_week": 7, "workout_type": "recovery",  "duration_min": 45,   "target_tss": 25,   "description": "Active recovery spin"},
]

_PEAK_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 2, "workout_type": "vo2max",    "duration_min": 60,   "target_tss": 90,   "description": "5×5 min VO2max intervals"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_tss": 40,   "description": "Zone 2 aerobic"},
    {"day_of_week": 4, "workout_type": "threshold", "duration_min": 90,   "target_tss": 100,  "description": "3×20 min threshold"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 150,  "target_tss": 120,  "description": "Long endurance with race-pace effort"},
    {"day_of_week": 7, "workout_type": "recovery",  "duration_min": 45,   "target_tss": 25,   "description": "Active recovery"},
]

_RECOVERY_WEEK: list[dict] = [
    {"day_of_week": 1, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 2, "workout_type": "recovery",  "duration_min": 45,   "target_tss": 25,   "description": "Easy spin"},
    {"day_of_week": 3, "workout_type": "recovery",  "duration_min": 60,   "target_tss": 35,   "description": "Zone 2"},
    {"day_of_week": 4, "workout_type": "tempo",     "duration_min": 60,   "target_tss": 55,   "description": "Moderate tempo"},
    {"day_of_week": 5, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
    {"day_of_week": 6, "workout_type": "endurance", "duration_min": 90,   "target_tss": 65,   "description": "Shorter long ride"},
    {"day_of_week": 7, "workout_type": "rest",      "duration_min": None, "target_tss": None, "description": None},
]

# Base parameters per workout type: (duration_min, target_tss, description)
_BASE_PARAMS: dict[str, tuple[int, int, str]] = {
    "easy":           (60,  40,  "Zone 2 aerobic endurance"),
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
    Return a multiplier (0.6 – 1.3) that scales TSS/duration week over week.

    Patterns:
    - base_building:  gentle 3-week ramp + 1 recovery, reaching ~1.1× at peak
    - race_prep:      aggressive ramp to 1.3× with a taper in the final 2 weeks
    - maintenance:    flat at 1.0× with recovery weeks every 4th week
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


def build_week_from_config(config: PlanConfig, week_num: int, total_weeks: int) -> list[dict]:
    """Build a week's workout list from the user's PlanConfig."""
    prog = progression_factor(week_num, total_weeks, config.periodization)
    scale = prog * intensity_multiplier(config.intensity_preference)

    configured_days = {dc.day_of_week: dc for dc in config.day_configs}
    week = []
    for day_num in range(1, 8):
        if day_num not in configured_days:
            week.append({
                "day_of_week": day_num,
                "workout_type": "rest",
                "duration_min": None,
                "target_tss": None,
                "description": None,
            })
        else:
            dc = configured_days[day_num]
            base = _BASE_PARAMS.get(dc.workout_type, _BASE_PARAMS["easy"])
            base_dur, base_tss, base_desc = base

            is_recovery = (week_num % 4 == 0 or week_num == total_weeks)
            if is_recovery and dc.workout_type in ("strength", "yoga", "rest"):
                duration = base_dur or None
                tss = base_tss or None
            else:
                duration = round(base_dur * scale) if base_dur else None
                tss = round(base_tss * scale) if base_tss else None

            week.append({
                "day_of_week": day_num,
                "workout_type": dc.workout_type,
                "duration_min": duration,
                "target_tss": tss,
                "description": dc.notes or base_desc,
            })
    return week
