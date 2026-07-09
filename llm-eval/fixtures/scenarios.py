"""Representative evaluation scenarios for each LLM call site in openkoutsi.

Every scenario is built from the *same* in-memory ORM objects and config the
backend uses at runtime; the prompt files hand these straight to the real
prompt builders (``backend.app.services.llm_*``), so the text sent to a model
under evaluation is byte-identical to production. SQLAlchemy models are plain
attribute holders — instantiating them without a session is enough because the
builders only read attributes.

Add a scenario by adding an entry to the relevant ``*_SCENARIOS`` dict and a
matching test row in ``promptfooconfig.yaml``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402  (sets SECRET_KEY + repo root before backend import)

from datetime import date, datetime, timezone  # noqa: E402

from backend.app.models.user_orm import (  # noqa: E402
    Activity,
    ActivityInterval,
    Athlete,
    DailyMetric,
    Goal,
    PlannedWorkout,
    TrainingPlan,
)
from backend.app.schemas.plans import DayConfig, PlanConfig  # noqa: E402


def _athlete(**kw) -> Athlete:
    kw.setdefault("global_user_id", "eval-athlete")
    return Athlete(**kw)


# ── Family 1: training-plan generation (JSON) ────────────────────────────────
PLAN_SCENARIOS: dict[str, dict] = {
    "beginner_base_build": {
        "config": PlanConfig(
            days_per_week=4,
            day_configs=[
                DayConfig(day_of_week=2, workout_type="endurance"),
                DayConfig(day_of_week=4, workout_type="threshold", notes="key session"),
                DayConfig(day_of_week=6, workout_type="long"),
                DayConfig(day_of_week=7, workout_type="recovery"),
            ],
            periodization="base_building",
            intensity_preference="moderate",
            long_description="First structured block after a winter off the bike.",
        ),
        "goal": None,
        "num_weeks": 4,
        "ftp": 210,
        "ctl": 42.0,
    },
    "race_prep_gran_fondo": {
        "config": PlanConfig(
            days_per_week=5,
            day_configs=[
                DayConfig(day_of_week=1, workout_type="recovery"),
                DayConfig(day_of_week=2, workout_type="vo2max", notes="short, sharp"),
                DayConfig(day_of_week=4, workout_type="threshold"),
                DayConfig(day_of_week=6, workout_type="long", notes="ride the course profile"),
                DayConfig(day_of_week=7, workout_type="endurance"),
            ],
            periodization="race_prep",
            intensity_preference="high",
            long_description="Building toward a hilly 140 km gran fondo; taper the final week.",
        ),
        "goal": "Gran Fondo (140 km, 2500 m climbing) in 8 weeks",
        "num_weeks": 8,
        "ftp": 285,
        "ctl": 72.0,
    },
}

# ── Family 2: structured workout synthesis (JSON) ────────────────────────────
WORKOUT_SCENARIOS: dict[str, dict] = {
    "vo2max_intervals": {
        "planned": PlannedWorkout(
            workout_type="vo2max",
            description="5 x 4 min at VO2max with 4 min recoveries",
            duration_min=75,
            target_tss=95,
        ),
        "ftp": 265,
        "sport": "Ride",
    },
    "endurance_long_ride": {
        "planned": PlannedWorkout(
            workout_type="long",
            description="Steady endurance ride, mostly zone 2 with a few tempo surges",
            duration_min=180,
            target_tss=150,
        ),
        "ftp": 240,
        "sport": "Ride",
    },
    "sweetspot_over_unders": {
        "planned": PlannedWorkout(
            workout_type="threshold",
            description="3 x 12 min over-unders alternating 90s at 95% and 30s at 105% FTP",
            duration_min=70,
            target_tss=85,
        ),
        "ftp": 300,
        "sport": "Ride",
    },
}

# ── Family 3: activity analysis (prose + MOOD) ───────────────────────────────
_pr_ride = Activity(
    sport_type="Ride",
    start_time=datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
    duration_s=5400,
    distance_m=52000,
    elevation_m=680,
    avg_power=238,
    normalized_power=255,
    intensity_factor=0.93,
    tss=129.0,
    avg_hr=156,
    max_hr=182,
    labels=["race"],
    notes="Felt strong the whole way, attacked the final climb.",
)
_pr_ride.intervals = [
    ActivityInterval(interval_number=1, start_offset_s=0, duration_s=1200, avg_hr=148, avg_power=225),
    ActivityInterval(interval_number=2, start_offset_s=1200, duration_s=600, avg_hr=172, avg_power=290),
    ActivityInterval(interval_number=3, start_offset_s=1800, duration_s=300, avg_hr=178, avg_power=340),
]

_easy_ride = Activity(
    sport_type="Ride",
    start_time=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
    duration_s=2700,
    distance_m=20000,
    avg_power=120,
    normalized_power=128,
    intensity_factor=0.46,
    tss=24.0,
    avg_hr=118,
    max_hr=135,
    labels=["recovery"],
)

ACTIVITY_SCENARIOS: dict[str, dict] = {
    "pr_hard_ride": {
        "activity": _pr_ride,
        "athlete": _athlete(ftp=275, max_hr=188),
        "fatigue": DailyMetric(date=date(2026, 7, 4), ctl=78.0, atl=65.0, tsb=13.0),
        "power_pr_badges": {60: {"all_time": "gold"}, 300: {"3mo": "silver"}},
        "distance_pr_badges": None,
        "locale": None,
    },
    "recovery_ride_finnish": {
        "activity": _easy_ride,
        "athlete": _athlete(ftp=260, max_hr=185),
        "fatigue": DailyMetric(date=date(2026, 7, 5), ctl=80.0, atl=92.0, tsb=-12.0),
        "power_pr_badges": None,
        "distance_pr_badges": None,
        "locale": "fi",
    },
}

# ── Family 4: daily training-status (prose + MOOD) ───────────────────────────
_now = datetime(2026, 7, 9, 7, 30, tzinfo=timezone.utc)


def _status_common(coaching_style, locale, adhering: bool) -> dict:
    athlete = _athlete(
        ftp=250,
        max_hr=186,
        app_settings={"coaching_style": coaching_style, "locale": locale, "timezone": "UTC"},
    )
    recent = [
        Activity(sport_type="Ride", start_time=datetime(2026, 7, 6, 17, tzinfo=timezone.utc), duration_s=3600, tss=68.0),
        Activity(sport_type="Ride", start_time=datetime(2026, 7, 8, 17, tzinfo=timezone.utc), duration_s=4500, tss=92.0),
    ]
    metric = DailyMetric(date=date(2026, 7, 8), ctl=64.0, atl=71.0, tsb=-7.0)
    plan = TrainingPlan(name="Base to Build", start_date=date(2026, 6, 29), end_date=date(2026, 8, 24), weeks=8, status="active")
    # Current plan week is 2 (plan started Mon 2026-06-29).
    if adhering:
        week = [
            PlannedWorkout(week_number=2, day_of_week=1, workout_type="recovery", target_tss=30, completed_activity_id="a1"),
            PlannedWorkout(week_number=2, day_of_week=3, workout_type="threshold", target_tss=85, completed_activity_id="a2"),
            PlannedWorkout(week_number=2, day_of_week=4, workout_type="endurance", target_tss=60),  # today, not yet done
            PlannedWorkout(week_number=2, day_of_week=6, workout_type="long", target_tss=120),
        ]
    else:
        week = [
            PlannedWorkout(week_number=2, day_of_week=1, workout_type="recovery", target_tss=30),  # missed, no reason
            PlannedWorkout(week_number=2, day_of_week=3, workout_type="threshold", target_tss=85, skip_reason="felt tired"),
            PlannedWorkout(week_number=2, day_of_week=4, workout_type="endurance", target_tss=60),  # today, not yet done
            PlannedWorkout(week_number=2, day_of_week=6, workout_type="long", target_tss=120),
        ]
    goals = [Goal(title="Reach FTP 275 W before September", target_date=date(2026, 9, 1), status="active", target_value=275, current_value=250)]
    return {
        "athlete": athlete,
        "recent_activities": recent,
        "current_metric": metric,
        "active_plan": plan,
        "this_week_workouts": week,
        "active_goals": goals,
        "now": _now,
        "coaching_style": coaching_style,
        "locale": locale,
    }


STATUS_SCENARIOS: dict[str, dict] = {
    "on_track_friendly": _status_common("friendly", "en", adhering=True),
    "missed_sessions_stern": _status_common("stern", "en", adhering=False),
}
