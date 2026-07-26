"""Achievement / streak service (issue #33).

Deterministic and always-on (never gated behind the LLM subscription). Wraps the
pure catalogue and streak math in :mod:`openkoutsi.achievements` with the DB
orchestration:

- reduce the athlete's activities, plans and goals to the facts the rules need,
- work out which tiers are earned and *when* they were earned,
- reconcile that against the stored ``achievement_unlocks`` rows.

The reconcile is a full rewrite, not an append: rows are inserted, re-dated or
**deleted** so the table always matches the data, exactly like
``metrics_engine.catch_up_metrics`` and ``plan_adherence.catch_up_adherence``
self-heal their snapshots. Deleting an activity can therefore revoke a tier —
that is the deliberate trade for never showing a badge the history no longer
supports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.user_orm import (
    AchievementUnlock,
    Activity,
    Athlete,
    Goal,
    PlannedWorkout,
    TrainingPlan,
)
from backend.app.services.notifications import ACHIEVEMENT_UNLOCKED, notify_user
from backend.app.services.plan_adherence import score_plan
from openkoutsi.achievements import (
    CATALOGUE,
    CATALOGUE_BY_ID,
    ActivityFact,
    StreakState,
    bucket_by,
    comeback_date,
    cumulative_tier_dates,
    distinct_tier_dates,
    qualifies_active,
    qualifies_climbing,
    qualifies_distance,
    qualifies_multisport,
    qualifies_volume,
    streak_state,
    streak_tier_dates,
    threshold_tier_dates,
)
from openkoutsi.sport_matching import activity_category

log = logging.getLogger(__name__)

# Streak achievement id → (predicate, monthly?)
_STREAK_RULES: dict[str, tuple[Callable, bool]] = {
    "streak_active_weeks": (qualifies_active, False),
    "streak_volume_weeks": (qualifies_volume, False),
    "streak_multisport_weeks": (qualifies_multisport, False),
    "streak_distance_weeks": (qualifies_distance, False),
    "streak_climbing_weeks": (qualifies_climbing, False),
    "streak_active_months": (qualifies_active, True),
}


def gamification_enabled(athlete: Athlete) -> bool:
    """Whether the athlete wants achievements at all (default on).

    Mirrors ``ask_for_rpe``: the preference is offered in the onboarding wizard
    and in settings, and an unset value means on. Because the recompute is a
    pure function of the data, switching this back on restores every unlock
    exactly — there is no backfill step to run.
    """
    return (athlete.app_settings or {}).get("gamification") is not False


def _local_day(dt: Optional[datetime], tz: Optional[ZoneInfo]) -> Optional[date]:
    """Calendar date of *dt* in the athlete's timezone.

    Week boundaries have to match the athlete's own calendar, otherwise a Sunday
    evening ride in UTC+3 lands in the wrong week and silently breaks a streak.
    """
    if dt is None:
        return None
    aware = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return (aware.astimezone(tz) if tz else aware).date()


def _zone(athlete: Athlete) -> Optional[ZoneInfo]:
    tz_str = (athlete.app_settings or {}).get("timezone")
    if not tz_str:
        return None
    try:
        return ZoneInfo(tz_str)
    except (ZoneInfoNotFoundError, ValueError):
        return None


@dataclass
class AchievementComputation:
    """Everything the API needs, computed from the athlete's current data."""

    # achievement_id → {tier: date first met}
    tier_dates: dict[str, dict[float, date]] = field(default_factory=dict)
    # achievement_id → current value, in the definition's own unit
    progress: dict[str, float] = field(default_factory=dict)
    # streak achievement_id → current/longest/in_progress
    streaks: dict[str, StreakState] = field(default_factory=dict)
    # requirement keys the athlete's data actually supports ("distance", …)
    available: set[str] = field(default_factory=set)
    # achievement_id → deep-link payload for the earning activity, when there is one
    context: dict[str, dict] = field(default_factory=dict)

    def is_available(self, achievement_id: str) -> bool:
        requires = CATALOGUE_BY_ID[achievement_id].requires
        return requires is None or requires in self.available


def _first_activity_reaching(
    facts: list[ActivityFact], value_of: Callable[[ActivityFact], float], tier: float
) -> Optional[str]:
    """Id of the earliest activity whose value clears *tier* (for deep-linking)."""
    for fact in sorted(facts, key=lambda f: f.day):
        if value_of(fact) >= tier:
            return fact.activity_id
    return None


async def _load_facts(
    athlete_id: str, tz: Optional[ZoneInfo], session: AsyncSession
) -> list[ActivityFact]:
    """Reduce the athlete's activities to the columns the rules actually need."""
    result = await session.execute(
        select(
            Activity.id,
            Activity.start_time,
            Activity.duration_s,
            Activity.distance_m,
            Activity.elevation_m,
            Activity.load,
            Activity.sport_type,
            Activity.rpe,
            Activity.notes,
            Activity.labels,
        ).where(Activity.athlete_id == athlete_id)
    )

    facts: list[ActivityFact] = []
    for row in result.all():
        day = _local_day(row.start_time, tz)
        if day is None:
            # An activity with no start time can't be placed on the calendar, so
            # it can't contribute to counts, streaks or unlock dates.
            continue
        labels = tuple(row.labels) if isinstance(row.labels, list) else ()
        facts.append(
            ActivityFact(
                day=day,
                duration_s=row.duration_s or 0,
                distance_m=row.distance_m or 0.0,
                elevation_m=row.elevation_m or 0.0,
                load=row.load or 0.0,
                sport=activity_category(row.sport_type),
                has_rpe=row.rpe is not None,
                has_notes=bool((row.notes or "").strip()),
                labels=labels,
                activity_id=row.id,
            )
        )
    return facts


def _tiers(achievement_id: str) -> tuple[float, ...]:
    return CATALOGUE_BY_ID[achievement_id].tiers


def _compute_activity_rules(
    facts: list[ActivityFact], comp: AchievementComputation, today: date
) -> None:
    """Everything derived from the activity list alone."""
    # ── Volume ───────────────────────────────────────────────────────────────
    comp.tier_dates["activity_count"] = cumulative_tier_dates(
        ((f.day, 1) for f in facts), _tiers("activity_count")
    )
    comp.progress["activity_count"] = float(len(facts))

    hours = [(f.day, f.duration_s / 3600) for f in facts]
    comp.tier_dates["long_activity"] = threshold_tier_dates(hours, _tiers("long_activity"))
    comp.progress["long_activity"] = max((h for _, h in hours), default=0.0)

    comp.tier_dates["total_hours"] = cumulative_tier_dates(hours, _tiers("total_hours"))
    comp.progress["total_hours"] = sum(h for _, h in hours)

    km = [(f.day, f.distance_m / 1000) for f in facts]
    comp.tier_dates["single_ride_distance"] = threshold_tier_dates(
        km, _tiers("single_ride_distance")
    )
    comp.progress["single_ride_distance"] = max((d for _, d in km), default=0.0)

    comp.tier_dates["total_distance"] = cumulative_tier_dates(km, _tiers("total_distance"))
    comp.progress["total_distance"] = sum(d for _, d in km)

    load = [(f.day, f.load) for f in facts]
    comp.tier_dates["total_load"] = cumulative_tier_dates(load, _tiers("total_load"))
    comp.progress["total_load"] = sum(v for _, v in load)

    # ── Climbing ─────────────────────────────────────────────────────────────
    climb = [(f.day, f.elevation_m) for f in facts]
    comp.tier_dates["single_ride_elevation"] = threshold_tier_dates(
        climb, _tiers("single_ride_elevation")
    )
    comp.progress["single_ride_elevation"] = max((m for _, m in climb), default=0.0)

    comp.tier_dates["total_elevation"] = cumulative_tier_dates(
        climb, _tiers("total_elevation")
    )
    comp.progress["total_elevation"] = sum(m for _, m in climb)

    comp.tier_dates["everesting"] = threshold_tier_dates(climb, _tiers("everesting"))
    comp.progress["everesting"] = max((m for _, m in climb), default=0.0)

    # ── Variety ──────────────────────────────────────────────────────────────
    comp.tier_dates["multisport"] = distinct_tier_dates(
        ((f.day, f.sport) for f in facts), _tiers("multisport")
    )
    comp.progress["multisport"] = float(len({f.sport for f in facts if f.sport}))

    races = [(f.day, 1) for f in facts if "race" in f.labels]
    comp.tier_dates["race_day"] = cumulative_tier_dates(races, _tiers("race_day"))
    comp.progress["race_day"] = float(len(races))

    commutes = [(f.day, 1) for f in facts if "commute" in f.labels]
    comp.tier_dates["commuter"] = cumulative_tier_dates(commutes, _tiers("commuter"))
    comp.progress["commuter"] = float(len(commutes))

    returned = comeback_date(f.day for f in facts)
    comp.tier_dates["comeback"] = {1: returned} if returned else {}
    comp.progress["comeback"] = 1.0 if returned else 0.0

    # ── Engagement ───────────────────────────────────────────────────────────
    rpes = [(f.day, 1) for f in facts if f.has_rpe]
    comp.tier_dates["rpe_recorded"] = cumulative_tier_dates(rpes, _tiers("rpe_recorded"))
    comp.progress["rpe_recorded"] = float(len(rpes))

    noted = [(f.day, 1) for f in facts if f.has_notes]
    comp.tier_dates["notes_written"] = cumulative_tier_dates(noted, _tiers("notes_written"))
    comp.progress["notes_written"] = float(len(noted))

    # ── Streaks ──────────────────────────────────────────────────────────────
    weekly = bucket_by(facts)
    monthly = bucket_by(facts, monthly=True)
    for achievement_id, (qualifies, is_monthly) in _STREAK_RULES.items():
        buckets = monthly if is_monthly else weekly
        state = streak_state(buckets, qualifies, today, monthly=is_monthly)
        comp.streaks[achievement_id] = state
        comp.progress[achievement_id] = float(state.current)
        comp.tier_dates[achievement_id] = streak_tier_dates(
            buckets, qualifies, _tiers(achievement_id), today, monthly=is_monthly
        )

    # ── Deep links for the single-activity badges ────────────────────────────
    for achievement_id, value_of in (
        ("long_activity", lambda f: f.duration_s / 3600),
        ("single_ride_distance", lambda f: f.distance_m / 1000),
        ("single_ride_elevation", lambda f: f.elevation_m),
        ("everesting", lambda f: f.elevation_m),
    ):
        earned = comp.tier_dates.get(achievement_id) or {}
        if not earned:
            continue
        best_tier = max(earned)
        activity_id = _first_activity_reaching(facts, value_of, best_tier)
        if activity_id:
            comp.context[achievement_id] = {"activity_id": activity_id}

    if returned:
        match = next((f for f in facts if f.day == returned), None)
        if match and match.activity_id:
            comp.context["comeback"] = {"activity_id": match.activity_id}

    # ── Availability ─────────────────────────────────────────────────────────
    if any(f.distance_m for f in facts):
        comp.available.add("distance")
    if any(f.elevation_m for f in facts):
        comp.available.add("elevation")
    if any(f.load for f in facts):
        comp.available.add("load")


async def _compute_plan_rules(
    athlete_id: str, comp: AchievementComputation, today: date, session: AsyncSession
) -> None:
    """Plan achievements, scored with the existing adherence engine.

    ``plan_adherence_daily`` only tracks *active* plans, so a finished plan is
    scored directly with :func:`score_plan` as of its end date. Reusing that one
    function means a badge can never disagree with the adherence number shown on
    the plan page.
    """
    result = await session.execute(
        select(TrainingPlan)
        .where(TrainingPlan.athlete_id == athlete_id)
        .options(
            selectinload(TrainingPlan.workouts).selectinload(
                PlannedWorkout.linked_activities
            )
        )
        .execution_options(populate_existing=True)
    )
    plans = list(result.scalars().all())
    if plans:
        comp.available.add("plan")

    completed: list[tuple[date, int]] = []
    flawless: list[date] = []
    scores: list[tuple[date, float]] = []
    plan_context: dict[str, str] = {}

    for plan in plans:
        # A plan counts once it is over — a plan still running hasn't been
        # "completed", however well it is going.
        if plan.start_date is None or plan.end_date is None or plan.end_date >= today:
            continue
        ps = score_plan(plan, plan.end_date)
        if ps.completed <= 0:
            # An empty plan that simply elapsed is not an achievement.
            continue
        completed.append((plan.end_date, 1))
        if ps.missed == 0 and ps.skipped == 0:
            flawless.append(plan.end_date)
            plan_context.setdefault("plan_flawless", plan.id)
        if ps.score is not None:
            scores.append((plan.end_date, ps.score))

    comp.tier_dates["plans_completed"] = cumulative_tier_dates(
        completed, _tiers("plans_completed")
    )
    comp.progress["plans_completed"] = float(len(completed))

    comp.tier_dates["plan_flawless"] = {1: min(flawless)} if flawless else {}
    comp.progress["plan_flawless"] = float(len(flawless))

    comp.tier_dates["plan_adherence"] = threshold_tier_dates(
        scores, _tiers("plan_adherence")
    )
    comp.progress["plan_adherence"] = max((s for _, s in scores), default=0.0)

    for achievement_id, plan_id in plan_context.items():
        comp.context[achievement_id] = {"plan_id": plan_id}


async def _compute_goal_rules(
    athlete_id: str, comp: AchievementComputation, session: AsyncSession
) -> None:
    result = await session.execute(
        select(Goal).where(Goal.athlete_id == athlete_id, Goal.status == "achieved")
    )
    goals = list(result.scalars().all())

    # Goals carry no "achieved at" timestamp, so the target date is the best
    # available stand-in, falling back to when the goal was set.
    events: list[tuple[date, int]] = []
    for goal in goals:
        when = goal.target_date or (
            goal.created_at.date() if goal.created_at else None
        )
        if when:
            events.append((when, 1))

    comp.tier_dates["goals_reached"] = cumulative_tier_dates(
        events, _tiers("goals_reached")
    )
    comp.progress["goals_reached"] = float(len(goals))


async def compute_achievements(
    athlete: Athlete, session: AsyncSession, today: Optional[date] = None
) -> AchievementComputation:
    """Work out every earned tier, current progress and streak state."""
    tz = _zone(athlete)
    if today is None:
        today = datetime.now(tz or timezone.utc).date()

    comp = AchievementComputation()
    facts = await _load_facts(athlete.id, tz, session)
    _compute_activity_rules(facts, comp, today)
    await _compute_plan_rules(athlete.id, comp, today, session)
    await _compute_goal_rules(athlete.id, comp, session)

    # Never date an unlock in the future, whatever the source data says.
    for tiers in comp.tier_dates.values():
        for tier, when in list(tiers.items()):
            if when > today:
                tiers[tier] = today

    return comp


async def recompute_achievements(
    athlete_id: str, session: AsyncSession, today: Optional[date] = None
) -> list[AchievementUnlock]:
    """Reconcile stored unlocks with what the athlete's data currently earns.

    Inserts newly earned tiers, corrects an ``achieved_on`` that history has
    moved (back-filling an old ride makes a badge *older*, not newer), and
    deletes tiers the data no longer supports. Idempotent: a second run over
    unchanged data writes nothing.

    Returns the rows created by this call, so the caller can notify on them.
    """
    athlete = (
        await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    ).scalar_one_or_none()
    if athlete is None or not gamification_enabled(athlete):
        return []

    comp = await compute_achievements(athlete, session, today)

    existing = {
        (row.achievement_id, row.tier): row
        for row in (
            await session.execute(
                select(AchievementUnlock).where(
                    AchievementUnlock.athlete_id == athlete_id
                )
            )
        ).scalars()
    }

    earned: dict[tuple[str, float], date] = {}
    for definition in CATALOGUE:
        if not comp.is_available(definition.id):
            continue
        for tier, when in (comp.tier_dates.get(definition.id) or {}).items():
            earned[(definition.id, float(tier))] = when

    created: list[AchievementUnlock] = []
    changed = False

    for key, when in earned.items():
        achievement_id, tier = key
        row = existing.get(key)
        if row is None:
            row = AchievementUnlock(
                athlete_id=athlete_id,
                achievement_id=achievement_id,
                tier=tier,
                achieved_on=when,
                notified=False,
                context=comp.context.get(achievement_id),
            )
            session.add(row)
            created.append(row)
            changed = True
        elif row.achieved_on != when:
            row.achieved_on = when
            changed = True

    for key, row in existing.items():
        if key not in earned:
            await session.delete(row)
            changed = True

    if changed:
        await session.commit()
    if created:
        await _notify(athlete, created, session)
    return created


async def _notify(
    athlete: Athlete, created: list[AchievementUnlock], session: AsyncSession
) -> None:
    """Put one inbox message in front of the athlete for a batch of unlocks.

    One message per *recompute*, not per tier: importing a season of history at
    once can earn a dozen tiers, and a dozen separate messages would read as
    spam. A single unlock names itself; a batch is summarised by count, and the
    frontend picks the template from ``count``.
    """
    try:
        data: dict = {"count": len(created)}
        if len(created) == 1:
            data["achievement_id"] = created[0].achievement_id
            data["tier"] = created[0].tier

        # Flag first: a message we fail to deliver is a far smaller problem than
        # one delivered again on every subsequent recompute.
        for row in created:
            row.notified = True
        await session.commit()

        await notify_user(athlete.global_user_id, ACHIEVEMENT_UNLOCKED, data)
    except Exception:
        log.warning(
            "Failed to notify athlete %s about new achievements", athlete.id, exc_info=True
        )


async def recompute_achievements_safe(athlete_id: str, session: AsyncSession) -> None:
    """Fire-and-forget wrapper for the ingest paths.

    Achievements are a nice-to-have on top of an upload; a failure here must
    never fail the upload itself or block a sync, so it is logged and swallowed.
    """
    try:
        await recompute_achievements(athlete_id, session)
    except Exception:
        log.warning("Achievement recompute failed for athlete %s", athlete_id, exc_info=True)
