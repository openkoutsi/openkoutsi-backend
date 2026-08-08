"""``get_plan_status`` — is the athlete actually doing the plan? (issue #42)

Adherence is the question the fixed-blob prompts got wrong most often, because
it is the one that needs the *calendar* rather than a list: a session dated
tomorrow is not missed, today's empty session is not missed yet, and a rest day
is not something to complete at all. Getting any of those wrong produces a coach
that scolds an athlete who is on track, which is worse than saying nothing.

So this tool answers with dates and states attached, using the same
:mod:`backend.app.services.plan_adherence` scoring the API and the dashboard use.
Nothing is recomputed differently here; a plan's adherence score means one thing
across the whole platform.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import round_or_none
from backend.app.models.user_orm import PlannedWorkout, TrainingPlan
from backend.app.services.plan_adherence import score_plan, workout_date
from openkoutsi.sport_matching import is_rest_workout

#: ``PlannedSession`` has a field called ``date``, which shadows the type inside
#: the class body; the annotation uses this alias instead.
_Date = date

#: Plans returned in one call. An athlete has one or two active plans; the cap
#: exists so an archived-history request cannot become unbounded.
MAX_PLANS = 5


class PlanStatusArgs(ToolArgs):
    include_archived: bool = Field(
        False,
        description=(
            "Include archived plans as well as active ones. Archived plans are "
            "history — useful for 'what did the last block look like', not for "
            "judging what the athlete should do today."
        ),
    )
    week_window_days: int = Field(
        7,
        ge=1,
        le=28,
        description=(
            "How many days of planned sessions, from today forward, to list per "
            "plan. 7 gives the week ahead; raise it to see a taper or a peak week."
        ),
    )


class PlannedSession(BaseModel):
    date: _Date = Field(..., description="Calendar date this session falls on.")
    workout_type: Optional[str] = Field(
        None, description="Session type as the plan names it, e.g. endurance, threshold, rest."
    )
    description: Optional[str] = Field(None, description="What the plan asks for, in its own words.")
    target_load: Optional[int] = Field(None, description="Prescribed Load (unitless Load points).")
    duration_min: Optional[int] = Field(None, description="Prescribed duration in minutes (min).")
    state: Literal["completed", "skipped", "missed", "due_today", "upcoming", "rest"] = Field(
        ...,
        description=(
            "completed = at least one activity is linked; skipped = deliberately "
            "not done, with a reason; missed = a past session with nothing "
            "against it; due_today = today's session, still open and NOT missed; "
            "upcoming = dated in the future and not due yet; rest = an intentional "
            "rest day with nothing to complete. Never treat due_today, upcoming "
            "or rest as a failure."
        ),
    )
    skip_reason: Optional[str] = Field(
        None, description="Why a skipped session was skipped, in the athlete's words."
    )
    match_score: Optional[float] = Field(
        None,
        description=(
            "How well the performed activities hit this session's targets, 0–100 "
            "(unitless). Over- and under-shooting both cost; a completed session "
            "never scores below 50. Null when not yet scorable."
        ),
    )
    actual_load: Optional[float] = Field(
        None, description="Load actually recorded against this session (unitless Load points)."
    )
    actual_duration_s: Optional[int] = Field(
        None, description="Time actually recorded against this session, in seconds (s)."
    )
    linked_activities: int = Field(
        0,
        description=(
            "How many activities completed this session (count). More than one "
            "means a single session was recorded in parts — judge them combined."
        ),
    )


class PlanStatus(BaseModel):
    plan_id: str = Field(..., description="Identifier of the training plan.")
    name: str = Field(..., description="Plan name as the athlete sees it.")
    goal: Optional[str] = Field(None, description="What the plan is built towards, in the athlete's words.")
    status: str = Field(..., description="'active' or 'archived'.")
    start_date: Optional[date] = Field(None, description="First calendar date of the plan.")
    end_date: Optional[date] = Field(None, description="Last calendar date of the plan.")
    weeks: Optional[int] = Field(None, description="Planned length in weeks (count).")
    current_week: Optional[int] = Field(
        None,
        description=(
            "Which week of the plan today falls in, 1-based (count). Null when "
            "the plan has not started or has already finished."
        ),
    )
    phase: Optional[str] = Field(
        None, description="This week's focus note from the plan's own metadata, when it has one."
    )
    adherence_score: Optional[float] = Field(
        None,
        description=(
            "Load-weighted adherence over the elapsed part of the plan, 0–100 "
            "(unitless). Missed sessions count zero; skips are softened by "
            "reason. Null when nothing is scorable yet."
        ),
    )
    completed: int = Field(0, description="Sessions completed so far (count).")
    missed: int = Field(0, description="Past sessions with nothing against them (count).")
    skipped: int = Field(0, description="Sessions deliberately skipped (count).")
    remaining: int = Field(
        0, description="Sessions still to do from today onward, today's open session included (count)."
    )
    upcoming: list[PlannedSession] = Field(
        default_factory=list, description="Sessions from today forward, within the requested window."
    )
    recent: list[PlannedSession] = Field(
        default_factory=list, description="The last seven days of sessions, so adherence can be read rather than asserted."
    )


class PlanStatusResult(BaseModel):
    plans: list[PlanStatus] = Field(
        default_factory=list, description="Matching plans, most recently created first."
    )
    returned: int = Field(0, description="How many plans are in this response (count).")
    total: int = Field(0, description="How many plans matched in total (count).")
    truncated: bool = Field(False, description="True when plans were cut off by the cap.")


def _state(workout: PlannedWorkout, when: date, today: date) -> str:
    if is_rest_workout(workout.workout_type):
        return "rest"
    if workout.linked_activities:
        return "completed"
    if workout.skip_reason:
        return "skipped"
    if when > today:
        return "upcoming"
    if when == today:
        return "due_today"
    return "missed"


def _session(
    workout: PlannedWorkout, when: date, today: date, match_scores: dict
) -> PlannedSession:
    linked = workout.linked_activities
    return PlannedSession(
        date=when,
        workout_type=workout.workout_type,
        description=workout.description,
        target_load=workout.target_load,
        duration_min=workout.duration_min,
        state=_state(workout, when, today),
        skip_reason=workout.skip_reason,
        match_score=round_or_none(match_scores.get(workout.id), 1),
        actual_load=round_or_none(sum((a.load or 0.0) for a in linked), 1) if linked else None,
        actual_duration_s=sum((a.duration_s or 0) for a in linked) if linked else None,
        linked_activities=len(linked),
    )


def _phase_note(plan: TrainingPlan, week_number: Optional[int]) -> Optional[str]:
    """The focus note the generator wrote for this week, if there is one."""
    if week_number is None or not plan.week_meta:
        return None
    index = week_number - 1
    if not (0 <= index < len(plan.week_meta)):
        return None
    meta = plan.week_meta[index]
    if not isinstance(meta, dict):
        return None
    note = meta.get("focus") or meta.get("note")
    kind = meta.get("kind") or meta.get("type")
    if note and kind:
        return f"{kind}: {note}"
    return note or kind


@tool(
    name="get_plan_status",
    title="Training plan status",
    scopes={"plans:read"},
    arguments=PlanStatusArgs,
    returns=PlanStatusResult,
)
async def get_plan_status(run: ToolRun, args: PlanStatusArgs) -> PlanStatusResult:
    """The athlete's training plans with their adherence: how far through each
    plan they are, the Load-weighted adherence score so far, how many sessions
    are completed, missed and skipped, how many remain, and the individual
    sessions of the last week and the week ahead.

    Read the per-session 'state' carefully before judging adherence. Only
    'missed' is a failure. 'due_today' is today's session with the day still to
    run, 'upcoming' is not due yet, and 'rest' is a deliberate part of the plan
    with nothing to complete — an athlete taking a scheduled rest day is
    following the plan, not skipping it. A whole week of 'upcoming' means the
    week has not happened yet, not that the athlete is behind.

    Where a session was skipped, the athlete's reason is included. A genuine one
    (illness, injury, travel) should temper the response; a pattern of thin ones
    should not.
    """
    query = select(TrainingPlan).where(TrainingPlan.athlete_id == run.athlete.id)
    if not args.include_archived:
        query = query.where(TrainingPlan.status == "active")

    plans = (
        await run.session.execute(
            query.options(
                selectinload(TrainingPlan.workouts).selectinload(
                    PlannedWorkout.linked_activities
                )
            ).order_by(TrainingPlan.created_at.desc())
        )
    ).scalars().all()

    total = len(plans)
    today = run.today
    results: list[PlanStatus] = []

    for plan in plans[:MAX_PLANS]:
        scored = score_plan(plan, today)

        current_week: Optional[int] = None
        if plan.start_date is not None and plan.start_date <= today:
            if plan.end_date is None or today <= plan.end_date:
                current_week = (today - plan.start_date).days // 7 + 1

        upcoming: list[PlannedSession] = []
        recent: list[PlannedSession] = []
        if plan.start_date is not None:
            for workout in plan.workouts:
                when = workout_date(plan.start_date, workout.week_number, workout.day_of_week)
                delta = (when - today).days
                if 0 <= delta <= args.week_window_days:
                    upcoming.append(_session(workout, when, today, scored.match_scores))
                elif -7 <= delta < 0:
                    recent.append(_session(workout, when, today, scored.match_scores))
            upcoming.sort(key=lambda s: s.date)
            recent.sort(key=lambda s: s.date)

        results.append(
            PlanStatus(
                plan_id=plan.id,
                name=plan.name,
                goal=plan.goal,
                status=plan.status,
                start_date=plan.start_date,
                end_date=plan.end_date,
                weeks=plan.weeks,
                current_week=current_week,
                phase=_phase_note(plan, current_week),
                adherence_score=round_or_none(scored.score, 1),
                completed=scored.completed,
                missed=scored.missed,
                skipped=scored.skipped,
                remaining=scored.future + scored.pending,
                upcoming=upcoming,
                recent=recent,
            )
        )

    return PlanStatusResult(
        plans=results,
        returned=len(results),
        total=total,
        truncated=total > len(results),
    )
