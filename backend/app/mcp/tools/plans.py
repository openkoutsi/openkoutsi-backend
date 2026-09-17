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

``propose_training_plan`` / ``propose_plan_change`` — offering, not doing (#72)
------------------------------------------------------------------------------
The two tools below are the only ones in the registry that write anything, and
what they write is **one row in ``plan_proposals``**. They cannot reach
``training_plans`` or ``planned_workouts`` — no tool can, through any door — so
what they produce is an offer sitting in front of the athlete, not a change to
their training. The only code that writes a plan is
:func:`~backend.app.services.plan_proposals.apply_proposal`, behind an HTTP route
that requires the athlete's own session and a proposal id.

That is the control, and it is structural because nothing softer would hold:
Koutsi calls tools as a session credential, which means every scope implicitly,
so a ``plans:write`` scope would gate an external client and gate Koutsi by one
sentence of prompt. See :mod:`backend.app.services.plan_proposals`.

Neither tool is *published*: they are ``internal_only``, because half of this
feature is an approval control in openkoutsi's own UI and an external client
drafting a proposal would leave it in a thread nobody has open.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.errors import ToolError
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import round_or_none
from backend.app.models.user_orm import PlannedWorkout, TrainingPlan
from backend.app.schemas.plan_proposals import (
    BUILT_BY_LLM,
    BUILT_BY_RULES,
    KIND_CREATE_PLAN,
    KIND_UPDATE_PLAN,
    KIND_UPDATE_WORKOUT,
    ArchivedPlanPreview,
    PlanProposalResult,
    PlanProposalSummary,
    ProposedChange,
    ProposedSession,
    ProposedWeek,
)
from backend.app.services.llm_agent import MAX_TOOL_RESULT_CHARS
from backend.app.services.llm_plan_generator import generate_plan_weeks_with_cfg
from backend.app.services.plan_adherence import score_plan, workout_date
from backend.app.services.plan_lifecycle import (
    LIVE_STATUSES,
    overlapping_active_plans,
    plan_end_date,
)
from backend.app.services.plan_proposals import (
    draft_proposal,
    occupant_of,
    session_date,
    stranded_sessions,
)
from openkoutsi.plan_builder import build_week_from_config, week_meta_from_weeks
from openkoutsi.plan_schema import DayConfig, PlanConfig, clamp_plan_params
from openkoutsi.sport_matching import is_rest_workout

log = logging.getLogger(__name__)

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
            "Include archived plans as well. Plans the athlete is following and "
            "plans that have run their course are both returned either way; "
            "archived ones were filed away by hand or superseded by a newer "
            "plan — useful for 'what did the last block look like', not for "
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
    status: str = Field(
        ...,
        description=(
            "'active' while the plan is being followed, 'completed' once its "
            "last day has passed, or 'archived' if it was filed away."
        ),
    )
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
        # A plan that finished yesterday is exactly what the coach needs to see
        # to talk about the block just gone, so `completed` stays in the default
        # set; only the filed-away ones are held back.
        query = query.where(TrainingPlan.status.in_(LIVE_STATUSES))

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


# ── Proposing a plan (issue #72) ────────────────────────────────────────────


#: Longest plan a proposal may draft, in weeks. The cap is about the *preview*
#: as much as the plan: every week adds a row to the summary the model reads
#: back, and the whole result has to stay well inside ``MAX_TOOL_RESULT_CHARS``.
MAX_PROPOSAL_WEEKS = 24

#: How much of a session's own words survive into the preview. Long enough to
#: recognise the session, short enough that seven of them plus a 24-week table
#: fit inside the loop's 6 000-character tool-result budget — which is the bound
#: that actually applies here, far tighter than ``call_tool``'s 64 KiB.
MAX_DESCRIPTION_CHARS = 140


#: Archived plans named individually in a **tool result**. The card shows them
#: all; this caps only what the model is handed, because `archives` is the last
#: field of the summary and `note` the last of the result — so an overflow is
#: truncated exactly where the consent instruction lives. See `_for_model`.
MAX_ARCHIVES_IN_RESULT = 4

#: How much of the goal survives into a **tool result**. The stored summary and
#: the payload keep it whole; this is a context-window bound, not a data one.
MAX_GOAL_CHARS_IN_RESULT = 200

#: A reasonable week when the athlete has not said which days they train. Taken
#: in order and then sorted, so "four days" lands on Tue/Thu/Sat/Sun rather than
#: Mon–Thu, which is what an athlete with a job and a weekend actually rides.
_DEFAULT_WEEK = [
    (2, "threshold"),
    (4, "endurance"),
    (6, "long"),
    (7, "endurance"),
    (3, "tempo"),
    (5, "recovery"),
    (1, "endurance"),
]

_DAY_NAMES = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday",
}


class ProposedTrainingDay(ToolArgs):
    day_of_week: int = Field(
        ..., ge=1, le=7, description="1 = Monday through 7 = Sunday."
    )
    workout_type: str = Field(
        ...,
        description=(
            "What this day is for: recovery, tempo, threshold, vo2max, "
            "endurance, long, strength, yoga, cross-training."
        ),
    )
    notes: Optional[str] = Field(
        None,
        description="Anything particular about this day, e.g. 'club ride', 'before work'.",
    )


class ProposePlanArgs(ToolArgs):
    name: str = Field(
        ...,
        max_length=120,
        description="What to call the plan, in the athlete's own terms, e.g. 'October gran fondo build'.",
    )
    start_date: date = Field(
        ...,
        description=(
            "First calendar date the plan runs. Must be today or later — a plan "
            "cannot begin in the past, and the athlete's approval may come "
            "tomorrow, so prefer the coming Monday over today unless they asked "
            "to start now."
        ),
    )
    weeks: int = Field(
        ...,
        ge=1,
        le=MAX_PROPOSAL_WEEKS,
        description=f"How long the plan runs, in weeks (count, 1–{MAX_PROPOSAL_WEEKS}).",
    )
    goal: Optional[str] = Field(
        None,
        max_length=500,
        description=(
            "The event or outcome the plan is built towards, in the athlete's "
            "words — 'hilly gran fondo, 140 km, 3 000 m' rather than 'get fitter'."
        ),
    )
    training_days: Optional[list[ProposedTrainingDay]] = Field(
        None,
        max_length=7,
        description=(
            "Which days the athlete trains and what each is for. Ask them if you "
            "do not know; omitting it falls back to a sensible week built from "
            "'days_per_week', which is a guess about their life rather than a "
            "plan for it."
        ),
    )
    days_per_week: int = Field(
        4,
        ge=1,
        le=7,
        description="How many days a week they train. Only used when 'training_days' is omitted.",
    )
    periodization: Literal["base_building", "race_prep", "maintenance"] = Field(
        "base_building",
        description=(
            "'race_prep' builds towards a date and tapers at the end; "
            "'base_building' grows aerobic volume; 'maintenance' holds fitness."
        ),
    )
    intensity_preference: Literal["low", "moderate", "high"] = Field(
        "moderate",
        description="How hard the plan should lean. Match it to their experience and their history.",
    )
    weekly_hours_low: Optional[float] = Field(
        None,
        ge=0,
        le=40,
        description="Fewest hours a week they can train, in hours (h). Recovery weeks sit near this.",
    )
    weekly_hours_high: Optional[float] = Field(
        None,
        ge=0,
        le=40,
        description="Most hours a week they can train, in hours (h). Peak build weeks sit near this.",
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description=(
            "Anything else that should shape the plan — terrain, a holiday in "
            "week five, a recent injury they have recovered from."
        ),
    )


class ProposePlanChangeArgs(ToolArgs):
    plan_id: str = Field(
        ...,
        description="Which plan to change. Call get_plan_status first; do not guess an id.",
    )
    change: Literal[
        "update_plan", "update_workout", "skip_workout", "unskip_workout"
    ] = Field(
        ...,
        description=(
            "'update_plan' for the plan as a whole (name, goal, start date, "
            "length, archive or reopen); 'update_workout' to change one planned "
            "session; 'skip_workout' to mark one session deliberately not done, "
            "with a reason; 'unskip_workout' to undo that."
        ),
    )
    name: Optional[str] = Field(
        None, max_length=120, description="New plan name, for 'update_plan'."
    )
    goal: Optional[str] = Field(
        None, max_length=500, description="New goal for the plan, for 'update_plan'."
    )
    start_date: Optional[date] = Field(
        None,
        description=(
            "New first calendar date, for 'update_plan'. Moving it moves every "
            "session with it — this is how 'push the whole thing back a week' is done."
        ),
    )
    weeks: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_PROPOSAL_WEEKS,
        description=(
            f"New length in weeks (count, 1–{MAX_PROPOSAL_WEEKS}), for "
            "'update_plan'. This moves the plan's end date; it does not write "
            "sessions for weeks that have none."
        ),
    )
    status: Optional[Literal["active", "completed", "archived"]] = Field(
        None,
        description=(
            "New plan status, for 'update_plan'. 'archived' files the plan away; "
            "'active' reopens it, which archives any active plan it overlaps."
        ),
    )
    workout_id: Optional[str] = Field(
        None,
        description=(
            "Which planned session to change. Required for everything except "
            "'update_plan'. Take it from get_plan_status rather than guessing."
        ),
    )
    workout_type: Optional[str] = Field(
        None, description="New session type, e.g. endurance, threshold, rest."
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="New session description. Make it match the duration and Load you set.",
    )
    duration_min: Optional[int] = Field(
        None, ge=0, le=1440, description="New prescribed duration in minutes (min)."
    )
    target_load: Optional[int] = Field(
        None, ge=0, le=1000, description="New prescribed Load (unitless Load points)."
    )
    day_of_week: Optional[int] = Field(
        None,
        ge=1,
        le=7,
        description="Move the session to this day: 1 = Monday through 7 = Sunday.",
    )
    skip_reason: Optional[str] = Field(
        None,
        max_length=200,
        description=(
            "Why the session is being skipped, in the athlete's words, for "
            "'skip_workout'. A genuine reason (illness, travel) is scored more "
            "kindly than a missed session, so record what they actually said."
        ),
    )


# ── Building the draft ──────────────────────────────────────────────────────


def _plan_config(args: ProposePlanArgs) -> PlanConfig:
    """The athlete's week, as the plan builder and the generator both want it."""
    if args.training_days:
        days = [
            DayConfig(
                day_of_week=day.day_of_week,
                workout_type=day.workout_type,
                notes=day.notes,
            )
            for day in sorted(args.training_days, key=lambda d: d.day_of_week)
        ]
    else:
        chosen = sorted(_DEFAULT_WEEK[: args.days_per_week], key=lambda pair: pair[0])
        days = [
            DayConfig(day_of_week=day, workout_type=workout_type)
            for day, workout_type in chosen
        ]

    return clamp_plan_params(
        PlanConfig(
            days_per_week=len(days),
            day_configs=days,
            periodization=args.periodization,
            intensity_preference=args.intensity_preference,
            long_description=args.notes,
            weekly_hours_min=args.weekly_hours_low,
            weekly_hours_max=args.weekly_hours_high,
        )
    )


def _rule_based_weeks(config: PlanConfig, num_weeks: int) -> list[list[dict]]:
    """The deterministic fallback.

    ``openkoutsi.plan_builder`` rather than ``services.plan_generator``: the
    latter's ``build_workout_rows`` returns ``PlannedWorkout`` *ORM instances*,
    and a tool that must never touch the plan tables has no business
    constructing their rows even unattached. This returns plain dicts, which is
    the same shape the generator's LLM path produces.
    """
    return [
        build_week_from_config(config, week_num, num_weeks)
        for week_num in range(1, num_weeks + 1)
    ]


def _trim(text: Optional[str], limit: int = MAX_DESCRIPTION_CHARS) -> Optional[str]:
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _weekly_summary(weeks_data: list[list[dict]], week_meta: list[dict]) -> list[ProposedWeek]:
    summary: list[ProposedWeek] = []
    for index, days in enumerate(weeks_data):
        meta = week_meta[index] if index < len(week_meta) else {}
        summary.append(
            ProposedWeek(
                week_number=index + 1,
                # `week_type` only, deliberately, and not the builder's prose
                # `focus` note beside it. Two reasons and they point the same
                # way: the note is English, and this preview is rendered in the
                # athlete's own language; and one per week is the single biggest
                # thing in a 24-week summary, which has to fit the loop's 6 000
                # characters. The plan itself still carries the full `week_meta`.
                week_type=meta.get("week_type"),
                sessions=sum(
                    1 for day in days if not is_rest_workout(day.get("workout_type"))
                ),
                total_load=sum(int(day.get("target_load") or 0) for day in days) or None,
                total_duration_min=sum(
                    int(day.get("duration_min") or 0) for day in days
                )
                or None,
            )
        )
    return summary


def _first_week(days: list[dict]) -> list[ProposedSession]:
    return [
        ProposedSession(
            day_of_week=int(day["day_of_week"]),
            workout_type=day.get("workout_type"),
            description=_trim(day.get("description")),
            duration_min=day.get("duration_min"),
            target_load=day.get("target_load"),
        )
        for day in sorted(days, key=lambda d: d["day_of_week"])
    ]


def _archives(plans: list[TrainingPlan]) -> list[ArchivedPlanPreview]:
    return [
        ArchivedPlanPreview(
            plan_id=plan.id,
            name=plan.name,
            start_date=plan.start_date,
            end_date=plan.end_date,
        )
        for plan in sorted(plans, key=lambda p: (p.start_date or date.min, p.name))
    ]


def _rendered_size(summary: PlanProposalSummary) -> int:
    """How many characters this summary costs the model, near enough.

    The result wrapper and the note add a few hundred on top, which
    :data:`_RESULT_OVERHEAD` covers — measuring the real thing would mean
    building a `PlanProposalResult` per candidate, and the point is to be
    conservative rather than exact.
    """
    return len(
        json.dumps(summary.model_dump(mode="json"), default=str, ensure_ascii=False)
    )


#: Headroom left for the result wrapper (ids, status, expiry) and `_note`, whose
#: length varies with how many plans it names.
_RESULT_OVERHEAD = 1200


def _for_model(summary: PlanProposalSummary) -> PlanProposalSummary:
    """The preview, shed down until it fits the loop's tool-result budget.

    The card reads the **stored** summary and carries every week and every plan
    an approval would archive. This is the copy the *model* gets, and it has a
    hard bound: `MAX_TOOL_RESULT_CHARS` is 6 000, while the tool's own arguments
    allow a 120-character name, a 500-character goal and 24 weeks at once, and
    `archives` grows with the athlete's plans.

    Left uncapped it overflows — and because `archives` is the summary's last
    field and `note` the result's last, the truncation lands precisely on the
    list of what would be filed away and on the sentence telling the model to say
    so. The informed-consent claim failing in the one case where it is
    load-bearing.

    So detail is shed in a deliberate order, cheapest first, and each reduction
    *says* it happened rather than quietly shrinking the answer:

    1. the archive list is capped and the rest counted (`archives_omitted`);
    2. the goal is trimmed — the payload keeps it whole;
    3. the week table is cut from the end (`weeks_omitted`), since the shape of
       the first weeks is what prose needs;
    4. the first week's descriptions are trimmed hard.

    A test drives the tool at its own declared maxima and asserts the bound, so
    this is checked rather than reasoned about.
    """
    named = summary.archives[:MAX_ARCHIVES_IN_RESULT]
    capped = summary.model_copy(
        update={
            "goal": _trim(summary.goal, MAX_GOAL_CHARS_IN_RESULT),
            "archives": named,
            "archives_omitted": len(summary.archives) - len(named),
        }
    )

    budget = MAX_TOOL_RESULT_CHARS - _RESULT_OVERHEAD
    if _rendered_size(capped) <= budget:
        return capped

    # 3. Shed weeks from the end until it fits, keeping at least the first.
    weekly = list(capped.weekly)
    while len(weekly) > 1 and _rendered_size(capped) > budget:
        weekly = weekly[:-1]
        capped = capped.model_copy(
            update={
                "weekly": weekly,
                "weeks_omitted": len(summary.weekly) - len(weekly),
            }
        )
    if _rendered_size(capped) <= budget:
        return capped

    # 4. Last resort: the first week's own words. The type, duration and Load
    #    stay, so the shape of the week survives even here.
    return capped.model_copy(
        update={
            "first_week": [
                day.model_copy(update={"description": _trim(day.description, 40)})
                for day in capped.first_week
            ]
        }
    )


def _note(summary: PlanProposalSummary) -> str:
    """What the model must understand before it writes its reply."""
    lines = [
        "Nothing has been changed. This is an offer: the athlete sees a card "
        "under your reply with 'yes' and 'no', and only their yes writes "
        "anything.",
        "Tell them what you have drafted, in prose, and that it is theirs to "
        "accept or decline.",
    ]
    if summary.archives:
        names = ", ".join(f"'{entry.name}'" for entry in summary.archives)
        if summary.archives_omitted:
            names += f" and {summary.archives_omitted} more"
        lines.append(
            f"Say plainly that accepting would archive {names}. They can "
            "unarchive a plan afterwards, but they should know before they click, "
            "not after."
        )
    if summary.stranded_sessions:
        lines.append(
            f"Say that {summary.stranded_sessions} planned session(s) fall "
            "beyond the plan's new last day. They are not deleted and they keep "
            "being scored, as missed — so a shorter plan costs the athlete "
            "adherence unless they clear those days themselves."
        )
    if summary.built_by == BUILT_BY_RULES and summary.kind == KIND_CREATE_PLAN:
        lines.append(
            "These weeks came from openkoutsi's own builder rather than from you, "
            "because the drafting call did not come back usable. Describe the "
            "plan as drafted, do not claim to have written each session."
        )
    return " ".join(lines)


@tool(
    name="propose_training_plan",
    title="Propose a training plan",
    scopes={"plans:read", "athlete:read"},
    arguments=ProposePlanArgs,
    returns=PlanProposalResult,
    internal_only=True,
    # Drafting makes a second, schema-constrained model call, which is seconds
    # rather than milliseconds. The loop's 30 s default would cancel a healthy
    # draft; see `Tool.timeout_s`.
    #
    # Deliberately **longer than the nested client's own 120 s** (`call_llm`).
    # Equal budgets race: a provider that hangs trips both at once, and if this
    # one wins the tool is cancelled *after* `draft_proposal` has committed but
    # before its result reaches the model — leaving a card under a reply that
    # never mentions a plan. With the inner timeout firing first, the nested call
    # raises, the fallback builder runs, and the athlete gets a proposal the
    # model has actually described. `chat_stuck_minutes` is 10, so this is still
    # far inside what `settle_stuck_turns` allows.
    timeout_s=150.0,
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
async def propose_training_plan(
    run: ToolRun, args: ProposePlanArgs
) -> PlanProposalResult:
    """Draft a whole training plan and offer it to the athlete for approval.
    **This does not create a plan.** It builds the plan you are proposing, puts
    it in front of the athlete as a yes/no card under your reply, and changes
    nothing at all unless they accept it.

    Use it only when the athlete has asked for a plan — "build me eight weeks for
    a gran fondo in October", "can you write me a winter block". A question about
    what a taper does, or what you would change about next week, is a question,
    not a request to build something. Propose at most one thing per turn.

    Call get_plan_status first. The result tells you what the athlete already has,
    and creating a plan **archives every active plan whose dates overlap it** —
    this tool reports exactly which, and your reply must say so, because a yes
    that did not know is not consent.

    Drafting takes a few seconds: the weeks are written properly rather than
    sketched. If that fails, openkoutsi's own builder produces them instead and
    the result says so.
    """
    config = _plan_config(args)
    num_weeks = args.weeks

    if args.start_date < run.today:
        raise ToolError(
            f"A plan cannot start on {args.start_date.isoformat()}, which is in "
            f"the past — today is {run.today.isoformat()}. Propose a start date "
            "of today or later; the athlete may not answer until tomorrow, so a "
            "date a few days out is usually the kind thing."
        )

    built_by = BUILT_BY_LLM
    fallback_reason: Optional[str] = None
    weeks_data: Optional[list[list[dict]]] = None

    if run.llm is None:
        built_by = BUILT_BY_RULES
        fallback_reason = "no model was available to this call"
    else:
        try:
            weeks_data = await generate_plan_weeks_with_cfg(
                run.athlete,
                config,
                num_weeks,
                args.goal,
                run.session,
                run.llm,
                user_id=run.caller.user_id,
                # The athlete's date, not the server's: the recent-training
                # window the draft is written against is theirs.
                today=run.today,
                # One completion, not two. See `generate_plan_weeks_with_cfg`:
                # the retry can outlast this tool's own 120 s budget, and a
                # cancelled draft produces nothing where a skipped retry falls
                # back to the builder and still produces a proposal.
                allow_retry=False,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never fail
            # Same posture as `coaching_stream`: a worse draft now beats no
            # draft at all, and `built_by` keeps the difference visible instead
            # of silently passing a rule-built plan off as a written one.
            log.warning("propose_training_plan: falling back to the builder: %s", exc)
            built_by = BUILT_BY_RULES
            fallback_reason = "the drafting call did not come back usable"

    if weeks_data is None:
        weeks_data = _rule_based_weeks(config, num_weeks)

    week_meta = week_meta_from_weeks(config, weeks_data)
    end_date = plan_end_date(args.start_date, num_weeks)
    archives = await overlapping_active_plans(
        run.session, run.athlete.id, args.start_date, end_date
    )

    summary = PlanProposalSummary(
        kind=KIND_CREATE_PLAN,
        built_by=built_by,
        fallback_reason=fallback_reason,
        plan_name=args.name,
        goal=args.goal,
        start_date=args.start_date,
        end_date=end_date,
        weeks=num_weeks,
        weekly=_weekly_summary(weeks_data, week_meta),
        first_week=_first_week(weeks_data[0]) if weeks_data else [],
        remaining_weeks=max(0, len(weeks_data) - 1),
        archives=_archives(archives),
    )

    proposal = await draft_proposal(
        run.session,
        kind=KIND_CREATE_PLAN,
        payload={
            "name": args.name,
            "goal": args.goal,
            "start_date": args.start_date.isoformat(),
            "weeks": num_weeks,
            "config": config.model_dump(),
            "week_meta": week_meta,
            "weeks_data": weeks_data,
            "generation_method": "llm" if built_by == BUILT_BY_LLM else "rule_based",
        },
        summary=summary,
        built_by=built_by,
        user_id=run.caller.user_id,
        conversation_id=run.conversation_id,
        message_id=run.message_id,
    )

    for_model = _for_model(summary)
    return PlanProposalResult(
        proposal_id=proposal.id,
        status=proposal.status,
        expires_at=proposal.expires_at,
        summary=for_model,
        note=_note(for_model),
    )


def _change_str(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@tool(
    name="propose_plan_change",
    title="Propose a change to a plan",
    scopes={"plans:read"},
    arguments=ProposePlanChangeArgs,
    returns=PlanProposalResult,
    internal_only=True,
    # No nested model call on this path, but the same headroom: a change is
    # drafted against a plan whose every session has to be loaded.
    timeout_s=150.0,
    annotations={"readOnlyHint": False, "idempotentHint": False},
)
async def propose_plan_change(
    run: ToolRun, args: ProposePlanChangeArgs
) -> PlanProposalResult:
    """Draft a change to a plan the athlete already has, and offer it for
    approval. **This does not change anything.** It puts the change in front of
    the athlete as a yes/no card under your reply, and only their yes writes it.

    Covers the plan as a whole — name, goal, start date, length, archiving or
    reopening it — and one planned session at a time: its type, description,
    duration, target Load or the day it falls on, plus skipping and unskipping it
    with a reason. "Push the whole thing back a week" is a new start date; "make
    Thursday easier" is one session.

    Call get_plan_status first and change what is actually there: ids come from
    its result, never from memory or a guess. Use it only when the athlete has
    asked for a change — "what would you change about next week?" is a request
    for your opinion, not for an edit. Propose at most one thing per turn.

    It cannot regenerate a plan's remaining weeks, and it cannot link an activity
    to a planned session or mark one done: that is a claim about what happened
    rather than about what is planned, and a wrong one quietly corrupts the
    athlete's adherence score.
    """
    plan = (
        await run.session.execute(
            select(TrainingPlan)
            .where(
                TrainingPlan.id == args.plan_id,
                TrainingPlan.athlete_id == run.athlete.id,
            )
            .options(selectinload(TrainingPlan.workouts))
        )
    ).scalar_one_or_none()
    if plan is None:
        raise ToolError(
            f"No plan with id '{args.plan_id}'. Call get_plan_status to see the "
            "athlete's plans and use an id from its result."
        )

    if args.change == "update_plan":
        summary, payload, workout_id = await _draft_plan_update(run, args, plan)
        kind = KIND_UPDATE_PLAN
    else:
        summary, payload, workout_id = _draft_workout_update(run, args, plan)
        kind = KIND_UPDATE_WORKOUT

    proposal = await draft_proposal(
        run.session,
        kind=kind,
        payload=payload,
        summary=summary,
        built_by=BUILT_BY_RULES,
        user_id=run.caller.user_id,
        conversation_id=run.conversation_id,
        message_id=run.message_id,
        target_plan_id=plan.id,
        target_workout_id=workout_id,
    )

    for_model = _for_model(summary)
    return PlanProposalResult(
        proposal_id=proposal.id,
        status=proposal.status,
        expires_at=proposal.expires_at,
        summary=for_model,
        note=_note(for_model),
    )


async def _draft_plan_update(
    run: ToolRun, args: ProposePlanChangeArgs, plan: TrainingPlan
) -> tuple[PlanProposalSummary, dict, Optional[str]]:
    changes: dict = {}
    shown: list[ProposedChange] = []

    def record(field: str, before, after) -> None:
        changes[field] = after.isoformat() if isinstance(after, date) else after
        shown.append(
            ProposedChange(
                field=field, before=_change_str(before), after=_change_str(after)
            )
        )

    if args.name is not None and args.name != plan.name:
        record("name", plan.name, args.name)
    if args.goal is not None and args.goal != plan.goal:
        record("goal", plan.goal, args.goal)
    if args.start_date is not None and args.start_date != plan.start_date:
        record("start_date", plan.start_date, args.start_date)
    if args.weeks is not None and args.weeks != plan.weeks:
        record("weeks", plan.weeks, args.weeks)
    if args.status is not None and args.status != plan.status:
        record("status", plan.status, args.status)

    if not changes:
        raise ToolError(
            "That change would leave the plan exactly as it is. Say what should "
            "be different — a new name, goal, start date, length, or status — or "
            "tell the athlete there is nothing to change."
        )

    # What the plan would look like afterwards, which is what decides both the
    # overlap set and whether this reopens something.
    start = (
        date.fromisoformat(changes["start_date"])
        if "start_date" in changes
        else plan.start_date
    )
    weeks = changes.get("weeks", plan.weeks)
    end = plan_end_date(start, weeks) if (start and weeks) else plan.end_date
    status_after = changes.get("status", plan.status)
    if "status" not in changes and plan.status == "completed" and (
        "start_date" in changes or "weeks" in changes
    ):
        status_after = "active"
    reopens = status_after == "active" and plan.status != "active"

    archives: list[TrainingPlan] = []
    if reopens:
        # Only a **reopen** files anything away, exactly as
        # `POST /plans/{id}/unarchive` does. A plan that is already active and is
        # merely being renamed archives nothing — computing the overlap set for
        # every change would quietly turn "fix this typo" into "archive the plan
        # you are also following". And never the plan itself: a plan always
        # overlaps its own dates.
        archives = [
            other
            for other in await overlapping_active_plans(
                run.session, run.athlete.id, start, end
            )
            if other.id != plan.id
        ]

    summary = PlanProposalSummary(
        kind=KIND_UPDATE_PLAN,
        built_by=BUILT_BY_RULES,
        plan_name=plan.name,
        goal=plan.goal,
        start_date=start,
        end_date=end,
        weeks=weeks,
        changes=shown,
        target_plan_id=plan.id,
        reopens_plan=reopens,
        archives=_archives(archives),
        # Shortening a plan leaves its later sessions where they are — the REST
        # path does the same — and `score_plan` has no end-date filter, so they
        # go on being scored as misses once their dates pass. The write is not
        # this feature's to change, but a preview that showed "weeks: 4 → 1" and
        # said nothing about the sessions that become misses would be collecting
        # exactly the uninformed yes this design exists to prevent.
        stranded_sessions=len(stranded_sessions(plan, weeks)),
    )
    return summary, {"changes": changes}, None


def _draft_workout_update(
    run: ToolRun, args: ProposePlanChangeArgs, plan: TrainingPlan
) -> tuple[PlanProposalSummary, dict, Optional[str]]:
    if not args.workout_id:
        raise ToolError(
            "'workout_id' is required for a change to one session. Call "
            "get_plan_status and use the session you mean from its result."
        )
    workout = next((w for w in plan.workouts if w.id == args.workout_id), None)
    if workout is None:
        raise ToolError(
            f"No planned session with id '{args.workout_id}' in plan "
            f"'{plan.name}'. Call get_plan_status again — the plan may have "
            "changed since you last looked."
        )
    if workout.is_completed:
        raise ToolError(
            "That session is already completed — an activity is linked to it — "
            "so it cannot be edited or skipped. Say so to the athlete rather "
            "than proposing a change they cannot accept."
        )

    changes: dict = {}
    shown: list[ProposedChange] = []

    def record(field: str, before, after) -> None:
        changes[field] = after.isoformat() if isinstance(after, date) else after
        shown.append(
            ProposedChange(
                field=field, before=_change_str(before), after=_change_str(after)
            )
        )

    if args.change == "skip_workout":
        if not args.skip_reason:
            raise ToolError(
                "Skipping a session needs a reason, in the athlete's own words. "
                "A genuine one is scored more kindly than a missed session, so a "
                "skip with nothing behind it is worse for them than leaving it."
            )
        record("skip_reason", workout.skip_reason, args.skip_reason)
    elif args.change == "unskip_workout":
        if not workout.skip_reason:
            raise ToolError(
                "That session is not marked as skipped, so there is nothing to "
                "undo."
            )
        record("skip_reason", workout.skip_reason, None)
    else:
        for field, value in (
            ("workout_type", args.workout_type),
            ("description", args.description),
            ("duration_min", args.duration_min),
            ("target_load", args.target_load),
            ("day_of_week", args.day_of_week),
        ):
            if value is not None and value != getattr(workout, field):
                record(field, getattr(workout, field), value)
        if not changes:
            raise ToolError(
                "That change would leave the session exactly as it is. Say what "
                "should be different — its type, description, duration, target "
                "Load, or the day it falls on."
            )
        moved_to = changes.get("day_of_week")
        if moved_to is not None:
            # Nothing stops two sessions sharing a day — there is no unique key
            # on (plan, week, day) and no REST endpoint that edits these fields,
            # so no precedent quietly covers it. Stacked, both get prescribed and
            # `score_plan` counts both, which is not what "move Thursday to
            # Saturday" means. Refused with a sentence naming the occupant, the
            # same posture the completed-session refusal above takes.
            occupant = occupant_of(plan, workout.week_number, moved_to, workout.id)
            if occupant is not None:
                raise ToolError(
                    f"There is already a {occupant.workout_type or 'session'} on "
                    f"{_DAY_NAMES.get(moved_to, 'that day')} of that week, and a "
                    "plan does not stack two sessions on one day. Move or clear "
                    "that one first, or pick a free day — get_plan_status shows "
                    "which are free."
                )

    # The day the session would end up on — not the one it is on now. Both the
    # date and the name are derived from it, because a card showing the old
    # slot's date under the new slot's name describes two different sessions and
    # neither is the one being agreed to.
    day_after = changes.get("day_of_week", workout.day_of_week)
    when = (
        workout_date(plan.start_date, workout.week_number, day_after)
        if plan.start_date is not None
        else None
    )

    # The weekday name comes from that computed date rather than from the
    # integer. `day_of_week` is documented as 1 = Monday, but `workout_date`
    # treats it as an offset from `plan.start_date`, which `create_plan` takes as
    # given and never normalises to a Monday — so on a Wednesday-started plan the
    # two readings disagree. This is the first place a weekday name is rendered
    # for the **athlete** rather than for a model, and deriving it from the date
    # is right under either reading.
    label = " ".join(
        part
        for part in (
            when.strftime("%A") if when is not None else None,
            workout.workout_type or "session",
        )
        if part
    )

    summary = PlanProposalSummary(
        kind=KIND_UPDATE_WORKOUT,
        built_by=BUILT_BY_RULES,
        plan_name=plan.name,
        changes=shown,
        target_plan_id=plan.id,
        target_workout_id=workout.id,
        target_date=when,
        target_label=label,
    )
    return summary, {"changes": changes}, workout.id
