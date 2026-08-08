"""Activity tools: ``list_recent_activities``, ``find_activity``, ``get_activity_detail``.

Three tools rather than one, because they answer three different questions and a
single ``get_activities(**everything)`` would make the model guess which shape it
was going to get back.

The hard constraint here is size. A three-hour ride at 1 Hz carries roughly
eleven thousand samples *per stream*, and the platform stores several; none of
that ever leaves this module. What a coach reads off a ride — how long, how hard,
how it was paced, whether the aerobic signal held up — is a few dozen numbers,
and those are computed on the way out. ``get_activity_detail`` goes furthest and
still returns intervals, time-in-zone and the per-ride power bests, all of which
are already aggregates in the database.

Location data is stripped by policy: no tool here returns coordinates, and none
of these models has a field that could carry one. Where a ride was is not
something a coaching model needs, and it is the single most sensitive thing in
the record.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.errors import ToolError
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import hhmm, page, pct, round_or_none
from backend.app.models.user_orm import (
    Activity,
    ActivityInterval,
    ActivityPowerBest,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)
from openkoutsi.training_math import efficiency_factor, variability_index

#: ``ActivitySummary`` carries a field literally called ``date``, which shadows
#: the type inside the class body; the annotation uses this alias instead.
_Date = date

#: Labels the platform recognises on an activity. Kept in step with
#: ``backend.app.api.activities._VALID_LABELS``; a test asserts they agree, so a
#: third label added there cannot become silently unfilterable here.
VALID_LABELS = ("race", "commute")

#: Interval rows returned by ``get_activity_detail``. A structured session might
#: have forty; an auto-split of a long ride can run to hundreds, and past the
#: first few dozen the model is reading noise.
MAX_INTERVALS = 40


class ActivitySummary(BaseModel):
    """One activity, reduced to the fields a coaching decision turns on."""

    activity_id: str = Field(..., description="Identifier to pass to get_activity_detail.")
    date: Optional[_Date] = Field(None, description="Calendar date the activity started.")
    name: Optional[str] = Field(None, description="Activity title as recorded or named by the athlete.")
    sport_type: Optional[str] = Field(None, description="Sport, e.g. Ride, VirtualRide, Run.")
    workout_category: Optional[str] = Field(
        None,
        description=(
            "Classified session type: recovery, endurance, tempo, threshold, "
            "vo2max, anaerobic, sprint, strength, yoga or cross_training. Null "
            "when it could not be classified (usually no power data)."
        ),
    )
    duration_s: Optional[int] = Field(None, description="Duration in seconds (s).")
    duration_text: Optional[str] = Field(None, description="Duration as readable text, e.g. '2 h 04'.")
    distance_m: Optional[float] = Field(None, description="Distance in metres (m).")
    elevation_m: Optional[float] = Field(None, description="Total ascent in metres (m).")
    avg_power_w: Optional[float] = Field(None, description="Average power in watts (W).")
    weighted_power_w: Optional[float] = Field(
        None,
        description=(
            "Weighted (normalised) average power in watts (W) — the intensity a "
            "steady ride of the same physiological cost would have been."
        ),
    )
    avg_hr_bpm: Optional[float] = Field(None, description="Average heart rate in beats per minute (bpm).")
    max_hr_bpm: Optional[float] = Field(None, description="Peak heart rate in beats per minute (bpm).")
    avg_cadence_rpm: Optional[float] = Field(None, description="Average cadence in revolutions per minute (rpm).")
    load: Optional[float] = Field(
        None, description="Training Load for the session (unitless Load points; ~100 = one hour at threshold)."
    )
    intensity: Optional[float] = Field(
        None, description="Intensity factor: weighted power ÷ FTP (ratio, unitless). 1.0 = threshold."
    )
    rpe: Optional[int] = Field(
        None, description="Athlete's own rating of perceived exertion, 1–10 (unitless). Null when unrated."
    )
    labels: list[str] = Field(default_factory=list, description="Athlete labels on the ride, e.g. race, commute.")
    has_power: bool = Field(False, description="Whether the ride recorded power at all.")


class AerobicResponse(BaseModel):
    """Efficiency and durability, with reason codes preserved rather than nulled."""

    efficiency_factor: Optional[float] = Field(
        None,
        description=(
            "Weighted power per heartbeat (watts per bpm). Rising over time at "
            "constant load is aerobic progress. Null without both power and HR."
        ),
    )
    variability_index: Optional[float] = Field(
        None,
        description=(
            "Weighted ÷ average power (ratio, unitless). Near 1.0 is steady; "
            "above ~1.10 indicates interval or punchy riding."
        ),
    )
    decoupling_pct: Optional[float] = Field(
        None,
        description=(
            "Aerobic decoupling: how far the power:heart-rate ratio drifted "
            "between the first and second half of the ride, as a percentage (%). "
            "Under ~5% is good durability. Null when not measurable — read "
            "'decoupling_reason' instead of treating null as zero."
        ),
    )
    decoupling_reason: Optional[str] = Field(
        None,
        description=(
            "Why there is no decoupling figure: too_short, no_power, no_hr, "
            "degenerate_hr, stream_mismatch, variable_effort or uneven_pacing. "
            "Null when a figure was computed."
        ),
    )
    cp_w: Optional[float] = Field(
        None,
        description=(
            "Critical power in watts (W) fit from the athlete's power curve as "
            "it stood on this ride's date, and frozen. Null when no plausible "
            "fit was available then."
        ),
    )
    w_prime_j: Optional[float] = Field(
        None, description="Anaerobic work capacity W′ in joules (J), from the same frozen fit."
    )


class ZoneTime(BaseModel):
    zone: str = Field(..., description="Zone name as configured by the athlete, e.g. Z2.")
    seconds: int = Field(0, description="Time accumulated in this zone, in seconds (s).")
    pct: float = Field(0.0, description="Share of the ride's zoned time spent here, as a percentage (%).")


class IntervalRow(BaseModel):
    number: int = Field(..., description="Interval's position in the ride, 1-based (count).")
    start_offset_s: int = Field(0, description="Seconds (s) from the start of the ride to this interval.")
    duration_s: int = Field(0, description="Interval length in seconds (s).")
    distance_m: Optional[float] = Field(None, description="Interval distance in metres (m).")
    avg_power_w: Optional[float] = Field(None, description="Average power over the interval, in watts (W).")
    avg_hr_bpm: Optional[float] = Field(None, description="Average heart rate over the interval, in bpm.")
    avg_cadence_rpm: Optional[float] = Field(None, description="Average cadence over the interval, in rpm.")
    auto_split: bool = Field(
        False, description="True when this interval was detected automatically rather than pressed by the athlete."
    )


class LinkedWorkout(BaseModel):
    """The planned session this activity was performed against, if any."""

    plan_name: str = Field(..., description="Name of the training plan the planned workout belongs to.")
    workout_type: Optional[str] = Field(None, description="Planned session type, e.g. endurance, threshold, rest.")
    description: Optional[str] = Field(None, description="What the plan asked for, in the plan's own words.")
    target_load: Optional[int] = Field(None, description="Load the plan prescribed (unitless Load points).")
    duration_min: Optional[int] = Field(None, description="Duration the plan prescribed, in minutes (min).")


class PowerBestRow(BaseModel):
    duration_s: int = Field(..., description="Effort length in seconds (s).")
    power_w: float = Field(..., description="Best average power sustained for that length in this ride, in watts (W).")


class ActivityDetail(ActivitySummary):
    notes: Optional[str] = Field(None, description="Free-text notes the athlete attached to the ride.")
    aerobic: AerobicResponse = Field(..., description="Efficiency, variability and durability for this ride.")
    power_zones: list[ZoneTime] = Field(
        default_factory=list,
        description="Time in each power zone, frozen at processing time using the zones in effect then.",
    )
    hr_zones: list[ZoneTime] = Field(
        default_factory=list, description="Time in each heart-rate zone, frozen the same way."
    )
    intervals: list[IntervalRow] = Field(
        default_factory=list, description=f"Intervals within the ride, at most {MAX_INTERVALS} of them."
    )
    intervals_total: int = Field(0, description="How many intervals the ride has in total (count).")
    power_bests: list[PowerBestRow] = Field(
        default_factory=list, description="Best sustained power at each standard duration within this ride."
    )
    linked_workout: Optional[LinkedWorkout] = Field(
        None, description="The planned workout this ride was performed against, when it is linked to one."
    )


class RecentActivitiesArgs(ToolArgs):
    limit: int = Field(10, ge=1, le=50, description="How many activities to return, newest first (count).")
    days: Optional[int] = Field(
        None,
        ge=1,
        le=3650,
        description="Only include activities from the past N days. Omit for no date restriction.",
    )
    sport_type: Optional[str] = Field(
        None, description="Restrict to one sport, e.g. Ride or Run. Omit for every sport."
    )
    exclude_labels: list[Literal["race", "commute"]] = Field(
        default_factory=list,
        description="Drop activities carrying any of these labels. Excluding 'commute' is the usual way to see real training only.",
    )


class RecentActivities(BaseModel):
    items: list[ActivitySummary] = Field(default_factory=list, description="Matching activities, newest first.")
    returned: int = Field(0, description="How many activities are in this response (count).")
    total: int = Field(0, description="How many matched in total (count); more than 'returned' means results were cut off.")
    truncated: bool = Field(False, description="True when results were cut off by 'limit'.")


class FindActivityArgs(ToolArgs):
    on_date: Optional[date] = Field(
        None, description="Exact calendar date to look on. Use for 'what did I ride on Tuesday'."
    )
    start: Optional[date] = Field(None, description="Earliest calendar date to include (inclusive).")
    end: Optional[date] = Field(None, description="Latest calendar date to include (inclusive).")
    sport_type: Optional[str] = Field(None, description="Restrict to one sport, e.g. Ride.")
    workout_category: Optional[str] = Field(
        None, description="Restrict to one session type, e.g. threshold, endurance, vo2max."
    )
    label: Optional[Literal["race", "commute"]] = Field(
        None, description="Only activities carrying this label."
    )
    name_contains: Optional[str] = Field(
        None, max_length=100, description="Case-insensitive substring match on the activity's name."
    )
    min_duration_s: Optional[int] = Field(
        None, ge=0, description="Only activities lasting at least this many seconds (s)."
    )
    limit: int = Field(10, ge=1, le=25, description="How many matches to return, newest first (count).")


class FoundActivities(RecentActivities):
    pass


class ActivityDetailArgs(ToolArgs):
    activity_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifier from list_recent_activities or find_activity.",
    )


def _labels_of(activity: Activity) -> list[str]:
    return [lbl for lbl in (activity.labels or []) if isinstance(lbl, str)]


def _summarize(activity: Activity) -> ActivitySummary:
    return ActivitySummary(
        activity_id=activity.id,
        date=activity.start_time.date() if activity.start_time else None,
        name=activity.name,
        sport_type=activity.sport_type,
        workout_category=activity.workout_category,
        duration_s=activity.duration_s,
        duration_text=hhmm(activity.duration_s),
        distance_m=round_or_none(activity.distance_m, 1),
        elevation_m=round_or_none(activity.elevation_m, 1),
        avg_power_w=round_or_none(activity.avg_power, 1),
        weighted_power_w=round_or_none(activity.weighted_power, 1),
        avg_hr_bpm=round_or_none(activity.avg_hr, 1),
        max_hr_bpm=round_or_none(activity.max_hr, 1),
        avg_cadence_rpm=round_or_none(activity.avg_cadence, 1),
        load=round_or_none(activity.load, 1),
        intensity=round_or_none(activity.intensity, 3),
        rpe=activity.rpe,
        labels=_labels_of(activity),
        has_power=activity.avg_power is not None,
    )


def _like_literal(text: str) -> str:
    """Escape LIKE's wildcards so a substring search is one.

    ``%`` and ``_`` are wildcards, so an unescaped ``name_contains="_intervals"``
    also matches ``4x8 intervals`` and ``"100%"`` matches anything starting
    ``100``. The field's own description promises a plain case-insensitive
    substring match, and that is what the model will believe it got — the same
    failure ``ToolArgs(extra="forbid")`` exists to prevent, where more rows come
    back than were asked for and get reported as a filtered answer.

    The backslash goes first, or it would escape the escapes added after it.
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _has_label(label: str):
    """Correlated EXISTS over the JSON ``labels`` array — see ``api.activities``."""
    entries = func.json_each(Activity.labels).table_valued("value")
    return (
        select(1)
        .select_from(entries)
        .where(entries.c.value == label)
        .correlate(Activity)
        .exists()
    )


def _describe(activity: Activity) -> str:
    """``2026-07-13 (endurance, 2 h 04)`` — for the 'nearest rides' sentence."""
    day = activity.start_time.date().isoformat() if activity.start_time else "undated"
    kind = activity.workout_category or activity.sport_type or "activity"
    return f"{day} ({kind}, {hhmm(activity.duration_s) or 'unknown length'})"


async def _page_of(run: ToolRun, query, limit: int) -> dict:
    """Count, then take the newest ``limit`` — the shared collection shape."""
    total = (
        await run.session.execute(select(func.count()).select_from(query.subquery()))
    ).scalar_one()
    rows = (
        await run.session.execute(
            query.order_by(Activity.start_time.desc()).limit(limit)
        )
    ).scalars().all()
    return page([_summarize(a) for a in rows], int(total))


@tool(
    name="list_recent_activities",
    title="Recent activities",
    scopes={"activities:read"},
    arguments=RecentActivitiesArgs,
    returns=RecentActivities,
)
async def list_recent_activities(run: ToolRun, args: RecentActivitiesArgs) -> RecentActivities:
    """The athlete's most recent activities, newest first, with the fields a
    coaching judgement actually turns on: duration, distance, Load, intensity,
    average and weighted power, heart rate, session category, labels and the
    athlete's own RPE.

    This is the tool for "what has this athlete been doing lately". Ask
    find_activity instead when you know roughly *when* or *what* you are looking
    for, and get_activity_detail when one specific ride needs unpacking.

    Excluding the 'commute' label is usually right when judging training: a
    twenty-minute ride to work is real Load but it is not a session, and left in
    it makes a week look busier than it trained.

    Never returns raw data streams or locations.
    """
    query = select(Activity).where(Activity.athlete_id == run.athlete.id)
    if args.days is not None:
        cutoff = datetime.combine(run.today - timedelta(days=args.days), time.min)
        query = query.where(Activity.start_time >= cutoff)
    if args.sport_type:
        query = query.where(Activity.sport_type == args.sport_type)
    for label in args.exclude_labels:
        query = query.where(~_has_label(label))

    return RecentActivities(**await _page_of(run, query, args.limit))


@tool(
    name="find_activity",
    title="Find activities",
    scopes={"activities:read"},
    arguments=FindActivityArgs,
    returns=FoundActivities,
)
async def find_activity(run: ToolRun, args: FindActivityArgs) -> FoundActivities:
    """Search the athlete's activities by date, sport, session category, label,
    name or minimum duration. Returns the same summary shape as
    list_recent_activities, newest first.

    Use 'on_date' for a specific day ("what did I do on the 14th"), or
    'start'/'end' for a range. When a date turns up nothing, the failure names
    the nearest rides on either side rather than just saying no — so a query
    that was one day off can be fixed in a single further call.

    Never returns raw data streams or locations.
    """
    if args.start and args.end and args.start > args.end:
        raise ToolError(
            f"The window is inverted: start ({args.start}) is after end "
            f"({args.end}). Swap them."
        )

    query = select(Activity).where(Activity.athlete_id == run.athlete.id)
    if args.on_date is not None:
        query = query.where(
            Activity.start_time >= datetime.combine(args.on_date, time.min),
            Activity.start_time <= datetime.combine(args.on_date, time.max),
        )
    if args.start is not None:
        query = query.where(Activity.start_time >= datetime.combine(args.start, time.min))
    if args.end is not None:
        query = query.where(Activity.start_time <= datetime.combine(args.end, time.max))
    if args.sport_type:
        query = query.where(Activity.sport_type == args.sport_type)
    if args.workout_category:
        query = query.where(Activity.workout_category == args.workout_category)
    if args.label:
        query = query.where(_has_label(args.label))
    if args.name_contains:
        query = query.where(
            Activity.name.ilike(f"%{_like_literal(args.name_contains)}%", escape="\\")
        )
    if args.min_duration_s is not None:
        query = query.where(Activity.duration_s >= args.min_duration_s)

    result = await _page_of(run, query, args.limit)
    if not result["items"] and args.on_date is not None:
        raise ToolError(
            f"No activity on {args.on_date.isoformat()}.",
            suggestions=await _nearest_rides(run, args.on_date),
        )
    return FoundActivities(**result)


async def _nearest_rides(run: ToolRun, missed: date) -> list[str]:
    """The closest activity either side of a date with nothing on it."""
    midnight = datetime.combine(missed, time.min)

    before = (
        await run.session.execute(
            select(Activity)
            .where(Activity.athlete_id == run.athlete.id, Activity.start_time < midnight)
            .order_by(Activity.start_time.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    after = (
        await run.session.execute(
            select(Activity)
            .where(
                Activity.athlete_id == run.athlete.id,
                Activity.start_time > datetime.combine(missed, time.max),
            )
            .order_by(Activity.start_time)
            .limit(1)
        )
    ).scalar_one_or_none()

    nearby = [_describe(a) for a in (before, after) if a is not None]
    if not nearby:
        return ["There are no activities recorded on either side of that date."]
    return [f"Nearest rides: {' and '.join(nearby)}."]


def _zone_rows(zone_times: Optional[dict], kind: str) -> list[ZoneTime]:
    times = (zone_times or {}).get(kind) or {}
    total = sum(times.values())
    return [
        ZoneTime(zone=name, seconds=int(seconds), pct=pct(seconds, total))
        for name, seconds in times.items()
    ]


@tool(
    name="get_activity_detail",
    title="Activity detail",
    scopes={"activities:read"},
    arguments=ActivityDetailArgs,
    returns=ActivityDetail,
)
async def get_activity_detail(run: ToolRun, args: ActivityDetailArgs) -> ActivityDetail:
    """Everything worth knowing about one activity: the summary figures, the
    athlete's notes and RPE, time spent in each power and heart-rate zone, the
    intervals within the ride, the best sustained power at each standard
    duration, the aerobic response (efficiency factor, variability index,
    decoupling), and the planned workout it was performed against if it is
    linked to one.

    Where a figure is absent, the reason is given rather than a bare null — a
    ride with no decoupling number carries a 'decoupling_reason' such as
    too_short or no_power. Treat those as facts, not as missing data.

    Raw per-second streams are deliberately not available through any tool: a
    long ride holds tens of thousands of samples, and everything they support is
    already summarised here.
    """
    activity = (
        await run.session.execute(
            select(Activity).where(
                Activity.id == args.activity_id,
                Activity.athlete_id == run.athlete.id,
            )
        )
    ).scalar_one_or_none()
    if activity is None:
        raise ToolError(
            f"No activity with id '{args.activity_id}' belongs to this athlete.",
            suggestions=[
                "Ids come from list_recent_activities or find_activity; call one "
                "of those and use an id from its results."
            ],
        )

    intervals_total = (
        await run.session.execute(
            select(func.count())
            .select_from(ActivityInterval)
            .where(ActivityInterval.activity_id == activity.id)
        )
    ).scalar_one()
    interval_rows = (
        await run.session.execute(
            select(ActivityInterval)
            .where(ActivityInterval.activity_id == activity.id)
            .order_by(ActivityInterval.interval_number)
            .limit(MAX_INTERVALS)
        )
    ).scalars().all()

    bests = (
        await run.session.execute(
            select(ActivityPowerBest)
            .where(ActivityPowerBest.activity_id == activity.id)
            .order_by(ActivityPowerBest.duration_s)
        )
    ).scalars().all()

    linked = (
        await run.session.execute(
            select(PlannedWorkout, TrainingPlan)
            .join(
                PlannedWorkoutActivity,
                PlannedWorkoutActivity.planned_workout_id == PlannedWorkout.id,
            )
            .join(TrainingPlan, TrainingPlan.id == PlannedWorkout.plan_id)
            .where(PlannedWorkoutActivity.activity_id == activity.id)
        )
    ).first()

    summary = _summarize(activity)
    return ActivityDetail(
        **summary.model_dump(),
        notes=activity.notes,
        aerobic=AerobicResponse(
            efficiency_factor=round_or_none(
                efficiency_factor(activity.weighted_power, activity.avg_hr), 3
            ),
            variability_index=round_or_none(
                variability_index(activity.weighted_power, activity.avg_power), 3
            ),
            decoupling_pct=round_or_none(activity.decoupling_pct, 2),
            decoupling_reason=activity.decoupling_reason,
            cp_w=round_or_none(activity.cp_w, 1),
            w_prime_j=round_or_none(activity.w_prime_j, 0),
        ),
        power_zones=_zone_rows(activity.zone_times, "power"),
        hr_zones=_zone_rows(activity.zone_times, "hr"),
        intervals=[
            IntervalRow(
                number=iv.interval_number,
                start_offset_s=iv.start_offset_s,
                duration_s=iv.duration_s,
                distance_m=round_or_none(iv.distance_m, 1),
                avg_power_w=round_or_none(iv.avg_power, 1),
                avg_hr_bpm=round_or_none(iv.avg_hr, 1),
                avg_cadence_rpm=round_or_none(iv.avg_cadence, 1),
                auto_split=iv.is_auto_split,
            )
            for iv in interval_rows
        ],
        intervals_total=int(intervals_total),
        power_bests=[
            PowerBestRow(duration_s=b.duration_s, power_w=round(b.power_w, 1))
            for b in bests
        ],
        linked_workout=(
            LinkedWorkout(
                plan_name=linked[1].name,
                workout_type=linked[0].workout_type,
                description=linked[0].description,
                target_load=linked[0].target_load,
                duration_min=linked[0].duration_min,
            )
            if linked is not None
            else None
        ),
    )
