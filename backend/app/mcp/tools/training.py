"""``get_training_status`` — where the athlete stands on a given day (issue #42).

The one tool a coaching turn almost always starts from, and the reason the tool
layer exists at all: the daily-feedback prompt builder currently assembles this
same picture ahead of time and hopes it guessed right. Here the model asks, gets
the numbers *and the context needed to read them*, and decides for itself what
to look at next.

That context is why this tool asks for ``athlete:read`` alongside
``metrics:read``. A Form of -18 means one thing for a first-season rider and
another for someone with an FTP of 340 W and a decade behind them, and a tool
that returned the number without the frame would be inviting the model to invent
the frame.

``as_of`` (issue #48) moves the whole picture back in time. Everything here was
anchored to today, which answers "where am I" but not "where was I" — and the
second question is the one a coach reaches for constantly: what shape was the
athlete in before last year's event, how loaded were they when the block
started, is this build steeper than the last one. The data to answer it was
already stored; only the anchor was hard-coded. Anchoring **every** figure on the
same date is the part that matters: a form number from June beside a volume
total from today would be a comparison the model could not see was invalid.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.errors import ToolError
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import int_or_none, round_or_none
from backend.app.models.user_orm import Activity, DailyMetric
from backend.app.schemas.metrics import _form_to_label
from backend.app.services.athlete_experience import experience_level
from backend.app.services.metrics_engine import catch_up_metrics
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES


class TrainingStatusArgs(ToolArgs):
    as_of: Optional[date] = Field(
        None,
        description=(
            "Calendar date to report the athlete's standing as of. Defaults to "
            "today. Give a past date to ask 'where was I then' — before an "
            "event, at the start of a block, this time last year — and every "
            "figure moves with it, including the trailing volume window and the "
            "Fitness trend. Cannot be in the future."
        ),
    )
    window_days: int = Field(
        28,
        ge=7,
        le=180,
        description=(
            "Length of the trailing window the volume totals cover, in days, "
            "ending on 'as_of'. 28 (four weeks) is the coaching default; 7 "
            "answers 'how was that week', 90 answers 'how did that block go'."
        ),
    )


class AthleteContext(BaseModel):
    """The profile figures the fitness numbers are only meaningful against."""

    ftp_w: Optional[int] = Field(
        None, description="Functional threshold power on the profile, in watts (W). Null when never set."
    )
    max_hr_bpm: Optional[int] = Field(
        None, description="Maximum heart rate on the profile, in beats per minute (bpm)."
    )
    resting_hr_bpm: Optional[int] = Field(
        None, description="Resting heart rate on the profile, in beats per minute (bpm)."
    )
    weight_kg: Optional[float] = Field(
        None, description="Most recent recorded bodyweight, in kilograms (kg)."
    )
    experience_level: Optional[str] = Field(
        None,
        description=(
            "Self-reported experience: novice, intermediate, experienced, "
            "semi-pro or elite. Null when the athlete has not said. Tailor "
            "load and detail to it."
        ),
    )


class VolumeTotals(BaseModel):
    """What was actually ridden over the window."""

    days: int = Field(0, description="Length of the window these totals cover, in days.")
    activities: int = Field(0, description="Number of cycling activities in the window (count).")
    duration_s: int = Field(0, description="Total moving/elapsed time in the window, in seconds (s).")
    distance_m: float = Field(0.0, description="Total distance ridden in the window, in metres (m).")
    load_total: float = Field(
        0.0,
        description=(
            "Sum of daily training Load over the window (unitless Load points; "
            "roughly 100 = one hour at threshold)."
        ),
    )
    load_weekly_avg: float = Field(
        0.0, description="Average Load per week over the window (Load points per week)."
    )


class TrainingStatus(BaseModel):
    as_of: date = Field(
        ...,
        description=(
            "Date the fitness figures actually describe (calendar date). This is "
            "the date asked for — today unless 'as_of' was given — unless nothing "
            "was stored on it, in which case it is the newest earlier date that "
            "was, and 'stale' is true."
        ),
    )
    requested_as_of: date = Field(
        ...,
        description=(
            "The date the figures were asked for (calendar date): the 'as_of' "
            "argument, or today when it was omitted. Compare with 'as_of' to see "
            "how far back the newest stored figures fall."
        ),
    )
    stale: bool = Field(
        False,
        description=(
            "True when the newest stored metrics predate the date asked about — "
            "the athlete recorded nothing up to it, so treat the numbers as of "
            "'as_of' rather than as of 'requested_as_of'."
        ),
    )
    fitness: float = Field(
        ..., description="Chronic training load / Fitness (unitless Load points, 42-day exponential average)."
    )
    fatigue: float = Field(
        ..., description="Acute training load / Fatigue (unitless Load points, 7-day exponential average)."
    )
    form: float = Field(
        ..., description="Form: Fitness minus Fatigue (unitless Load points). Positive = fresh, negative = loaded."
    )
    form_label: str = Field(
        ...,
        description=(
            "Form banded into words: peak, fresh, neutral, tired or overreached. "
            "Use this rather than inventing thresholds for the raw number."
        ),
    )
    load_today: float = Field(
        0.0,
        description=(
            "Load recorded for 'as_of' itself (unitless Load points) — that "
            "day's riding, not today's, when a past date was asked for."
        ),
    )
    fitness_change_7d: Optional[float] = Field(
        None,
        description=(
            "Change in Fitness over the last 7 days (Load points per week) — the "
            "ramp rate. Above roughly +8/week is a fast build; negative is "
            "detraining. Null when there is no metric from a week ago."
        ),
    )
    fitness_change_28d: Optional[float] = Field(
        None, description="Change in Fitness over the last 28 days (Load points)."
    )
    volume: VolumeTotals = Field(..., description="Riding actually done over the requested window.")
    athlete: AthleteContext = Field(..., description="Profile context the numbers should be read against.")


async def _first_metric_date(run: ToolRun) -> Optional[date]:
    """The earliest day this athlete has a stored metric for, if any."""
    return (
        await run.session.execute(
            select(func.min(DailyMetric.date)).where(
                DailyMetric.athlete_id == run.athlete.id
            )
        )
    ).scalar_one_or_none()


def _history_starts(first: Optional[date]) -> str:
    """The half-sentence naming where the recorded history does begin.

    A refusal that only says "nothing there" costs the model a call to find out
    where 'there' starts; naming it means the retry can be right first time.
    """
    if first is None:
        return " This athlete has no stored training metrics at all yet."
    return f" The recorded history starts on {first.isoformat()}."


@tool(
    name="get_training_status",
    title="Training status on a given day",
    scopes={"metrics:read", "athlete:read"},
    arguments=TrainingStatusArgs,
    returns=TrainingStatus,
)
async def get_training_status(run: ToolRun, args: TrainingStatusArgs) -> TrainingStatus:
    """Where the athlete stands: Fitness, Fatigue and Form with their recent
    trend, the volume actually ridden over a trailing window, and the profile
    context (FTP, max HR, weight, experience level) those numbers only mean
    something against.

    Start here. It is cheap, it needs no arguments, and it tells you whether the
    interesting question is about load, freshness, or something else entirely.
    Form is reported both as a number and as a word (peak / fresh / neutral /
    tired / overreached) — prefer the word over inventing your own thresholds.

    Pass 'as_of' to ask the same question about a **past** date: what shape the
    athlete was in before an event, at the start of a block, or this time last
    year. Everything moves with it — the trend, the volume window, that day's
    Load — so two calls with different 'as_of' dates are directly comparable.
    Call it twice rather than reasoning about how the numbers 'must have'
    changed; only the second call is real evidence.

    Missing metric rows are filled in from stored Load before answering, so the
    figures are current whether or not anything has opened the dashboard today.
    """
    session, athlete = run.session, run.athlete
    today = run.today
    requested = args.as_of or today
    if requested > today:
        raise ToolError(
            f"Cannot report a training status for {requested.isoformat()}: it is "
            f"in the future, and today is {today.isoformat()}. Fitness and "
            f"Fatigue are computed from riding that has happened.",
            suggestions=[
                "Ask for a date no later than today, or omit 'as_of' entirely.",
                "For what is planned ahead, call get_plan_status instead.",
            ],
        )

    # Same catch-up the dashboard and the fitness forecast perform: without it
    # the answer depends on whether a browser happened to hit the API recently.
    # It fills forward to today, so it also settles any gap a past 'as_of' is
    # about to be read through.
    await catch_up_metrics(athlete.id, session)

    metric = (
        await session.execute(
            select(DailyMetric)
            .where(DailyMetric.athlete_id == athlete.id, DailyMetric.date <= requested)
            .order_by(DailyMetric.date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    if metric is None and args.as_of is not None:
        # Only for an *explicit* date. Answering "where do I stand" over an empty
        # database with zeros is honest — a new athlete has no history and the
        # profile context is still worth having. Answering "where was I on 3
        # March" with zeros is not: the model has no way to tell that apart from
        # a genuine week off, and would report a collapse that never happened.
        raise ToolError(
            f"No training metrics stored on or before "
            f"{requested.isoformat()}.{_history_starts(await _first_metric_date(run))}",
            suggestions=[
                "Ask for a date inside the recorded history, or omit 'as_of' "
                "for where the athlete stands today."
            ],
        )

    as_of = metric.date if metric else requested
    fitness = metric.fitness if metric else 0.0
    fatigue = metric.fatigue if metric else 0.0
    form = metric.form if metric else 0.0

    history = {
        row.date: row.fitness
        for row in (
            await session.execute(
                select(DailyMetric).where(
                    DailyMetric.athlete_id == athlete.id,
                    DailyMetric.date >= as_of - timedelta(days=28),
                    DailyMetric.date <= as_of,
                )
            )
        ).scalars()
    }

    def change_over(days: int) -> Optional[float]:
        previous = history.get(as_of - timedelta(days=days))
        return None if previous is None else round(fitness - previous, 1)

    # Both totals end on the date asked about, not on today. The upper bound on
    # the activity query is what makes that true: without it a past 'as_of'
    # would report the fitness of March beside every ride since, which reads as
    # a training block that never happened.
    window_start = requested - timedelta(days=args.window_days)
    load_total = (
        await session.execute(
            select(func.coalesce(func.sum(DailyMetric.load_day), 0.0)).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date > window_start,
                DailyMetric.date <= requested,
            )
        )
    ).scalar_one()

    count, duration, distance = (
        await session.execute(
            select(
                func.count(Activity.id),
                func.coalesce(func.sum(Activity.duration_s), 0),
                func.coalesce(func.sum(Activity.distance_m), 0.0),
            ).where(
                Activity.athlete_id == athlete.id,
                Activity.sport_type.in_(CYCLING_SPORT_TYPES),
                Activity.start_time >= datetime.combine(window_start, time.min),
                Activity.start_time <= datetime.combine(requested, time.max),
            )
        )
    ).one()

    weeks = max(args.window_days / 7.0, 1.0)
    return TrainingStatus(
        as_of=as_of,
        requested_as_of=requested,
        stale=as_of < requested,
        fitness=round(fitness, 1),
        fatigue=round(fatigue, 1),
        form=round(form, 1),
        form_label=_form_to_label(form),
        load_today=round(metric.load_day, 1) if metric else 0.0,
        fitness_change_7d=change_over(7),
        fitness_change_28d=change_over(28),
        volume=VolumeTotals(
            days=args.window_days,
            activities=int(count),
            duration_s=int(duration),
            distance_m=round(float(distance), 1),
            load_total=round(float(load_total), 1),
            load_weekly_avg=round(float(load_total) / weeks, 1),
        ),
        athlete=AthleteContext(
            ftp_w=athlete.ftp,
            max_hr_bpm=int_or_none(athlete.max_hr),
            resting_hr_bpm=int_or_none(athlete.resting_hr),
            weight_kg=round_or_none(athlete.weight_kg, 1),
            experience_level=experience_level(athlete.app_settings),
        ),
    )
