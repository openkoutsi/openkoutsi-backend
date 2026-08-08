"""``get_training_status`` — where the athlete stands today (issue #42).

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
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from backend.app.mcp.dispatch import ToolRun
from backend.app.mcp.registry import ToolArgs, tool
from backend.app.mcp.shaping import int_or_none, round_or_none
from backend.app.models.user_orm import Activity, DailyMetric
from backend.app.schemas.metrics import _form_to_label
from backend.app.services.athlete_experience import experience_level
from backend.app.services.metrics_engine import catch_up_metrics
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES


class TrainingStatusArgs(ToolArgs):
    window_days: int = Field(
        28,
        ge=7,
        le=180,
        description=(
            "Length of the trailing window the volume totals cover, in days. "
            "28 (four weeks) is the coaching default; 7 answers 'how was last "
            "week', 90 answers 'how has this block gone'."
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
    as_of: date = Field(..., description="Date the fitness figures describe (calendar date).")
    stale: bool = Field(
        False,
        description=(
            "True when the newest stored metrics predate today — the athlete has "
            "recorded nothing recently, so treat the numbers as of 'as_of', not "
            "as of now."
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
        0.0, description="Load recorded for 'as_of' itself (unitless Load points)."
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


@tool(
    name="get_training_status",
    title="Current training status",
    scopes={"metrics:read", "athlete:read"},
    arguments=TrainingStatusArgs,
    returns=TrainingStatus,
)
async def get_training_status(run: ToolRun, args: TrainingStatusArgs) -> TrainingStatus:
    """Where the athlete stands right now: Fitness, Fatigue and Form with their
    recent trend, the volume actually ridden over a trailing window, and the
    profile context (FTP, max HR, weight, experience level) those numbers only
    mean something against.

    Start here. It is cheap, it needs no arguments, and it tells you whether the
    interesting question is about load, freshness, or something else entirely.
    Form is reported both as a number and as a word (peak / fresh / neutral /
    tired / overreached) — prefer the word over inventing your own thresholds.

    Missing metric rows are filled in from stored Load before answering, so the
    figures are current whether or not anything has opened the dashboard today.
    """
    session, athlete = run.session, run.athlete
    today = run.today

    # Same catch-up the dashboard and the fitness forecast perform: without it
    # the answer depends on whether a browser happened to hit the API recently.
    await catch_up_metrics(athlete.id, session)

    metric = (
        await session.execute(
            select(DailyMetric)
            .where(DailyMetric.athlete_id == athlete.id, DailyMetric.date <= today)
            .order_by(DailyMetric.date.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    as_of = metric.date if metric else today
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

    window_start = today - timedelta(days=args.window_days)
    load_total = (
        await session.execute(
            select(func.coalesce(func.sum(DailyMetric.load_day), 0.0)).where(
                DailyMetric.athlete_id == athlete.id,
                DailyMetric.date > window_start,
                DailyMetric.date <= today,
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
            )
        )
    ).one()

    weeks = max(args.window_days / 7.0, 1.0)
    return TrainingStatus(
        as_of=as_of,
        stale=as_of < today,
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
