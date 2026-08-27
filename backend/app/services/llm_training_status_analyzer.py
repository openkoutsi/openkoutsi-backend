"""
LLM-based daily training status analysis service.

Streams a coaching overview of the athlete's recent training state from any
OpenAI-compatible chat completions API and persists the result incrementally
so local models that take several minutes never time out.

Uses the same LLM configuration as llm_activity_analyzer — the instance's
configured presets (``instance_settings.llm_models``, first entry = default).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator

from sqlalchemy import select, update

from ..core.timezones import local_now
from ..db.user_session import get_user_session_factory
from ..models.user_orm import Activity, Athlete, DailyMetric, Goal, PlannedWorkout, TrainingPlan
from ..schemas.metrics import IntensityDistributionResponse, _form_to_label
from .athlete_experience import EXPERIENCE_GUIDANCE, experience_level
from .intensity_distribution import (
    DEFAULT_WINDOW_DAYS,
    compute_intensity_distribution,
)
from .llm_agent import (
    MAX_ROUNDS_STATUS,
    AgentRequest,
    agentic_enabled,
    coaching_stream,
)
from .llm_streaming import (
    failure_recovery,
    stream_chat_completion,
    stream_into_db,
)
from .stranded_runs import run_is_current, settle_training_status

log = logging.getLogger(__name__)

_LOCALE_LANGUAGE: dict[str, str] = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
}

_STATUS_GUIDANCE = """\
You are Koutsi, an expert endurance sports coach. Review the athlete's overall \
training state and provide direct, actionable daily coaching feedback in 3-5 paragraphs. \
Cover: recent training load trend, current fitness and fatigue state, \
adherence to the active training plan(s) (if any), and 1-2 specific recommendations \
for the coming days. Write in plain prose — no markdown headers, no bullet points, \
no code blocks. Separate each paragraph with a single blank line.

When evaluating training plan adherence, apply these rules:
- Today's planned workouts that are not yet completed must never be treated as missed. \
The athlete still has time to complete them. Either assume they will be done later today, \
or encourage the athlete to get them done — but do not criticise or flag them as missed.
- Planned workouts dated later than today (marked "upcoming") are in the future and are \
not due yet. Never treat an upcoming workout as missed, incomplete, or a sign of poor \
adherence. If the entire remainder of the week is still upcoming, the athlete is on track, \
not behind — frame it as the week ahead, not a shortfall.
- Only workouts from previous days count as missed. If past days show incomplete sessions, \
be direct and stern about it.
- When an incomplete workout has a skip reason attached, take it into account. A legitimate \
reason (illness, injury, travel, rest) should temper your criticism, while a pattern of weak \
excuses warrants a firmer response.
- Rest days are a planned, intentional part of the training plan. They do not have to be \
performed and there is nothing to complete on them. Never treat a rest day as missed, skipped, \
or a sign of poor adherence — an athlete taking their scheduled rest is following the plan correctly.\
"""

# The format contract, kept as its own constant because the agentic path has to
# restate it (issue #43). `parseMoodAndParagraphs` reads this line to choose
# Koutsi's avatar and the llm-eval harness asserts on it, and models are
# measurably worse at obeying a leading-format rule on a turn that follows tool
# results than on a clean single-shot prompt — so the agent loop repeats it
# where the model is about to answer rather than trusting turn one to carry.
_MOOD_RULE = """\
Before the feedback paragraphs, output a single line in the format: MOOD:<mood>
where <mood> is one of: cheer, knowing, neutral, stern.
- cheer: athlete is training well and making great progress
- stern: athlete is missing planned sessions, undertraining, or not following their plan
- neutral: routine week with no strong positive or negative takeaway
- knowing: all other cases (default)
The MOOD line must be the very first line, followed by a blank line, then the paragraphs.\
"""

# Unchanged wording, assembled from the two halves above.
_SYSTEM_PROMPT_BASE = f"{_STATUS_GUIDANCE}\n\n{_MOOD_RULE}"

# What replaces the blob on the agentic path: the same coaching rules, plus how
# to go and get the facts. Deliberately explicit that the prompt carries no data
# — a model handed tools and a familiar-looking question will otherwise answer
# from the question alone.
_TOOL_GUIDANCE = """\
You have tools that read this athlete's own training data, and this prompt \
contains none of it. An answer written without calling any tool is guesswork, so \
gather what you need first, then write the feedback.
- Start broad and narrow down. get_training_status gives fitness, fatigue and \
form; list_recent_activities gives what the athlete has actually been doing; \
get_plan_status gives this week's planned sessions and whether they were done.
- Follow the thread. If a number looks off, go and look at the sessions behind \
it so you can say why, instead of restating the number.
- Do not call the same tool twice with the same arguments; the answer will not \
change. Prefer a few well-chosen calls over many.
- A tool that cannot answer replies with a sentence explaining why, often naming \
what is nearby. Read it and adjust rather than repeating the call.
- Every figure you quote must come from a tool result. Never invent one, and \
never fill a gap with a plausible number.\
"""

_COACHING_STYLE_PROMPTS: dict[str, str] = {
    "stern": "Be strict, demanding, and blunt. Hold the athlete to the highest standards and do not sugarcoat shortcomings.",
    "friendly": "Use a warm, conversational, and supportive tone. Be honest but always kind.",
    "encouraging": "Lead with positives. Celebrate wins, frame feedback constructively, and focus on building motivation and confidence.",
}


def _local_now(tz_str: str | None) -> datetime:
    """Kept as the historical import site; the rules live in one shared place.

    ``llm_goal_guidance`` and ``api/athlete`` already import this name, so it
    stays — but it now delegates, so the UTC fallback and the set of exceptions
    swallowed are decided once (``backend.app.core.timezones``) rather than
    drifting per consumer.
    """
    return local_now(tz_str)


def _decorate(prompt: str, locale: str | None, coaching_style: str | None) -> str:
    """Append the experience guidance, the house coaching style and the language.

    Shared by both paths so the agentic answer sounds like the single-shot one:
    the same experience calibration, the same tone the athlete chose, the same
    language. The ``MOOD:`` token stays English in every locale by design.
    """
    prompt += f"\n\n{EXPERIENCE_GUIDANCE}"
    if coaching_style and coaching_style in _COACHING_STYLE_PROMPTS:
        prompt += f"\n\n{_COACHING_STYLE_PROMPTS[coaching_style]}"
    if locale:
        lang = _LOCALE_LANGUAGE.get(locale.split("-")[0].lower())
        if lang:
            prompt += f" Respond in {lang}."
    return prompt


def _build_system_prompt(locale: str | None = None, coaching_style: str | None = None) -> str:
    return _decorate(_SYSTEM_PROMPT_BASE, locale, coaching_style)


def _build_agentic_system_prompt(
    locale: str | None = None, coaching_style: str | None = None
) -> str:
    """The system prompt for the tool-driven path (issue #43).

    The coaching rules and the format contract are byte-for-byte the ones the
    blob path uses — only the middle changes, from "here is everything" to "here
    is how to go and get it". Keeping the rules identical is what makes an
    ``llm-eval`` head-to-head between the two paths mean anything.
    """
    return _decorate(
        f"{_STATUS_GUIDANCE}\n\n{_TOOL_GUIDANCE}\n\n{_MOOD_RULE}", locale, coaching_style
    )


def _build_agentic_user_prompt(now: datetime) -> str:
    """The brief. Everything else the model wants, it asks for.

    Today's date is here rather than in a tool because every judgement in the
    answer is relative to it — "not due yet" versus "missed" turns on it — and a
    model that has to spend a round trip discovering what day it is will
    sometimes not bother.
    """
    today = now.date()
    tz_label = now.strftime("%Z") or "UTC"
    return (
        f"Today is {today.isoformat()} ({today.strftime('%A')}), "
        f"{now.strftime('%H:%M')} {tz_label} in the athlete's own timezone.\n\n"
        "Give this athlete their daily training-status feedback."
    )


_SHAPE_TEXT = {
    "polarized": "polarized — mostly easy, with the hard work genuinely hard",
    "pyramidal": "pyramidal — easy base, less tempo/threshold, least hard work",
    "threshold": "threshold-heavy — a large share of moderate, grey-zone work",
    "predominantly_low": "almost entirely low intensity, with very little above LT1",
}


def _distribution_lines(distribution: IntensityDistributionResponse | None) -> list[str]:
    """Render the block's intensity distribution for the prompt.

    Gives the coach the shape of the last twelve weeks, so it can say "you've
    spent eight weeks in a threshold distribution — that's why your form is
    flat" instead of reasoning from daily load alone. The method and coverage
    travel with the numbers: a distribution without its method stated is
    meaningless, and one drawn from a handful of rides shouldn't be argued from.
    """
    if distribution is None or distribution.classification is None:
        return []

    bands = {b.band: b for b in distribution.bands}
    lines = ["\nIntensity distribution (last 12 weeks, by time in zones):"]
    lines.append(f"  Shape: {_SHAPE_TEXT.get(distribution.classification, distribution.classification)}")
    lines.append(f"  Below LT1 (easy): {bands[1].pct:.0f}%")
    lines.append(f"  LT1–LT2 (tempo/threshold): {bands[2].pct:.0f}%")
    lines.append(f"  Above LT2 (hard): {bands[3].pct:.0f}%")

    coverage = distribution.coverage
    if coverage.activities_used < coverage.activities_total:
        lines.append(
            f"  Based on {coverage.activities_used} of {coverage.activities_total} "
            "rides — the rest had no usable zone data."
        )
    if distribution.zone_definitions_changed:
        lines.append(
            "  Note: the athlete's zones or FTP changed inside this window, so "
            "the band boundaries are not consistent across the whole period."
        )
    return lines


def _build_status_prompt(
    athlete: Athlete,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plans: list[tuple[TrainingPlan, list[PlannedWorkout]]],
    active_goals: list[Goal],
    now: datetime,
    distribution: IntensityDistributionResponse | None = None,
) -> str:
    today = now.date()
    tz_label = now.strftime("%Z") or "UTC"
    day_name = today.strftime("%A")
    lines = [f"Training status report — {today.isoformat()} ({day_name})  {now.strftime('%H:%M')} {tz_label}"]

    if athlete.ftp:
        lines.append(f"Athlete FTP: {athlete.ftp} W")
    if athlete.max_hr:
        lines.append(f"Athlete max HR: {athlete.max_hr} bpm")
    level = experience_level(athlete.app_settings)
    if level:
        lines.append(f"Athlete self-reported experience level: {level}")

    if current_metric:
        lines.append("\nCurrent fitness state:")
        lines.append(f"  Fitness: {current_metric.fitness:.1f}")
        lines.append(f"  Fatigue: {current_metric.fatigue:.1f}")
        lines.append(
            f"  Form: {current_metric.form:.1f} ({_form_to_label(current_metric.form)})"
        )

    lines.extend(_distribution_lines(distribution))

    lines.append("\nLast 28 days of training:")
    if recent_activities:
        for act in recent_activities:
            act_date = act.start_time.date() if act.start_time else "?"
            mins = int((act.duration_s or 0) // 60)
            load = f"{act.load:.0f} Load" if act.load else "no Load"
            lines.append(
                f"  {act_date}  {act.sport_type or 'unknown'}  {mins}min  {load}"
            )
    else:
        lines.append("  (no activities recorded)")

    for plan, this_week_workouts in active_plans:
        # Ended-but-not-archived plans are no longer relevant to today's status.
        if plan.end_date is not None and today > plan.end_date:
            continue
        # Upcoming plans (start in the future) are noted for context only, with no
        # "current week" or this-week workouts.
        if plan.start_date is not None and plan.start_date > today:
            lines.append(f"\nUpcoming training plan: {plan.name}")
            lines.append(
                f"  Period: {plan.start_date} → {plan.end_date or 'open-ended'}"
            )
            continue
        plan_start = plan.start_date or today
        week_num = max(1, (today - plan_start).days // 7 + 1)
        lines.append(f"\nActive training plan: {plan.name}")
        lines.append(
            f"  Period: {plan.start_date} → {plan.end_date or 'open-ended'}"
        )
        lines.append(f"  Current week: {week_num}")
        if this_week_workouts:
            # Start of the current plan week (a rolling 7-day block from plan_start).
            week_start = plan_start + timedelta(days=(week_num - 1) * 7)
            lines.append("  This week's planned workouts:")
            for w in this_week_workouts:
                # Map day_of_week (1=Mon..7=Sun, isoweekday convention) to the actual
                # calendar date within this plan week, so the weekday label is explicit
                # and unambiguous in the athlete's local timezone.
                workout_date = next(
                    (
                        week_start + timedelta(days=offset)
                        for offset in range(7)
                        if (week_start + timedelta(days=offset)).isoweekday()
                        == w.day_of_week
                    ),
                    week_start,
                )
                weekday_name = workout_date.strftime("%A")
                upcoming = workout_date > today
                if workout_date == today:
                    today_marker = " (today)"
                elif upcoming:
                    today_marker = " (upcoming)"
                else:
                    today_marker = ""
                # Rest days are intentional and have nothing to perform, so they
                # carry no completed/skipped status — otherwise "not completed"
                # reads as a missed session to the model.
                if (w.workout_type or "").strip().lower() == "rest":
                    lines.append(
                        f"    {weekday_name} {workout_date.isoformat()}{today_marker}: "
                        f"rest day — nothing to complete, no action required"
                    )
                    continue
                is_completed = w.is_completed
                if is_completed:
                    completed = "completed"
                elif upcoming:
                    # Future sessions are not due yet; wording it as "not
                    # completed" reads as a missed session to the model.
                    completed = "upcoming — not due yet"
                else:
                    completed = "not completed"
                # When a workout was completed by several activities (for example a
                # ride recorded in two parts), report the combined actual so the
                # coach sees the aggregate that met the goal.
                if is_completed:
                    n = len(w.linked_activities)
                    total_load = sum(a.load or 0 for a in w.linked_activities)
                    total_min = round(
                        sum(a.duration_s or 0 for a in w.linked_activities) / 60
                    )
                    if n > 1:
                        completed = (
                            f"completed across {n} activities "
                            f"(combined {round(total_load)} Load, {total_min} min)"
                        )
                tss_str = f", target Load {w.target_load}" if w.target_load else ""
                skip_str = (
                    f" (skipped — reason: {w.skip_reason.strip()})"
                    if not is_completed and w.skip_reason and w.skip_reason.strip()
                    else ""
                )
                lines.append(
                    f"    {weekday_name} {workout_date.isoformat()}{today_marker}: "
                    f"{w.workout_type or 'workout'}{tss_str} — {completed}{skip_str}"
                )
        else:
            lines.append("  No workouts planned for this week")

    if active_goals:
        lines.append("\nActive goals:")
        for g in active_goals:
            target_str = f", target {g.target_value}" if g.target_value is not None else ""
            current_str = f", current {g.current_value}" if g.current_value is not None else ""
            date_str = f" (by {g.target_date})" if g.target_date else ""
            lines.append(f"  {g.title}{date_str}: {g.status}{target_str}{current_str}")

    return "\n".join(lines)


def _stream_status_analysis(
    athlete: Athlete,
    user_id: str,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plans: list[tuple[TrainingPlan, list[PlannedWorkout]]],
    active_goals: list[Goal],
    now: datetime,
    locale: str | None = None,
    coaching_style: str | None = None,
    usage_out: dict | None = None,
    distribution: IntensityDistributionResponse | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE."""
    return stream_chat_completion(
        athlete,
        user_id,
        system_prompt=_build_system_prompt(locale, coaching_style),
        user_prompt=_build_status_prompt(
            athlete, recent_activities, current_metric, active_plans,
            active_goals, now, distribution,
        ),
        usage_out=usage_out,
    )


async def analyze_training_status_bg(
    athlete_id: str,
    user_id: str,
    locale: str | None = None,
    *,
    allow_agentic: bool = True,
    run_id: str | None = None,
) -> None:
    """
    Background task: stream LLM training status → write chunks to DB every 500 ms
    → set final training_status_status to 'done' or 'error'.

    ``allow_agentic=False`` forces the single-shot blob prompt regardless of what
    the athlete opted into (issue #43). Used by the provider-sync paths: a
    backlog import fires one of these per athlete alongside a task per imported
    activity, and multiplying that by an agent loop's four-to-six calls is a real
    bill on the one path nobody is watching the output of.

    ``run_id`` is the token this run owns ``training_status*`` by (issue #50):
    a row settled or re-triggered under it clears the token, and this run then
    discards its own writes rather than committing over its replacement.
    ``None`` keeps the old behaviour.
    """

    async def _clear_pending(recovery_session) -> None:
        result = await recovery_session.execute(
            select(Athlete).where(Athlete.id == athlete_id)
        )
        stuck = result.scalar_one_or_none()
        if stuck is not None and settle_training_status(stuck):
            # Stamping the date stops the auto-refresh re-firing the run that
            # just failed on the very next read.
            stuck.training_status_date = datetime.now(timezone.utc).date()

    async with failure_recovery(
        user_id, f"Training status analysis for athlete {athlete_id}", _clear_pending
    ):
        async with get_user_session_factory(user_id)() as session:
            athlete_result = await session.execute(
                select(Athlete).where(Athlete.id == athlete_id)
            )
            athlete = athlete_result.scalar_one()

            app_cfg = athlete.app_settings or {}
            resolved_locale = locale or app_cfg.get("locale")
            coaching_style = app_cfg.get("coaching_style")
            now = _local_now(app_cfg.get("timezone"))
            today = now.date()
            window_start = today - timedelta(days=28)

            # Last 28 days of activities
            acts_result = await session.execute(
                select(Activity)
                .where(
                    Activity.athlete_id == athlete_id,
                    Activity.start_time >= datetime(
                        window_start.year, window_start.month, window_start.day,
                        tzinfo=timezone.utc,
                    ),
                )
                .order_by(Activity.start_time.asc())
            )
            recent_activities = list(acts_result.scalars().all())

            # Latest DailyMetric
            metric_result = await session.execute(
                select(DailyMetric)
                .where(DailyMetric.athlete_id == athlete_id)
                .order_by(DailyMetric.date.desc())
                .limit(1)
            )
            current_metric = metric_result.scalar_one_or_none()

            # Active training plans (issue #45): the app allows several
            # non-overlapping active plans to coexist, so consider all of them
            # rather than just the most recently created one. Each current plan
            # gets its own week's planned workouts; upcoming/ended plans are
            # passed through with an empty list and classified by the prompt
            # builder.
            plans_result = await session.execute(
                select(TrainingPlan)
                .where(
                    TrainingPlan.athlete_id == athlete_id,
                    TrainingPlan.status == "active",
                )
                .order_by(TrainingPlan.start_date.asc().nullsfirst())
            )
            active_plans: list[tuple[TrainingPlan, list[PlannedWorkout]]] = []
            for plan in plans_result.scalars().all():
                workouts: list[PlannedWorkout] = []
                # Only current plans (started and not yet ended) contribute this
                # week's workouts; upcoming/ended plans don't have a "this week".
                covers_today = (
                    plan.start_date is not None
                    and plan.start_date <= today
                    and (plan.end_date is None or today <= plan.end_date)
                )
                if covers_today:
                    current_week = max(1, (today - plan.start_date).days // 7 + 1)
                    pw_result = await session.execute(
                        select(PlannedWorkout)
                        .where(
                            PlannedWorkout.plan_id == plan.id,
                            PlannedWorkout.week_number == current_week,
                        )
                        .order_by(PlannedWorkout.day_of_week)
                    )
                    workouts = list(pw_result.scalars().all())
                active_plans.append((plan, workouts))

            # Active goals
            goals_result = await session.execute(
                select(Goal)
                .where(
                    Goal.athlete_id == athlete_id,
                    Goal.status == "active",
                )
                .order_by(Goal.target_date.asc().nullslast())
            )
            active_goals = list(goals_result.scalars().all())

            # Intensity distribution over the last block (issue #38). Uses the
            # default time-in-zone method so the coach's numbers match the
            # chart the athlete is looking at.
            distribution = await compute_intensity_distribution(
                athlete,
                session,
                start=today - timedelta(days=DEFAULT_WINDOW_DAYS),
                end=today,
            )

            def _touch() -> None:
                # `PENDING_TIMEOUT_MINUTES` compares against this column, and it
                # was calibrated when the path made exactly one completion. An
                # agentic run makes up to seven, and `_STREAM_TIMEOUT` is a
                # *read* timeout — between chunks — so it bounds no turn's total
                # duration. A slow local model can therefore cross the budget
                # while perfectly healthy. Touching the timestamp on every
                # progress commit turns it into "no progress for N minutes"
                # rather than "started N minutes ago", which is what it should
                # always have meant. Free: `stream_into_db` commits right after
                # every callback anyway.
                athlete.training_status_updated_at = datetime.now(timezone.utc)

            def _set_status(text: str) -> None:
                athlete.training_status = text
                # Text is progress too (issue #91). Touching the clock only on
                # agent *steps* left the blob path — and every provider that
                # can't do tool calling — with the original shape: one
                # completion streaming steadily past the budget would be
                # declared dead underneath itself, the card would flip to
                # `error` while the run was healthy, and `_finish` would then
                # overwrite that error with `done`.
                _touch()

            def _set_step(code: str | None) -> None:
                athlete.training_status_progress = code
                _touch()

            def _finish(text: str) -> None:
                athlete.training_status = text
                athlete.training_status_status = "done"
                # The card must end up looking exactly as it did before the
                # agentic path existed; a leftover step line under a finished
                # answer would be the tell that it doesn't.
                athlete.training_status_progress = None
                athlete.training_status_date = today
                athlete.training_status_updated_at = datetime.now(timezone.utc)

            def _fail() -> None:
                athlete.training_status_status = "error"
                athlete.training_status_progress = None
                athlete.training_status_date = today
                athlete.training_status_updated_at = datetime.now(timezone.utc)

            request = (
                AgentRequest(
                    athlete=athlete,
                    user_id=user_id,
                    system_prompt=_build_agentic_system_prompt(
                        resolved_locale, coaching_style
                    ),
                    user_prompt=_build_agentic_user_prompt(now),
                    feature="training_status",
                    max_rounds=MAX_ROUNDS_STATUS,
                    format_rule=_MOOD_RULE,
                    today=today,
                )
                if allow_agentic and agentic_enabled(athlete)
                else None
            )

            await stream_into_db(
                session,
                lambda usage_out: coaching_stream(
                    request=request,
                    blob=lambda blob_usage: _stream_status_analysis(
                        athlete, user_id,
                        recent_activities, current_metric,
                        active_plans, active_goals,
                        now, locale=resolved_locale, coaching_style=coaching_style,
                        usage_out=blob_usage, distribution=distribution,
                    ),
                    usage_out=usage_out,
                ),
                on_progress=_set_status,
                on_done=_finish,
                on_error=_fail,
                on_step=_set_step,
                user_id=user_id,
                feature="training_status",
                label=f"Training status analysis for athlete {athlete_id}",
            )

            # `stream_into_db` has committed by now, so a run that lost its
            # claim takes its own writes back out; the callbacks cannot, being
            # synchronous and unable to see another session's commit. `training_status_date` goes too, so the next read
            # regenerates rather than showing a hole.
            if not await run_is_current(
                session, Athlete, athlete_id, Athlete.training_status_run_id, run_id
            ):
                # Only when **nobody** owns the row. The check above fails for two
                # opposite reasons: the row was settled (token cleared), so these
                # writes are unwanted; or it was re-triggered and a live run holds
                # the token and is writing these very columns. Clearing in the
                # second case would destroy that run's work.
                cleared = await session.execute(
                    update(Athlete)
                    .where(
                        Athlete.id == athlete_id,
                        Athlete.training_status_run_id.is_(None),
                    )
                    .values(
                        training_status=None,
                        training_status_status=None,
                        training_status_progress=None,
                        training_status_date=None,
                        training_status_updated_at=None,
                    )
                )
                await session.commit()
                if cleared.rowcount:
                    log.info(
                        "Discarded a superseded training status for athlete %s",
                        athlete_id,
                    )
                else:
                    log.info(
                        "Training status for athlete %s was superseded by a live "
                        "run — leaving that run's columns alone",
                        athlete_id,
                    )
