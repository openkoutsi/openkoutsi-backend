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
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import AsyncIterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from sqlalchemy import select

from ..core.ssrf import check_url_safe
from ..db.registry import _RegistrySessionLocal
from ..db.user_session import get_user_session_factory
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Activity, Athlete, DailyMetric, Goal, PlannedWorkout, TrainingPlan
from ..schemas.metrics import _form_to_label
from .athlete_experience import EXPERIENCE_GUIDANCE, experience_level
from .llm_access import record_llm_usage, usage_from_sse_data
from .llm_client import (
    apply_body_extras,
    merge_llm_headers,
    raise_for_llm_status,
    resolve_llm_config,
    temperature_param,
)

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

_SYSTEM_PROMPT_BASE = """\
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
- Only workouts from previous days count as missed. If past days show incomplete sessions, \
be direct and stern about it.
- When an incomplete workout has a skip reason attached, take it into account. A legitimate \
reason (illness, injury, travel, rest) should temper your criticism, while a pattern of weak \
excuses warrants a firmer response.
- Rest days are a planned, intentional part of the training plan. They do not have to be \
performed and there is nothing to complete on them. Never treat a rest day as missed, skipped, \
or a sign of poor adherence — an athlete taking their scheduled rest is following the plan correctly.

Before the feedback paragraphs, output a single line in the format: MOOD:<mood>
where <mood> is one of: cheer, knowing, neutral, stern.
- cheer: athlete is training well and making great progress
- stern: athlete is missing planned sessions, undertraining, or not following their plan
- neutral: routine week with no strong positive or negative takeaway
- knowing: all other cases (default)
The MOOD line must be the very first line, followed by a blank line, then the paragraphs.\
"""

_COACHING_STYLE_PROMPTS: dict[str, str] = {
    "stern": "Be strict, demanding, and blunt. Hold the athlete to the highest standards and do not sugarcoat shortcomings.",
    "friendly": "Use a warm, conversational, and supportive tone. Be honest but always kind.",
    "encouraging": "Lead with positives. Celebrate wins, frame feedback constructively, and focus on building motivation and confidence.",
}


def _local_now(tz_str: str | None) -> datetime:
    if tz_str:
        try:
            return datetime.now(ZoneInfo(tz_str))
        except ZoneInfoNotFoundError:
            pass
    return datetime.now(timezone.utc)


def _build_system_prompt(locale: str | None = None, coaching_style: str | None = None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    prompt += f"\n\n{EXPERIENCE_GUIDANCE}"
    if coaching_style and coaching_style in _COACHING_STYLE_PROMPTS:
        prompt += f"\n\n{_COACHING_STYLE_PROMPTS[coaching_style]}"
    if locale:
        lang = _LOCALE_LANGUAGE.get(locale.split("-")[0].lower())
        if lang:
            prompt += f" Respond in {lang}."
    return prompt


def _build_status_prompt(
    athlete: Athlete,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plans: list[tuple[TrainingPlan, list[PlannedWorkout]]],
    active_goals: list[Goal],
    now: datetime,
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
                today_marker = " (today)" if workout_date == today else ""
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
                completed = "completed" if is_completed else "not completed"
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


async def _stream_status_analysis(
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
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE.

    When ``usage_out`` is provided it is populated with ``{"cfg", "usage"}`` so
    the caller can record the instance-paid token usage (issue #9).
    """
    instance: InstanceSettings | None = None
    async with _RegistrySessionLocal() as reg:
        result = await reg.execute(select(InstanceSettings).limit(1))
        instance = result.scalar_one_or_none()

    # Resolve the athlete's effective LLM config, exactly like the chat proxy:
    # their own BYOK server if configured, else their selected instance preset
    # (``app_settings["llm_model"]``), else the instance default (first preset).
    # BYOK calls resolve to ``source == "user"`` and are skipped by usage
    # recording (the hoster pays nothing for them).
    cfg = resolve_llm_config(athlete, instance, user_id)
    if usage_out is not None:
        usage_out["cfg"] = cfg

    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    headers = merge_llm_headers(headers, cfg.extra_headers)

    prompt = _build_status_prompt(
        athlete, recent_activities, current_metric, active_plans,
        active_goals, now,
    )
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(locale, coaching_style)},
        {"role": "user", "content": prompt},
    ]
    analysis_context = getattr(instance, "llm_analysis_context", None)
    if analysis_context and analysis_context.strip():
        messages.insert(1, {"role": "system", "content": analysis_context.strip()})

    def _payload(include_usage: bool) -> dict:
        base: dict = {
            "model": cfg.model,
            "messages": messages,
            **temperature_param(),
            "stream": True,
        }
        if include_usage:
            base["stream_options"] = {"include_usage": True}
        return apply_body_extras(base, cfg.extra_body)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0)
    ) as client:
        # Ask for a trailing usage chunk; retry once without it if the upstream
        # rejects stream_options (Ollama-family tolerance).
        cm = client.stream("POST", url, json=_payload(True), headers=headers)
        resp = await cm.__aenter__()
        if getattr(resp, "is_error", False):
            await cm.__aexit__(None, None, None)
            cm = client.stream("POST", url, json=_payload(False), headers=headers)
            resp = await cm.__aenter__()
        try:
            await raise_for_llm_status(resp, url)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                if usage_out is not None:
                    usage = usage_from_sse_data(data)
                    if usage is not None:
                        usage_out["usage"] = usage
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        finally:
            await cm.__aexit__(None, None, None)


async def analyze_training_status_bg(
    athlete_id: str,
    user_id: str,
    locale: str | None = None,
) -> None:
    """
    Background task: stream LLM training status → write chunks to DB every 500 ms
    → set final training_status_status to 'done' or 'error'.
    """
    try:
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

            buffer: list[str] = []
            last_flush = time.monotonic()
            accumulated = ""
            usage_out: dict = {}
            started = time.monotonic()

            try:
                async for chunk in _stream_status_analysis(
                    athlete, user_id,
                    recent_activities, current_metric,
                    active_plans, active_goals,
                    now, locale=resolved_locale, coaching_style=coaching_style,
                    usage_out=usage_out,
                ):
                    buffer.append(chunk)
                    if time.monotonic() - last_flush >= 0.5:
                        accumulated += "".join(buffer)
                        buffer.clear()
                        last_flush = time.monotonic()
                        athlete.training_status = accumulated
                        await session.commit()

                accumulated += "".join(buffer)
                athlete.training_status = accumulated
                athlete.training_status_status = "done"
                athlete.training_status_date = today
                athlete.training_status_updated_at = datetime.now(timezone.utc)
                await session.commit()
                log.info("Training status analysis complete for athlete %s", athlete_id)

            except Exception:
                log.exception("Training status analysis failed for athlete %s", athlete_id)
                athlete.training_status_status = "error"
                athlete.training_status_date = today
                athlete.training_status_updated_at = datetime.now(timezone.utc)
                await session.commit()
            finally:
                # Record instance-paid token usage (issue #9). Fire-and-forget.
                cfg = usage_out.get("cfg")
                if cfg is not None:
                    await record_llm_usage(
                        user_id=user_id,
                        feature="training_status",
                        cfg=cfg,
                        usage=usage_out.get("usage"),
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )

    except Exception:
        # Session acquisition or early DB query failed — open a fresh session to
        # clear the pending state so the user can retry.
        log.exception(
            "Training status background task failed outside inner try for athlete %s",
            athlete_id,
        )
        try:
            async with get_user_session_factory(user_id)() as recovery_session:
                result = await recovery_session.execute(
                    select(Athlete).where(Athlete.id == athlete_id)
                )
                athlete = result.scalar_one_or_none()
                if athlete:
                    athlete.training_status_status = "error"
                    athlete.training_status_updated_at = datetime.now(timezone.utc)
                    athlete.training_status_date = datetime.now(timezone.utc).date()
                    await recovery_session.commit()
        except Exception:
            log.exception(
                "Recovery session also failed for athlete %s — status may remain stuck",
                athlete_id,
            )
