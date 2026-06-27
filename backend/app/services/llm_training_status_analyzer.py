"""
LLM-based daily training status analysis service.

Streams a coaching overview of the athlete's recent training state from any
OpenAI-compatible chat completions API and persists the result incrementally
so local models that take several minutes never time out.

Uses the same LLM configuration as llm_activity_analyzer:
  LLM_BASE_URL  e.g. "http://localhost:11434/v1"
  LLM_API_KEY   empty string is fine for local models
  LLM_MODEL     e.g. "llama3.2", "gpt-4o-mini"
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

from ..core.config import settings
from ..core.ssrf import check_url_safe
from ..db.registry import _RegistrySessionLocal
from ..db.team_session import get_team_session_factory
from ..models.registry_orm import Team
from ..models.team_orm import Activity, Athlete, DailyMetric, Goal, PlannedWorkout, TrainingPlan
from ..schemas.metrics import _tsb_to_form

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
adherence to the active training plan (if any), and 1-2 specific recommendations \
for the coming days. Write in plain prose — no markdown headers, no bullet points, \
no code blocks. Separate each paragraph with a single blank line.

When evaluating training plan adherence, apply these rules:
- Today's planned workouts that are not yet completed must never be treated as missed. \
The athlete still has time to complete them. Either assume they will be done later today, \
or encourage the athlete to get them done — but do not criticise or flag them as missed.
- Only workouts from previous days count as missed. If past days show incomplete sessions, \
be direct and stern about it.

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
    active_plan: TrainingPlan | None,
    this_week_workouts: list[PlannedWorkout],
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

    if current_metric:
        lines.append("\nCurrent fitness state:")
        lines.append(f"  Fitness (CTL): {current_metric.ctl:.1f}")
        lines.append(f"  Fatigue (ATL): {current_metric.atl:.1f}")
        lines.append(
            f"  Form (TSB): {current_metric.tsb:.1f} ({_tsb_to_form(current_metric.tsb)})"
        )

    lines.append("\nLast 28 days of training:")
    if recent_activities:
        for act in recent_activities:
            act_date = act.start_time.date() if act.start_time else "?"
            mins = int((act.duration_s or 0) // 60)
            tss = f"{act.tss:.0f} TSS" if act.tss else "no TSS"
            lines.append(
                f"  {act_date}  {act.sport_type or 'unknown'}  {mins}min  {tss}"
            )
    else:
        lines.append("  (no activities recorded)")

    if active_plan:
        plan_start = active_plan.start_date or today
        week_num = max(1, (today - plan_start).days // 7 + 1)
        lines.append(f"\nActive training plan: {active_plan.name}")
        lines.append(
            f"  Period: {active_plan.start_date} → {active_plan.end_date or 'open-ended'}"
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
                completed = "completed" if w.completed_activity_id else "not completed"
                tss_str = f", target TSS {w.target_tss}" if w.target_tss else ""
                lines.append(
                    f"    {weekday_name} {workout_date.isoformat()}{today_marker}: "
                    f"{w.workout_type or 'workout'}{tss_str} — {completed}"
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
    team_id: str,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plan: TrainingPlan | None,
    this_week_workouts: list[PlannedWorkout],
    active_goals: list[Goal],
    now: datetime,
    locale: str | None = None,
    coaching_style: str | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE."""
    team: Team | None = None
    async with _RegistrySessionLocal() as reg:
        result = await reg.execute(select(Team).where(Team.id == team_id))
        team = result.scalar_one_or_none()

    base_url = (team.llm_base_url.strip() if team and team.llm_base_url else None) or (settings.llm_base_url or "").strip()
    model = (team.llm_model.strip() if team and team.llm_model else None) or (settings.llm_model or "").strip()

    if not base_url or not model:
        raise ValueError("LLM base URL and model must be configured in Settings → AI / LLM")

    url = f"{base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if team and getattr(team, "llm_api_key_enc", None):
        try:
            from backend.app.core.file_encryption import decrypt_team_secret
            api_key = decrypt_team_secret(str(team.llm_api_key_enc), team_id)
            headers["Authorization"] = f"Bearer {api_key}"
        except Exception:
            log.warning("Could not decrypt team LLM API key for team %s", team_id)

    prompt = _build_status_prompt(
        athlete, recent_activities, current_metric, active_plan,
        this_week_workouts, active_goals, now,
    )
    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(locale, coaching_style)},
        {"role": "user", "content": prompt},
    ]
    analysis_context = getattr(team, "llm_analysis_context", None)
    if analysis_context and analysis_context.strip():
        messages.insert(1, {"role": "system", "content": analysis_context.strip()})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0)
    ) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


async def analyze_training_status_bg(
    athlete_id: str,
    team_id: str,
    locale: str | None = None,
) -> None:
    """
    Background task: stream LLM training status → write chunks to DB every 500 ms
    → set final training_status_status to 'done' or 'error'.
    """
    try:
        async with get_team_session_factory(team_id)() as session:
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

            # Active training plan
            plan_result = await session.execute(
                select(TrainingPlan)
                .where(
                    TrainingPlan.athlete_id == athlete_id,
                    TrainingPlan.status == "active",
                )
                .order_by(TrainingPlan.created_at.desc())
                .limit(1)
            )
            active_plan = plan_result.scalar_one_or_none()

            # This week's planned workouts (if plan exists)
            this_week_workouts: list[PlannedWorkout] = []
            if active_plan and active_plan.start_date:
                current_week = max(1, (today - active_plan.start_date).days // 7 + 1)
                pw_result = await session.execute(
                    select(PlannedWorkout)
                    .where(
                        PlannedWorkout.plan_id == active_plan.id,
                        PlannedWorkout.week_number == current_week,
                    )
                    .order_by(PlannedWorkout.day_of_week)
                )
                this_week_workouts = list(pw_result.scalars().all())

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

            try:
                async for chunk in _stream_status_analysis(
                    athlete, team_id,
                    recent_activities, current_metric,
                    active_plan, this_week_workouts, active_goals,
                    now, locale=resolved_locale, coaching_style=coaching_style,
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

    except Exception:
        # Session acquisition or early DB query failed — open a fresh session to
        # clear the pending state so the user can retry.
        log.exception(
            "Training status background task failed outside inner try for athlete %s",
            athlete_id,
        )
        try:
            async with get_team_session_factory(team_id)() as recovery_session:
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
