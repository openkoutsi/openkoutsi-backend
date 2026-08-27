"""
LLM-based per-goal guidance service.

Judges how realistic a single goal is for its timeline given the athlete's
current fitness and trend, and streams concrete coaching prose on how to reach
it from any OpenAI-compatible chat completions API. The prose is persisted
incrementally so local models that take several minutes never time out.

Structured like ``llm_training_status_analyzer`` — same LLM configuration
(the instance's configured presets, first entry = default), same streaming and
usage-recording plumbing, and the same "Koutsi" coach voice with a leading
machine-readable tag line (``REALISM:`` here, mirroring the ``MOOD:`` convention).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from sqlalchemy import select, update

from ..db.user_session import get_user_session_factory
from ..models.user_orm import Activity, Athlete, DailyMetric, Goal, TrainingPlan
from ..schemas.metrics import _form_to_label
from .athlete_experience import EXPERIENCE_GUIDANCE, experience_level
from .llm_streaming import (
    failure_recovery,
    stream_chat_completion,
    stream_into_db,
)
from .llm_training_status_analyzer import (
    _COACHING_STYLE_PROMPTS,
    _LOCALE_LANGUAGE,
    _local_now,
)
from .stranded_runs import run_is_current, settle_goal_guidance

log = logging.getLogger(__name__)

# The model must lead with one of three verdict tokens — fixed English even when
# the prose is localized (same rule as the MOOD line): realistic / ambitious /
# unrealistic.
_REALISM_RE = re.compile(
    r"^REALISM:\s?(realistic|ambitious|unrealistic)\s*$", re.IGNORECASE
)
# When the model omits or mangles the tag, fall back to the cautious middle.
_FALLBACK_VERDICT = "ambitious"

_SYSTEM_PROMPT_BASE = """\
You are Koutsi, an expert endurance sports coach. The athlete has set a single \
training/event goal. Assess whether the goal is realistic for its timeline given \
the athlete's current fitness and recent trend, then give concrete, actionable \
guidance on how to reach it. Write 2-4 paragraphs of plain prose — no markdown \
headers, no bullet points, no code blocks. Separate each paragraph with a single \
blank line. Ground your judgement in the athlete's numbers (FTP, Fitness/Fatigue/Form, \
recent training volume) and the days remaining until the target date; be honest \
when a goal is over-aggressive, but always give a realistic path forward.

Before the guidance paragraphs, output a single line in the format: REALISM:<verdict>
where <verdict> is one of: realistic, ambitious, unrealistic.
- realistic: the goal is well within reach on the current timeline and trajectory
- ambitious: the goal is a real stretch but achievable with focused, consistent work
- unrealistic: the timeline or target is not attainable without change (extend the \
date, adjust the target, or dramatically increase training)
The REALISM line must be the very first line, followed by a blank line, then the \
paragraphs. Keep the REALISM token in English even when the guidance is written in \
another language.\
"""


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


def _build_goal_prompt(
    athlete: Athlete,
    goal: Goal,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plan: TrainingPlan | None,
    now: datetime,
) -> str:
    today = now.date()
    tz_label = now.strftime("%Z") or "UTC"
    lines = [f"Goal guidance request — {today.isoformat()} ({today.strftime('%A')})  {now.strftime('%H:%M')} {tz_label}"]

    lines.append("\nGoal:")
    lines.append(f"  Title: {goal.title}")
    if goal.description and goal.description.strip():
        lines.append(f"  Description: {goal.description.strip()}")
    if goal.metric:
        lines.append(f"  Metric: {goal.metric}")
    if goal.target_value is not None:
        lines.append(f"  Target value: {goal.target_value}")
    if goal.current_value is not None:
        lines.append(f"  Current value: {goal.current_value}")
    if goal.target_date:
        days_remaining = (goal.target_date - today).days
        if days_remaining >= 0:
            lines.append(f"  Target date: {goal.target_date.isoformat()} ({days_remaining} days remaining)")
        else:
            lines.append(
                f"  Target date: {goal.target_date.isoformat()} "
                f"({abs(days_remaining)} days ago — already past)"
            )
    else:
        lines.append("  Target date: (none set)")

    lines.append("\nAthlete:")
    if athlete.ftp:
        lines.append(f"  FTP: {athlete.ftp} W")
    if athlete.max_hr:
        lines.append(f"  Max HR: {athlete.max_hr} bpm")
    level = experience_level(athlete.app_settings)
    if level:
        lines.append(f"  Self-reported experience level: {level}")

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

    if active_plan:
        lines.append(f"\nActive training plan: {active_plan.name}")
        lines.append(
            f"  Period: {active_plan.start_date} → {active_plan.end_date or 'open-ended'}"
        )
    else:
        lines.append("\nNo active training plan.")

    return "\n".join(lines)


def _parse_verdict(text: str) -> tuple[str, str]:
    """Split a leading ``REALISM:<verdict>`` line off the streamed prose.

    Returns ``(verdict, prose)``. When the first non-empty line isn't a valid
    REALISM tag, the fallback verdict is used and the prose is returned intact —
    the coaching text is still worth showing.
    """
    lines = text.splitlines()
    # Skip any leading blank lines the model may emit before the tag.
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines):
        match = _REALISM_RE.match(lines[idx].strip())
        if match:
            rest = lines[idx + 1:]
            if rest and rest[0].strip() == "":
                rest = rest[1:]
            return match.group(1).lower(), "\n".join(rest).strip()
    return _FALLBACK_VERDICT, text.strip()


def _stream_display_prose(text: str) -> str:
    """Tag-free prose to persist mid-stream, so a poller never sees the raw tag.

    The leading ``REALISM:`` line arrives one token at a time; showing the
    partially-formed tag would flicker ``REAL…`` at the top of the card. We hold
    back until the first line is terminated by a newline, then strip a recognised
    tag via :func:`_parse_verdict` (which returns the text unchanged if the first
    line turns out to be ordinary prose). This keeps the persisted ``guidance``
    tag-free in both the ``pending`` and ``done`` states.
    """
    if "\n" not in text:
        return ""
    _, prose = _parse_verdict(text)
    return prose


def _stream_goal_guidance(
    athlete: Athlete,
    user_id: str,
    goal: Goal,
    recent_activities: list[Activity],
    current_metric: DailyMetric | None,
    active_plan: TrainingPlan | None,
    now: datetime,
    locale: str | None = None,
    coaching_style: str | None = None,
    usage_out: dict | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE."""
    return stream_chat_completion(
        athlete,
        user_id,
        system_prompt=_build_system_prompt(locale, coaching_style),
        user_prompt=_build_goal_prompt(
            athlete, goal, recent_activities, current_metric, active_plan, now
        ),
        usage_out=usage_out,
    )


async def generate_goal_guidance_bg(
    athlete_id: str,
    goal_id: str,
    user_id: str,
    locale: str | None = None,
    run_id: str | None = None,
) -> None:
    """
    Background task: stream per-goal LLM guidance → write prose to DB every 500 ms
    → parse the leading REALISM verdict → set final guidance_status 'done'/'error'.

    ``run_id`` is the token this run owns ``guidance*`` by (issue #50): a row
    settled or re-triggered under it clears the token, and this run then
    discards its own writes rather than committing over its replacement.
    ``None`` keeps the old behaviour.
    """

    async def _clear_pending(recovery_session) -> None:
        result = await recovery_session.execute(select(Goal).where(Goal.id == goal_id))
        stuck = result.scalar_one_or_none()
        if stuck is not None:
            settle_goal_guidance(stuck)

    async with failure_recovery(
        user_id, f"Goal guidance for goal {goal_id}", _clear_pending
    ):
        async with get_user_session_factory(user_id)() as session:
            athlete_result = await session.execute(
                select(Athlete).where(Athlete.id == athlete_id)
            )
            athlete = athlete_result.scalar_one()

            goal_result = await session.execute(
                select(Goal).where(
                    Goal.id == goal_id, Goal.athlete_id == athlete_id
                )
            )
            goal = goal_result.scalar_one()

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

            def _set_prose(text: str) -> None:
                # Persist tag-free prose so a mid-stream poll never returns the
                # raw REALISM: line (see _stream_display_prose).
                goal.guidance = _stream_display_prose(text)
                # And touch the clock the pending timeout reads (issue #91):
                # without this the budget counts from the trigger, so a single
                # completion streaming steadily past it is declared dead while
                # it is still writing — the card flips to `error`, the run keeps
                # spending, and `_finish` then overwrites the error with `done`.
                goal.guidance_updated_at = datetime.now(timezone.utc)

            def _finish(text: str) -> None:
                verdict, prose = _parse_verdict(text)
                goal.guidance = prose
                goal.guidance_verdict = verdict
                goal.guidance_status = "done"
                goal.guidance_updated_at = datetime.now(timezone.utc)

            def _fail() -> None:
                goal.guidance_status = "error"
                goal.guidance_updated_at = datetime.now(timezone.utc)

            await stream_into_db(
                session,
                lambda usage_out: _stream_goal_guidance(
                    athlete, user_id, goal,
                    recent_activities, current_metric, active_plan,
                    now, locale=resolved_locale, coaching_style=coaching_style,
                    usage_out=usage_out,
                ),
                on_progress=_set_prose,
                on_done=_finish,
                on_error=_fail,
                user_id=user_id,
                feature="goal_guidance",
                label=f"Goal guidance for goal {goal_id}",
            )

            # `stream_into_db` has committed by now, so a run that lost its
            # claim takes its own writes back out; the callbacks cannot, being
            # synchronous and unable to see another session's commit.
            if not await run_is_current(
                session, Goal, goal_id, Goal.guidance_run_id, run_id
            ):
                # Only when **nobody** owns the row. The check above fails for two
                # opposite reasons: the row was settled (token cleared), so these
                # writes are unwanted; or it was re-triggered and a live run holds
                # the token and is writing these very columns. Clearing in the
                # second case would destroy that run's work.
                cleared = await session.execute(
                    update(Goal)
                    .where(
                        Goal.id == goal_id,
                        Goal.guidance_run_id.is_(None),
                    )
                    .values(
                        guidance=None,
                        guidance_verdict=None,
                        guidance_status=None,
                        guidance_updated_at=None,
                    )
                )
                await session.commit()
                if cleared.rowcount:
                    log.info("Discarded a superseded guidance for goal %s", goal_id)
                else:
                    log.info(
                        "Guidance for goal %s was superseded by a live run — "
                        "leaving that run's columns alone",
                        goal_id,
                    )
