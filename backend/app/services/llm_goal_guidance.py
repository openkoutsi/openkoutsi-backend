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
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import httpx
from sqlalchemy import select

from ..core.ssrf import check_url_safe
from ..db.registry import _RegistrySessionLocal
from ..db.user_session import get_user_session_factory
from ..models.registry_orm import InstanceSettings
from ..models.user_orm import Activity, Athlete, DailyMetric, Goal, TrainingPlan
from ..schemas.metrics import _tsb_to_form
from .llm_access import record_llm_usage, usage_from_sse_data
from .llm_client import (
    apply_body_extras,
    merge_llm_headers,
    raise_for_llm_status,
    resolve_llm_config,
    temperature_param,
)
from .llm_training_status_analyzer import (
    _COACHING_STYLE_PROMPTS,
    _LOCALE_LANGUAGE,
    _local_now,
)

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
blank line. Ground your judgement in the athlete's numbers (FTP, CTL/ATL/TSB, \
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


async def _stream_goal_guidance(
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
    """Yield text chunks from the LLM via streaming SSE.

    When ``usage_out`` is provided it is populated with ``{"cfg", "usage"}`` so
    the caller can record the instance-paid token usage (issue #9).
    """
    instance: InstanceSettings | None = None
    async with _RegistrySessionLocal() as reg:
        result = await reg.execute(select(InstanceSettings).limit(1))
        instance = result.scalar_one_or_none()

    # Resolve the athlete's effective LLM config, exactly like the other
    # analyzers: their own BYOK server if configured, else their selected
    # instance preset, else the instance default (first preset).
    cfg = resolve_llm_config(athlete, instance, user_id)
    if usage_out is not None:
        usage_out["cfg"] = cfg

    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    check_url_safe(url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    headers = merge_llm_headers(headers, cfg.extra_headers)

    prompt = _build_goal_prompt(
        athlete, goal, recent_activities, current_metric, active_plan, now
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


async def generate_goal_guidance_bg(
    athlete_id: str,
    goal_id: str,
    user_id: str,
    locale: str | None = None,
) -> None:
    """
    Background task: stream per-goal LLM guidance → write prose to DB every 500 ms
    → parse the leading REALISM verdict → set final guidance_status 'done'/'error'.
    """
    try:
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

            buffer: list[str] = []
            last_flush = time.monotonic()
            accumulated = ""
            usage_out: dict = {}
            started = time.monotonic()

            try:
                async for chunk in _stream_goal_guidance(
                    athlete, user_id, goal,
                    recent_activities, current_metric, active_plan,
                    now, locale=resolved_locale, coaching_style=coaching_style,
                    usage_out=usage_out,
                ):
                    buffer.append(chunk)
                    if time.monotonic() - last_flush >= 0.5:
                        accumulated += "".join(buffer)
                        buffer.clear()
                        last_flush = time.monotonic()
                        # Persist tag-free prose so a mid-stream poll never
                        # returns the raw REALISM: line (see _stream_display_prose).
                        goal.guidance = _stream_display_prose(accumulated)
                        await session.commit()

                accumulated += "".join(buffer)
                verdict, prose = _parse_verdict(accumulated)
                goal.guidance = prose
                goal.guidance_verdict = verdict
                goal.guidance_status = "done"
                goal.guidance_updated_at = datetime.now(timezone.utc)
                await session.commit()
                log.info("Goal guidance complete for goal %s", goal_id)

            except Exception:
                log.exception("Goal guidance failed for goal %s", goal_id)
                goal.guidance_status = "error"
                goal.guidance_updated_at = datetime.now(timezone.utc)
                await session.commit()
            finally:
                # Record instance-paid token usage (issue #9). Fire-and-forget.
                cfg = usage_out.get("cfg")
                if cfg is not None:
                    await record_llm_usage(
                        user_id=user_id,
                        feature="goal_guidance",
                        cfg=cfg,
                        usage=usage_out.get("usage"),
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )

    except Exception:
        # Session acquisition or early DB query failed — open a fresh session to
        # clear the pending state so the user can retry.
        log.exception(
            "Goal guidance background task failed outside inner try for goal %s",
            goal_id,
        )
        try:
            async with get_user_session_factory(user_id)() as recovery_session:
                result = await recovery_session.execute(
                    select(Goal).where(Goal.id == goal_id)
                )
                goal = result.scalar_one_or_none()
                if goal:
                    goal.guidance_status = "error"
                    goal.guidance_updated_at = datetime.now(timezone.utc)
                    await recovery_session.commit()
        except Exception:
            log.exception(
                "Recovery session also failed for goal %s — status may remain stuck",
                goal_id,
            )
