"""
LLM-based workout analysis service.

Streams a coaching analysis from any OpenAI-compatible chat completions API
and persists the result incrementally to the database so local models that
take several minutes never time out and the frontend can show live progress.

Uses the same LLM configuration as llm_plan_generator:
  LLM_BASE_URL  e.g. "http://localhost:11434/v1"
  LLM_API_KEY   empty string is fine for local models
  LLM_MODEL     e.g. "llama3.2", "gpt-4o-mini"
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator

import httpx
from sqlalchemy import select

from ..core.config import settings
from ..core.ssrf import check_url_safe
from ..db.registry import _RegistrySessionLocal
from ..db.team_session import get_team_session_factory
from ..models.registry_orm import Team
from ..models.team_orm import Activity, Athlete, DailyMetric
from .pr_detection import detect_pr_badges

if TYPE_CHECKING:
    pass

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
You are Koutsi, an expert endurance sports coach. Analyse the following workout data and \
provide actionable coaching feedback in 3-5 paragraphs. Cover: effort quality and pacing, \
power/heart-rate relationship if data is available, the athlete's current fatigue state and \
what it means for recovery, and 1-2 specific recommendations for the athlete's next sessions.
Write in plain prose — no markdown headers, no bullet points, no code blocks.
Separate each paragraph with a single blank line.

Before the feedback paragraphs, output a single line in the format: MOOD:<mood>
where <mood> is one of: cheer, knowing, neutral, stern.
- cheer: great session, personal records set, athlete exceeded expectations
- stern: poor effort, insufficient intensity, or counterproductive session
- neutral: routine session with no strong positive or negative takeaway
- knowing: all other cases (default)
The MOOD line must be the very first line, followed by a blank line, then the paragraphs.\
"""


def _build_system_prompt(locale: str | None = None) -> str:
    prompt = _SYSTEM_PROMPT_BASE
    if locale:
        lang = _LOCALE_LANGUAGE.get(locale.split("-")[0].lower())
        if lang:
            prompt += f" Respond in {lang}."
    return prompt


_WINDOW_LABELS: dict[str, str] = {
    "all_time": "all-time",
    "12mo": "12-month",
    "6mo": "6-month",
    "3mo": "3-month",
}


def _format_duration_label(duration_s: int) -> str:
    if duration_s < 60:
        return f"{duration_s}s"
    mins = duration_s // 60
    if mins < 60:
        return f"{mins}min"
    return f"{mins // 60}h{mins % 60:02d}min" if mins % 60 else f"{mins // 60}h"


def _format_distance_label(distance_m: int) -> str:
    if distance_m < 1000:
        return f"{distance_m}m"
    km = distance_m / 1000
    return f"{km:.0f}km" if km == int(km) else f"{km}km"


def _build_prompt(
    activity: Activity,
    athlete: Athlete,
    fatigue: DailyMetric | None = None,
    power_pr_badges: dict | None = None,
    distance_pr_badges: dict | None = None,
) -> str:
    lines = [f"Workout summary for a {activity.sport_type or 'unknown sport'} session:"]

    if activity.start_time:
        lines.append(f"  Date: {activity.start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    if activity.duration_s:
        mins, secs = divmod(activity.duration_s, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            lines.append(f"  Duration: {hours}h {mins}m {secs}s")
        else:
            lines.append(f"  Duration: {mins}m {secs}s")
    if activity.distance_m:
        lines.append(f"  Distance: {activity.distance_m / 1000:.2f} km")
    if activity.elevation_m:
        lines.append(f"  Elevation gain: {activity.elevation_m:.0f} m")
    if activity.avg_power:
        lines.append(f"  Average power: {activity.avg_power:.0f} W")
    if activity.normalized_power:
        lines.append(f"  Normalized power: {activity.normalized_power:.0f} W")
    if activity.intensity_factor:
        lines.append(f"  Intensity factor: {activity.intensity_factor:.3f}")
    if activity.tss:
        lines.append(f"  Training stress score (TSS): {activity.tss:.1f}")
    if activity.avg_hr:
        lines.append(f"  Average heart rate: {activity.avg_hr:.0f} bpm")
    if activity.max_hr:
        lines.append(f"  Peak heart rate: {activity.max_hr:.0f} bpm")
    if athlete.ftp:
        lines.append(f"  Athlete FTP: {athlete.ftp} W")
    if athlete.max_hr:
        lines.append(f"  Athlete max HR: {athlete.max_hr} bpm")

    if fatigue:
        from ..schemas.metrics import _tsb_to_form
        lines.append("\nAthlete fatigue state prior to this workout:")
        lines.append(f"  Fitness (CTL): {fatigue.ctl:.1f}")
        lines.append(f"  Fatigue (ATL): {fatigue.atl:.1f}")
        lines.append(f"  Form (TSB): {fatigue.tsb:.1f} ({_tsb_to_form(fatigue.tsb)})")

    if activity.intervals:
        lines.append("\nInterval breakdown:")
        for iv in activity.intervals:
            mins, secs = divmod(iv.duration_s, 60)
            line = f"  Interval {iv.interval_number}: {mins}m {secs}s"
            if iv.avg_hr:
                line += f", avg HR {iv.avg_hr:.0f} bpm"
            if iv.avg_power:
                line += f", avg power {iv.avg_power:.0f} W"
            if iv.avg_speed_ms:
                line += f", avg speed {iv.avg_speed_ms * 3.6:.1f} km/h"
            if iv.is_auto_split:
                line += " (auto-split)"
            lines.append(line)

    if getattr(activity, "labels", None):
        lines.append(f"\nActivity labels: {', '.join(activity.labels)}")
    if getattr(activity, "notes", None) and activity.notes.strip():
        lines.append(f"\nAthlete notes: {activity.notes.strip()}")

    pr_lines: list[str] = []
    for duration_s, badges in (power_pr_badges or {}).items():
        label = _format_duration_label(int(duration_s))
        parts = [
            f"{_WINDOW_LABELS.get(w, w)} {tier}"
            for w, tier in badges.items()
            if w in _WINDOW_LABELS
        ]
        if parts:
            pr_lines.append(f"  {label} power: {', '.join(parts)}")
    for distance_m, badges in (distance_pr_badges or {}).items():
        label = _format_distance_label(int(distance_m))
        parts = [
            f"{_WINDOW_LABELS.get(w, w)} {tier}"
            for w, tier in badges.items()
            if w in _WINDOW_LABELS
        ]
        if parts:
            pr_lines.append(f"  {label} distance: {', '.join(parts)}")
    if pr_lines:
        lines.append("\nPersonal Records in this activity:")
        lines.extend(pr_lines)

    return "\n".join(lines)


async def _stream_analysis(
    activity: Activity,
    athlete: Athlete,
    team_id: str,
    fatigue: DailyMetric | None = None,
    locale: str | None = None,
    power_pr_badges: dict | None = None,
    distance_pr_badges: dict | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE."""
    # Fetch team for LLM config
    team: Team | None = None
    async with _RegistrySessionLocal() as reg:
        result = await reg.execute(select(Team).where(Team.id == team_id))
        team = result.scalar_one_or_none()

    # Priority: team settings → global env vars
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
            log.warning("Could not decrypt team LLM API key for team %s — proceeding without auth", team_id)

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(locale)},
    ]
    analysis_context = getattr(team, "llm_analysis_context", None)
    if analysis_context and analysis_context.strip():
        messages.append({"role": "system", "content": analysis_context.strip()})
    messages.append(
        {"role": "user", "content": _build_prompt(activity, athlete, fatigue, power_pr_badges, distance_pr_badges)}
    )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,
    }

    # Local models can take several minutes; use a generous but finite timeout.
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0)  # 5-minute read timeout
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


async def analyze_activity_bg(
    activity_id: str, athlete_id: str, team_id: str, locale: str | None = None
) -> None:
    """
    Background task: stream LLM analysis → write chunks to DB every 500 ms
    → set final analysis_status to 'done' or 'error'.

    Lives in the service layer so it can be imported from both api/activities.py
    and services/strava_sync.py without circular dependencies.
    """
    async with get_team_session_factory(team_id)() as session:
        activity_result = await session.execute(
            select(Activity).where(Activity.id == activity_id)
        )
        activity = activity_result.scalar_one()

        athlete_result = await session.execute(
            select(Athlete).where(Athlete.id == athlete_id)
        )
        athlete = athlete_result.scalar_one()

        # Resolve locale: explicit arg → athlete app_settings → None (defaults to English)
        resolved_locale = locale or (athlete.app_settings or {}).get("locale")

        # Fetch fatigue metrics for the day before the workout
        workout_date = activity.start_time.date() if activity.start_time else None
        fatigue: DailyMetric | None = None
        if workout_date:
            fat_res = await session.execute(
                select(DailyMetric)
                .where(
                    DailyMetric.athlete_id == athlete.id,
                    DailyMetric.date < workout_date,
                )
                .order_by(DailyMetric.date.desc())
                .limit(1)
            )
            fatigue = fat_res.scalar_one_or_none()

        power_pr_badges, distance_pr_badges = await detect_pr_badges(
            athlete.id, activity.id, activity.start_time, activity.sport_type, session
        )

        buffer: list[str] = []
        last_flush = time.monotonic()
        accumulated = ""

        try:
            async for chunk in _stream_analysis(
                activity, athlete, team_id, fatigue=fatigue, locale=resolved_locale,
                power_pr_badges=power_pr_badges, distance_pr_badges=distance_pr_badges,
            ):
                buffer.append(chunk)
                if time.monotonic() - last_flush >= 0.5:
                    accumulated += "".join(buffer)
                    buffer.clear()
                    last_flush = time.monotonic()
                    activity.analysis = accumulated
                    await session.commit()

            # Final flush
            accumulated += "".join(buffer)
            activity.analysis = accumulated
            activity.analysis_status = "done"
            await session.commit()
            log.info("Analysis complete for activity %s", activity_id)

        except Exception:
            log.exception("Analysis failed for activity %s", activity_id)
            activity.analysis_status = "error"
            await session.commit()
