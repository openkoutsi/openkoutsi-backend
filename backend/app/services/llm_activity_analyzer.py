"""
LLM-based workout analysis service.

Streams a coaching analysis from any OpenAI-compatible chat completions API
and persists the result incrementally to the database so local models that
take several minutes never time out and the frontend can show live progress.

Uses the same LLM configuration as llm_plan_generator — the instance's
configured presets (``instance_settings.llm_models``, first entry = default).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, AsyncIterator

from sqlalchemy import select

from ..core.timezones import local_now
from ..db.user_session import get_user_session_factory
from ..models.user_orm import Activity, Athlete, DailyMetric, PlannedWorkout
from .athlete_experience import EXPERIENCE_GUIDANCE, experience_level
from .llm_agent import (
    MAX_ROUNDS_ACTIVITY,
    AgentRequest,
    agentic_enabled,
    coaching_stream,
)
from .llm_streaming import failure_recovery, stream_chat_completion, stream_into_db
from .pr_detection import detect_pr_badges
from .stranded_runs import settle_activity_analysis

from openkoutsi.sport_matching import CYCLING_SPORT_TYPES
from openkoutsi.training_math import efficiency_factor, variability_index

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

_ANALYSIS_GUIDANCE = """\
You are Koutsi, an expert endurance sports coach. Analyse the following workout data and \
provide actionable coaching feedback in 3-5 paragraphs. Cover: effort quality and pacing, \
power/heart-rate relationship if data is available, the athlete's current fatigue state and \
what it means for recovery, and 1-2 specific recommendations for the athlete's next sessions.
If the activity is linked to a planned workout, explicitly assess how well the session matched it \
— intent, intensity and duration — and what any deviation (over- or under-doing it, or a missed \
target) means for the athlete's training.
Write in plain prose — no markdown headers, no bullet points, no code blocks.
Separate each paragraph with a single blank line.\
"""

# The format contract, its own constant so the agent loop can restate it on the
# answering turn (issue #43) — see the note on `_MOOD_RULE` in
# `llm_training_status_analyzer` for why that restatement is not optional.
_MOOD_RULE = """\
Before the feedback paragraphs, output a single line in the format: MOOD:<mood>
where <mood> is one of: cheer, knowing, neutral, stern.
- cheer: great session, personal records set, athlete exceeded expectations
- stern: poor effort, insufficient intensity, or counterproductive session
- neutral: routine session with no strong positive or negative takeaway
- knowing: all other cases (default)
The MOOD line must be the very first line, followed by a blank line, then the paragraphs.\
"""

# Unchanged wording, assembled from the two halves above.
_SYSTEM_PROMPT_BASE = f"{_ANALYSIS_GUIDANCE}\n\n{_MOOD_RULE}"

_SUPPLEMENTAL_GUIDANCE = """\
You are Koutsi, an encouraging cycling coach. The athlete's primary sport is cycling; \
the workout below is a different sport, so treat it as supplemental / cross-training \
rather than their main focus. Do NOT give a detailed coaching breakdown — no pacing, \
power or heart-rate analysis, no multi-paragraph feedback. Instead respond with a \
short (1-2 sentences), warm acknowledgement that recognises the work the athlete put in \
and encourages them to keep it up.
Write in plain prose — no markdown headers, no bullet points, no code blocks.\
"""

_SUPPLEMENTAL_MOOD_RULE = """\
Before the acknowledgement, output a single line in the format: MOOD:<mood>
where <mood> is one of: cheer, knowing, neutral, stern. Use cheer for a strong effort \
and knowing (the default) otherwise.
The MOOD line must be the very first line, followed by a blank line, then the acknowledgement.\
"""

_SYSTEM_PROMPT_SUPPLEMENTAL = f"{_SUPPLEMENTAL_GUIDANCE}\n\n{_SUPPLEMENTAL_MOOD_RULE}"

# The agentic path's replacement for the blob. Narrower than the training-status
# version because the question is: one activity, whose id is already known. The
# interesting calls are the second ones — the comparison a fixed prompt could
# never make, because it never knew what to compare against.
_TOOL_GUIDANCE = """\
You have tools that read this athlete's own training data, and this prompt \
contains none of it. Call get_activity_detail with the activity id below first — \
everything you need to describe the session is in that one result.
- Then, only if it would change what you say, look wider: \
list_recent_activities or find_activity to compare this session against similar \
recent ones, get_training_status for the fatigue state it lands in, \
get_power_profile if a power number looks like a breakthrough or a slump.
- Do not call the same tool twice with the same arguments; the answer will not \
change. Two or three calls is plenty for one session.
- A tool that cannot answer replies with a sentence explaining why, often naming \
what is nearby. Read it and adjust rather than repeating the call.
- Every figure you quote must come from a tool result. Never invent one, and \
never fill a gap with a plausible number.\
"""


def _language_suffix(locale: str | None) -> str:
    if not locale:
        return ""
    lang = _LOCALE_LANGUAGE.get(locale.split("-")[0].lower())
    return f" Respond in {lang}." if lang else ""


def _build_system_prompt(
    locale: str | None = None, sport_type: str | None = None
) -> str:
    # Cycling is the athlete's primary sport → full coaching analysis. Everything
    # else is supplemental training and only gets a short acknowledgement (issue #52).
    if sport_type in CYCLING_SPORT_TYPES:
        prompt = _SYSTEM_PROMPT_BASE
        prompt += f"\n\n{EXPERIENCE_GUIDANCE}"
    else:
        prompt = _SYSTEM_PROMPT_SUPPLEMENTAL
    return prompt + _language_suffix(locale)


def mood_rule_for(sport_type: str | None) -> str:
    """The format contract this sport's analysis is held to (issue #43).

    A supplemental session gets a one-line acknowledgement, not paragraphs, so
    restating the *paragraph* rule on its final turn would contradict the prompt
    it started from. Two rules, picked the same way the system prompt is.
    """
    return (
        _MOOD_RULE if sport_type in CYCLING_SPORT_TYPES else _SUPPLEMENTAL_MOOD_RULE
    )


def _build_agentic_system_prompt(
    locale: str | None = None, sport_type: str | None = None
) -> str:
    """The system prompt for the tool-driven path (issue #43).

    The coaching rules and the format contract are the blob path's, unchanged;
    only the "here is everything" middle becomes "here is how to get it". A
    supplemental session keeps its short-acknowledgement framing — an agent loop
    does not make a swim worth four paragraphs.
    """
    if sport_type in CYCLING_SPORT_TYPES:
        prompt = f"{_ANALYSIS_GUIDANCE}\n\n{_TOOL_GUIDANCE}\n\n{_MOOD_RULE}"
        prompt += f"\n\n{EXPERIENCE_GUIDANCE}"
    else:
        prompt = (
            f"{_SUPPLEMENTAL_GUIDANCE}\n\n{_TOOL_GUIDANCE}\n\n{_SUPPLEMENTAL_MOOD_RULE}"
        )
    return prompt + _language_suffix(locale)


def _build_agentic_user_prompt(activity: Activity) -> str:
    """The brief: which activity, and enough to recognise it if a lookup fails.

    The id is what the tools key on, but the date and sport are here too so a
    model whose ``get_activity_detail`` call misses has something to search with
    rather than a dead end.
    """
    when = (
        activity.start_time.strftime("%Y-%m-%d")
        if activity.start_time
        else "an unknown date"
    )
    return (
        f'Analyse the athlete\'s activity with id "{activity.id}" — '
        f"a {activity.sport_type or 'unknown sport'} session on {when}.\n\n"
        "Start by calling get_activity_detail with that id."
    )


# Plain-language renderings of the decoupling gate's reason codes, so the coach
# knows why the figure is missing rather than inventing one.
_DECOUPLING_REASON_TEXT: dict[str, str] = {
    "too_short": "the ride was too short for a meaningful drift measurement",
    "no_power": "no power data",
    "no_hr": "no heart-rate data",
    "degenerate_hr": "the heart-rate data was unusable",
    "variable_effort": "this was interval or otherwise surging riding, where the "
                       "measurement describes the intervals rather than aerobic durability",
    "uneven_pacing": "the two halves were ridden at markedly different intensities "
                     "(a ramp or negative split), so any drift figure would reflect "
                     "the pacing choice rather than aerobic durability",
    "stream_mismatch": "the power and heart-rate recordings don't line up well enough "
                       "to pair them sample by sample",
}


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
    planned: PlannedWorkout | None = None,
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
    if activity.weighted_power:
        lines.append(f"  Weighted power: {activity.weighted_power:.0f} W")
    if activity.intensity:
        lines.append(f"  Intensity factor: {activity.intensity:.3f}")
    if activity.load:
        lines.append(f"  Training load: {activity.load:.1f}")
    if activity.avg_hr:
        lines.append(f"  Average heart rate: {activity.avg_hr:.0f} bpm")
    if activity.max_hr:
        lines.append(f"  Peak heart rate: {activity.max_hr:.0f} bpm")

    # Aerobic response metrics (issue #37). A coach that can say "3% drift over
    # three hours, your durability is holding up" is markedly more credible than
    # one working from load and duration alone.
    vi = variability_index(activity.weighted_power, activity.avg_power)
    if vi is not None:
        lines.append(
            f"  Variability index: {vi:.2f} "
            "(weighted power / average power; 1.00 is perfectly steady, "
            "above 1.10 means surging or interval riding)"
        )
    ef = efficiency_factor(activity.weighted_power, activity.avg_hr)
    if ef is not None:
        lines.append(
            f"  Efficiency factor: {ef:.2f} W/bpm "
            "(weighted power per heartbeat; rising over time at the same "
            "training load indicates improving aerobic fitness)"
        )
    if activity.decoupling_pct is not None:
        lines.append(
            f"  Aerobic decoupling: {activity.decoupling_pct:.1f}% "
            "(how far the power:heart-rate ratio drifted from the first half of "
            "the ride to the second; under ~5% is generally considered good "
            "aerobic durability, though heat, dehydration and caffeine also "
            "push it up)"
        )
    elif activity.decoupling_reason:
        lines.append(
            "  Aerobic decoupling: not measured for this ride "
            f"({_DECOUPLING_REASON_TEXT.get(activity.decoupling_reason, activity.decoupling_reason)}) "
            "— do not speculate about a drift figure"
        )
    if athlete.ftp:
        lines.append(f"  Athlete FTP: {athlete.ftp} W")
    if athlete.max_hr:
        lines.append(f"  Athlete max HR: {athlete.max_hr} bpm")
    level = experience_level(athlete.app_settings)
    if level:
        lines.append(f"  Athlete self-reported experience level: {level}")

    if fatigue:
        from ..schemas.metrics import _form_to_label
        lines.append("\nAthlete fatigue state prior to this workout:")
        lines.append(f"  Fitness: {fatigue.fitness:.1f}")
        lines.append(f"  Fatigue: {fatigue.fatigue:.1f}")
        lines.append(f"  Form: {fatigue.form:.1f} ({_form_to_label(fatigue.form)})")

    if planned is not None:
        lines.append("\nPlanned workout this activity is linked to:")
        if planned.workout_type:
            lines.append(f"  Type: {planned.workout_type}")
        if planned.description and planned.description.strip():
            lines.append(f"  Description: {planned.description.strip()}")
        if planned.duration_min:
            lines.append(f"  Planned duration: {planned.duration_min} min")
        if planned.target_load:
            lines.append(f"  Target training load: {planned.target_load}")

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
    if getattr(activity, "rpe", None) is not None:
        lines.append(
            f"\nAthlete-rated perceived effort (RPE): {activity.rpe}/10 "
            "(subjective 1–10 scale; compare against measured intensity)"
        )

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


def _stream_analysis(
    activity: Activity,
    athlete: Athlete,
    user_id: str,
    fatigue: DailyMetric | None = None,
    locale: str | None = None,
    power_pr_badges: dict | None = None,
    distance_pr_badges: dict | None = None,
    planned: PlannedWorkout | None = None,
    usage_out: dict | None = None,
) -> AsyncIterator[str]:
    """Yield text chunks from the LLM via streaming SSE."""
    return stream_chat_completion(
        athlete,
        user_id,
        system_prompt=_build_system_prompt(locale, activity.sport_type),
        user_prompt=_build_prompt(
            activity, athlete, fatigue, power_pr_badges, distance_pr_badges, planned
        ),
        usage_out=usage_out,
    )


async def analyze_activity_bg(
    activity_id: str,
    athlete_id: str,
    user_id: str,
    locale: str | None = None,
    *,
    allow_agentic: bool = True,
) -> None:
    """
    Background task: stream LLM analysis → write chunks to DB every 500 ms
    → set final analysis_status to 'done' or 'error'.

    Lives in the service layer so it can be imported from both api/activities.py
    and services/strava_sync.py without circular dependencies.

    ``allow_agentic=False`` forces the single-shot blob prompt whatever the
    athlete opted into (issue #43). The provider-sync paths pass it: a backlog
    import creates one of these per imported activity, and a few hundred
    activities at four-to-six calls each is both a real bill and a lot of
    concurrent loops against one local model that serialises requests — on the
    one path where nobody reads the output one analysis at a time.
    """

    async def _clear_pending(recovery_session) -> None:
        result = await recovery_session.execute(
            select(Activity).where(Activity.id == activity_id)
        )
        stuck = result.scalar_one_or_none()
        if stuck is not None:
            settle_activity_analysis(stuck)

    # `stream_into_db` settles the status itself, but only once it is running —
    # and only while its session still works. Unlike the training-status
    # surface, this one has no pending timeout to fall back on: `trigger_analysis`
    # early-returns for `analysis_status == "pending"`, so a row that never
    # settles is an activity that can never be analysed again. The agentic path
    # adds ways to die between `pending` and settled (a cancelled tool call can
    # invalidate the very session `on_error` would commit through), which is
    # what makes the net worth having here as well.
    async with failure_recovery(
        user_id, f"Analysis for activity {activity_id}", _clear_pending
    ):
        await _analyze_activity(
            activity_id, athlete_id, user_id, locale, allow_agentic=allow_agentic
        )


async def _analyze_activity(
    activity_id: str,
    athlete_id: str,
    user_id: str,
    locale: str | None,
    *,
    allow_agentic: bool,
) -> None:
    async with get_user_session_factory(user_id)() as session:
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
        # The athlete's own date, for the tools to reckon "the last N days" from
        # (issue #43). The server's is a day out for anyone far enough from UTC,
        # and this surface reaches the same date-sensitive tools the status card
        # does.
        athlete_today = local_now((athlete.app_settings or {}).get("timezone")).date()

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

        # Include the planned workout this activity is linked to (if any) so the
        # coach can comment on plan adherence (issue #31). Only an explicit link
        # is used — we never guess a mapping from the date, which would wrongly
        # attribute e.g. a commute spin to a key session another ride completed.
        from .activity_workout_matcher import resolve_planned_workout_for_activity
        planned = await resolve_planned_workout_for_activity(session, activity)

        def _touch() -> None:
            # The clock the pending timeout reads (issue #91). Touched on every
            # progress commit, not just at the start, so the budget means "no
            # progress for N minutes" rather than "started N minutes ago" — a
            # long agentic run stays alive as long as it is visibly working, and
            # a run whose process died stops looking alive the moment it does.
            activity.analysis_updated_at = datetime.now(timezone.utc)

        def _set_analysis(text: str) -> None:
            activity.analysis = text
            _touch()

        def _set_step(code: str | None) -> None:
            activity.analysis_progress = code
            _touch()

        def _finish(text: str) -> None:
            activity.analysis = text
            activity.analysis_status = "done"
            # Cleared so a finished analysis renders exactly as it did before
            # the agentic path existed.
            activity.analysis_progress = None
            _touch()

        def _fail() -> None:
            activity.analysis_status = "error"
            activity.analysis_progress = None
            _touch()

        request = (
            AgentRequest(
                athlete=athlete,
                user_id=user_id,
                system_prompt=_build_agentic_system_prompt(
                    resolved_locale, activity.sport_type
                ),
                user_prompt=_build_agentic_user_prompt(activity),
                feature="activity_analysis",
                max_rounds=MAX_ROUNDS_ACTIVITY,
                format_rule=mood_rule_for(activity.sport_type),
                today=athlete_today,
            )
            if allow_agentic and agentic_enabled(athlete)
            else None
        )

        await stream_into_db(
            session,
            lambda usage_out: coaching_stream(
                request=request,
                blob=lambda blob_usage: _stream_analysis(
                    activity, athlete, user_id, fatigue=fatigue, locale=resolved_locale,
                    power_pr_badges=power_pr_badges, distance_pr_badges=distance_pr_badges,
                    planned=planned, usage_out=blob_usage,
                ),
                usage_out=usage_out,
            ),
            on_progress=_set_analysis,
            on_done=_finish,
            on_error=_fail,
            on_step=_set_step,
            user_id=user_id,
            feature="activity_analysis",
            label=f"Analysis for activity {activity_id}",
        )
