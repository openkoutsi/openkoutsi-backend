"""LLM-written pacing plan for an analysed course (issue #55).

The physics computes the numbers; this service asks the model to write the
day: pacing keyed to the segment table, the key climbs, fuelling grounded in
the predicted duration and intensity, and decision points. The model **never
does arithmetic** and is constrained to reason only from the supplied table —
which is also why the prompt is built from the ``Course`` row, its
``CourseSegment`` rows and the ``Athlete``: types that carry no coordinates,
so the track cannot leak into the context even by accident.

Structured like ``llm_goal_guidance`` — same streaming and usage-recording
plumbing, same Koutsi voice, and the ``MOOD:`` tag-line convention the daily
card established.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Sequence

from sqlalchemy import select, update

from openkoutsi import course as course_math

from ..db.user_session import get_user_session_factory
from ..models.user_orm import Athlete, Course, CourseSegment
from .athlete_experience import experience_level
from .llm_streaming import failure_recovery, stream_chat_completion, stream_into_db
from .llm_training_status_analyzer import _decorate, _local_now, _MOOD_RULE
from .stranded_runs import settle_course_plan

log = logging.getLogger(__name__)

# How often the stream checks that it still owns the row. Matches the
# 500 ms commit cadence in `stream_into_db`, so it is one extra indexed
# read per commit rather than one per chunk.
_GUARD_INTERVAL_S = 0.5

_MOOD_RE = re.compile(r"^MOOD:\s?(cheer|knowing|neutral|stern)\s*$", re.IGNORECASE)
_FALLBACK_MOOD = "knowing"

_SYSTEM_PROMPT_BASE = f"""\
You are Koutsi, an expert endurance sports coach. The athlete has uploaded a \
course for an upcoming ride or event, and the app has computed a physics-based \
pacing model for it: the course split into gradient segments, each with a power \
target and a predicted split solved from the athlete's own FTP and weight. \
Write a practical pacing plan in 3-5 paragraphs of plain prose — no markdown \
headers, no bullet points, no code blocks. Separate paragraphs with a single \
blank line.

Cover, grounded strictly in the table you are given: how to ride each phase of \
the course (refer to places as "the climb starting at km X" — the table's \
distances are the only geography you have); where the day is won or lost; a \
fuelling and drinking schedule appropriate to the predicted duration and \
intensity; and one or two decision points where the athlete should check \
themselves and adjust.

Hard rules: never invent local knowledge about roads, weather, or scenery — \
you know nothing about this course beyond the table. Never do arithmetic: \
every number you state must appear in the table. State plainly, once, that all \
speed and time predictions assume still air and a dry paved surface, and that \
wind will move them. If the pacing model marked the athlete's target time as \
not achievable, open with that: say why in the model's terms (the power it \
would take against what is sustainable) and build the plan around the fastest \
realistic ride instead.

{_MOOD_RULE}\
"""


def _build_system_prompt(locale: str | None, coaching_style: str | None) -> str:
    return _decorate(_SYSTEM_PROMPT_BASE, locale, coaching_style)


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _segment_plans(segments: Sequence[CourseSegment]) -> list[course_math.SegmentPlan]:
    """Rebuild core dataclasses from the rows, for `key_climbs` grouping."""
    plans = []
    for row in segments:
        plans.append(
            course_math.SegmentPlan(
                segment=course_math.Segment(
                    index=row.segment_index,
                    start_distance_m=row.start_distance_m,
                    end_distance_m=row.end_distance_m,
                    length_m=row.length_m,
                    avg_gradient=row.avg_gradient,
                    elevation_change_m=row.elevation_change_m,
                    segment_type=row.segment_type,
                ),
                power_w=row.power_w or 0.0,
                speed_ms=row.speed_ms or 0.0,
                duration_s=row.duration_s or 0.0,
                start_offset_s=row.start_offset_s or 0.0,
                speed_capped=row.speed_capped,
            )
        )
    return plans


def _build_course_prompt(
    athlete: Athlete,
    course: Course,
    segments: Sequence[CourseSegment],
    now: datetime,
) -> str:
    today = now.date()
    tz_label = now.strftime("%Z") or "UTC"
    lines = [
        f"Course plan request — {today.isoformat()} ({today.strftime('%A')})  "
        f"{now.strftime('%H:%M')} {tz_label}"
    ]

    lines.append("\nCourse:")
    lines.append(
        f"  {course.name} — {course.distance_m / 1000:.1f} km, "
        f"{course.elevation_gain_m or 0:.0f} m up / {course.elevation_loss_m or 0:.0f} m down"
    )
    if course.start_time:
        local_start = course.start_time.astimezone(now.tzinfo or timezone.utc)
        lines.append(f"  Event start: {local_start.strftime('%Y-%m-%d %H:%M')}")
    else:
        lines.append("  Event start: (not set)")

    lines.append("\nAthlete:")
    ftp = course.ftp_w_used or (athlete.ftp or 0)
    weight = course.weight_kg_used or (athlete.weight_kg or 0)
    wkg = f" ({ftp / weight:.1f} W/kg)" if ftp and weight else ""
    lines.append(f"  FTP: {ftp:.0f} W, weight: {weight:.0f} kg{wkg}")
    level = experience_level(athlete.app_settings)
    if level:
        lines.append(f"  Self-reported experience level: {level}")

    lines.append("\nPacing model (still air, dry pavement, bike+kit mass included):")
    if course.target_power_w:
        share = (
            f" ({course.target_power_w / course.ftp_w_used * 100:.0f}% of FTP)"
            if course.ftp_w_used
            else ""
        )
        lines.append(
            f"  Requested target: hold an average of {course.target_power_w:.0f} W"
            f"{share} for the whole ride — the finish time below is what that produces, "
            "not something the athlete asked for."
        )
    elif course.target_time_s:
        lines.append(f"  Requested target time: {_hms(course.target_time_s)}")
    else:
        lines.append("  Requested target: (none — a steady sustainable effort)")

    if course.feasible is False:
        required = f"{course.required_intensity:.2f}" if course.required_intensity else "?"
        # The ceiling is a function of how long the ride lasts. For a time
        # target that is the target itself; for a power target it is the time
        # the requested watts produce.
        duration_s = (
            course.target_time_s
            if course.target_time_s and not course.target_power_w
            else course.predicted_time_s
        )
        sustainable = (
            f"{course_math.max_sustainable_intensity(duration_s):.2f}" if duration_s else "?"
        )
        if course.refusal_reason == "target_faster_than_physics":
            lines.append("  ⚠ The requested target is NOT achievable:")
            lines.append(
                "    reason: faster than the physics allows at any human power. "
                f"Fastest modelled ride: {_hms(course.predicted_time_s)}"
            )
        elif course.target_power_w:
            lines.append(
                f"  ⚠ That average power is above what anyone sustains for this long: "
                f"{required} × FTP against a ceiling of about {sustainable} × FTP for "
                f"{_hms(course.predicted_time_s)} of riding. The splits below are still "
                "exactly what was asked for — say plainly that holding them to the finish "
                "is unlikely, and what to give up first if it starts to come apart."
            )
        else:
            lines.append("  ⚠ The requested target is NOT achievable:")
            lines.append(
                f"    reason: it would take an average intensity of {required} × FTP, "
                f"against a sustainable ceiling of about {sustainable} × FTP for that duration."
            )

    # An impossible finish time leaves no splits to talk about; an
    # unsustainable *power* leaves a complete plan, warned about above. So the
    # numbers are stated in every case except the first.
    if not (course.feasible is False and not course.target_power_w):
        if course.intensity is not None:
            lines.append(f"  Intensity: {course.intensity:.2f} × FTP")
        lines.append(f"  Predicted total time: {_hms(course.predicted_time_s)}")

    lines.append(
        "\nSegments (start km | length km | avg grade | type | target power | est. speed | split | cumulative):"
    )
    for row in segments:
        power = f"{row.power_w:.0f} W" if row.power_w else "coast"
        if row.power_w and course.ftp_w_used:
            power += f" ({row.power_w / course.ftp_w_used * 100:.0f}% FTP)"
        speed = f"{(row.speed_ms or 0) * 3.6:.1f} km/h"
        lines.append(
            f"  {row.start_distance_m / 1000:5.1f} | {row.length_m / 1000:4.1f} | "
            f"{row.avg_gradient * 100:+.1f}% | {row.segment_type:7s} | {power} | "
            f"{speed} | {_hms(row.duration_s)} | {_hms(row.start_offset_s)}"
        )

    climbs = course_math.key_climbs(_segment_plans(segments))
    if climbs:
        lines.append("\nKey climbs:")
        for climb in climbs:
            at = f"{climb.start_distance_m / 1000:.1f}"
            power = f" at {climb.avg_power_w:.0f} W" if climb.avg_power_w else ""
            lines.append(
                f"  - climb starting at km {at}: {climb.length_m / 1000:.1f} km at "
                f"{climb.avg_gradient * 100:.1f}%, {climb.elevation_gain_m:.0f} m gain, "
                f"est. {_hms(climb.duration_s)}{power}"
            )

    return "\n".join(lines)


def _parse_mood(text: str) -> tuple[str, str]:
    """Split a leading ``MOOD:<mood>`` line off the streamed prose."""
    lines = text.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx < len(lines):
        match = _MOOD_RE.match(lines[idx].strip())
        if match:
            rest = lines[idx + 1:]
            if rest and rest[0].strip() == "":
                rest = rest[1:]
            return match.group(1).lower(), "\n".join(rest).strip()
    return _FALLBACK_MOOD, text.strip()


def _stream_display_prose(text: str) -> str:
    """Tag-free prose to persist mid-stream — same holdback as goal guidance."""
    if "\n" not in text:
        return ""
    _, prose = _parse_mood(text)
    return prose


def _stream_course_plan(
    athlete: Athlete,
    user_id: str,
    course: Course,
    segments: Sequence[CourseSegment],
    now: datetime,
    locale: str | None = None,
    coaching_style: str | None = None,
    usage_out: dict | None = None,
) -> AsyncIterator[str]:
    return stream_chat_completion(
        athlete,
        user_id,
        system_prompt=_build_system_prompt(locale, coaching_style),
        user_prompt=_build_course_prompt(athlete, course, segments, now),
        usage_out=usage_out,
    )


async def _run_is_current(session, course_id: str, run_id: str | None) -> bool:
    """Does the row still belong to this plan run?

    A fresh SELECT rather than the loaded instance's attribute: the point is to
    see what *another* session committed while this one was streaming.
    """
    if run_id is None:
        return True  # a caller that predates run tokens keeps the old behaviour
    current = (
        await session.execute(select(Course.plan_run_id).where(Course.id == course_id))
    ).scalar_one_or_none()
    return current == run_id


async def generate_course_plan_bg(
    athlete_id: str,
    course_id: str,
    user_id: str,
    locale: str | None = None,
    run_id: str | None = None,
) -> None:
    """Background task: stream the written plan into ``courses.plan`` every
    500 ms, parse the leading MOOD tag, settle to 'done'/'error'.

    This runs on **its own session** and commits after the request that started
    it, so it can outlive the state it was started for: re-analysing a course
    mid-stream clears the plan columns, and without a guard the next progress
    commit would put prose describing the *old* segment table straight back —
    ending on ``done`` with power targets keyed to distances that no longer
    exist. ``run_id`` is the guard: the trigger stamps it, re-analysis clears
    it, and a run whose token no longer matches discards its own writes.
    """

    async def _clear_pending(recovery_session) -> None:
        result = await recovery_session.execute(
            select(Course).where(Course.id == course_id)
        )
        stuck = result.scalar_one_or_none()
        if stuck is not None:
            settle_course_plan(stuck)

    async with failure_recovery(
        user_id, f"Course plan for course {course_id}", _clear_pending
    ):
        async with get_user_session_factory(user_id)() as session:
            athlete = (
                await session.execute(select(Athlete).where(Athlete.id == athlete_id))
            ).scalar_one()
            course = (
                await session.execute(
                    select(Course).where(
                        Course.id == course_id, Course.athlete_id == athlete_id
                    )
                )
            ).scalar_one()
            segments = (
                await session.execute(
                    select(CourseSegment)
                    .where(CourseSegment.course_id == course_id)
                    .order_by(CourseSegment.segment_index)
                )
            ).scalars().all()

            app_cfg = athlete.app_settings or {}
            resolved_locale = locale or app_cfg.get("locale")
            coaching_style = app_cfg.get("coaching_style")
            now = _local_now(app_cfg.get("timezone"))

            def _set_prose(text: str) -> None:
                course.plan = _stream_display_prose(text)
                # Touch the inactivity clock the pending timeout reads
                # (issue #91) so a healthy slow stream is not declared dead.
                course.plan_updated_at = datetime.now(timezone.utc)

            def _finish(text: str) -> None:
                mood, prose = _parse_mood(text)
                course.plan = prose
                course.plan_mood = mood
                course.plan_status = "done"
                course.plan_updated_at = datetime.now(timezone.utc)

            def _fail() -> None:
                course.plan_status = "error"
                course.plan_updated_at = datetime.now(timezone.utc)

            async def _guarded(usage_out: dict):
                """The plan stream, stopped early once this run is superseded.

                Checked on roughly the flush cadence rather than per chunk: one
                cheap indexed read per commit, which is what bounds how long a
                superseded run keeps spending tokens. The reconciliation below
                is what makes the guarantee — this only stops the waste.
                """
                last_check = time.monotonic()
                async for chunk in _stream_course_plan(
                    athlete, user_id, course, segments,
                    now, locale=resolved_locale, coaching_style=coaching_style,
                    usage_out=usage_out,
                ):
                    if time.monotonic() - last_check >= _GUARD_INTERVAL_S:
                        last_check = time.monotonic()
                        if not await _run_is_current(session, course_id, run_id):
                            log.info(
                                "Course plan for course %s superseded mid-stream; stopping",
                                course_id,
                            )
                            return
                    yield chunk

            await stream_into_db(
                session,
                _guarded,
                on_progress=_set_prose,
                on_done=_finish,
                on_error=_fail,
                user_id=user_id,
                feature="course_plan",
                label=f"Course plan for course {course_id}",
            )

            # The guarantee. `stream_into_db` has committed by now, whichever
            # way it ended, so a run that lost its claim has to take its own
            # writes back out — the callbacks cannot do it themselves, being
            # synchronous and unable to read another session's commit.
            if not await _run_is_current(session, course_id, run_id):
                # Only when **nobody** owns the row. This surface had run tokens
                # first and the other three copied this block from it, including
                # the defect: keyed on the id alone, a slow run that lost its
                # claim to a *re-trigger* blanked the columns the live run was
                # writing. When a newer run holds the token the correct action
                # is none at all — it overwrites every one of these itself.
                cleared = await session.execute(
                    update(Course)
                    .where(
                        Course.id == course_id,
                        Course.plan_run_id.is_(None),
                    )
                    .values(
                        plan=None, plan_mood=None, plan_status=None,
                        plan_updated_at=None,
                    )
                )
                await session.commit()
                if cleared.rowcount:
                    log.info(
                        "Discarded a superseded course plan for course %s", course_id
                    )
                else:
                    log.info(
                        "Course plan for course %s was superseded by a live run — "
                        "leaving that run's columns alone",
                        course_id,
                    )
