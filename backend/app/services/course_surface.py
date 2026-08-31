"""The background surface pass for a stored course (issue #56, Stage 2).

Two phases, because a matched surface is a **different analysis**, not an
annotation: it moves segment boundaries and it changes rolling resistance, so
the segment table has to be rebuilt rather than decorated.

1. The upload (and every re-analysis) returns a Stage 1 result synchronously,
   exactly as before. Latency character preserved: a course that would have
   taken 300 ms still does, and an instance without a sidecar never waits.
2. This runs afterwards, on its own session — match the stored track, store
   what the matcher said, then re-solve and replace the segments.

Because phase 2 works from the *stored* track, matching a new course and
enriching one uploaded years ago are the same function. That is most of the
value of turning the sidecar on: an instance that enables it later can pick up
every course it already holds, with no re-upload.

The status columns copy ``plan_*`` exactly, run token included, so
``stranded_runs`` settles a match a redeploy interrupted instead of leaving a
course pending for ever.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from openkoutsi import course as course_math

from backend.app.db.registry import registry_session
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import Athlete, Course, CourseTrack
from backend.app.services import course_analysis
from backend.app.services.instance_features import course_recon_enabled
from backend.app.services.llm_streaming import failure_recovery
from backend.app.services.surface_matcher import (
    SurfaceMatcher,
    get_surface_matcher,
    match_track,
)

log = logging.getLogger(__name__)

#: ``courses.surface_status`` values. ``None`` means "never attempted", which
#: is the state of every course on an instance with no sidecar and is an
#: absence rather than a failure.
PENDING = "pending"
DONE = "done"
#: The matcher was asked and could not answer — unreachable, or it snapped
#: nothing. Distinct from ``None`` so the UI can offer a retry rather than
#: implying the instance has no matcher at all, and still not an error the
#: athlete is shown: the course itself is fine.
UNAVAILABLE = "unavailable"


#: Matches in flight in this process.
#:
#: A plain counter rather than an :class:`asyncio.Semaphore`, for the reason
#: ``llm_agent._active_runs`` sets out: the one thing this guard must never do
#: is *wait*, and waiting is a semaphore's whole purpose. Testing ``.locked()``
#: before ``async with`` looks non-blocking and is not. asyncio is cooperative
#: and there is no ``await`` between reading this and incrementing it, so the
#: check and the claim are one indivisible step.
#:
#: It bounds two things at once: how much of the athlete's small connection
#: pool the optional background pass can occupy, and how many simultaneous
#: requests are aimed at a sidecar that is one container on a two-core box.
_active_matches = 0
MAX_CONCURRENT_MATCHES = 2


def _max_concurrent() -> int:
    return max(1, MAX_CONCURRENT_MATCHES)


def _try_claim_slot() -> bool:
    """Take a slot if one is free. Never waits, never raises."""
    global _active_matches
    if _active_matches >= _max_concurrent():
        return False
    _active_matches += 1
    return True


def _release_slot() -> None:
    global _active_matches
    _active_matches = max(0, _active_matches - 1)


def settle_course_surface(course, now: Optional[datetime] = None) -> bool:
    """Settle a match that a restart left ``pending``. Returns True when it did.

    Registered with ``stranded_runs`` alongside the plan settler. Lands on
    ``unavailable`` rather than ``error``: nothing about the course is wrong,
    the match simply did not finish, and the athlete still has a complete
    Stage 1 result in front of them.
    """
    if course.surface_status != PENDING:
        return False
    course.surface_status = UNAVAILABLE
    # Retire the token too: a run declared dead must not be able to come back
    # and overwrite the settled state if it was merely slow.
    course.surface_run_id = None
    course.surface_updated_at = now or datetime.now(timezone.utc)
    return True


async def match_course_surface(
    athlete_id: str,
    course_id: str,
    user_id: str,
    run_id: str | None = None,
    matcher: SurfaceMatcher | None = None,
) -> None:
    """Background task: match a stored course and re-solve it with the result.

    Degrades rather than fails at every step. An absent matcher, an unreachable
    one, one that times out or snaps nothing — each leaves the course exactly
    as Stage 1 left it, with a status saying so and no error surfaced to the
    athlete. A course analysed without surface data is a complete and useful
    thing and must not be presented as broken.
    """
    matcher = matcher or get_surface_matcher()

    # The capability check lives here, in the job, not only on the route that
    # schedules it (issue #56). A switch honoured at the entry point alone is
    # the mistake `allow_personal_access_tokens` made by checking issuance and
    # leaving /mcp open: this task can also be started by a retry, a future
    # caller, or a sweep, and none of those goes through the router.
    async with registry_session() as registry:
        if not await course_recon_enabled(registry):
            log.info(
                "Course recon is disabled on this instance; not matching course %s",
                course_id,
            )
            return

    async def _clear_pending(recovery_session) -> None:
        stuck = (
            await recovery_session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one_or_none()
        if stuck is not None:
            settle_course_surface(stuck)

    if not _try_claim_slot():
        # Every slot busy. Settling to `unavailable` rather than queueing is
        # the honest answer: the athlete has a complete Stage 1 course, and a
        # status this feature already has and already handles beats holding a
        # database connection open waiting for a turn.
        log.info(
            "All %d surface-match slots busy; not matching course %s now",
            _max_concurrent(),
            course_id,
        )
        async with failure_recovery(
            user_id, f"Surface match for course {course_id}", _clear_pending
        ):
            async with get_user_session_factory(user_id)() as session:
                course = await _load_course(session, course_id, athlete_id)
                if course is not None:
                    await _settle(session, course, run_id, UNAVAILABLE)
        return

    try:
        async with failure_recovery(
            user_id, f"Surface match for course {course_id}", _clear_pending
        ):
            # ── read ────────────────────────────────────────────────────────
            # Everything the match needs is read and the session closed before
            # the sidecar is called. Holding a pooled connection across up to
            # TOTAL_BUDGET_S of network I/O would let a handful of concurrent
            # matches exhaust this athlete's pool (pool_size=3, max_overflow=2)
            # and make their *interactive* requests queue behind a routing
            # engine — the optional feature degrading the app around it.
            async with get_user_session_factory(user_id)() as session:
                course = await _load_course(session, course_id, athlete_id)
                if course is None:
                    return
                if run_id is not None and course.surface_run_id != run_id:
                    # Superseded before we started — a re-analysis, or a second
                    # match request. The run holding the token owns the columns.
                    return

                track = await session.get(CourseTrack, course_id)
                if track is None or not track.points:
                    await _settle(session, course, run_id, UNAVAILABLE)
                    return

                stored_points = list(track.points)
                target_time_s = course.target_time_s
                target_power_w = course.target_power_w
                bike = await _bike_params(session, course)
                athlete = (
                    await session.execute(
                        select(Athlete).where(Athlete.id == athlete_id)
                    )
                ).scalar_one_or_none()
                rider = (
                    course_math.RiderParams(
                        ftp_w=float(athlete.ftp), weight_kg=float(athlete.weight_kg)
                    )
                    if athlete is not None and athlete.ftp and athlete.weight_kg
                    else None
                )

            # ── match, holding no database connection ───────────────────────
            points = [(row[0], row[1]) for row in stored_points]
            distances = [row[3] for row in stored_points]
            surfaces = await match_track(matcher, points, distances)

            solved = None
            if surfaces is not None and any(entry[0] for entry in surfaces) and rider:
                solved, reason = await asyncio.to_thread(
                    course_analysis.analyze_stored_track,
                    stored_points,
                    rider,
                    bike,
                    target_time_s,
                    target_power_w,
                    surfaces,
                )
                if solved is None:
                    log.warning(
                        "Course %s could not be re-solved after matching (%s)",
                        course_id,
                        reason,
                    )

            # ── write ───────────────────────────────────────────────────────
            async with get_user_session_factory(user_id)() as session:
                course = await _load_course(session, course_id, athlete_id)
                if course is None:
                    return
                # The token re-check that already guarded a re-analysis landing
                # mid-match now also spans the session boundary, which is what
                # makes reopening safe. It is a truer check here than before:
                # it reads outside the transaction it is trying to detect a
                # change against.
                if run_id is not None and course.surface_run_id != run_id:
                    log.info(
                        "Surface match for course %s superseded; discarding", course_id
                    )
                    return

                if surfaces is None or not any(entry[0] for entry in surfaces):
                    # Nothing identified is not "the whole route is unknown":
                    # drawing a full-length grey band would claim we had looked
                    # and found something when we had only looked.
                    await _settle(session, course, run_id, UNAVAILABLE)
                    return

                track = await session.get(CourseTrack, course_id)
                if track is None:
                    await _settle(session, course, run_id, UNAVAILABLE)
                    return
                track.surfaces = surfaces
                track.surface_matched_at = datetime.now(timezone.utc)

                if solved is None or rider is None:
                    # The matcher answered but the physics could not re-solve —
                    # no FTP or weight on the profile, most likely. Keep what
                    # the matcher said, since it is still true about the road,
                    # rather than half-updating the segments.
                    await _settle(session, course, run_id, UNAVAILABLE)
                    return

                await course_analysis.persist_analysis(
                    course, solved, session, rider=rider
                )
                course.surface_status = DONE
                course.surface_run_id = None
                course.surface_updated_at = datetime.now(timezone.utc)
                await session.commit()
    finally:
        _release_slot()


async def _load_course(session: AsyncSession, course_id: str, athlete_id: str):
    return (
        await session.execute(
            select(Course).where(
                Course.id == course_id, Course.athlete_id == athlete_id
            )
        )
    ).scalar_one_or_none()


async def _settle(
    session: AsyncSession, course: Course, run_id: str | None, status: str
) -> None:
    if run_id is not None and course.surface_run_id != run_id:
        return
    course.surface_status = status
    course.surface_run_id = None
    course.surface_updated_at = datetime.now(timezone.utc)
    await session.commit()


async def _bike_params(session: AsyncSession, course: Course) -> course_math.BikeParams:
    from backend.app.models.user_orm import Bike

    bike = await session.get(Bike, course.bike_id) if course.bike_id else None
    if bike is None:
        return course_math.BikeParams(
            tyre_width_mm=None, riding_position=course_math.DEFAULT_POSITION
        )
    return course_math.BikeParams(
        tyre_width_mm=bike.tyre_width_mm, riding_position=bike.riding_position
    )


def claim_run(course: Course) -> str:
    """Stamp a fresh run token on a course and mark the match pending.

    Called on the request thread before the background task is scheduled, so
    the token is committed with the rest of the request and a task that starts
    late cannot mistake an older run's columns for its own.
    """
    run_id = str(uuid.uuid4())
    course.surface_status = PENDING
    course.surface_run_id = run_id
    course.surface_updated_at = datetime.now(timezone.utc)
    return run_id
