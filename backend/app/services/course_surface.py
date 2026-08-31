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


async def surface_matching_available(
    registry: AsyncSession,
    matcher: SurfaceMatcher | None = None,
) -> bool:
    """Whether this instance can classify a surface at all.

    Both halves have to be true and they mean different things — the admin
    switched the capability on, *and* a sidecar is actually configured — so
    they are kept apart internally and only ANDed here, where a caller wants
    the single effective answer.
    """
    if not await course_recon_enabled(registry):
        return False
    return (matcher or get_surface_matcher()).is_configured


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

    async with failure_recovery(
        user_id, f"Surface match for course {course_id}", _clear_pending
    ):
        async with get_user_session_factory(user_id)() as session:
            course = (
                await session.execute(
                    select(Course).where(
                        Course.id == course_id, Course.athlete_id == athlete_id
                    )
                )
            ).scalar_one_or_none()
            if course is None:
                return
            if run_id is not None and course.surface_run_id != run_id:
                # Superseded before we started — a re-analysis, or a second
                # match request. The run that holds the token owns the columns.
                return

            track = await session.get(CourseTrack, course_id)
            if track is None or not track.points:
                await _settle(session, course, run_id, UNAVAILABLE)
                return

            points = [(row[0], row[1]) for row in track.points]
            distances = [row[3] for row in track.points]
            surfaces = await match_track(matcher, points, distances)
            if surfaces is None or not any(entry[0] for entry in surfaces):
                # Nothing identified is not the same as "the whole route is
                # unknown": drawing a full-length grey band would claim we had
                # looked and found something when we had only looked. The
                # client already collapses this case, but asserting it here
                # too means the property holds whichever layer produced it.
                await _settle(session, course, run_id, UNAVAILABLE)
                return

            athlete = (
                await session.execute(select(Athlete).where(Athlete.id == athlete_id))
            ).scalar_one_or_none()
            if athlete is None or not athlete.ftp or not athlete.weight_kg:
                # The physics cannot re-solve without these. Keep what the
                # matcher said — it is still true about the road — but leave
                # the segments alone rather than half-updating them.
                track.surfaces = surfaces
                track.surface_matched_at = datetime.now(timezone.utc)
                await _settle(session, course, run_id, UNAVAILABLE)
                return

            bike = await _bike_params(session, course)
            rider = course_math.RiderParams(
                ftp_w=float(athlete.ftp), weight_kg=float(athlete.weight_kg)
            )
            analysis, reason = await asyncio.to_thread(
                course_analysis.analyze_stored_track,
                track.points,
                rider,
                bike,
                course.target_time_s,
                course.target_power_w,
                surfaces,
            )
            if analysis is None:
                log.warning(
                    "Course %s could not be re-solved after matching (%s)",
                    course_id,
                    reason,
                )
                await _settle(session, course, run_id, UNAVAILABLE)
                return

            # Re-check the token before writing: a re-analysis may have landed
            # while we were out at the sidecar, and its segment table is the
            # current one. Same guarantee `generate_course_plan_bg` makes.
            await session.refresh(course)
            if run_id is not None and course.surface_run_id != run_id:
                log.info("Surface match for course %s superseded; discarding", course_id)
                return

            track.surfaces = surfaces
            track.surface_matched_at = datetime.now(timezone.utc)
            await course_analysis.persist_analysis(
                course, analysis, session, rider=rider
            )
            course.surface_status = DONE
            course.surface_run_id = None
            course.surface_updated_at = datetime.now(timezone.utc)
            await session.commit()


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
