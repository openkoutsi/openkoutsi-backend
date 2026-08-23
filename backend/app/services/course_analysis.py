"""Parsing, analysing and storing courses (issue #55).

The bridge between the pure math in :mod:`openkoutsi.course` and the backend's
storage. Three concerns live here, and the coordinate boundary is the point of
the layout:

* **Parsing and analysis** — synchronous, CPU-bound functions the API calls
  through ``asyncio.to_thread``. The :class:`openkoutsi.gpx.Route` and
  :class:`openkoutsi.course.CourseTrack` never escape this module except as
  the ``course_tracks`` JSON row, which nothing user-facing reads.
* **The encrypted blob** — the raw GPX stored exactly as FIT files are
  (:mod:`backend.app.core.file_encryption`, same derived key), under an
  **opaque storage key**: a bare filename resolved against the user's upload
  directory at read time. Never an absolute path in the database — issue #51
  exists to purge those, and a new blob type must not add a third one.
* **Persistence** — writing an analysis onto a ``Course`` row, replacing its
  segments wholesale and clearing any written plan, which is stale the moment
  the segment table it reasoned over changes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.file_encryption import decrypt_file, encrypt_file
from backend.app.models.user_orm import Course, CourseSegment
from backend.app.models.user_orm import CourseTrack as CourseTrackRow
from openkoutsi import course as course_math
from openkoutsi import gpx


@dataclass(frozen=True)
class ParsedCourse:
    """What a successful parse+analysis hands back to the API layer."""

    track: course_math.CourseTrack
    name: str | None
    analysis: course_math.CourseAnalysis


def parse_and_analyze(
    gpx_bytes: bytes,
    rider: course_math.RiderParams,
    bike: course_math.BikeParams,
    target_time_s: int | None,
    target_power_w: int | None = None,
) -> tuple[ParsedCourse | None, str | None]:
    """Parse a GPX course and run the full analysis. Sync and CPU-bound —
    call through ``asyncio.to_thread``.

    Returns ``(parsed, None)`` or ``(None, reason)`` with the profile reason
    codes from :func:`openkoutsi.course.course_profile`. A file that is not
    GPX at all raises :class:`openkoutsi.gpx.ActivityParseError`, which the
    API maps to a 400.
    """
    route = gpx.extract_route(io.BytesIO(gpx_bytes))
    track = course_math.thin_track(route)
    profile, reason = course_math.course_profile(track)
    if profile is None:
        return None, reason
    analysis = course_math.analyze_course(
        profile, rider, bike, target_time_s, target_power_w
    )
    return ParsedCourse(track=track, name=route.name, analysis=analysis), None


def analyze_stored_track(
    points: list,
    rider: course_math.RiderParams,
    bike: course_math.BikeParams,
    target_time_s: int | None,
    target_power_w: int | None = None,
) -> tuple[course_math.CourseAnalysis | None, str | None]:
    """Re-analysis without re-upload: run on the ``course_tracks`` JSON row.

    This is the path every target change takes — a new target time, a switch
    to a target power, or clearing both — which is the whole reason the track
    is stored at all.

    Sync and CPU-bound — call through ``asyncio.to_thread``.
    """
    track = track_from_points(points)
    profile, reason = course_math.course_profile(track)
    if profile is None:
        return None, reason
    return (
        course_math.analyze_course(profile, rider, bike, target_time_s, target_power_w),
        None,
    )


def track_points_json(track: course_math.CourseTrack) -> list:
    """The ``course_tracks.points`` series: ``[[lat, lon, elevation_m, distance_m], …]``."""
    return [
        [p.latitude, p.longitude, p.elevation_m, p.distance_m] for p in track.points
    ]


def track_from_points(points: list) -> course_math.CourseTrack:
    return course_math.CourseTrack(
        points=[
            course_math.TrackPoint(
                latitude=lat, longitude=lon, distance_m=dist, elevation_m=ele
            )
            for lat, lon, ele, dist in points
        ]
    )


# ── the encrypted blob ────────────────────────────────────────────────────────


def store_course_blob(gpx_bytes: bytes, user_id: str, course_id: str) -> str:
    """Write the original GPX encrypted to disk; return its opaque storage key.

    The key is a bare filename — resolution happens in
    :func:`resolve_course_blob`, so the database never learns where this
    node keeps its files.
    """
    key = f"course-{course_id}.gpx"
    storage_dir = settings.user_fit_dir(user_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / key
    path.write_bytes(gpx_bytes)
    try:
        encrypt_file(path, user_id)
    except Exception:
        # The file is on disk as plaintext GPX until `encrypt_file` returns, and
        # it is *designed* to raise hard on a missing or rotated ENCRYPTION_KEY.
        # Leaving it would strand a readable track that no row points at — so
        # invisible to per-course delete and to the GDPR export, both of which
        # iterate Course rows. Take it with us.
        path.unlink(missing_ok=True)
        raise
    return key


def resolve_course_blob(key: str, user_id: str):
    """The on-disk path for a storage key, containment-checked.

    Raises ``ValueError`` when the key resolves outside the user's upload
    directory — a row tampered into a traversal must not become a read.
    """
    expected_dir = settings.user_fit_dir(user_id).resolve()
    path = (expected_dir / key).resolve()
    if not path.is_relative_to(expected_dir):
        raise ValueError("course blob key resolves outside the user's storage")
    return path


def read_course_blob(course: Course, user_id: str) -> bytes:
    path = resolve_course_blob(course.gpx_file_key, user_id)
    if course.gpx_file_encrypted:
        return decrypt_file(path, user_id)
    return path.read_bytes()


def delete_blob_by_key(key: str, user_id: str) -> None:
    """Remove one stored blob by key. Safe to call for a file that is not there."""
    try:
        resolve_course_blob(key, user_id).unlink(missing_ok=True)
    except ValueError:
        # A key that fails containment names nothing we should touch.
        pass


def delete_course_blob(course: Course, user_id: str) -> None:
    delete_blob_by_key(course.gpx_file_key, user_id)


# ── persistence ───────────────────────────────────────────────────────────────


async def persist_analysis(
    course: Course,
    analysis: course_math.CourseAnalysis,
    session: AsyncSession,
    *,
    rider: course_math.RiderParams,
) -> None:
    """Write an analysis onto a course row. Does not commit — the caller does.

    Segments are replaced wholesale (they are derived state, like intervals),
    and any written plan is cleared: prose reasoned over the old segment table
    is stale by definition. Clearing includes ``plan_run_id``, which is what
    makes the clear stick against a run that is still streaming.
    """
    course.status = "ready"
    course.error = None
    course.ftp_w_used = rider.ftp_w
    course.weight_kg_used = rider.weight_kg
    course.distance_m = analysis.total_distance_m
    course.elevation_gain_m = analysis.elevation_gain_m
    course.elevation_loss_m = analysis.elevation_loss_m
    course.min_elevation_m = analysis.min_elevation_m
    course.max_elevation_m = analysis.max_elevation_m
    course.profile = [
        [p.distance_m, p.elevation_m, p.gradient] for p in analysis.profile
    ]

    pacing = analysis.pacing
    course.predicted_time_s = pacing.predicted_time_s
    course.intensity = pacing.intensity
    course.required_intensity = pacing.required_intensity
    course.feasible = pacing.feasible
    course.refusal_reason = pacing.refusal_reason

    course.plan = None
    course.plan_mood = None
    course.plan_status = None
    course.plan_updated_at = None
    # Clearing the token is what actually stops an in-flight run: the generator
    # holds its own session and commits after this one, so nulling the prose
    # alone would let it write the stale plan straight back onto the new
    # segment table. See `llm_course_plan.generate_course_plan_bg`.
    course.plan_run_id = None

    await session.execute(
        delete(CourseSegment).where(CourseSegment.course_id == course.id)
    )
    for plan in pacing.splits:
        seg = plan.segment
        session.add(
            CourseSegment(
                course_id=course.id,
                segment_index=seg.index,
                start_distance_m=seg.start_distance_m,
                end_distance_m=seg.end_distance_m,
                length_m=seg.length_m,
                avg_gradient=seg.avg_gradient,
                elevation_change_m=seg.elevation_change_m,
                segment_type=seg.segment_type,
                power_w=plan.power_w,
                speed_ms=plan.speed_ms,
                duration_s=plan.duration_s,
                start_offset_s=plan.start_offset_s,
                speed_capped=plan.speed_capped,
            )
        )
    # A refused target still deserves a segment table — the athlete should see
    # the course even when the requested time is not achievable.
    if not pacing.splits and analysis.segments:
        for seg in analysis.segments:
            session.add(
                CourseSegment(
                    course_id=course.id,
                    segment_index=seg.index,
                    start_distance_m=seg.start_distance_m,
                    end_distance_m=seg.end_distance_m,
                    length_m=seg.length_m,
                    avg_gradient=seg.avg_gradient,
                    elevation_change_m=seg.elevation_change_m,
                    segment_type=seg.segment_type,
                )
            )


async def store_track(course: Course, track: course_math.CourseTrack, session: AsyncSession) -> None:
    """Upsert the thinned track row for a course."""
    existing = await session.get(CourseTrackRow, course.id)
    if existing is not None:
        existing.points = track_points_json(track)
    else:
        session.add(CourseTrackRow(course_id=course.id, points=track_points_json(track)))
