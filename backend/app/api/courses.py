"""Courses — upload a GPX course, get a segment table and a pacing plan (issue #55).

Processing is synchronous: parse + thin + smooth + segment + solve is pure
arithmetic in the low hundreds of milliseconds for a typical course, so the
POST returns the finished analysis and errors are immediate 400/422s rather
than a job to poll. The CPU-bound work still runs off the event loop via
``asyncio.to_thread``. The LLM-written plan is the opposite: always
asynchronous, copying the goal-guidance shape exactly (trigger → pending →
poll, stranded runs settled at boot).

No response in this module carries a coordinate. The stored track and the
encrypted GPX exist for re-analysis, Stage 2 and the GDPR export — not for
the API, the LLM, or MCP.
"""
import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.consent import require_consent
from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_forbidden, pat_scopes
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Bike, Course, CourseSegment, Goal
from backend.app.models.user_orm import CourseTrack as CourseTrackRow
from backend.app.schemas.courses import (
    CoursePlanBody,
    CoursePlanResponse,
    CourseDetailResponse,
    CourseReanalyzeBody,
    CourseSummaryResponse,
)
from backend.app.schemas.pagination import Page, PageParams, paginate_params
from backend.app.services import course_analysis
from backend.app.services.stranded_runs import pending_timed_out, settle_course_plan
from openkoutsi.course import BikeParams, RiderParams
from openkoutsi.gpx import ActivityParseError

router = APIRouter(
    prefix="/courses",
    tags=["courses"],
    dependencies=[pat_scopes(read="courses:read", write="courses:write")],
)

# A GPX course is text; even a very long one is a few MB. The cap bounds the
# in-memory read, not a legitimate file.
_MAX_COURSE_BYTES = 20 * 1024 * 1024

_REASON_MESSAGES = {
    "no_elevation_data": "This course has no elevation data, so gradients cannot be derived.",
    "course_too_short": "This course is too short to analyse.",
    "missing_rider_data": "Set your FTP and weight on your profile first — the pacing physics needs both.",
    "conflicting_targets": "Pace this course to a finish time or to an average power, not to both.",
}


def _reason_error(status_code: int, reason: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": reason, "message": _REASON_MESSAGES.get(reason, reason)},
    )


def _rider_params(athlete) -> RiderParams:
    if not athlete.ftp or not athlete.weight_kg:
        raise _reason_error(422, "missing_rider_data")
    return RiderParams(ftp_w=float(athlete.ftp), weight_kg=float(athlete.weight_kg))


async def _get_owned_course(course_id: str, athlete, session: AsyncSession) -> Course:
    result = await session.execute(
        select(Course).where(Course.id == course_id, Course.athlete_id == athlete.id)
    )
    course = result.scalar_one_or_none()
    if course is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return course


async def _get_owned_bike(bike_id: str, athlete, session: AsyncSession) -> Bike:
    result = await session.execute(
        select(Bike).where(Bike.id == bike_id, Bike.athlete_id == athlete.id)
    )
    bike = result.scalar_one_or_none()
    if bike is None:
        raise HTTPException(status_code=404, detail="Bike not found")
    return bike


async def _check_owned_goal(goal_id: str, athlete, session: AsyncSession) -> None:
    result = await session.execute(
        select(Goal).where(Goal.id == goal_id, Goal.athlete_id == athlete.id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Goal not found")


async def _detail(course_id: str, session: AsyncSession) -> CourseDetailResponse:
    # Re-select so the selectin-loaded segments are fresh after a commit — and
    # `populate_existing` is what makes that true. These sessions run with
    # `expire_on_commit=False`, so a plain re-select finds the row already in
    # the identity map and hands back its *loaded* collection untouched: after
    # a re-analysis that is the segment table from before the change, deleted
    # in the database but still hanging off the object. The response would
    # then carry the new summary numbers over the old splits.
    result = await session.execute(
        select(Course)
        .where(Course.id == course_id)
        .execution_options(populate_existing=True)
    )
    course = result.scalar_one()
    return CourseDetailResponse.model_validate(course)


@router.post("", response_model=CourseDetailResponse, status_code=201,
             dependencies=[Depends(require_consent)],
             operation_id="createCourse", summary="Upload and analyse a course")
@limiter.limit("30/hour")
async def create_course(
    request: Request,
    file: UploadFile = File(...),
    bike_id: str = Form(...),
    # Bounded like every other name in the tree (BikeCreate 100, activities
    # and workouts 200). It lands in every course response and verbatim in
    # the LLM prompt; the parsed fallback is already capped in the parser,
    # so this caller-supplied field was the only open-ended one.
    name: str | None = Form(None, min_length=1, max_length=200),
    goal_id: str | None = Form(None),
    # The two targets are alternatives, and the course can be re-pointed at
    # either one afterwards without re-uploading — see `reanalyze_course`.
    target_time_s: int | None = Form(None, gt=0),
    target_power_w: int | None = Form(None, gt=0),
    start_time: datetime | None = Form(None),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    if target_time_s is not None and target_power_w is not None:
        raise _reason_error(422, "conflicting_targets")
    bike = await _get_owned_bike(bike_id, athlete, session)
    if goal_id:
        await _check_owned_goal(goal_id, athlete, session)
    rider = _rider_params(athlete)

    gpx_bytes = await file.read(_MAX_COURSE_BYTES + 1)
    if len(gpx_bytes) > _MAX_COURSE_BYTES:
        raise HTTPException(status_code=413, detail="Course file too large")

    bike_params = BikeParams(
        tyre_width_mm=bike.tyre_width_mm, riding_position=bike.riding_position
    )
    try:
        parsed, reason = await asyncio.to_thread(
            course_analysis.parse_and_analyze,
            gpx_bytes,
            rider,
            bike_params,
            target_time_s,
            target_power_w,
        )
    except ActivityParseError as exc:
        raise HTTPException(status_code=400, detail=f"Not a readable GPX file: {exc}")
    if parsed is None:
        raise _reason_error(422, reason)

    course_id = str(uuid.uuid4())
    key = course_analysis.store_course_blob(gpx_bytes, ctx.user_id, course_id)

    # From here the blob exists but nothing references it yet. Both delete and
    # the GDPR export iterate Course rows, so a blob whose row never lands is
    # unreachable by either — and this feature's contract is that deleting a
    # course removes the rows *and* the file. Anything that fails before the
    # commit takes the file with it.
    try:
        course = Course(
            id=course_id,
            athlete_id=athlete.id,
            name=name or parsed.name or "Course",
            goal_id=goal_id,
            bike_id=bike.id,
            gpx_file_key=key,
            # True by construction, not by luck: `store_course_blob` either
            # returns with the file encrypted or raises having removed it.
            gpx_file_encrypted=True,
            target_time_s=target_time_s,
            target_power_w=target_power_w,
            start_time=start_time,
            distance_m=parsed.analysis.total_distance_m,
        )
        session.add(course)
        await session.flush()
        await course_analysis.persist_analysis(course, parsed.analysis, session, rider=rider)
        await course_analysis.store_track(course, parsed.track, session)
        await session.commit()
    except Exception:
        await session.rollback()
        course_analysis.delete_blob_by_key(key, ctx.user_id)
        raise
    return await _detail(course_id, session)


@router.get("", response_model=Page[CourseSummaryResponse],
            operation_id="listCourses", summary="List courses")
async def list_courses(
    ctx_athlete=Depends(get_ctx_session_athlete),
    params: PageParams = Depends(paginate_params),
):
    ctx, session, athlete = ctx_athlete
    from sqlalchemy import func

    total = (await session.execute(
        select(func.count()).select_from(Course).where(Course.athlete_id == athlete.id)
    )).scalar_one()
    result = await session.execute(
        select(Course)
        .where(Course.athlete_id == athlete.id)
        .order_by(Course.created_at.desc())
        .offset(params.offset)
        .limit(params.page_size)
    )
    items = [CourseSummaryResponse.model_validate(c) for c in result.scalars().all()]
    return Page.build(items, total, params.page, params.page_size)


@router.get("/{course_id}", response_model=CourseDetailResponse,
            operation_id="getCourse", summary="Get a course with its segment table")
async def get_course(
    course_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    course = await _get_owned_course(course_id, athlete, session)
    return CourseDetailResponse.model_validate(course)


@router.delete("/{course_id}", status_code=204,
               operation_id="deleteCourse", summary="Delete a course")
async def delete_course(
    course_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    course = await _get_owned_course(course_id, athlete, session)
    # The blob first — rows and file must go together, and the file is the
    # copy delete exists to remove.
    course_analysis.delete_course_blob(course, ctx.user_id)
    # Explicit child deletes: PRAGMA foreign_keys is off on these connections,
    # so the declared cascades document intent rather than execute it.
    await session.execute(delete(CourseSegment).where(CourseSegment.course_id == course.id))
    await session.execute(delete(CourseTrackRow).where(CourseTrackRow.course_id == course.id))
    await session.execute(delete(Course).where(Course.id == course.id))
    await session.commit()


@router.post("/{course_id}/reanalyze", response_model=CourseDetailResponse,
             dependencies=[Depends(require_consent)],
             operation_id="reanalyzeCourse", summary="Re-analyse a course without re-upload")
@limiter.limit("60/hour")
async def reanalyze_course(
    request: Request,
    course_id: str,
    body: CourseReanalyzeBody,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    course = await _get_owned_course(course_id, athlete, session)

    updates = body.model_dump(exclude_unset=True)
    if "bike_id" in updates and updates["bike_id"]:
        await _get_owned_bike(updates["bike_id"], athlete, session)
    if "goal_id" in updates and updates["goal_id"]:
        await _check_owned_goal(updates["goal_id"], athlete, session)

    # Re-pointing a course at the other kind of target is one request, not
    # two: setting a target time clears any target power and vice versa. A
    # request naming both is refused rather than resolved by precedence —
    # there is no reading of it that is obviously what the athlete meant.
    setting_time = updates.get("target_time_s") is not None
    setting_power = updates.get("target_power_w") is not None
    if setting_time and setting_power:
        raise _reason_error(422, "conflicting_targets")
    if setting_time:
        updates["target_power_w"] = None
    elif setting_power:
        updates["target_time_s"] = None

    for field, value in updates.items():
        setattr(course, field, value)

    if not course.bike_id:
        raise HTTPException(
            status_code=409,
            detail={"code": "no_bike", "message": "Pick a bike for this course first."},
        )
    bike = await _get_owned_bike(course.bike_id, athlete, session)
    rider = _rider_params(athlete)

    track_row = await session.get(CourseTrackRow, course.id)
    if track_row is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "no_stored_track", "message": "This course has no stored track."},
        )

    bike_params = BikeParams(
        tyre_width_mm=bike.tyre_width_mm, riding_position=bike.riding_position
    )
    analysis, reason = await asyncio.to_thread(
        course_analysis.analyze_stored_track,
        track_row.points,
        rider,
        bike_params,
        course.target_time_s,
        course.target_power_w,
    )
    if analysis is None:
        raise _reason_error(422, reason)

    await course_analysis.persist_analysis(course, analysis, session, rider=rider)
    await session.commit()
    return await _detail(course_id, session)


@router.get("/{course_id}/plan", response_model=CoursePlanResponse,
            operation_id="getCoursePlan", summary="Get the written pacing plan")
async def get_course_plan(
    course_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    course = await _get_owned_course(course_id, athlete, session)

    # Stuck-pending recovery, identical to goal guidance: the window is an
    # inactivity budget — the generator touches the timestamp on every
    # progress commit — so a slow but healthy stream is not declared dead.
    if course.plan_status == "pending":
        now_utc = datetime.now(timezone.utc)
        if pending_timed_out(course.plan_updated_at, now_utc):
            settle_course_plan(course, now_utc)
            await session.commit()

    return CoursePlanResponse(
        status=course.plan_status,
        mood=course.plan_mood,
        plan=course.plan,
        updated_at=course.plan_updated_at,
    )


@router.post("/{course_id}/plan", status_code=202, dependencies=[pat_forbidden()],
             operation_id="triggerCoursePlan", summary="Trigger the written pacing plan")
async def trigger_course_plan(
    course_id: str,
    body: CoursePlanBody = CoursePlanBody(),
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete
    course = await _get_owned_course(course_id, athlete, session)
    if course.status != "ready":
        raise HTTPException(status_code=409, detail="Course is not ready for a plan")

    # Issue #9 gate — a course plan is one more instance-paid caller.
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error(access)

    if course.plan_status == "pending":
        return {"status": "pending"}

    # The token this run owns its columns by. Re-analysis (or a settle) clears
    # it, and the generator then discards its own writes rather than putting a
    # plan for the old segment table back on the row.
    run_id = str(uuid.uuid4())
    course.plan_status = "pending"
    course.plan = None
    course.plan_mood = None
    course.plan_run_id = run_id
    course.plan_updated_at = datetime.now(timezone.utc)
    await session.commit()

    from backend.app.services.llm_course_plan import generate_course_plan_bg
    asyncio.create_task(
        generate_course_plan_bg(athlete.id, course.id, ctx.user_id, body.locale, run_id)
    )
    return {"status": "pending"}
