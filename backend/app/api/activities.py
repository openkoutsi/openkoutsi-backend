import asyncio
import shutil
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.consent import require_consent
from backend.app.core.auth import get_current_user
from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.file_encryption import decrypt_file, encrypt_file
from backend.app.db.registry import get_registry_session
from backend.app.db.user_session import get_user_session_factory
from backend.app.models.user_orm import (
    Activity,
    ActivityDistanceBest,
    ActivityInterval,
    ActivityPowerBest,
    ActivitySource,
    ActivityStream,
    Athlete,
    Bike,
    ImportJob,
)
from backend.app.schemas.imports import (
    ImportJobListResponse,
    ImportJobResponse,
    ImportJobSummary,
)
from backend.app.schemas.activities import (
    ActivityDetailResponse,
    ActivityListResponse,
    ActivityResponse,
    ActivityStreamsResponse,
    ActivityUpdate,
    AnalyzeBody,
    FrontendAnalysisBody,
    IntervalResponse,
    ManualActivityCreate,
    RpeQueueResponse,
    CommuteScanResponse,
    CommuteRuleProposal,
    CommuteFeedback,
)
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_forbidden, pat_scopes
from backend.app.services.activity_import import (
    is_in_flight,
    reap_stale_staging,
    run_import_job,
)
from backend.app.services.fit_processor import process_fit_file, read_fit_start_time
from backend.app.services.metrics_engine import recalculate_from
from backend.app.services.pr_detection import detect_pr_badges
from backend.app.services.stranded_runs import (
    begin_activity_analysis_run,
    begin_training_status_run,
    pending_timed_out,
    settle_activity_analysis_if_timed_out,
)
from backend.app.services.provider_sync import (
    _add_distance_bests,
    _add_power_bests,
    _source_priority,
    activity_create_guard,
    rebuild_intervals,
)
from backend.app.services.weight import load_weight_log
from backend.app.services.aerobic_metrics import apply_aerobic_metrics, replace_w_bal_stream
from backend.app.services import commute as commute_service
from backend.app.services import garage as garage_service
from openkoutsi.commute import MIN_SAMPLES_FOR_PROPOSAL, propose_rule
from openkoutsi.training_math import calculate_load, variability_index
from openkoutsi.categorization import WorkoutCategory, classify_workout
from openkoutsi.sport_matching import CYCLING_SPORT_TYPES
from backend.app.services.activity_workout_matcher import find_and_link_workout
from backend.app.services.plan_adherence import catch_up_adherence
from backend.app.services.achievements import mark_achievements_dirty

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
# The bulk-import request cap, which is a different question from the
# single-upload one: this is a whole training history arriving at once. A Strava
# export of a decade of daily riding is well under this; the archive's
# *expanded* size is bounded separately in `services.activity_archive`, because
# a compressed size bounds nothing on its own.
_MAX_IMPORT_BYTES = 500 * 1024 * 1024  # 500 MB
_FIT_MAGIC = b".FIT"
_DUPLICATE_WINDOW = timedelta(minutes=5)
_VALID_LABELS = {"race", "commute"}

log = logging.getLogger(__name__)


def _import_source_name(files: list[UploadFile]) -> str:
    """A label for what the athlete handed over, for the list of past imports."""
    names = [f.filename for f in files if f.filename]
    if len(names) == 1:
        return names[0]
    if names:
        return f"{len(files)} files"
    return f"{len(files)} uploads"


router = APIRouter(
    prefix="/activities",
    tags=["activities"],
    dependencies=[pat_scopes(read="activities:read", write="activities:write")],
)


def _has_pending_suggestion_clause(label: str):
    """Correlated EXISTS: is ``label`` suggested on this Activity and unanswered?

    ``label_suggestions`` is a JSON object keyed by label, so unlike the array
    membership test below this reads a nested field: ``json_extract`` pulls
    ``$.<label>.state`` and we compare it to ``pending``. Activities with a NULL
    column, no entry for this label, or an answered one all fail the comparison
    and are excluded, which is exactly the set the review screen wants.
    """
    return (
        func.json_extract(Activity.label_suggestions, f"$.{label}.state") == "pending"
    )


def _has_label_clause(label: str):
    """Correlated EXISTS: does this Activity's JSON ``labels`` array contain ``label``?

    ``labels`` is a JSON list column; SQLite's ``json_each`` table-valued
    function lets us test array membership. Activities with a NULL/empty
    ``labels`` produce no rows and so never match.
    """
    entries = func.json_each(Activity.labels).table_valued("value")
    return (
        select(1)
        .select_from(entries)
        .where(entries.c.value == label)
        .correlate(Activity)
        .exists()
    )


def _maybe_auto_analyze(activity: Activity, athlete: Athlete) -> Optional[str]:
    """Claim this activity's analysis columns if the athlete opted in.

    Returns the run token, or ``None`` when auto-analysis is off. Like
    ``_maybe_auto_training_status`` below, this only marks the row — the caller
    commits and *then* starts the task, so the task cannot write a result before
    the `pending` state it is answering has been persisted.
    """
    if not (athlete.app_settings or {}).get("auto_analyze"):
        return None
    return begin_activity_analysis_run(activity)


def _maybe_auto_training_status(athlete: Athlete) -> Optional[str]:
    """Marks athlete as pending for training status analysis if eligible.

    Returns the run token the caller must hand to
    ``analyze_training_status_bg``, or ``None`` when it did not claim the row.
    The caller must commit the session and start the task *after* the commit,
    to avoid a race where the task writes "error" before the pending state is
    persisted.
    """
    if (athlete.app_settings or {}).get("auto_training_status") and athlete.training_status_status != "pending":
        return begin_training_status_run(athlete)
    return None


async def _bg_process_and_recalculate(
    file_path: str, athlete_id: str, activity_id: str,
    user_id: str, global_user_id: str,
) -> None:
    async with get_user_session_factory(user_id)() as session:
        athlete_result = await session.execute(
            select(Athlete).where(Athlete.id == athlete_id)
        )
        athlete = athlete_result.scalar_one()

        activity_result = await session.execute(
            select(Activity).where(Activity.id == activity_id)
        )
        activity = activity_result.scalar_one()

        src_result = await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
        upload_src = src_result.scalar_one()

        try:
            await process_fit_file(file_path, athlete, activity, session)

            target_act = activity
            if activity.start_time is not None:
                existing_result = await session.execute(
                    select(Activity).where(
                        Activity.athlete_id == athlete_id,
                        Activity.id != activity_id,
                        Activity.start_time >= activity.start_time - _DUPLICATE_WINDOW,
                        Activity.start_time <= activity.start_time + _DUPLICATE_WINDOW,
                    )
                )
                existing_act = existing_result.scalar_one_or_none()

                if existing_act is not None:
                    for attr in (
                        "name", "sport_type", "start_time", "duration_s", "distance_m",
                        "elevation_m", "avg_power", "weighted_power", "avg_hr", "max_hr",
                        "avg_speed_ms", "avg_cadence", "load", "intensity",
                        "workout_category", "status",
                    ):
                        setattr(existing_act, attr, getattr(activity, attr))

                    await session.execute(
                        delete(ActivityStream).where(ActivityStream.activity_id == existing_act.id)
                    )
                    await session.execute(
                        delete(ActivityPowerBest).where(ActivityPowerBest.activity_id == existing_act.id)
                    )
                    await session.execute(
                        delete(ActivityDistanceBest).where(ActivityDistanceBest.activity_id == existing_act.id)
                    )
                    await session.execute(
                        delete(ActivityInterval).where(ActivityInterval.activity_id == existing_act.id)
                    )
                    await session.flush()

                    await session.execute(
                        update(ActivityStream)
                        .where(ActivityStream.activity_id == activity_id)
                        .values(activity_id=existing_act.id)
                    )
                    await session.execute(
                        update(ActivityPowerBest)
                        .where(ActivityPowerBest.activity_id == activity_id)
                        .values(activity_id=existing_act.id)
                    )
                    await session.execute(
                        update(ActivityDistanceBest)
                        .where(ActivityDistanceBest.activity_id == activity_id)
                        .values(activity_id=existing_act.id)
                    )
                    await session.execute(
                        update(ActivityInterval)
                        .where(ActivityInterval.activity_id == activity_id)
                        .values(activity_id=existing_act.id)
                    )
                    await session.flush()

                    upload_src.activity_id = existing_act.id
                    await session.flush()

                    await session.execute(
                        delete(Activity).where(Activity.id == activity_id)
                    )
                    await session.flush()
                    target_act = existing_act

            start_date = (
                target_act.start_time.date()
                if target_act.start_time and hasattr(target_act.start_time, "date")
                else date.today()
            )

            try:
                encrypt_file(Path(file_path), user_id)
                upload_src.fit_file_encrypted = True
            except Exception:
                log.warning(
                    "Failed to encrypt FIT file %s — left in plaintext",
                    file_path,
                    exc_info=True,
                )

            # Issue #9: on a gated instance, skip the instance-paid auto hooks for
            # denied users (their settings stay saved but inert — debug log only).
            from backend.app.services.llm_access import auto_analysis_allowed
            llm_ok = await auto_analysis_allowed(global_user_id, athlete)
            if not llm_ok and (athlete.app_settings or {}).get("auto_analyze"):
                log.debug("Auto-analyze skipped for user %s — LLM access denied", global_user_id)

            analysis_run = _maybe_auto_analyze(target_act, athlete) if llm_ok else None
            status_run = _maybe_auto_training_status(athlete) if llm_ok else None
            await session.commit()
            # Both tasks start only after the commit, so neither can settle a
            # `pending` state that is not yet on disk.
            if analysis_run is not None:
                from backend.app.services.llm_activity_analyzer import analyze_activity_bg
                asyncio.create_task(
                    analyze_activity_bg(
                        target_act.id, athlete.id, user_id, run_id=analysis_run
                    )
                )
            if status_run is not None:
                from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
                asyncio.create_task(
                    analyze_training_status_bg(athlete.id, user_id, run_id=status_run)
                )
            await find_and_link_workout(session, athlete_id, target_act)
            await recalculate_from(athlete_id, start_date, session)
            await catch_up_adherence(athlete_id, session)
            await mark_achievements_dirty(athlete_id, session)

        except Exception:
            try:
                err_result = await session.execute(
                    select(Activity).where(Activity.id == activity_id)
                )
                err_act = err_result.scalar_one_or_none()
                if err_act is not None:
                    err_act.status = "error"
                    await session.commit()
            except Exception:
                pass
            raise


async def _bg_recalculate(athlete_id: str, from_date: date, user_id: str) -> None:
    async with get_user_session_factory(user_id)() as session:
        await recalculate_from(athlete_id, from_date, session)


async def _bg_attach_fit_and_reprocess(
    file_path: str, activity_id: str, user_id: str, global_user_id: str,
) -> None:
    """After attaching a user-uploaded FIT to an existing synced activity,
    replace its intervals with lap data from the device file."""
    from backend.app.core.file_encryption import encrypt_file

    async with get_user_session_factory(user_id)() as session:
        act_result = await session.execute(select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one_or_none()
        if activity is None:
            return

        streams_result = await session.execute(
            select(ActivityStream).where(ActivityStream.activity_id == activity_id)
        )
        stream_map = {s.stream_type: s.data for s in streams_result.scalars()}
        await rebuild_intervals(activity, session, file_path, stream_map, replace=True)

        try:
            src_result = await session.execute(
                select(ActivitySource).where(
                    ActivitySource.activity_id == activity_id,
                    ActivitySource.provider == "upload",
                )
            )
            upload_src = src_result.scalar_one()
            encrypt_file(Path(file_path), user_id)
            upload_src.fit_file_encrypted = True
        except Exception:
            log.warning("Failed to encrypt attached FIT file %s", file_path, exc_info=True)

        await session.commit()


@router.post("/upload", response_model=ActivityResponse, status_code=201,
             dependencies=[Depends(require_consent)])
@limiter.limit("30/hour")
async def upload_activity(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    storage_dir = settings.user_fit_dir(ctx.user_id)
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / f"{uuid.uuid4()}.fit"

    written = 0
    with file_path.open("wb") as out:
        while True:
            chunk = await file.read(65536)
            if not chunk:
                break
            written += len(chunk)
            if written > _MAX_UPLOAD_BYTES:
                file_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                )
            out.write(chunk)

    with file_path.open("rb") as f:
        header = f.read(12)
    if len(header) < 12 or header[8:12] != _FIT_MAGIC:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not a valid FIT file")

    # Off the event loop: `getStartTime` stops at the first timestamped record,
    # so it is far cheaper than a full parse — but "cheap" is not "bounded". A
    # file whose first record sits behind a long header still costs real time,
    # and this runs inline in the request handler, where every millisecond is
    # one every other request in the process waits (issue #101 §2.1, #102 F-05).
    fit_start = await asyncio.to_thread(read_fit_start_time, str(file_path))

    # ── Find-or-create under the same two guards every other writer uses ─────
    #
    # A Wahoo webhook or Strava backfill landing while an athlete uploads the
    # same ride would have both writers see an empty window and each create an
    # activity. The window here is one request rather than the tens of minutes a
    # bulk import holds it open, which is why this was the last of the original
    # three to be closed.
    #
    # Both branches below commit inside the block, which is the invariant the
    # guard requires.
    attached_to: Optional[Activity] = None
    activity: Optional[Activity] = None

    async with activity_create_guard(session, ctx.user_id, athlete.id):
        if fit_start is not None:
            dupe_result = await session.execute(
                select(Activity).where(
                    Activity.athlete_id == athlete.id,
                    Activity.start_time >= fit_start - _DUPLICATE_WINDOW,
                    Activity.start_time <= fit_start + _DUPLICATE_WINDOW,
                )
            )
            duplicate = dupe_result.scalars().first()
            if duplicate is not None:
                already_uploaded = any(s.provider == "upload" for s in duplicate.sources)
                if already_uploaded:
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=409,
                        detail="An activity starting at this time already exists.",
                    )
                # Existing activity from a sync source — attach this FIT and
                # reprocess intervals.
                upload_src = ActivitySource(
                    activity_id=duplicate.id,
                    provider="upload",
                    fit_file_path=str(file_path),
                    format="fit",
                )
                session.add(upload_src)
                await session.commit()
                attached_to = duplicate

        if attached_to is None:
            activity = Activity(
                id=str(uuid.uuid4()),
                athlete_id=athlete.id,
                status="pending",
            )
            session.add(activity)
            await session.flush()

            upload_src = ActivitySource(
                activity_id=activity.id,
                provider="upload",
                fit_file_path=str(file_path),
                format="fit",
            )
            session.add(upload_src)
            await session.commit()
            await session.refresh(activity)

    # Scheduled outside the block: the work these do is the processing, not the
    # find-or-create, and holding the lease across it would serialise an upload
    # against every sync for the length of a FIT parse.
    if attached_to is not None:
        background_tasks.add_task(
            _bg_attach_fit_and_reprocess,
            str(file_path), attached_to.id, ctx.user_id, ctx.user_id,
        )
        return ActivityResponse.model_validate(attached_to)

    background_tasks.add_task(
        _bg_process_and_recalculate,
        str(file_path), athlete.id, activity.id, ctx.user_id, ctx.user_id,
    )

    return ActivityResponse.model_validate(activity)


@router.post("/import", response_model=ImportJobResponse, status_code=202,
             operation_id="importActivities",
             summary="Import many activity files, or an archive of them",
             dependencies=[Depends(require_consent)])
@limiter.limit("5/hour")
async def import_activities(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Activity files, .gz files, or .zip archives"),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Start a bulk import and return the job to poll (issue #36).

    Accepts `.fit`, `.gpx`, `.tcx`, any of those gzipped, and `.zip` archives
    containing a mix of them — which is exactly the shape of a Strava bulk
    export. The response is the job, not the activities: an export is thousands
    of files and tens of minutes of parsing, so the work happens in the
    background and the client polls `GET /activities/imports/{id}`.

    **On the rate limit.** The single-file upload is limited to 30/hour, which
    is what makes importing a history one file at a time impossible. This limit
    is on *jobs* rather than files, since a job is the unit of work an athlete
    asks for and may legitimately carry three thousand files. A second job while
    one is running is refused (409) rather than queued — two imports writing to
    one SQLite database interleave badly.
    """
    ctx, session, athlete = ctx_athlete

    if not files:
        raise HTTPException(status_code=422, detail="No files were uploaded")

    unfinished = await session.execute(
        select(ImportJob).where(
            ImportJob.athlete_id == athlete.id,
            ImportJob.status.in_(("pending", "running")),
        )
    )
    # `is_in_flight` rather than the status alone: a job whose process died
    # cannot clear its own status, and without the staleness check that athlete
    # could never import anything again.
    if any(is_in_flight(job) for job in unfinished.scalars()):
        raise HTTPException(
            status_code=409,
            detail="An import is already running. Wait for it to finish before starting another.",
        )

    # Committed *before* the request body is staged, so the check above is not
    # separated from the row it depends on by an upload that takes minutes. With
    # the insert last, two tabs — or a retry after a timeout the server is still
    # working through — both pass the check and both schedule a job, which is
    # exactly what this endpoint refuses to do. The `pending` status already
    # means "no files walked yet", so a row that exists during the upload fits
    # the model rather than bending it.
    job = ImportJob(
        id=str(uuid.uuid4()),
        athlete_id=athlete.id,
        status="pending",
        source_name=_import_source_name(files),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Anything left by an import whose process died is finished with; imports
    # run one at a time, so this is the natural moment to sweep it up.
    await reap_stale_staging(session, athlete.id, ctx.user_id, keep=job.id)

    work_dir = settings.user_fit_dir(ctx.user_id) / "imports" / job.id
    work_dir.mkdir(parents=True, exist_ok=True)

    uploads: list[tuple[Path, str]] = []
    written = 0
    try:
        for index, upload in enumerate(files):
            part_path = work_dir / f"part-{index}"
            with part_path.open("wb") as out:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MAX_IMPORT_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"Upload exceeds the {_MAX_IMPORT_BYTES // (1024 * 1024)} MB "
                                "limit. Split the archive and import it in parts."
                            ),
                        )
                    out.write(chunk)
            uploads.append((part_path, upload.filename or f"upload-{index}"))
    except Exception as exc:
        # The job row is already visible, so failing the upload has to settle it
        # — otherwise a 413 leaves an athlete with a `pending` job that never
        # runs and blocks the next import until it goes stale. It also gives
        # them something to look at, which "413 and nothing" did not.
        shutil.rmtree(work_dir, ignore_errors=True)
        job.status = "failed"
        job.error = exc.detail if isinstance(exc, HTTPException) else "Upload failed"
        job.completed_at = datetime.now(timezone.utc)
        await session.commit()
        raise

    background_tasks.add_task(
        run_import_job, job.id, athlete.id, ctx.user_id, uploads, work_dir,
    )
    return ImportJobResponse.model_validate(job)


@router.get("/imports", response_model=ImportJobListResponse,
            operation_id="listImports", summary="Recent bulk imports")
async def list_imports(
    limit: int = Query(20, ge=1, le=100),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Recent import jobs, newest first, without the per-file detail.

    The detail list can hold thousands of rows, which is worth fetching for the
    one job being looked at and not for a list of them.
    """
    ctx, session, athlete = ctx_athlete

    total_result = await session.execute(
        select(func.count()).select_from(
            select(ImportJob).where(ImportJob.athlete_id == athlete.id).subquery()
        )
    )
    result = await session.execute(
        select(ImportJob)
        .where(ImportJob.athlete_id == athlete.id)
        .order_by(ImportJob.created_at.desc())
        .limit(limit)
    )
    return ImportJobListResponse(
        items=[ImportJobSummary.model_validate(j) for j in result.scalars().all()],
        total=total_result.scalar_one(),
    )


@router.get("/imports/{import_id}", response_model=ImportJobResponse,
            operation_id="getImport", summary="Progress and per-file outcome of an import")
async def get_import(
    import_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(ImportJob).where(
            ImportJob.id == import_id, ImportJob.athlete_id == athlete.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Import not found")
    return ImportJobResponse.model_validate(job)


@router.post("", response_model=ActivityResponse, status_code=201,
             operation_id="createActivity", summary="Create a manual activity")
async def create_manual_activity(
    payload: ManualActivityCreate,
    background_tasks: BackgroundTasks,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    load: Optional[float] = None
    if payload.load is not None:
        load = payload.load
    elif payload.duration_s is not None and payload.rpe is not None:
        # rpe/avg_hr derivations both need a duration to scale by.
        load = (payload.duration_s / 3600) * (payload.rpe ** 2) * 10
    elif payload.duration_s is not None and payload.avg_hr is not None:
        load, _ = calculate_load(
            payload.duration_s, None, payload.avg_hr, None, athlete.max_hr
        )

    default_name = (
        f"{payload.sport_type} Activity" if payload.sport_type else "Manual Activity"
    )
    activity = Activity(
        id=str(uuid.uuid4()),
        athlete_id=athlete.id,
        name=payload.name or default_name,
        sport_type=payload.sport_type,
        start_time=payload.start_time,
        duration_s=payload.duration_s,
        avg_hr=payload.avg_hr,
        max_hr=payload.max_hr,
        avg_power=payload.avg_power,
        avg_cadence=payload.avg_cadence,
        distance_m=payload.distance_m,
        elevation_m=payload.elevation_m,
        load=load,
        rpe=payload.rpe,
        status="processed",
    )
    session.add(activity)
    await session.flush()

    manual_src = ActivitySource(activity_id=activity.id, provider="manual")
    session.add(manual_src)
    await session.commit()
    await session.refresh(activity)

    # A hand-logged ride still describes a commute, and this path has no
    # processing pass to hang the detector off (issue #63).
    if await commute_service.evaluate_activity(session, athlete, activity):
        await session.commit()

    # Same gap for the garage (issue #64): a ride logged by hand is a ride on a
    # bike, and leaving this path out is how a per-bike total ends up right for
    # rides that arrived from Strava and quietly short for the rest.
    if await garage_service.assign_bike(session, athlete, activity):
        await session.commit()
        await session.refresh(activity)

    # Workout matching and the fitness recalc are both keyed on the date.
    if payload.start_time is not None:
        await find_and_link_workout(session, athlete.id, activity)
        await catch_up_adherence(athlete.id, session)
        await mark_achievements_dirty(athlete.id, session)

    if load is not None and payload.start_time is not None:
        start_date = (
            payload.start_time.date()
            if hasattr(payload.start_time, "date")
            else payload.start_time
        )
        background_tasks.add_task(_bg_recalculate, athlete.id, start_date, ctx.user_id)

    return ActivityResponse.model_validate(activity)


@router.get("", response_model=ActivityListResponse,
            operation_id="listActivities", summary="List activities")
async def list_activities(
    q: Optional[str] = Query(None, description="Fuzzy search on activity name"),
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    sport_type: Optional[str] = Query(None),
    workout_category: Optional[str] = Query(None),
    labels: Optional[list[str]] = Query(
        None,
        description="Only include activities carrying at least one of these labels (e.g. race, commute)",
    ),
    suggested_label: Optional[str] = Query(
        None,
        description=(
            "Only activities with an unanswered suggestion for this label "
            "(issue #63). What the bulk-review screen filters on."
        ),
    ),
    exclude_labels: Optional[list[str]] = Query(
        None,
        description="Exclude activities carrying any of these labels (e.g. commute)",
    ),
    min_duration: Optional[int] = Query(None, ge=0, description="Minimum duration in seconds"),
    max_duration: Optional[int] = Query(None, ge=0, description="Maximum duration in seconds"),
    min_distance: Optional[float] = Query(None, ge=0, description="Minimum distance in meters"),
    max_distance: Optional[float] = Query(None, ge=0, description="Maximum distance in meters"),
    min_tss: Optional[float] = Query(None, ge=0, description="Minimum Load"),
    max_tss: Optional[float] = Query(None, ge=0, description="Maximum Load"),
    has_power: Optional[bool] = Query(None, description="Only activities with power data"),
    wahoo_device_only: bool = Query(False, alias="wahoo_device_only"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    base_query = select(Activity).where(Activity.athlete_id == athlete.id)
    if q:
        base_query = base_query.where(Activity.name.ilike(f"%{q}%"))
    if start:
        base_query = base_query.where(Activity.start_time >= datetime.combine(start, time.min))
    if end:
        base_query = base_query.where(Activity.start_time <= datetime.combine(end, time.max))
    if sport_type:
        base_query = base_query.where(Activity.sport_type == sport_type)
    if workout_category:
        base_query = base_query.where(Activity.workout_category == workout_category)
    if labels:
        bad = [lbl for lbl in labels if lbl not in _VALID_LABELS]
        if bad:
            raise HTTPException(status_code=422, detail=f"Unknown labels: {bad}")
        base_query = base_query.where(or_(*[_has_label_clause(lbl) for lbl in labels]))
    if suggested_label:
        if suggested_label not in _VALID_LABELS:
            raise HTTPException(
                status_code=422, detail=f"Unknown labels: [{suggested_label!r}]"
            )
        base_query = base_query.where(_has_pending_suggestion_clause(suggested_label))
    if exclude_labels:
        bad = [lbl for lbl in exclude_labels if lbl not in _VALID_LABELS]
        if bad:
            raise HTTPException(status_code=422, detail=f"Unknown labels: {bad}")
        base_query = base_query.where(
            and_(*[~_has_label_clause(lbl) for lbl in exclude_labels])
        )
    if min_duration is not None:
        base_query = base_query.where(Activity.duration_s >= min_duration)
    if max_duration is not None:
        base_query = base_query.where(Activity.duration_s <= max_duration)
    if min_distance is not None:
        base_query = base_query.where(Activity.distance_m >= min_distance)
    if max_distance is not None:
        base_query = base_query.where(Activity.distance_m <= max_distance)
    if min_tss is not None:
        base_query = base_query.where(Activity.load >= min_tss)
    if max_tss is not None:
        base_query = base_query.where(Activity.load <= max_tss)
    if has_power is True:
        base_query = base_query.where(Activity.avg_power.isnot(None))
    elif has_power is False:
        base_query = base_query.where(Activity.avg_power.is_(None))
    if wahoo_device_only:
        non_wahoo_exists = (
            select(ActivitySource.id)
            .where(
                ActivitySource.activity_id == Activity.id,
                ActivitySource.provider != "wahoo",
            )
            .exists()
        )
        base_query = base_query.where(
            or_(Activity.duration_s.isnot(None), non_wahoo_exists)
        )

    count_result = await session.execute(
        select(func.count()).select_from(base_query.subquery())
    )
    total = count_result.scalar_one()

    items_result = await session.execute(
        base_query.order_by(Activity.start_time.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [ActivityResponse.model_validate(a) for a in items_result.scalars().all()]
    return ActivityListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/rpe-queue", response_model=RpeQueueResponse,
            operation_id="getRpeQueue", summary="Pending RPE-rating queue")
async def get_rpe_queue(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Qualifying cycling activities still awaiting an RPE rating (issue #28).

    Returns activities that are cycling sports, ingested after the athlete's
    ``rpe_head`` cursor, and still lack an ``rpe`` — oldest-first (by ingestion
    ``created_at``). ``commute``-labelled rides are excluded so easy spins don't
    nag.

    A ride with a *pending* commute suggestion is deliberately still in the
    queue (issue #63): the prompt is where the athlete answers it, with the
    client reading ``label_suggestions`` off each item to pre-tick its "This was
    a commute" box. Only an answered suggestion, which by then has applied the
    label, takes a ride out.

    The cursor is stored in ``app_settings.rpe_head``. On the very first call
    (cursor unset) it is pinned to the athlete's most recent activity so the
    backlog of their entire history is *not* surfaced; only rides ingested from
    then on are prompted. Advancing the cursor is done by the caller via
    ``PATCH /api/athlete`` (set ``app_settings.rpe_head`` to the handled
    activity's ``created_at``).
    """
    ctx, session, athlete = ctx_athlete

    app_settings = dict(athlete.app_settings or {})
    rpe_head_raw = app_settings.get("rpe_head")

    if rpe_head_raw is None:
        # First load after the feature ships: pin the cursor to the most recent
        # activity so we don't ask the athlete to backfill their whole career.
        latest_result = await session.execute(
            select(func.max(Activity.created_at)).where(
                Activity.athlete_id == athlete.id
            )
        )
        latest_created_at = latest_result.scalar_one_or_none()
        rpe_head = (
            latest_created_at.isoformat()
            if latest_created_at is not None
            else datetime.now(timezone.utc).isoformat()
        )
        athlete.app_settings = {**app_settings, "rpe_head": rpe_head}
        await session.commit()
        return RpeQueueResponse(items=[], rpe_head=rpe_head)

    try:
        cursor = datetime.fromisoformat(rpe_head_raw)
    except (TypeError, ValueError):
        cursor = None

    query = select(Activity).where(
        Activity.athlete_id == athlete.id,
        Activity.sport_type.in_(CYCLING_SPORT_TYPES),
        Activity.rpe.is_(None),
        ~_has_label_clause("commute"),
    )
    if cursor is not None:
        query = query.where(Activity.created_at > cursor)
    query = query.order_by(Activity.created_at.asc())

    result = await session.execute(query)
    items = [ActivityResponse.model_validate(a) for a in result.scalars().all()]
    return RpeQueueResponse(items=items, rpe_head=rpe_head_raw)


# ── Commute detection (issue #63) ────────────────────────────────────────────
# Declared before `/{activity_id}`: FastAPI matches in declaration order, so a
# literal path has to come first or the path parameter swallows it.


@router.post("/commute/scan", response_model=CommuteScanResponse,
             operation_id="scanCommutes", summary="Look for commutes in the whole history")
# A full-history read plus a write transaction — at least as costly as an
# import, which carries the same limit two endpoints up. Nothing else stops a
# client, or a retry loop behind a slow response, calling it back to back.
@limiter.limit("5/hour")
async def scan_commutes(
    request: Request,
    force: bool = Query(
        False,
        description=(
            "Re-examine activities the athlete has already answered. Off by "
            "default: a dismissal is meant to be durable."
        ),
    ),
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Run the athlete's commute rules over their entire activity history.

    The answer to an imported back catalogue — a decade of rides that arrived
    with no provider flag and no labels, which the per-ingest hook will never
    see. Deliberately an explicit request rather than something a rule edit does
    on its own: it can touch tens of thousands of rows.

    Suggests; does not label. Only a rule the athlete marked ``auto_apply``
    writes to `labels`, and Strava's own flag, which is handled at sync time.
    """
    ctx, session, athlete = ctx_athlete
    result = await commute_service.scan_history(session, athlete, force=force)
    if result["applied"]:
        await mark_achievements_dirty(athlete.id, session)
    return CommuteScanResponse(**result)


@router.get("/commute/proposal", response_model=CommuteRuleProposal,
            operation_id="getCommuteRuleProposal",
            summary="A commute rule derived from your own labelled rides")
async def get_commute_rule_proposal(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Propose rule parameters from the commutes the athlete has already labelled.

    What makes rule configuration something other than a chore: nobody is going
    to hand-type "between 4.2 and 6.8 km, 06:41–08:12", but they will happily
    nudge those numbers once something has proposed them.

    Built from `labels`, never from our own suggestions — a proposal should be
    derived from what the athlete confirmed, not from what we guessed.
    """
    ctx, session, athlete = ctx_athlete
    samples = await commute_service.labelled_samples(session, athlete)
    rule = propose_rule(samples)
    return CommuteRuleProposal(
        rule=rule.as_dict() if rule else None,
        sample_count=len(samples),
        min_samples=MIN_SAMPLES_FOR_PROPOSAL,
    )


@router.get("/commute/feedback", response_model=CommuteFeedback,
            operation_id="getCommuteFeedback",
            summary="Where your commute rules look wrong")
async def get_commute_feedback(ctx_athlete=Depends(get_ctx_session_athlete)):
    """Rides the rules missed, and rules whose suggestions keep being dismissed.

    Two signals read straight off the suggestion column rather than kept as
    counters — `source` already records which rule fired, so there is nothing to
    keep in sync and nothing to drift.
    """
    ctx, session, athlete = ctx_athlete
    return CommuteFeedback(**await commute_service.rule_feedback(session, athlete))


@router.get("/{activity_id}", response_model=ActivityDetailResponse)
async def get_activity(
    activity_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Same recovery the training-status and goal-guidance cards have had all
    # along (issue #91): a run that has shown no progress for the whole budget
    # is not coming back, so show the error and its retry rather than a spinner
    # that never resolves.
    if settle_activity_analysis_if_timed_out(activity):
        await session.commit()

    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    streams = {s.stream_type: s.data for s in streams_result.scalars()}

    bests_result = await session.execute(
        select(ActivityPowerBest).where(ActivityPowerBest.activity_id == activity_id)
    )
    power_bests = {b.duration_s: b.power_w for b in bests_result.scalars()}

    dbests_result = await session.execute(
        select(ActivityDistanceBest).where(ActivityDistanceBest.activity_id == activity_id)
    )
    distance_bests = {b.distance_m: b.time_s for b in dbests_result.scalars()}

    ivs_result = await session.execute(
        select(ActivityInterval)
        .where(ActivityInterval.activity_id == activity_id)
        .order_by(ActivityInterval.interval_number)
    )
    intervals = [
        IntervalResponse.model_validate(iv, from_attributes=True)
        for iv in ivs_result.scalars()
    ]

    power_pr_badges, distance_pr_badges = await detect_pr_badges(
        athlete.id, activity_id, activity.start_time, activity.sport_type, session
    )

    return ActivityDetailResponse.from_orm_and_streams(
        activity, streams, power_bests, distance_bests, intervals,
        power_pr_badges=power_pr_badges,
        distance_pr_badges=distance_pr_badges,
    )


@router.get("/{activity_id}/streams", response_model=ActivityStreamsResponse)
async def get_activity_streams(
    activity_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    streams = {s.stream_type: s.data for s in streams_result.scalars()}
    return ActivityStreamsResponse(streams=streams)


@router.get("/{activity_id}/fit")
async def download_fit_file(
    activity_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Download the original activity file.

    Still spelled `/fit` — that is the path the web client has always called and
    the one an athlete's bookmark points at — but since issue #36 the file it
    serves is whatever was uploaded: a FIT, a GPX or a TCX. The extension of the
    downloaded file follows the source's stored format, so an imported GPX comes
    back as a GPX rather than as something that claims to be a FIT and is not.
    """
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    fit_sources = [s for s in activity.sources if s.fit_file_path]
    if not fit_sources:
        raise HTTPException(status_code=404, detail="No activity file for this activity")

    best = min(fit_sources, key=lambda s: _source_priority(s.provider, True))
    fit_path = Path(best.fit_file_path).resolve()
    expected_dir = settings.user_fit_dir(ctx.user_id).resolve()
    if not fit_path.is_relative_to(expected_dir):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not fit_path.exists():
        raise HTTPException(status_code=404, detail="Activity file not found on disk")

    safe_name = "".join(
        c if c.isalnum() or c in " _-" else "_"
        for c in (activity.name or activity.id)
    ).strip()
    filename = f"{safe_name}.{best.file_format}"

    if best.fit_file_encrypted:
        content = decrypt_file(fit_path, ctx.user_id)
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return FileResponse(
        path=str(fit_path),
        media_type="application/octet-stream",
        filename=filename,
    )


@router.post("/{activity_id}/reprocess", response_model=ActivityDetailResponse)
async def reprocess_activity(
    activity_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Recompute Weighted Power/Load/Intensity/bests/intervals from stored streams using current athlete settings."""
    import io
    from sqlalchemy import delete as sa_delete
    from openkoutsi.training_math import weighted_power, compute_torque_stream

    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")
    if activity.status != "processed":
        raise HTTPException(status_code=400, detail="Activity has not been processed yet")

    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    # Recomputed from the streams as stored, never re-parsed from the device
    # file. An activity keeps the streams it was ingested with, including their
    # shape: rides stored before issue #76 stay on the old dense convention,
    # where the index is a sample rather than a second, and reach the shared
    # clock only by being ingested again. Every consumer below reads both shapes.
    stream_map = {s.stream_type: s.data for s in streams_result.scalars()}

    power_data: list[float | None] = stream_map.get("power") or []
    speed_data: list[float | None] = stream_map.get("speed") or []
    cadence_data: list[float | None] = stream_map.get("cadence") or []

    # Derive and persist torque from stored power + cadence so activities
    # uploaded before torque existed pick it up on reprocess.
    torque_data = compute_torque_stream(power_data, cadence_data)
    await session.execute(
        sa_delete(ActivityStream).where(
            ActivityStream.activity_id == activity_id,
            ActivityStream.stream_type == "torque",
        )
    )
    if torque_data:
        session.add(
            ActivityStream(
                id=str(uuid.uuid4()),
                activity_id=activity_id,
                stream_type="torque",
                data=torque_data,
            )
        )
        stream_map["torque"] = torque_data
    else:
        stream_map.pop("torque", None)

    # Recompute Weighted Power, Load, Intensity from stored streams using current athlete FTP/max_HR
    wp = (
        weighted_power(power_data)
        if len(power_data) >= 30
        else (activity.avg_power)
    )
    load, intensity = calculate_load(
        activity.duration_s or 0,
        wp,
        activity.avg_hr,
        athlete.ftp,
        athlete.max_hr,
    )
    activity.load = load
    activity.intensity = intensity
    if wp is not None:
        activity.weighted_power = wp

    # Rebuild power bests. Carry the weight already snapshotted on this activity
    # across the rebuild, so reprocessing never re-attributes an old effort to a
    # weight the athlete only logged later. Only when the rows never had one (or
    # this is the first processing) do we look it up from the log.
    if power_data:
        prev = await session.execute(
            select(ActivityPowerBest.weight_kg)
            .where(
                ActivityPowerBest.activity_id == activity_id,
                ActivityPowerBest.weight_kg.is_not(None),
            )
            .limit(1)
        )
        weight = prev.scalar_one_or_none()
        await session.execute(
            sa_delete(ActivityPowerBest).where(ActivityPowerBest.activity_id == activity_id)
        )
        _add_power_bests(
            activity, athlete, session, power_data,
            await load_weight_log(athlete.id, session) if weight is None else None,
            weight=weight,
        )

    # Rebuild distance bests
    if speed_data:
        await session.execute(
            sa_delete(ActivityDistanceBest).where(ActivityDistanceBest.activity_id == activity_id)
        )
        _add_distance_bests(activity, athlete, session, speed_data)

    # Re-extract intervals from the original file, or auto-split when there is
    # none. Which parser reads the laps depends on the format the original was
    # stored in (issue #36): a TCX carries the athlete's own splits like a FIT
    # does, a GPX has no lap concept and always auto-splits.
    fileish = None
    fmt = "fit"
    fit_sources = [s for s in (activity.sources or []) if s.fit_file_path]
    if fit_sources:
        best = min(fit_sources, key=lambda s: _source_priority(s.provider, True))
        fmt = best.file_format
        fit_path = Path(best.fit_file_path).resolve()
        expected_dir = settings.user_fit_dir(ctx.user_id).resolve()
        if fit_path.is_relative_to(expected_dir) and fit_path.exists():
            if best.fit_file_encrypted:
                fileish = io.BytesIO(decrypt_file(fit_path, ctx.user_id))
            else:
                fileish = str(fit_path)

    await rebuild_intervals(activity, session, fileish, stream_map, replace=True, fmt=fmt)

    # Recalculate workout category
    vi = variability_index(activity.weighted_power, activity.avg_power)
    category = classify_workout(activity.intensity, vi)
    activity.workout_category = category.value if category else None

    # Re-derive the aerobic metrics and W' balance from the stored streams, so
    # activities processed before this existed pick them up on reprocess — the
    # same backfill route torque uses. Runs after the category is recalculated
    # (the decoupling gate reads it) and after the power bests are rebuilt (the
    # CP fit sees them via autoflush).
    w_bal_data = await apply_aerobic_metrics(activity, athlete, stream_map, session)
    await replace_w_bal_stream(session, activity_id, w_bal_data)
    if w_bal_data:
        stream_map["w_bal"] = w_bal_data
    else:
        stream_map.pop("w_bal", None)

    # Re-run the commute rules too, so a ride processed before the athlete wrote
    # a rule picks one up (issue #63). An answered suggestion is left alone:
    # `evaluate` refuses to overwrite a terminal state, which is what makes a
    # dismissal survive a reprocess rather than coming back every time.
    await commute_service.evaluate_activity(session, athlete, activity)

    # Re-run the bike mapping too, so a ride processed before the athlete
    # described their bikes picks one up (issue #64). A bike set by hand is
    # never overwritten — `bike_source` is what makes that correction durable
    # across exactly this call.
    await garage_service.assign_bike(session, athlete, activity)

    await session.commit()

    await find_and_link_workout(session, athlete.id, activity)
    await catch_up_adherence(athlete.id, session)
    await mark_achievements_dirty(athlete.id, session)

    # Update fitness metrics from this activity's date forward
    if activity.start_time is not None:
        act_date = (
            activity.start_time.date()
            if hasattr(activity.start_time, "date")
            else activity.start_time
        )
        await recalculate_from(athlete.id, act_date, session)

    bests_result = await session.execute(
        select(ActivityPowerBest).where(ActivityPowerBest.activity_id == activity_id)
    )
    power_bests = {b.duration_s: b.power_w for b in bests_result.scalars()}
    dbests_result = await session.execute(
        select(ActivityDistanceBest).where(ActivityDistanceBest.activity_id == activity_id)
    )
    distance_bests = {b.distance_m: b.time_s for b in dbests_result.scalars()}
    ivs_result = await session.execute(
        select(ActivityInterval)
        .where(ActivityInterval.activity_id == activity_id)
        .order_by(ActivityInterval.interval_number)
    )
    intervals = [
        IntervalResponse.model_validate(iv, from_attributes=True)
        for iv in ivs_result.scalars()
    ]
    power_pr_badges, distance_pr_badges = await detect_pr_badges(
        athlete.id, activity_id, activity.start_time, activity.sport_type, session
    )
    return ActivityDetailResponse.from_orm_and_streams(
        activity, stream_map, power_bests, distance_bests, intervals,
        power_pr_badges=power_pr_badges,
        distance_pr_badges=distance_pr_badges,
    )


@router.patch("/{activity_id}", response_model=ActivityResponse)
async def update_activity(
    activity_id: str,
    payload: ActivityUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    if payload.name is not None:
        activity.name = payload.name.strip()
    if "workout_category" in payload.model_fields_set:
        if payload.workout_category is None:
            activity.workout_category = None
        else:
            try:
                activity.workout_category = WorkoutCategory(payload.workout_category).value
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown workout category: {payload.workout_category}",
                )
    if "labels" in payload.model_fields_set:
        labels = payload.labels or []
        bad = [lbl for lbl in labels if lbl not in _VALID_LABELS]
        if bad:
            raise HTTPException(status_code=422, detail=f"Unknown labels: {bad}")
        # A label set by hand is the athlete's final word on this ride, so any
        # suggestion still hanging over it is answered by the same edit (issue
        # #63) — otherwise unticking a suggested label would leave the
        # suggestion pending and the ride would be proposed all over again.
        for label in _VALID_LABELS:
            state = commute_service.suggestion_state(activity, label)
            if state is None:
                # Nothing was ever suggested for this label, so there is nothing
                # to answer. Without this guard, hand-labelling a ride writes a
                # suggestion entry out of thin air — `{"labels": ["race"]}` on
                # an account with no rules at all would record a `race`
                # suggestion, and `race` has no detector. The column means
                # "labels openkoutsi thinks apply"; it must not become a log of
                # what the athlete applied by hand.
                continue
            if label in labels:
                if state != commute_service.STATE_ACCEPTED:
                    commute_service.answer_suggestion(
                        activity, label, commute_service.STATE_ACCEPTED
                    )
            else:
                # Any answered-or-pending suggestion, not just a pending one.
                # Un-ticking an *accepted* label is the athlete rejecting it
                # just as much as un-ticking a pending one, and leaving it at
                # `accepted` hides that rejection from `rule_feedback`, whose
                # "this rule is too wide" signal counts dismissals. The same
                # athlete action would otherwise produce different feedback
                # depending on whether the client sent `labels` or
                # `label_answers`.
                if state != commute_service.STATE_DISMISSED:
                    commute_service.answer_suggestion(
                        activity, label, commute_service.STATE_DISMISSED
                    )
        activity.labels = labels
    if "label_answers" in payload.model_fields_set:
        # Answering a suggestion without restating the whole label list. Accept
        # applies the label and records the answer in one write, so the two can
        # never drift apart.
        for label, answer in (payload.label_answers or {}).items():
            if label not in _VALID_LABELS:
                raise HTTPException(status_code=422, detail=f"Unknown labels: [{label!r}]")
            if answer not in (commute_service.STATE_ACCEPTED, commute_service.STATE_DISMISSED):
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown answer for {label}: {answer}",
                )
            if answer == commute_service.STATE_ACCEPTED:
                commute_service.apply_label(activity, label)
            else:
                commute_service.remove_label(activity, label)
            commute_service.answer_suggestion(activity, label, answer)
    if "bike_id" in payload.model_fields_set:
        # The athlete's correction of the automatic guess (issue #64). Stamping
        # `manual` is the whole mechanism: without it the next reprocess or
        # provider sync would put the guess straight back, which is precisely
        # the failure this override exists to prevent.
        if payload.bike_id is None:
            # "None of mine" is a choice, so it is stamped like any other. It
            # has to be: `(NULL, NULL)` is the exact predicate automapping
            # reads as free to fill, so clearing without the marker would let
            # the next reprocess or re-sync put the guess straight back — the
            # one correction that did not survive the events this override
            # exists to survive. The bike stays NULL, so no total moves.
            #
            # `delete_bike` deliberately writes both NULL instead: there the
            # bike is gone and no choice was made, so there is nothing to
            # record and automapping should be free to fill the gap.
            activity.bike_id = None
            activity.bike_source = garage_service.SOURCE_MANUAL
        else:
            bike = (
                await session.execute(
                    select(Bike).where(
                        Bike.id == payload.bike_id, Bike.athlete_id == athlete.id
                    )
                )
            ).scalar_one_or_none()
            if bike is None:
                raise HTTPException(status_code=404, detail="Bike not found")
            # Retired and unclaimed bikes are deliberately accepted here even
            # though they are hidden from the course picker: correcting an old
            # ride onto the bike it was actually done on is the point of the
            # override, and that bike is often exactly the one since sold.
            activity.bike_id = bike.id
            activity.bike_source = garage_service.SOURCE_MANUAL
    if "notes" in payload.model_fields_set:
        activity.notes = payload.notes
    if "rpe" in payload.model_fields_set:
        activity.rpe = payload.rpe

    await session.commit()
    await session.refresh(activity)
    # RPE, notes and labels all feed achievements, so an edit can earn (or
    # un-earn) a badge just as an upload can.
    await mark_achievements_dirty(athlete.id, session)
    return ActivityResponse.model_validate(activity)


@router.delete("/{activity_id}", status_code=204)
async def delete_activity(
    activity_id: str,
    background_tasks: BackgroundTasks,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete

    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    for src in activity.sources:
        if src.fit_file_path:
            p = Path(src.fit_file_path)
            if p.exists():
                p.unlink()

    start_date = (
        activity.start_time.date()
        if activity.start_time and hasattr(activity.start_time, "date")
        else None
    )

    await session.delete(activity)
    await session.commit()

    # Unlocks are derived state: losing the activity that earned a tier must
    # revoke it, not leave a badge the history no longer supports.
    await mark_achievements_dirty(athlete.id, session)

    if start_date:
        background_tasks.add_task(_bg_recalculate, athlete.id, start_date, ctx.user_id)


@router.post("/{activity_id}/analyze", status_code=202, dependencies=[pat_forbidden()])
async def trigger_analysis(
    activity_id: str,
    background_tasks: BackgroundTasks,
    body: AnalyzeBody = AnalyzeBody(),
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    from backend.app.services.llm_activity_analyzer import analyze_activity_bg
    from backend.app.models.registry_orm import InstanceSettings

    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    # Issue #9 gate (the analysis is always instance-paid).
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error(access)

    # A run still making progress owns the row; one that has gone quiet for the
    # whole budget does not, and before issue #91 this early return was
    # unconditional — an analysis whose process died left the activity
    # permanently un-analysable, because every route back in came through here.
    if activity.analysis_status == "pending" and not pending_timed_out(
        activity.analysis_updated_at
    ):
        return {"status": "pending"}

    # Claiming the row here is what supersedes a previous run: the token it was
    # holding is gone, so if its process is alive and merely slow it discards
    # its own writes rather than committing a stale answer over this one.
    run_id = begin_activity_analysis_run(activity)
    await session.commit()

    background_tasks.add_task(
        analyze_activity_bg,
        activity_id,
        athlete.id,
        ctx.user_id,
        body.locale,
        run_id=run_id,
    )
    return {"status": "pending"}


@router.patch("/{activity_id}/analysis", response_model=ActivityDetailResponse)
async def save_frontend_analysis(
    activity_id: str,
    body: FrontendAnalysisBody,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(Activity).where(Activity.id == activity_id, Activity.athlete_id == athlete.id)
    )
    activity = result.scalar_one_or_none()
    if activity is None:
        raise HTTPException(status_code=404, detail="Activity not found")

    activity.analysis = body.analysis
    activity.analysis_status = "done"
    await session.commit()

    streams_result = await session.execute(
        select(ActivityStream).where(ActivityStream.activity_id == activity_id)
    )
    streams = {s.stream_type: s.data for s in streams_result.scalars()}
    return ActivityDetailResponse.from_orm_and_streams(activity, streams)
