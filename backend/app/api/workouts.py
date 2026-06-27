import uuid
from datetime import datetime, time, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.deps import get_ctx_and_session
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import ProviderConnection
from backend.app.models.team_orm import Athlete, WahooWorkoutUpload, WorkoutDefinition
from backend.app.schemas.workouts import (
    ExportFormatInfo,
    WahooPushRequest,
    WahooPushResponse,
    WorkoutDefinitionCreate,
    WorkoutDefinitionResponse,
    WorkoutDefinitionUpdate,
)
from backend.app.services.provider_sync import ensure_fresh_token
from backend.app.services.providers.wahoo import WahooClient, workout_type_id_for
from openkoutsi.workout_estimator import estimate_duration_s, estimate_tss
from openkoutsi.workout_formats.registry import EXPORTERS
from openkoutsi.workout_formats.wahoo_plan import build_wahoo_plan
from openkoutsi.workout_schema import WorkoutStepOrRepeat

router = APIRouter(prefix="/workouts", tags=["workouts"])

# Wahoo only displays plans attached to a workout scheduled today → +6 days.
_WAHOO_VISIBILITY_DAYS = 6


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(
        select(Athlete).where(Athlete.global_user_id == global_user_id)
    )
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


async def _get_workout(
    workout_id: str, athlete_id: str, session: AsyncSession
) -> WorkoutDefinition:
    result = await session.execute(
        select(WorkoutDefinition).where(
            WorkoutDefinition.id == workout_id,
            WorkoutDefinition.athlete_id == athlete_id,
        )
    )
    workout = result.scalar_one_or_none()
    if workout is None:
        raise HTTPException(status_code=404, detail="Workout not found")
    return workout


def _validate_steps(steps_raw: list[WorkoutStepOrRepeat]) -> list[dict]:
    """Validate and serialise steps, enforcing max repeat-nesting depth of 1."""
    from openkoutsi.workout_schema import RepeatBlock

    for step in steps_raw:
        if isinstance(step, RepeatBlock) and step.max_depth() > 1:
            raise HTTPException(
                status_code=422,
                detail="Repeat blocks may not contain nested repeat blocks.",
            )
    return [s.model_dump() for s in steps_raw]


@router.get("/export/formats", response_model=list[ExportFormatInfo])
async def list_export_formats():
    return [
        ExportFormatInfo(
            key=cls.meta.key,
            label=cls.meta.label,
            file_extension=cls.meta.file_extension,
            mime_type=cls.meta.mime_type,
        )
        for cls in EXPORTERS.values()
    ]


@router.get("/", response_model=list[WorkoutDefinitionResponse])
async def list_workouts(ctx_session=Depends(get_ctx_and_session)):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(WorkoutDefinition)
        .where(WorkoutDefinition.athlete_id == athlete.id)
        .order_by(WorkoutDefinition.created_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=WorkoutDefinitionResponse, status_code=201)
async def create_workout(
    body: WorkoutDefinitionCreate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    steps = _validate_steps(body.steps)
    workout = WorkoutDefinition(
        id=str(uuid.uuid4()),
        athlete_id=athlete.id,
        name=body.name,
        description=body.description,
        sport_type=body.sport_type,
        steps=steps,
        estimated_duration_s=estimate_duration_s(steps),
        estimated_tss=estimate_tss(steps, athlete.ftp),
    )
    session.add(workout)
    await session.commit()
    await session.refresh(workout)
    return workout


@router.get("/{workout_id}", response_model=WorkoutDefinitionResponse)
async def get_workout(
    workout_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    return await _get_workout(workout_id, athlete.id, session)


@router.put("/{workout_id}", response_model=WorkoutDefinitionResponse)
async def update_workout(
    workout_id: str,
    body: WorkoutDefinitionUpdate,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    workout = await _get_workout(workout_id, athlete.id, session)

    update = body.model_dump(exclude_unset=True)
    if "steps" in update:
        update["steps"] = _validate_steps(body.steps)
        update["estimated_duration_s"] = estimate_duration_s(update["steps"])
        update["estimated_tss"] = estimate_tss(update["steps"], athlete.ftp)

    for field, value in update.items():
        setattr(workout, field, value)

    await session.commit()
    await session.refresh(workout)
    return workout


@router.delete("/{workout_id}", status_code=204)
async def delete_workout(
    workout_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    workout = await _get_workout(workout_id, athlete.id, session)
    await session.delete(workout)
    await session.commit()


@router.get("/{workout_id}/export/{format_key}")
async def export_workout(
    workout_id: str,
    format_key: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    workout = await _get_workout(workout_id, athlete.id, session)

    exporter_cls = EXPORTERS.get(format_key)
    if exporter_cls is None:
        raise HTTPException(status_code=404, detail=f"Unknown export format: {format_key}")

    exporter = exporter_cls()
    try:
        data = exporter.export(
            steps=workout.steps,
            workout_name=workout.name,
            workout_description=workout.description,
            athlete_ftp=athlete.ftp,
            athlete_power_zones=athlete.power_zones,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in workout.name)
    filename = f"{safe_name}.{exporter_cls.meta.file_extension}"
    return Response(
        content=data,
        media_type=exporter_cls.meta.mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{workout_id}/push/wahoo", response_model=WahooPushResponse)
async def push_workout_to_wahoo(
    workout_id: str,
    body: WahooPushRequest,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    """Push a structured workout to Wahoo as a plan + scheduled workout pair.

    Creates (or updates, by ``external_id``) a plan in the athlete's Wahoo
    library and a workout scheduled within the today→+6 day visibility window so
    it appears under "Planned Workouts" on ELEMNT / RIVAL devices.
    """
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    workout = await _get_workout(workout_id, athlete.id, session)

    # Resolve the Wahoo connection and a fresh access token.
    conn_result = await registry_session.execute(
        select(ProviderConnection).where(
            ProviderConnection.user_id == ctx.user_id,
            ProviderConnection.provider == "wahoo",
        )
    )
    conn = conn_result.scalar_one_or_none()
    if conn is None:
        raise HTTPException(status_code=400, detail="wahoo is not connected")
    access_token = await ensure_fresh_token(conn, registry_session)

    # Validate the schedule falls within Wahoo's visibility window.
    starts = body.starts or datetime.now(timezone.utc)
    if starts.tzinfo is None:
        starts = starts.replace(tzinfo=timezone.utc)
    today = datetime.now(timezone.utc).date()
    window_end = datetime.combine(
        today + timedelta(days=_WAHOO_VISIBILITY_DAYS), time.max, tzinfo=timezone.utc
    )
    if not (today <= starts.date() and starts <= window_end):
        raise HTTPException(
            status_code=422,
            detail=f"starts must be within today and {_WAHOO_VISIBILITY_DAYS} days from now",
        )

    plan_json = build_wahoo_plan(
        steps=workout.steps,
        workout_name=workout.name,
        workout_description=workout.description,
        sport_type=workout.sport_type,
        athlete_ftp=athlete.ftp,
        athlete_power_zones=athlete.power_zones,
    )

    external_id = f"okoutsi-wd-{workout.id}"

    # Look up any prior upload so we update in place instead of duplicating.
    upload_result = await session.execute(
        select(WahooWorkoutUpload).where(
            WahooWorkoutUpload.athlete_id == athlete.id,
            WahooWorkoutUpload.external_id == external_id,
        )
    )
    upload = upload_result.scalar_one_or_none()

    minutes = max(1, round((workout.estimated_duration_s or 0) / 60)) or 1
    client = WahooClient()
    try:
        plan_id = await client.create_or_update_plan(
            access_token,
            plan_json=plan_json,
            external_id=external_id,
            provider_updated_at=workout.updated_at,
            filename=f"{external_id}.json",
        )
        wahoo_workout_id = await client.create_or_update_workout(
            access_token,
            name=workout.name,
            workout_token=external_id,
            workout_type_id=workout_type_id_for(workout.sport_type),
            starts=starts,
            minutes=minutes,
            plan_id=plan_id,
            existing_id=upload.wahoo_workout_id if upload else None,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            raise HTTPException(status_code=403, detail="insufficient_scope")
        raise HTTPException(status_code=502, detail="Wahoo API error during push")

    if upload is None:
        upload = WahooWorkoutUpload(
            athlete_id=athlete.id,
            workout_definition_id=workout.id,
            external_id=external_id,
        )
        session.add(upload)
    upload.wahoo_plan_id = plan_id
    upload.wahoo_workout_id = wahoo_workout_id
    upload.starts = starts
    upload.provider_updated_at = workout.updated_at
    await session.commit()

    return WahooPushResponse(plan_id=plan_id, workout_id=wahoo_workout_id, starts=starts)
