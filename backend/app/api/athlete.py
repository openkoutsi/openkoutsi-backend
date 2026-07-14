import asyncio
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session
from backend.app.core.file_encryption import decrypt_file
from backend.app.core.ssrf import check_url_safe
from backend.app.db.registry import get_registry_session
from backend.app.api.consent import CURRENT_CONSENT_VERSION
from backend.app.models.registry_orm import InstanceSettings, ProviderConnection, User
from backend.app.models.user_orm import Activity, Athlete, WeightLog
from backend.app.schemas.athlete import AthleteResponse, AthleteUpdate, TrainingStatusBody, TrainingStatusResponse
from backend.app.services.athlete_experience import VALID_EXPERIENCE_LEVELS

_MAX_AVATAR_BYTES = 5 * 1024 * 1024  # 5 MB

_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": "jpg",
    "image/png":  "png",
    "image/gif":  "gif",
    "image/webp": "webp",
}


def _detect_image_type(data: bytes) -> str | None:
    """Return MIME type by inspecting magic bytes; None if not a recognised image."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None

router = APIRouter(prefix="/athlete", tags=["athlete"])


async def _get_athlete(global_user_id: str, session: AsyncSession) -> Athlete:
    result = await session.execute(
        select(Athlete).where(Athlete.global_user_id == global_user_id)
    )
    athlete = result.scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=404, detail="Athlete profile not found")
    return athlete


_MAX_LLM_URL_LEN = 2048

# Self-reported athlete experience level, stored in app_settings (see #18) and
# fed into the LLM coaching/generation prompts (see #32). The canonical tuple
# lives in ``services.athlete_experience`` so validation here and prompt building
# there share one source.


def _validate_llm_base_url(raw: str) -> str:
    """Validate and normalise a user's BYOK base URL at save time.

    Fails fast in the UI (instead of at the first LLM call): strips whitespace,
    caps the length, requires an ``http(s)://`` scheme, enforces the allow-list,
    and runs the SSRF guard (which resolves DNS and blocks metadata ranges).
    """
    url = raw.strip()
    if len(url) > _MAX_LLM_URL_LEN:
        raise HTTPException(status_code=400, detail="LLM base URL is too long.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="LLM base URL must start with http:// or https://.",
        )
    allowed = settings.llm_allowed_servers_list
    if allowed and url.rstrip("/") not in {a.rstrip("/") for a in allowed}:
        raise HTTPException(
            status_code=400,
            detail="That LLM server is not in the server's allowed list.",
        )
    check_url_safe(url)
    return url


def _safe_app_settings(athlete: Athlete) -> dict:
    raw: dict = dict(athlete.app_settings or {})
    safe = {k: v for k, v in raw.items() if k != "llm_api_key_enc"}
    safe["llm_api_key_set"] = bool(raw.get("llm_api_key_enc"))
    return safe


def _athlete_response(
    athlete: Athlete, connected_providers: list[str], consent_accepted: bool = False
) -> AthleteResponse:
    avatar_url = (
        f"{settings.api_url}/api/public/users/{athlete.global_user_id}/avatar"
        if athlete.avatar_path
        else None
    )
    return AthleteResponse(
        id=athlete.id,
        user_id=athlete.global_user_id,
        name=athlete.name,
        date_of_birth=athlete.date_of_birth,
        weight_kg=athlete.weight_kg,
        ftp=athlete.ftp,
        max_hr=athlete.max_hr,
        resting_hr=athlete.resting_hr,
        hr_zones=athlete.hr_zones or [],
        power_zones=athlete.power_zones or [],
        ftp_tests=athlete.ftp_tests or [],
        connected_providers=connected_providers,
        app_settings=_safe_app_settings(athlete),
        avatar_url=avatar_url,
        created_at=athlete.created_at,
        updated_at=athlete.updated_at,
        consent_accepted=consent_accepted,
    )


async def _get_connected_providers(
    global_user_id: str, registry_session: AsyncSession
) -> list[str]:
    result = await registry_session.execute(
        select(ProviderConnection).where(ProviderConnection.user_id == global_user_id)
    )
    return [c.provider for c in result.scalars().all()]


async def _get_consent_accepted(user_id: str, registry_session: AsyncSession) -> bool:
    result = await registry_session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    return (
        user is not None
        and user.consented_at is not None
        and user.consent_version == CURRENT_CONSENT_VERSION
    )


@router.get("", response_model=AthleteResponse,
            operation_id="getAthlete", summary="Get current athlete")
async def get_athlete(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    providers = await _get_connected_providers(ctx.user_id, registry_session)
    consent_ok = await _get_consent_accepted(ctx.user_id, registry_session)
    return _athlete_response(athlete, providers, consent_accepted=consent_ok)


@router.patch("", response_model=AthleteResponse,
              operation_id="updateAthlete", summary="Update current athlete")
async def update_athlete(
    body: AthleteUpdate,
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    if body.name is not None:
        athlete.name = body.name
    if body.date_of_birth is not None:
        athlete.date_of_birth = body.date_of_birth
    if body.weight_kg is not None:
        athlete.weight_kg = body.weight_kg
        today = datetime.now(timezone.utc).date()
        wl_result = await session.execute(
            select(WeightLog).where(
                WeightLog.athlete_id == athlete.id,
                WeightLog.effective_date == today,
            )
        )
        wl_entry = wl_result.scalar_one_or_none()
        if wl_entry:
            wl_entry.weight_kg = body.weight_kg
        else:
            session.add(WeightLog(
                athlete_id=athlete.id,
                effective_date=today,
                weight_kg=body.weight_kg,
            ))
    if body.ftp is not None:
        athlete.ftp = body.ftp
        tests = list(athlete.ftp_tests or [])
        tests.append({
            "date": datetime.now(timezone.utc).date().isoformat(),
            "ftp": body.ftp,
            "method": body.ftp_test_method or "manual",
        })
        athlete.ftp_tests = tests
    if body.max_hr is not None:
        athlete.max_hr = body.max_hr
    if body.resting_hr is not None:
        athlete.resting_hr = body.resting_hr
    if body.hr_zones is not None:
        athlete.hr_zones = [z.model_dump() for z in body.hr_zones]
    if body.power_zones is not None:
        athlete.power_zones = [z.model_dump() for z in body.power_zones]
    if body.app_settings is not None:
        new_settings: dict = dict(body.app_settings)
        new_settings.pop("llm_api_key_set", None)

        if "llm_base_url" in new_settings:
            raw_url = new_settings.get("llm_base_url")
            if raw_url and str(raw_url).strip():
                new_settings["llm_base_url"] = _validate_llm_base_url(str(raw_url))
            else:
                # Empty/blank clears the BYOK URL (merged-None deletes the key).
                new_settings["llm_base_url"] = None

        if "experience_level" in new_settings:
            level = new_settings.get("experience_level")
            if level and str(level).strip():
                if level not in VALID_EXPERIENCE_LEVELS:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid experience level.",
                    )
            else:
                # Empty/blank clears the setting (merged-None deletes the key).
                new_settings["experience_level"] = None

        if "llm_api_key" in new_settings:
            raw_key = new_settings.pop("llm_api_key")
            if raw_key:
                try:
                    from backend.app.core.file_encryption import encrypt_secret
                    new_settings["llm_api_key_enc"] = encrypt_secret(
                        str(raw_key), ctx.user_id
                    )
                except RuntimeError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Cannot encrypt API key — ENCRYPTION_KEY not set: {exc}",
                    )
            else:
                new_settings["llm_api_key_enc"] = None

        # Merge into existing settings. Explicit None values are treated as
        # deletions so callers can remove a key without a full-replace round-trip.
        merged = {**(athlete.app_settings or {}), **new_settings}
        athlete.app_settings = {k: v for k, v in merged.items() if v is not None}

    athlete.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(athlete)
    providers = await _get_connected_providers(ctx.user_id, registry_session)
    consent_ok = await _get_consent_accepted(ctx.user_id, registry_session)
    return _athlete_response(athlete, providers, consent_accepted=consent_ok)


@router.put("/avatar", response_model=AthleteResponse,
            operation_id="setAvatar", summary="Upload/replace own avatar")
@router.post("/avatar", response_model=AthleteResponse, include_in_schema=False)
async def upload_avatar(
    file: UploadFile = File(...),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session

    data = await file.read(_MAX_AVATAR_BYTES + 1)
    if len(data) > _MAX_AVATAR_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 5 MB).")

    detected_type = _detect_image_type(data)
    if detected_type is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPEG, PNG, WebP, or GIF.",
        )
    ext = _CONTENT_TYPE_TO_EXT[detected_type]
    athlete = await _get_athlete(ctx.user_id, session)

    avatar_dir = settings.user_avatar_dir(ctx.user_id)
    avatar_dir.mkdir(parents=True, exist_ok=True)
    dest = avatar_dir / f"{ctx.user_id}.{ext}"

    if athlete.avatar_path:
        old = Path(athlete.avatar_path)
        if old.exists() and old != dest:
            old.unlink(missing_ok=True)

    dest.write_bytes(data)
    athlete.avatar_path = str(dest)
    athlete.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(athlete)
    providers = await _get_connected_providers(ctx.user_id, registry_session)
    consent_ok = await _get_consent_accepted(ctx.user_id, registry_session)
    return _athlete_response(athlete, providers, consent_accepted=consent_ok)


@router.delete("/avatar", response_model=AthleteResponse,
               operation_id="deleteAvatar", summary="Delete own avatar")
async def delete_avatar(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    if athlete.avatar_path:
        Path(athlete.avatar_path).unlink(missing_ok=True)
        athlete.avatar_path = None
        athlete.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(athlete)
    providers = await _get_connected_providers(ctx.user_id, registry_session)
    consent_ok = await _get_consent_accepted(ctx.user_id, registry_session)
    return _athlete_response(athlete, providers, consent_accepted=consent_ok)


@router.get("/{athlete_id}/avatar",
            operation_id="getAthleteAvatar", summary="Get an athlete's avatar (auth)")
async def get_avatar(
    athlete_id: str,
    ctx_session=Depends(get_ctx_and_session),
):
    ctx, session = ctx_session
    result = await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    athlete = result.scalar_one_or_none()
    if athlete is None or not athlete.avatar_path:
        raise HTTPException(status_code=404, detail="No avatar set")
    path = Path(athlete.avatar_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Avatar file not found")
    return FileResponse(path)


_PENDING_TIMEOUT_MINUTES = 30


@router.get("/training-status", response_model=TrainingStatusResponse,
            operation_id="getTrainingStatus", summary="Get training-status feedback")
async def get_training_status(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    app_cfg = athlete.app_settings or {}
    from backend.app.services.llm_training_status_analyzer import _local_now
    now_utc = datetime.now(timezone.utc)
    today = _local_now(app_cfg.get("timezone")).date()
    stale = (
        athlete.training_status_date is None
        or athlete.training_status_date < today
    )

    # Recover from a stuck "pending" state: if the task hasn't completed within
    # the timeout window, reset to "error" so the user can retry.
    # A NULL updated_at with status "pending" (e.g. pre-migration row) is treated
    # as immediately timed out.
    if athlete.training_status_status == "pending":
        updated_at = athlete.training_status_updated_at
        if updated_at is not None:
            # Normalise to UTC regardless of whether the stored value is naive or aware
            aware = updated_at if updated_at.tzinfo else updated_at.replace(tzinfo=timezone.utc)
            timed_out = (now_utc - aware.astimezone(timezone.utc)).total_seconds() > _PENDING_TIMEOUT_MINUTES * 60
        else:
            timed_out = True  # pre-migration row with no timestamp — treat as timed out
        if timed_out:
            athlete.training_status_status = "error"
            athlete.training_status_updated_at = now_utc
            # Set training_status_date to today so stale=False and the auto-trigger
            # doesn't immediately re-fire after this error reset.
            athlete.training_status_date = today
            await session.commit()

    if app_cfg.get("auto_training_status") and stale and athlete.training_status_status != "pending":
        # Issue #9: skip the instance-paid auto refresh silently for denied users
        # on a gated instance (the toggle stays saved but inert).
        from backend.app.services.llm_access import check_llm_access
        instance = (
            await registry_session.execute(select(InstanceSettings).limit(1))
        ).scalar_one_or_none()
        access = await check_llm_access(ctx, athlete, instance, registry_session)
        if access.allowed:
            athlete.training_status_status = "pending"
            athlete.training_status = None
            athlete.training_status_updated_at = now_utc
            await session.commit()
            from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
            asyncio.create_task(analyze_training_status_bg(athlete.id, ctx.user_id))

    return TrainingStatusResponse(
        status=athlete.training_status_status,
        feedback=athlete.training_status,
        generated_date=athlete.training_status_date,
    )


@router.post("/training-status", status_code=202,
             operation_id="triggerTrainingStatus", summary="Trigger training-status analysis")
async def trigger_training_status(
    body: TrainingStatusBody = TrainingStatusBody(),
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    # Issue #9 gate (the training-status analysis is always instance-paid).
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error()

    if athlete.training_status_status == "pending":
        return {"status": "pending"}

    now_utc = datetime.now(timezone.utc)
    athlete.training_status_status = "pending"
    athlete.training_status = None
    athlete.training_status_updated_at = now_utc
    await session.commit()

    from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
    asyncio.create_task(analyze_training_status_bg(athlete.id, ctx.user_id, body.locale))
    return {"status": "pending"}


@router.get("/weight-log",
            operation_id="getWeightLog", summary="Get the athlete's weight log")
async def get_weight_log(ctx_session=Depends(get_ctx_and_session)):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)
    result = await session.execute(
        select(WeightLog)
        .where(WeightLog.athlete_id == athlete.id)
        .order_by(WeightLog.effective_date.desc())
    )
    entries = result.scalars().all()
    return [{"date": e.effective_date.isoformat(), "weight_kg": e.weight_kg} for e in entries]


@router.get("/export",
            operation_id="exportAthlete", summary="Export all athlete data as a zip")
async def export_athlete(
    ctx_session=Depends(get_ctx_and_session),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session = ctx_session
    athlete = await _get_athlete(ctx.user_id, session)

    user_result = await registry_session.execute(
        select(User).where(User.id == ctx.user_id)
    )
    user = user_result.scalar_one_or_none()
    username = user.username if user else ctx.user_id

    profile_data = {
        "id": athlete.id,
        "username": username,
        "name": athlete.name,
        "date_of_birth": athlete.date_of_birth.isoformat() if athlete.date_of_birth else None,
        "weight_kg": athlete.weight_kg,
        "ftp": athlete.ftp,
        "max_hr": athlete.max_hr,
        "resting_hr": athlete.resting_hr,
        "hr_zones": athlete.hr_zones or [],
        "power_zones": athlete.power_zones or [],
        "ftp_tests": athlete.ftp_tests or [],
        "created_at": athlete.created_at.isoformat(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    activities_result = await session.execute(
        select(Activity)
        .where(Activity.athlete_id == athlete.id)
        .order_by(Activity.start_time.asc())
    )
    activities = activities_result.scalars().all()

    activities_data = [
        {
            "id": a.id,
            "name": a.name,
            "sport_type": a.sport_type,
            "start_time": a.start_time.isoformat() if a.start_time else None,
            "duration_s": a.duration_s,
            "distance_m": a.distance_m,
            "elevation_m": a.elevation_m,
            "avg_power": a.avg_power,
            "normalized_power": a.normalized_power,
            "avg_hr": a.avg_hr,
            "max_hr": a.max_hr,
            "tss": a.tss,
            "intensity_factor": a.intensity_factor,
            "sources": [s.provider for s in (a.sources or [])],
            "status": a.status,
            "created_at": a.created_at.isoformat(),
            "has_fit_file": a.has_fit_file,
        }
        for a in activities
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("profile.json", json.dumps(profile_data, indent=2))
        zf.writestr("activities.json", json.dumps(activities_data, indent=2))
        for a in activities:
            fit_sources = [s for s in (a.sources or []) if s.fit_file_path]
            for src in fit_sources:
                fit_path = Path(src.fit_file_path)
                if fit_path.exists():
                    if src.fit_file_encrypted:
                        zf.writestr(
                            f"fit_files/{a.id}.fit",
                            decrypt_file(fit_path, ctx.user_id),
                        )
                    else:
                        zf.write(fit_path, f"fit_files/{a.id}.fit")
                    break
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=openkoutsi_export.zip"},
    )
