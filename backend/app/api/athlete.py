import asyncio
import io
import json
import zipfile
from collections.abc import Collection
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import inspect as sa_inspect, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.deps import get_ctx_and_session, get_ctx_session_athlete
from backend.app.core.file_encryption import decrypt_file
from backend.app.core.ssrf import check_url_safe
from backend.app.core.scopes import pat_forbidden, pat_scope, pat_scopes
from backend.app.db.registry import get_registry_session
from backend.app.api.consent import CURRENT_CONSENT_VERSION
from backend.app.api.distance import all_time_distance_bests
from backend.app.api.power import all_time_power_bests
from backend.app.models.chat_orm import ChatConversation, ChatMessage
from backend.app.models.message_orm import Message
from backend.app.models.registry_orm import (
    InstanceSettings,
    PersonalAccessToken,
    ProviderConnection,
    User,
)
from backend.app.models.user_orm import (
    AchievementUnlock,
    Activity,
    Athlete,
    Bike,
    Course,
    DailyMetric,
    Goal,
    TrainingPlan,
    WeightLog,
    WorkoutDefinition,
)
from backend.app.schemas.athlete import AthleteResponse, AthleteUpdate, TrainingStatusBody, TrainingStatusResponse
from backend.app.services import personal_access_tokens as pat_service
from backend.app.services.pat_expiry import (
    EMAIL_OPT_OUT_SETTING as PAT_EXPIRY_EMAIL_SETTING,
)
from backend.app.services.athlete_experience import VALID_EXPERIENCE_LEVELS
from backend.app.services.stranded_runs import (
    begin_training_status_run,
    pending_timed_out,
    settle_training_status,
)
from openkoutsi.plan_schema import HOURS_BOUNDS

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



router = APIRouter(
    prefix="/athlete",
    tags=["athlete"],
    dependencies=[pat_scopes(read="athlete:read", write="athlete:write")],
)


_MAX_LLM_URL_LEN = 2048

# Every `app_settings` key that steers where an LLM call goes or what
# authenticates it. Closed to personal access tokens (issue #46) — see the guard
# in `update_athlete`. `llm_models` is a per-athlete preset list whose entries
# each carry their own `base_url`, so it belongs here too.
_LLM_SETTING_KEYS = frozenset({
    "llm_base_url", "llm_api_key", "llm_api_key_enc", "llm_model", "llm_models",
})

# Self-reported athlete experience level, stored in app_settings (see #18) and
# fed into the LLM coaching/generation prompts (see #32). The canonical tuple
# lives in ``services.athlete_experience`` so validation here and prompt building
# there share one source.


def _validate_llm_base_url(raw: str) -> str:
    """Validate and normalise a user's BYOK base URL at save time.

    Fails fast in the UI (instead of at the first LLM call): strips whitespace,
    caps the length, requires an ``http(s)://`` scheme, enforces the allow-list,
    and runs the SSRF guard (which resolves DNS and blocks metadata, loopback
    and private ranges).
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
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete
    providers = await _get_connected_providers(ctx.user_id, registry_session)
    consent_ok = await _get_consent_accepted(ctx.user_id, registry_session)
    return _athlete_response(athlete, providers, consent_accepted=consent_ok)


@router.patch("", response_model=AthleteResponse,
              operation_id="updateAthlete", summary="Update current athlete")
async def update_athlete(
    body: AthleteUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete

    if body.name is not None:
        athlete.name = body.name
    if body.date_of_birth is not None:
        athlete.date_of_birth = body.date_of_birth
    weight_changed = False
    if body.weight_kg is not None:
        weight_changed = True
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

        # A personal access token may never touch the LLM *configuration*, not
        # just the endpoints that spend money (issue #46). Repointing
        # `llm_base_url` would make the user's own browser session ship their
        # training data to a host of the token holder's choosing on the next
        # analysis — every PAT control still green, because the token itself
        # never calls an LLM route. `check_url_safe` blocks internal ranges,
        # not hosts, so any publicly resolvable host of theirs would pass.
        if ctx.is_pat and _LLM_SETTING_KEYS & new_settings.keys():
            raise HTTPException(
                status_code=403,
                detail="LLM configuration cannot be changed with a personal access token.",
            )

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

        # Default weekly training-hours availability (a range), reused to prefill
        # the plan-generation dialog. Each endpoint is an optional number in the
        # allowed hours band; blank/None clears it.
        for _hours_key in ("weekly_hours_min", "weekly_hours_max"):
            if _hours_key in new_settings:
                raw_hours = new_settings.get(_hours_key)
                if raw_hours in (None, ""):
                    new_settings[_hours_key] = None
                    continue
                try:
                    hours = float(raw_hours)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {_hours_key}: must be a number.",
                    )
                lo, hi = HOURS_BOUNDS
                if not (lo <= hours <= hi):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {_hours_key}: must be between {lo:g} and {hi:g} hours.",
                    )
                new_settings[_hours_key] = hours
        merged_min = new_settings.get(
            "weekly_hours_min", (athlete.app_settings or {}).get("weekly_hours_min")
        )
        merged_max = new_settings.get(
            "weekly_hours_max", (athlete.app_settings or {}).get("weekly_hours_max")
        )
        if (
            merged_min is not None
            and merged_max is not None
            and float(merged_min) > float(merged_max)
        ):
            raise HTTPException(
                status_code=400,
                detail="weekly_hours_min cannot be greater than weekly_hours_max.",
            )

        # Expiry email for personal access tokens (issue #46). Opt-*out*: the
        # default is on and only email is affected — the inbox message that says
        # a credential is about to die is unconditional.
        if PAT_EXPIRY_EMAIL_SETTING in new_settings:
            raw_flag = new_settings.get(PAT_EXPIRY_EMAIL_SETTING)
            if raw_flag is None or isinstance(raw_flag, bool):
                pass
            elif isinstance(raw_flag, (int, str)) and str(raw_flag).lower() in {
                "true", "false", "1", "0",
            }:
                new_settings[PAT_EXPIRY_EMAIL_SETTING] = str(raw_flag).lower() in {
                    "true", "1",
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid {PAT_EXPIRY_EMAIL_SETTING}: must be true or false.",
                )
            # `False` must survive the merge below, which strips None but keeps
            # falsy values — opting out is the whole point of the setting.

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

    if weight_changed:
        # A weigh-in applies from its own date onward only: power bests that
        # already carry a weight keep it, so past activities stay attributed to
        # whatever the athlete weighed back then (or to no weight at all). This
        # pass just fills in rows still missing a weight — e.g. efforts recorded
        # earlier today, before the first-ever weigh-in landed.
        from backend.app.services.weight import backfill_missing_power_best_weights

        await backfill_missing_power_best_weights(athlete.id, session)

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
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete

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
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete
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


@router.get("/training-status", response_model=TrainingStatusResponse,
            operation_id="getTrainingStatus", summary="Get training-status feedback")
async def get_training_status(
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete
    app_cfg = athlete.app_settings or {}
    from backend.app.services.llm_training_status_analyzer import _local_now
    now_utc = datetime.now(timezone.utc)
    today = _local_now(app_cfg.get("timezone")).date()
    stale = (
        athlete.training_status_date is None
        or athlete.training_status_date < today
    )

    # Deterministic plan-adherence snapshots piggyback the daily first-read
    # cadence (issue #26). Always-on and not gated behind the LLM subscription,
    # so this runs regardless of auto_training_status or entitlement.
    if stale:
        from backend.app.services.plan_adherence import catch_up_adherence
        await catch_up_adherence(athlete.id, session)
        # Achievements piggyback the same daily first-read cadence (issue #33),
        # so streaks stay current for an athlete who hasn't uploaded in a while.
        # Deferred import for the same reason as catch_up_adherence above.
        from backend.app.services.achievements import recompute_achievements_safe
        await recompute_achievements_safe(athlete.id, session)

    # Recover from a stuck "pending" state: if the run hasn't shown progress
    # within the timeout window, reset to "error" so the user can retry. The
    # window is an inactivity budget — the analyzer touches the timestamp on
    # every progress commit (issue #91) — so a healthy run that simply takes a
    # long time is no longer declared dead underneath itself.
    # A NULL updated_at with status "pending" (e.g. pre-migration row) is treated
    # as immediately timed out.
    if athlete.training_status_status == "pending" and pending_timed_out(
        athlete.training_status_updated_at, now_utc
    ):
        settle_training_status(athlete, now_utc)
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
            run_id = begin_training_status_run(athlete, now_utc)
            await session.commit()
            from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
            asyncio.create_task(
                analyze_training_status_bg(athlete.id, ctx.user_id, run_id=run_id)
            )

    return TrainingStatusResponse(
        status=athlete.training_status_status,
        feedback=athlete.training_status,
        generated_date=athlete.training_status_date,
        # Only meaningful while the run is in flight (issue #43). A settled row
        # should have had it cleared by the analyzer, but a run killed between
        # its last progress commit and settling would leave one behind, and a
        # stale "Koutsi is checking your power curve…" under a finished answer
        # reads as a bug. Gating on the status here makes that impossible.
        progress=(
            athlete.training_status_progress
            if athlete.training_status_status == "pending"
            else None
        ),
    )


@router.post("/training-status", status_code=202, dependencies=[pat_forbidden()],
             operation_id="triggerTrainingStatus", summary="Trigger training-status analysis")
async def trigger_training_status(
    body: TrainingStatusBody = TrainingStatusBody(),
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete

    # Issue #9 gate (the training-status analysis is always instance-paid).
    from backend.app.services.llm_access import check_llm_access, subscription_required_error
    instance = (
        await registry_session.execute(select(InstanceSettings).limit(1))
    ).scalar_one_or_none()
    access = await check_llm_access(ctx, athlete, instance, registry_session)
    if not access.allowed:
        raise subscription_required_error(access)

    if athlete.training_status_status == "pending":
        return {"status": "pending"}

    run_id = begin_training_status_run(athlete, datetime.now(timezone.utc))
    await session.commit()

    from backend.app.services.llm_training_status_analyzer import analyze_training_status_bg
    asyncio.create_task(
        analyze_training_status_bg(
            athlete.id, ctx.user_id, body.locale, run_id=run_id
        )
    )
    return {"status": "pending"}


@router.get("/weight-log",
            operation_id="getWeightLog", summary="Get the athlete's weight log")
async def get_weight_log(ctx_athlete=Depends(get_ctx_session_athlete)):
    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(WeightLog)
        .where(WeightLog.athlete_id == athlete.id)
        .order_by(WeightLog.effective_date.desc())
    )
    entries = result.scalars().all()
    return [{"date": e.effective_date.isoformat(), "weight_kg": e.weight_kg} for e in entries]


# ── Data export ──────────────────────────────────────────────────────────────
#
# Every file in the export zip is "one row per record, all of its columns".
# Driving that off the mapper rather than a hand-written field list per model is
# what keeps the export honest: a column added to a model shows up in the user's
# download without anyone having to remember to add it here, which is exactly
# the failure mode a data export must not have. `exclude` drops internal
# plumbing (foreign keys the export's own structure already implies), and the
# keyword arguments supply values the columns can't: relationships, derived
# fields, and the redacted `app_settings`.


def _dump(obj, *, exclude: Collection[str] = (), **extra) -> dict:
    """Every mapped column of ``obj``, JSON-safe, plus any ``extra`` keys."""
    values = {
        attr.key: _json_safe(getattr(obj, attr.key))
        for attr in sa_inspect(type(obj)).mapper.column_attrs
        if attr.key not in exclude
    }
    return {**values, **extra}


def _json_safe(value):
    """Column values `json.dumps` can't take on its own — dates and datetimes."""
    return value.isoformat() if isinstance(value, (date, datetime)) else value


async def _export_rows(
    session: AsyncSession, query, *, exclude: Collection[str] = (), extra=None
) -> list[dict]:
    """Dump every row ``query`` returns; ``extra(row)`` adds per-row derived keys."""
    result = await session.execute(query)
    return [
        _dump(row, exclude=exclude, **(extra(row) if extra else {}))
        for row in result.scalars().all()
    ]


def _export_profile(athlete: Athlete, username: str) -> dict:
    """Full athlete profile, including LLM/analysis settings (BYOK key redacted)."""
    return _dump(
        athlete,
        # `global_user_id` and `avatar_path` are server-side plumbing: a registry
        # key and a filesystem path, neither meaningful outside this instance.
        exclude=("global_user_id", "avatar_path"),
        username=username,
        hr_zones=athlete.hr_zones or [],
        power_zones=athlete.power_zones or [],
        availability=athlete.availability or {},
        ftp_tests=athlete.ftp_tests or [],
        app_settings=_safe_app_settings(athlete),
        exported_at=datetime.now(timezone.utc).isoformat(),
    )


def _export_activity(a: Activity) -> dict:
    """One activity, including notes, labels and LLM analysis."""
    return _dump(
        a,
        exclude=("athlete_id",),
        labels=a.labels or [],
        sources=[s.provider for s in (a.sources or [])],
        has_fit_file=a.has_fit_file,
    )


async def _export_goals(athlete: Athlete, session: AsyncSession) -> list[dict]:
    return await _export_rows(
        session,
        select(Goal).where(Goal.athlete_id == athlete.id).order_by(Goal.created_at),
        exclude=("athlete_id",),
    )


async def _export_bikes(athlete: Athlete, session: AsyncSession) -> list[dict]:
    return await _export_rows(
        session,
        select(Bike).where(Bike.athlete_id == athlete.id).order_by(Bike.created_at),
        exclude=("athlete_id",),
    )


async def _export_courses(athlete: Athlete, session: AsyncSession) -> list[dict]:
    """Courses with their nested segment tables (issue #55).

    ``gpx_file_key`` is server-side plumbing and stays out; the file itself is
    delivered decrypted as ``courses/{id}.gpx``, which also carries the
    coordinates — the rows here are the coordinate-free derived data.
    """
    return await _export_rows(
        session,
        select(Course)
        .where(Course.athlete_id == athlete.id)
        .options(selectinload(Course.segments))
        .order_by(Course.created_at),
        exclude=("athlete_id", "gpx_file_key", "gpx_file_encrypted"),
        extra=lambda c: {
            "profile": c.profile or [],
            "segments": [
                _dump(s, exclude=("course_id",)) for s in c.segments
            ],
        },
    )


def _export_planned_workout(w) -> dict:
    return _dump(
        w,
        exclude=("plan_id",),
        linked_activity_ids=[a.id for a in w.linked_activities],
        completed_activity_id=(
            w.linked_activities[0].id if w.linked_activities else None
        ),
    )


async def _export_plans(athlete: Athlete, session: AsyncSession) -> list[dict]:
    return await _export_rows(
        session,
        select(TrainingPlan)
        .where(TrainingPlan.athlete_id == athlete.id)
        .options(selectinload(TrainingPlan.workouts))
        .order_by(TrainingPlan.created_at),
        exclude=("athlete_id",),
        extra=lambda p: {
            "config": p.config or {},
            "week_meta": p.week_meta or [],
            "planned_workouts": [
                _export_planned_workout(w)
                for w in sorted(p.workouts, key=lambda w: (w.week_number, w.day_of_week))
            ],
        },
    )


async def _export_workout_definitions(
    athlete: Athlete, session: AsyncSession
) -> list[dict]:
    return await _export_rows(
        session,
        select(WorkoutDefinition)
        .where(WorkoutDefinition.athlete_id == athlete.id)
        .order_by(WorkoutDefinition.created_at),
        exclude=("athlete_id",),
        extra=lambda w: {"steps": w.steps or []},
    )


async def _export_daily_metrics(athlete: Athlete, session: AsyncSession) -> list[dict]:
    """CTL/ATL/TSB per day (`load_day` also drives weekly TSS)."""
    return await _export_rows(
        session,
        select(DailyMetric)
        .where(DailyMetric.athlete_id == athlete.id)
        .order_by(DailyMetric.date),
        exclude=("athlete_id",),
    )


async def _export_inbox(session: AsyncSession) -> list[dict]:
    """In-app messages. The per-user DB identifies the recipient, so no filter."""
    return await _export_rows(
        session,
        select(Message).order_by(Message.created_at),
        extra=lambda m: {"data": m.data or {}},
    )


async def _export_chat(session: AsyncSession) -> list[dict]:
    """Koutsi conversations, nested message-in-thread (issue #44).

    Health-adjacent free text the athlete wrote about their own body, so it
    belongs in the export from the day the feature ships rather than after
    somebody notices — which is what issue #21 exists to remember.

    Nested rather than two flat files because the thread is the unit that means
    anything: a list of messages with conversation ids in it would be a join the
    reader has to perform to get back what they actually wrote. ``tool_names``
    comes along as the record of what Koutsi consulted; the tool *results* were
    never stored (see ``models.chat_orm``), so there is nothing else to give.
    """
    conversations = (
        (
            await session.execute(
                select(ChatConversation).order_by(ChatConversation.created_at)
            )
        )
        .scalars()
        .all()
    )
    messages = (
        (await session.execute(select(ChatMessage).order_by(ChatMessage.created_at)))
        .scalars()
        .all()
    )
    by_conversation: dict[str, list[dict]] = {}
    for message in messages:
        by_conversation.setdefault(message.conversation_id, []).append(
            _dump(message, exclude=("conversation_id",))
        )
    return [
        _dump(conversation, messages=by_conversation.get(conversation.id, []))
        for conversation in conversations
    ]


async def _export_weight_log(athlete: Athlete, session: AsyncSession) -> list[dict]:
    return await _export_rows(
        session,
        select(WeightLog)
        .where(WeightLog.athlete_id == athlete.id)
        .order_by(WeightLog.effective_date),
        exclude=("athlete_id",),
    )


async def _export_achievements(athlete: Athlete, session: AsyncSession) -> list[dict]:
    """Earned achievement tiers (issue #33).

    The catalogue itself is code, not data, so only the unlocks are exported —
    the ids are the stable machine keys the API uses.

    Reads the stored rows rather than recomputing: ``export_athlete`` settles the
    athlete first (issue #69), so by the time this runs the table is current.
    """
    return await _export_rows(
        session,
        select(AchievementUnlock)
        .where(AchievementUnlock.athlete_id == athlete.id)
        .order_by(AchievementUnlock.achieved_on),
        exclude=("athlete_id",),
    )


async def _export_personal_access_tokens(
    user_id: str, registry_session: AsyncSession
) -> list[dict]:
    """Personal access tokens, metadata only — never the hash (issue #46).

    The first registry-sourced entry in the export: every other file here is
    drawn from the per-user DB, and tokens live in the registry, which is why
    ``export_athlete`` threads a registry session through to this one.

    Covers dead tokens as well as live ones. Expired and revoked rows are
    retained rather than pruned, so leaving them out would make the export a
    less complete record than the app's own token list.
    """
    result = await registry_session.execute(
        select(PersonalAccessToken)
        .where(PersonalAccessToken.user_id == user_id)
        .order_by(PersonalAccessToken.created_at)
    )
    return [
        {
            "id": token.id,
            "name": token.name,
            "scopes": pat_service.scopes_of(token),
            "status": pat_service.status_of(token),
            "expires_at": token.expires_at.isoformat() if token.expires_at else None,
            "last_used_at": token.last_used_at.isoformat() if token.last_used_at else None,
            "revoked_at": token.revoked_at.isoformat() if token.revoked_at else None,
            "created_at": token.created_at.isoformat() if token.created_at else None,
        }
        for token in result.scalars().all()
    ]


@router.get("/export", dependencies=[pat_scope("athlete:export")],
            operation_id="exportAthlete", summary="Export all athlete data as a zip")
async def export_athlete(
    ctx_athlete=Depends(get_ctx_session_athlete),
    registry_session: AsyncSession = Depends(get_registry_session),
):
    ctx, session, athlete = ctx_athlete

    # Settle any pending achievement recompute before reading the unlock rows
    # (issue #69). The write paths only mark now, so an export taken straight
    # after an upload would otherwise omit the badges that upload earned — and an
    # export that under-reports is a worse failure than a slow one. Done first,
    # so `_export_achievements` below reads the settled table. Ordered before the
    # inbox read too, so a message this reconcile emits is in the export as well.
    from backend.app.services.achievements import recompute_achievements_safe
    await recompute_achievements_safe(athlete.id, session)

    user_result = await registry_session.execute(
        select(User).where(User.id == ctx.user_id)
    )
    user = user_result.scalar_one_or_none()
    username = user.username if user else ctx.user_id

    activities_result = await session.execute(
        select(Activity)
        .where(Activity.athlete_id == athlete.id)
        .order_by(Activity.start_time.asc())
    )
    activities = activities_result.scalars().all()

    power_bests = await all_time_power_bests(athlete, session)
    distance_bests = await all_time_distance_bests(athlete, session)
    personal_records = {
        "power_bests": [e.model_dump(mode="json") for e in power_bests],
        "distance_bests": [e.model_dump(mode="json") for e in distance_bests],
    }

    files: dict[str, object] = {
        "profile.json": _export_profile(athlete, username),
        "activities.json": [_export_activity(a) for a in activities],
        "goals.json": await _export_goals(athlete, session),
        "plans.json": await _export_plans(athlete, session),
        "workout_definitions.json": await _export_workout_definitions(athlete, session),
        "daily_metrics.json": await _export_daily_metrics(athlete, session),
        "personal_records.json": personal_records,
        "inbox.json": await _export_inbox(session),
        "chat.json": await _export_chat(session),
        "weight_log.json": await _export_weight_log(athlete, session),
        "achievements.json": await _export_achievements(athlete, session),
        "bikes.json": await _export_bikes(athlete, session),
        "courses.json": await _export_courses(athlete, session),
        "personal_access_tokens.json": await _export_personal_access_tokens(
            ctx.user_id, registry_session
        ),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, payload in files.items():
            zf.writestr(filename, json.dumps(payload, indent=2))
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
        # The uploaded course originals, decrypted — the one place the export
        # hands coordinates back, because they are the user's own file.
        from backend.app.services.course_analysis import resolve_course_blob

        courses = (
            await session.execute(select(Course).where(Course.athlete_id == athlete.id))
        ).scalars().all()
        for course in courses:
            try:
                blob = resolve_course_blob(course.gpx_file_key, ctx.user_id)
            except ValueError:
                continue
            if blob.exists():
                if course.gpx_file_encrypted:
                    zf.writestr(
                        f"courses/{course.id}.gpx", decrypt_file(blob, ctx.user_id)
                    )
                else:
                    zf.write(blob, f"courses/{course.id}.gpx")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=openkoutsi_export.zip"},
    )
