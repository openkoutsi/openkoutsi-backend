"""The garage — bikes an athlete owns, rides and maintains (issues #55, #64).

One record with two readers. A bike started as the small equipment concept the
pacing model needs (tyre width → rolling resistance, riding position → drag
area, both tabulated in ``openkoutsi.course``); issue #64 gave the same row a
baseline odometer, the sports it claims, a retirement date, a maintenance log
and a list of what is bolted to it. The garage edits exactly the rows the
route-analysis bike selector lists, so "bikes in the garage are entries in the
course picker" holds by construction — which is the thing a second
``garage_bikes`` table would have broken on day one.

Distance is **derived on read**, never stored: ``tracked_km`` is a ``SUM`` over
the rides assigned to the bike and ``lifetime_km`` adds the athlete's own
baseline. Reassigning a ride or correcting a baseline is therefore immediately
right everywhere, with no counter to drift.

No pagination anywhere here — a person owns a handful of bikes, and a
maintenance log is a page of a notebook, not a history of one.
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select, update

from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.limiter import limiter
from backend.app.core.scopes import pat_scopes
from backend.app.models.user_orm import (
    Activity,
    Bike,
    BikeAccessory,
    BikeMaintenance,
    Course,
)
from backend.app.schemas.bikes import (
    AccessoryCreate,
    AccessoryResponse,
    AccessoryUpdate,
    AssignHistoryResponse,
    BikeCreate,
    BikeDetailResponse,
    BikeResponse,
    BikeUpdate,
    MaintenanceCreate,
    MaintenanceResponse,
    MaintenanceUpdate,
)
from backend.app.services import garage as garage_service

router = APIRouter(
    prefix="/bikes",
    tags=["bikes"],
    dependencies=[pat_scopes(read="bikes:read", write="bikes:write")],
)
# Issue #55 gated this router with course recon, on the reasoning that "a bike
# exists for nothing but course pacing" and a form for an unavailable feature
# is worse than no form. Issue #64 made that reasoning false: a bike is now
# where an athlete's own kilometres, maintenance history and equipment live,
# none of which has anything to do with whether the self-hoster switched on
# GPX course analysis. The gate stays on `/courses` itself, where the
# capability actually is.


def _bike_response(
    bike: Bike, tracked: float, model=BikeResponse, **extra
) -> BikeResponse:
    return model(
        **{
            **{
                name: getattr(bike, name)
                for name in BikeResponse.model_fields
                if hasattr(bike, name)
            },
            "default_sports": list(bike.default_sports or []),
            "tracked_km": tracked,
            "lifetime_km": garage_service.lifetime_km(bike, tracked),
            **extra,
        }
    )


def _claim_conflict(exc: garage_service.SportClaimError) -> HTTPException:
    """A rejected ``default_sports`` list, as the status the client can act on.

    409 for a collision, naming the bike that already holds the sport: two
    bikes claiming ``GravelRide`` has no correct resolution, so picking one
    would be wrong half the time and silently. 422 for a sport that is not a
    cycling sport at all, which is an unusable claim rather than a contested
    one.
    """
    if exc.bike is not None:
        return HTTPException(
            status_code=409,
            detail={
                "code": "sport_already_claimed",
                "message": str(exc),
                "sport": exc.sport,
                "bike_id": exc.bike.id,
                "bike_name": exc.bike.name,
            },
        )
    return HTTPException(status_code=422, detail=str(exc))


@router.get("", response_model=list[BikeResponse],
            operation_id="listBikes", summary="List bikes")
async def list_bikes(ctx_athlete=Depends(get_ctx_session_athlete)):
    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(Bike).where(Bike.athlete_id == athlete.id).order_by(Bike.created_at)
    )
    bikes = result.scalars().all()
    # One grouped query for the whole fleet rather than one per bike — the
    # garage draws every total at once.
    tracked = await garage_service.tracked_km(session, athlete)
    return [_bike_response(b, tracked.get(b.id, 0.0)) for b in bikes]


@router.post("", response_model=BikeResponse, status_code=201,
             operation_id="createBike", summary="Create a bike")
async def create_bike(
    body: BikeCreate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    fields = body.model_dump()
    try:
        fields["default_sports"] = garage_service.normalise_default_sports(
            fields.get("default_sports")
        )
        await garage_service.check_sport_claims(
            session, athlete, fields["default_sports"]
        )
    except garage_service.SportClaimError as exc:
        raise _claim_conflict(exc)

    bike = Bike(id=str(uuid.uuid4()), athlete_id=athlete.id, **fields)
    session.add(bike)
    await session.commit()
    await session.refresh(bike)
    # A brand-new bike has no rides yet; history is picked up by the explicit
    # `assign-history` call, never inline here — see that endpoint.
    return _bike_response(bike, 0.0)


async def _get_owned_bike(bike_id: str, athlete, session) -> Bike:
    result = await session.execute(
        select(Bike).where(Bike.id == bike_id, Bike.athlete_id == athlete.id)
    )
    bike = result.scalar_one_or_none()
    if bike is None:
        raise HTTPException(status_code=404, detail="Bike not found")
    return bike


@router.post("/assign-history", response_model=AssignHistoryResponse,
             operation_id="assignBikeHistory",
             summary="Assign past rides to the bikes that claim their sport")
# A full-history read plus a write transaction, exactly like the commute scan
# two features over, which carries the same limit for the same reason: nothing
# else stops a client — or a retry loop behind a slow response — calling it
# back to back.
@limiter.limit("5/hour")
async def assign_history(
    request: Request,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Walk the back catalogue and attach unassigned rides to a bike.

    Adding a bike, or changing what it claims, has to be able to look backwards:
    a garage that only counts rides from today onward is empty exactly when the
    athlete first opens it.

    Deliberately an explicit request rather than something ``PATCH
    /api/bikes/{id}`` does inline — this can touch a decade of rides, and doing
    that inside an edit holds a request worker for the length of the scan.

    Only rides with no bike at all are touched. Anything already assigned, by
    hand or by an earlier automatic pass, is left exactly as it is.
    """
    ctx, session, athlete = ctx_athlete
    return AssignHistoryResponse(**await garage_service.assign_history(session, athlete))


@router.get("/{bike_id}", response_model=BikeDetailResponse,
            operation_id="getBike", summary="One bike, with its log and accessories")
async def get_bike(
    bike_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    tracked = await garage_service.tracked_km_for(session, athlete, bike.id)
    return _bike_response(
        bike,
        tracked,
        model=BikeDetailResponse,
        maintenance=_maintenance_log(bike, garage_service.lifetime_km(bike, tracked)),
        accessories=[
            AccessoryResponse.model_validate(a)
            for a in sorted(bike.accessories, key=lambda a: a.created_at)
        ],
    )


@router.patch("/{bike_id}", response_model=BikeResponse,
              operation_id="updateBike", summary="Update a bike")
async def update_bike(
    bike_id: str,
    body: BikeUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    fields = body.model_dump(exclude_unset=True)
    if "default_sports" in fields:
        try:
            fields["default_sports"] = garage_service.normalise_default_sports(
                fields["default_sports"]
            )
            await garage_service.check_sport_claims(
                session, athlete, fields["default_sports"], exclude_bike_id=bike.id
            )
        except garage_service.SportClaimError as exc:
            raise _claim_conflict(exc)
    for field, value in fields.items():
        setattr(bike, field, value)
    await session.commit()
    await session.refresh(bike)
    # Editing what a bike claims does *not* walk history here — see
    # `assign-history`, which is the endpoint for that and is rate-limited
    # because it can touch tens of thousands of rows.
    tracked = await garage_service.tracked_km_for(session, athlete, bike.id)
    return _bike_response(bike, tracked)


@router.delete("/{bike_id}", status_code=204,
               operation_id="deleteBike", summary="Delete a bike")
async def delete_bike(
    bike_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Delete a bike, keeping every ride and course that referenced it.

    Retiring is the better answer for an athlete who means "stop showing me
    this" — a retired bike keeps its distance and its log, and deleting cannot
    be undone. The UI offers retirement first for that reason.

    Every ``ON DELETE`` clause in this schema documents intent and does not
    execute: SQLite honours them only with ``PRAGMA foreign_keys`` on, which
    these connections do not set. So each one is enforced here in Python.
    """
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    # SET NULL, by hand: deleting a bike never deletes a course, it just leaves
    # the course needing a bike picked before re-analysis.
    await session.execute(
        update(Course).where(Course.bike_id == bike.id).values(bike_id=None)
    )
    # And never deletes a ride. `bike_source` goes with `bike_id` — leaving
    # "manual" behind on a row with no bike would claim the athlete had chosen
    # something that no longer exists, and would then block automapping from
    # ever filling the gap.
    await session.execute(
        update(Activity)
        .where(Activity.bike_id == bike.id)
        .values(bike_id=None, bike_source=None)
    )
    # The log and the accessories are part of the bike and go with it. That one
    # is carried by the `delete-orphan` cascade on `Bike.maintenance` /
    # `Bike.accessories`, which is an *ORM*-level cascade and so, unlike the SQL
    # clause, does not depend on the pragma either.
    await session.delete(bike)
    await session.commit()


# ── Maintenance ────────────────────────────────────────────────────────────
#
# Both sub-resources are per-bike collections reached through `_get_owned_bike`,
# so ownership is checked exactly once and in exactly one way.


def _maintenance_log(bike: Bike, lifetime: Optional[float]) -> list[MaintenanceResponse]:
    """The log newest-first, with the derived component spans attached."""
    entries = sorted(bike.maintenance, key=garage_service.maintenance_order)
    spans = garage_service.component_spans(entries, lifetime)
    return [
        MaintenanceResponse(
            **{
                name: getattr(entry, name)
                for name in MaintenanceResponse.model_fields
                if hasattr(entry, name)
            },
            **spans[entry.id],
        )
        for entry in reversed(entries)
    ]


async def _owned_entry(model, entry_id: str, bike: Bike, session):
    result = await session.execute(
        select(model).where(model.id == entry_id, model.bike_id == bike.id)
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=404, detail="Not found")
    return entry


@router.get("/{bike_id}/maintenance", response_model=list[MaintenanceResponse],
            operation_id="listBikeMaintenance", summary="A bike's maintenance log")
async def list_maintenance(
    bike_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    tracked = await garage_service.tracked_km_for(session, athlete, bike.id)
    return _maintenance_log(bike, garage_service.lifetime_km(bike, tracked))


@router.post("/{bike_id}/maintenance", response_model=MaintenanceResponse,
             status_code=201, operation_id="createBikeMaintenance",
             summary="Log something done to a bike")
async def create_maintenance(
    bike_id: str,
    body: MaintenanceCreate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    entry = BikeMaintenance(
        id=str(uuid.uuid4()),
        bike_id=bike.id,
        athlete_id=athlete.id,
        **body.model_dump(),
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    await session.refresh(bike, ["maintenance"])
    tracked = await garage_service.tracked_km_for(session, athlete, bike.id)
    spans = garage_service.component_spans(
        list(bike.maintenance), garage_service.lifetime_km(bike, tracked)
    )
    return MaintenanceResponse(
        **{
            name: getattr(entry, name)
            for name in MaintenanceResponse.model_fields
            if hasattr(entry, name)
        },
        **spans[entry.id],
    )


@router.patch("/{bike_id}/maintenance/{entry_id}", response_model=MaintenanceResponse,
              operation_id="updateBikeMaintenance", summary="Edit a maintenance entry")
async def update_maintenance(
    bike_id: str,
    entry_id: str,
    body: MaintenanceUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    entry = await _owned_entry(BikeMaintenance, entry_id, bike, session)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)
    await session.commit()
    await session.refresh(entry)
    await session.refresh(bike, ["maintenance"])
    tracked = await garage_service.tracked_km_for(session, athlete, bike.id)
    spans = garage_service.component_spans(
        list(bike.maintenance), garage_service.lifetime_km(bike, tracked)
    )
    return MaintenanceResponse(
        **{
            name: getattr(entry, name)
            for name in MaintenanceResponse.model_fields
            if hasattr(entry, name)
        },
        **spans[entry.id],
    )


@router.delete("/{bike_id}/maintenance/{entry_id}", status_code=204,
               operation_id="deleteBikeMaintenance", summary="Delete a maintenance entry")
async def delete_maintenance(
    bike_id: str,
    entry_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    entry = await _owned_entry(BikeMaintenance, entry_id, bike, session)
    await session.delete(entry)
    await session.commit()


# ── Accessories ────────────────────────────────────────────────────────────


@router.get("/{bike_id}/accessories", response_model=list[AccessoryResponse],
            operation_id="listBikeAccessories", summary="What is fitted to a bike")
async def list_accessories(
    bike_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    return [
        AccessoryResponse.model_validate(a)
        for a in sorted(bike.accessories, key=lambda a: a.created_at)
    ]


@router.post("/{bike_id}/accessories", response_model=AccessoryResponse,
             status_code=201, operation_id="createBikeAccessory",
             summary="Note something fitted to a bike")
async def create_accessory(
    bike_id: str,
    body: AccessoryCreate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    """Record an accessory — a child trailer, a rack, a set of lights.

    A plain record on purpose. A trailer genuinely changes mass and drag, but
    feeding that into the pacing model means touching ``BikeParams`` and
    deciding what happens to already-analysed courses; that is deferred, and
    noting the thing exists is what was asked for.
    """
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    accessory = BikeAccessory(
        id=str(uuid.uuid4()),
        bike_id=bike.id,
        athlete_id=athlete.id,
        **body.model_dump(),
    )
    session.add(accessory)
    await session.commit()
    await session.refresh(accessory)
    return AccessoryResponse.model_validate(accessory)


@router.patch("/{bike_id}/accessories/{accessory_id}", response_model=AccessoryResponse,
              operation_id="updateBikeAccessory", summary="Edit an accessory")
async def update_accessory(
    bike_id: str,
    accessory_id: str,
    body: AccessoryUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    accessory = await _owned_entry(BikeAccessory, accessory_id, bike, session)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(accessory, field, value)
    await session.commit()
    await session.refresh(accessory)
    return AccessoryResponse.model_validate(accessory)


@router.delete("/{bike_id}/accessories/{accessory_id}", status_code=204,
               operation_id="deleteBikeAccessory", summary="Remove an accessory")
async def delete_accessory(
    bike_id: str,
    accessory_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    accessory = await _owned_entry(BikeAccessory, accessory_id, bike, session)
    await session.delete(accessory)
    await session.commit()
