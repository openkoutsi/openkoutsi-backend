"""Bikes — the small equipment concept the course physics needs (issue #55).

A bike is described only as much as the pacing model consumes: tyre width
selects a rolling-resistance coefficient, riding position an aerodynamic drag
area (the tables live in ``openkoutsi.course``). No pagination on the list — a
person owns a handful of bikes, not a history of them.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update

from backend.app.core.deps import get_ctx_session_athlete
from backend.app.core.scopes import pat_scopes
from backend.app.models.user_orm import Bike, Course
from backend.app.schemas.bikes import BikeCreate, BikeResponse, BikeUpdate

router = APIRouter(
    prefix="/bikes",
    tags=["bikes"],
    dependencies=[pat_scopes(read="bikes:read", write="bikes:write")],
)


@router.get("", response_model=list[BikeResponse],
            operation_id="listBikes", summary="List bikes")
async def list_bikes(ctx_athlete=Depends(get_ctx_session_athlete)):
    ctx, session, athlete = ctx_athlete
    result = await session.execute(
        select(Bike).where(Bike.athlete_id == athlete.id).order_by(Bike.created_at)
    )
    return [BikeResponse.model_validate(b) for b in result.scalars().all()]


@router.post("", response_model=BikeResponse, status_code=201,
             operation_id="createBike", summary="Create a bike")
async def create_bike(
    body: BikeCreate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = Bike(id=str(uuid.uuid4()), athlete_id=athlete.id, **body.model_dump())
    session.add(bike)
    await session.commit()
    await session.refresh(bike)
    return bike


async def _get_owned_bike(bike_id: str, athlete, session) -> Bike:
    result = await session.execute(
        select(Bike).where(Bike.id == bike_id, Bike.athlete_id == athlete.id)
    )
    bike = result.scalar_one_or_none()
    if bike is None:
        raise HTTPException(status_code=404, detail="Bike not found")
    return bike


@router.patch("/{bike_id}", response_model=BikeResponse,
              operation_id="updateBike", summary="Update a bike")
async def update_bike(
    bike_id: str,
    body: BikeUpdate,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(bike, field, value)
    await session.commit()
    await session.refresh(bike)
    return bike


@router.delete("/{bike_id}", status_code=204,
               operation_id="deleteBike", summary="Delete a bike")
async def delete_bike(
    bike_id: str,
    ctx_athlete=Depends(get_ctx_session_athlete),
):
    ctx, session, athlete = ctx_athlete
    bike = await _get_owned_bike(bike_id, athlete, session)
    # The FK is ON DELETE SET NULL, but PRAGMA foreign_keys is off on these
    # connections, so enforce it here: deleting a bike never deletes a course,
    # it just leaves the course needing a bike picked before re-analysis.
    await session.execute(
        update(Course).where(Course.bike_id == bike.id).values(bike_id=None)
    )
    await session.delete(bike)
    await session.commit()
