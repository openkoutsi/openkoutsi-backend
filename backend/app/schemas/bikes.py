"""Bike, maintenance and accessory payloads (issues #55, #64).

A bike started as the small equipment concept the course physics reads; issue
#64 promoted the same row into the garage — what the athlete owns, how far it
has gone and what has been done to it. One record, two readers: the garage
edits exactly the rows the route-analysis bike selector lists, which is what
makes those two agree by construction rather than by synchronisation.
"""
from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

RidingPosition = Literal["tops", "hoods", "drops", "aero"]


def _reject_explicit_null(*fields: str):
    """Refuse an explicit ``null`` on a column that is NOT NULL.

    A PATCH schema declares every field ``Optional`` so it can be *omitted*, but
    ``exclude_unset=True`` keeps a value the client actually sent as ``null``, so
    it reaches ``setattr`` and dies at flush on the constraint — an unhandled 500
    over a rolled-back session, for an intent the API does not support anyway.

    A validator turns that into the 422 it always was. Only the NOT NULL columns
    are listed: ``retired_at``, ``odometer_base_km``, ``tyre_width_mm``,
    ``default_sports`` and ``note`` are genuinely nullable, and ``null`` on them
    is a request to clear — which is how a bike is un-retired.
    """

    @field_validator(*fields, mode="before")
    @classmethod
    def _validate(cls, value, info):
        if value is None:
            raise ValueError(f"{info.field_name} cannot be null")
        return value

    return _validate



class BikeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    tyre_width_mm: Optional[int] = Field(default=None, ge=10, le=80)
    riding_position: RidingPosition = "hoods"
    # Kilometres ridden before openkoutsi ever saw the bike. Without it every
    # wear figure reads low and every maintenance interval is wrong.
    odometer_base_km: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    # Cycling `sport_type` values this bike claims, for automapping. Normalised
    # server-side, so `gravel_ride` and `GravelRide` are the same claim; a
    # non-cycling sport is a 422 and a sport another bike already holds is a
    # 409 naming that bike.
    default_sports: Optional[list[str]] = None


class BikeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    tyre_width_mm: Optional[int] = Field(default=None, ge=10, le=80)
    riding_position: Optional[RidingPosition] = None
    odometer_base_km: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    default_sports: Optional[list[str]] = None
    _no_null = _reject_explicit_null("name", "riding_position")
    # When the bike left the fleet. A retired bike drops out of the pickers but
    # keeps its rides, its distance and its maintenance history — send `null`
    # to bring it back. Deleting is not the same thing and is not a substitute:
    # it would silently rewrite the athlete's past totals.
    retired_at: Optional[datetime] = None


class BikeResponse(BaseModel):
    id: str
    name: str
    tyre_width_mm: Optional[int] = None
    riding_position: str
    odometer_base_km: Optional[float] = None
    default_sports: list[str] = []
    retired_at: Optional[datetime] = None
    # Kept apart on purpose. `tracked_km` is what openkoutsi actually observed —
    # the sum of the rides assigned to this bike — while `lifetime_km` adds the
    # athlete's own baseline on top. A garage that blurred the two would be
    # unable to say where a disputed number came from.
    tracked_km: float = 0.0
    lifetime_km: float = 0.0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MaintenanceCreate(BaseModel):
    performed_on: date
    # Free-form so the vocabulary stays open, the way `Activity.labels` is
    # handled. `BikeMaintenance.COMPONENTS` is the suggested starting list and
    # is what the UI offers; anything else is stored as given.
    component: str = Field(min_length=1, max_length=50)
    # The absolute reading at the time — never an offset. It must not move when
    # history is re-imported, a baseline is corrected or a ride is reassigned.
    odometer_km: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    note: Optional[str] = Field(default=None, max_length=2000)


class MaintenanceUpdate(BaseModel):
    performed_on: Optional[date] = None
    component: Optional[str] = Field(default=None, min_length=1, max_length=50)
    odometer_km: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    note: Optional[str] = Field(default=None, max_length=2000)
    _no_null = _reject_explicit_null("performed_on", "component")


class MaintenanceResponse(BaseModel):
    id: str
    bike_id: str
    performed_on: date
    component: str
    odometer_km: Optional[float] = None
    note: Optional[str] = None
    # How far the part replaced *at this entry* had run: the gap to the previous
    # entry for the same component. Null when that span is genuinely unknown —
    # nothing of this component came before, or one of the two readings is
    # missing. Null is not zero, and reporting zero would be a lie about wear.
    previous_component_km: Optional[float] = None
    # How far the bike has run since this entry. On the newest entry for a
    # component this is the open-ended figure the athlete is really after: the
    # tyres fitted at 4 200 km on a bike now at 6 000 have done 1 800.
    km_since: Optional[float] = None
    is_current: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AccessoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    note: Optional[str] = Field(default=None, max_length=2000)


class AccessoryUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    note: Optional[str] = Field(default=None, max_length=2000)
    _no_null = _reject_explicit_null("name")


class AccessoryResponse(BaseModel):
    id: str
    bike_id: str
    name: str
    note: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BikeDetailResponse(BikeResponse):
    """One bike with everything hanging off it — the garage's detail view."""

    maintenance: list[MaintenanceResponse] = []
    accessories: list[AccessoryResponse] = []


class AssignHistoryResponse(BaseModel):
    """What ``POST /api/bikes/assign-history`` did.

    ``scanned`` counts the unassigned rides looked at, not the whole history:
    anything already carrying a bike is skipped before it is examined, because
    this pass must never re-home a ride.
    """

    scanned: int
    assigned: int
