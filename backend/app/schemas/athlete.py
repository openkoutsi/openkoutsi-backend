from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from openkoutsi.zones import HR_ZONE_COUNT, POWER_ZONE_COUNT


class ZoneSchema(BaseModel):
    low: int
    high: int
    name: str


def _validate_zone_list(zones: Optional[list[ZoneSchema]], expected: int, label: str):
    """Enforce the fixed zone model on write (issue #38).

    Zone lists used to be free-form: any length, any names. Anything built on
    top of them then had to guess what a given zone meant, which is what made
    the three-band intensity mapping hard. They are now fixed at
    ``POWER_ZONE_COUNT`` / ``HR_ZONE_COUNT`` entries, ascending and
    non-overlapping.

    The ordering checks used to live only in ``Zones.validate()``, which runs
    lazily from ``time_in_zones``. A malformed list saved through the API
    therefore passed here and blew up much later, while processing an activity.
    Validating on the way in turns that into a 422 at the point of the mistake.
    """
    if zones is None:
        return None

    if len(zones) != expected:
        raise ValueError(
            f"{label} must have exactly {expected} zones, got {len(zones)}"
        )

    for i, zone in enumerate(zones):
        if zone.high <= zone.low:
            raise ValueError(
                f"{label} Z{i + 1}: high ({zone.high}) must be above low ({zone.low})"
            )
        if i and zone.low < zones[i - 1].high:
            raise ValueError(
                f"{label} Z{i + 1}: low ({zone.low}) must not be below "
                f"Z{i} high ({zones[i - 1].high})"
            )

    return zones


class FtpTestSchema(BaseModel):
    date: str
    ftp: int
    method: str = "test"


class AthleteResponse(BaseModel):
    id: str
    user_id: str
    name: Optional[str] = None
    date_of_birth: Optional[date] = None
    weight_kg: Optional[float] = None
    ftp: Optional[int] = None
    max_hr: Optional[int] = None
    resting_hr: Optional[int] = None
    hr_zones: list[ZoneSchema] = []
    power_zones: list[ZoneSchema] = []
    ftp_tests: list[FtpTestSchema] = []
    connected_providers: list[str] = []
    app_settings: dict = {}
    avatar_url: Optional[str] = None
    consent_accepted: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("hr_zones", "power_zones", "ftp_tests", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        return v if v is not None else []

    @field_validator("app_settings", mode="before")
    @classmethod
    def coerce_app_settings(cls, v):
        return v if isinstance(v, dict) else {}

class AthleteUpdate(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[date] = None
    weight_kg: Optional[float] = None
    ftp: Optional[int] = None
    max_hr: Optional[int] = None
    resting_hr: Optional[int] = None
    hr_zones: Optional[list[ZoneSchema]] = None
    power_zones: Optional[list[ZoneSchema]] = None
    app_settings: Optional[dict] = None
    ftp_test_method: Optional[Literal["manual", "20min", "cp"]] = None

    @field_validator("hr_zones")
    @classmethod
    def _check_hr_zones(cls, v):
        return _validate_zone_list(v, HR_ZONE_COUNT, "hr_zones")

    @field_validator("power_zones")
    @classmethod
    def _check_power_zones(cls, v):
        return _validate_zone_list(v, POWER_ZONE_COUNT, "power_zones")


class TrainingStatusBody(BaseModel):
    locale: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    status: Optional[str] = None
    feedback: Optional[str] = None
    generated_date: Optional[date] = None
