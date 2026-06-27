from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class ZoneSchema(BaseModel):
    low: int
    high: int
    name: str


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


class TrainingStatusBody(BaseModel):
    locale: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    status: Optional[str] = None
    feedback: Optional[str] = None
    generated_date: Optional[date] = None
