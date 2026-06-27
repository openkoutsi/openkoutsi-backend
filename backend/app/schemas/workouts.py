from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from openkoutsi.workout_schema import WorkoutStepOrRepeat  # noqa: F401 — re-exported for API layer


class WorkoutDefinitionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    sport_type: str = "Ride"
    steps: list[WorkoutStepOrRepeat] = Field(default_factory=list)


class WorkoutDefinitionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    sport_type: Optional[str] = None
    steps: Optional[list[WorkoutStepOrRepeat]] = None


class WorkoutDefinitionResponse(BaseModel):
    id: str
    athlete_id: str
    name: str
    description: Optional[str] = None
    sport_type: str
    steps: list[dict]
    estimated_duration_s: Optional[int] = None
    estimated_tss: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExportFormatInfo(BaseModel):
    key: str
    label: str
    file_extension: str
    mime_type: str


class WahooPushRequest(BaseModel):
    starts: Optional[datetime] = Field(
        None,
        description="When to schedule the workout. Must be within today→+6 days. "
        "Defaults to now.",
    )


class WahooPushResponse(BaseModel):
    plan_id: str
    workout_id: str
    starts: datetime
