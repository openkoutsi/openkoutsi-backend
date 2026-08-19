from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

RidingPosition = Literal["tops", "hoods", "drops", "aero"]


class BikeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    tyre_width_mm: Optional[int] = Field(default=None, ge=10, le=80)
    riding_position: RidingPosition = "hoods"


class BikeUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    tyre_width_mm: Optional[int] = Field(default=None, ge=10, le=80)
    riding_position: Optional[RidingPosition] = None


class BikeResponse(BaseModel):
    id: str
    name: str
    tyre_width_mm: Optional[int] = None
    riding_position: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
