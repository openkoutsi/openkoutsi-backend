from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DistanceBestEntry(BaseModel):
    distance_m: int
    rank: int
    time_s: int
    activity_id: str
    activity_name: Optional[str]
    activity_start_time: Optional[datetime]


class AllTimeDistanceBestsResponse(BaseModel):
    bests: list[DistanceBestEntry]
