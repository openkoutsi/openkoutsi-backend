from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DayConfig(BaseModel):
    day_of_week: int  # 1=Mon … 7=Sun
    workout_type: str  # "threshold", "easy", "long", "strength", "yoga", …
    notes: Optional[str] = None


class PlanConfig(BaseModel):
    days_per_week: int
    day_configs: list[DayConfig]
    periodization: str = "base_building"  # "base_building" | "race_prep" | "maintenance"
    intensity_preference: str = "moderate"  # "low" | "moderate" | "high"
    long_description: Optional[str] = None  # free-text for LLM
