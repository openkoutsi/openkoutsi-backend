from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel


class GoalCreate(BaseModel):
    title: str
    description: Optional[str] = None
    target_date: Optional[date] = None
    metric: Optional[str] = None
    target_value: Optional[float] = None


class GoalUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_date: Optional[date] = None
    metric: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    status: Optional[str] = None
    outcome_note: Optional[str] = None


class GoalResponse(BaseModel):
    id: str
    athlete_id: str
    title: str
    description: Optional[str] = None
    target_date: Optional[date] = None
    metric: Optional[str] = None
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    status: str
    outcome_note: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
