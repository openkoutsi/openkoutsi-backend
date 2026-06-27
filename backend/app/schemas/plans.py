from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel

from openkoutsi.plan_schema import DayConfig, PlanConfig  # noqa: F401 — re-exported for API layer


class PlannedWorkoutResponse(BaseModel):
    id: str
    plan_id: str
    week_number: int
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_tss: Optional[int] = None
    completed_activity_id: Optional[str] = None
    skip_reason: Optional[str] = None

    model_config = {"from_attributes": True}


class SkipWorkoutRequest(BaseModel):
    reason: str


class WorkoutCreate(BaseModel):
    """A single workout day as returned by the frontend LLM."""
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_tss: Optional[int] = None


class TrainingPlanCreate(BaseModel):
    name: str
    start_date: date
    weeks: int = 8
    goal: Optional[str] = None
    config: Optional[PlanConfig] = None
    use_llm: bool = False
    llm_weeks: Optional[list[list[WorkoutCreate]]] = None


class TrainingPlanUpdate(BaseModel):
    status: Optional[str] = None
    name: Optional[str] = None
    goal: Optional[str] = None
    start_date: Optional[date] = None
    weeks: Optional[int] = None


class PlannedWorkoutUpdate(BaseModel):
    """Editable fields of a single planned workout."""
    workout_type: Optional[str] = None
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_tss: Optional[int] = None
    day_of_week: Optional[int] = None
    week_number: Optional[int] = None


class PlannedWorkoutCreate(BaseModel):
    """A new planned workout added to an existing plan."""
    week_number: int
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_tss: Optional[int] = None


class RegeneratePlanRequest(BaseModel):
    """Re-run generation for an existing plan, preserving completed workouts."""
    config: Optional[PlanConfig] = None
    use_llm: bool = False
    weeks: Optional[int] = None
    goal: Optional[str] = None
    llm_weeks: Optional[list[list[WorkoutCreate]]] = None


class LinkActivityRequest(BaseModel):
    activity_id: str


class GenerateUpcomingWorkoutsRequest(BaseModel):
    """Optional explicit date range (within the upcoming-week window) and refresh flag."""
    start: Optional[date] = None
    end: Optional[date] = None
    refresh: bool = False  # regenerate cached workout definitions instead of reusing


class GenerateUpcomingResultItem(BaseModel):
    planned_workout_id: str
    date: date
    workout_type: Optional[str] = None
    workout_definition_id: Optional[str] = None
    status: str  # "generated" | "skipped" | "failed"
    reason: Optional[str] = None


class GenerateUpcomingWorkoutsResponse(BaseModel):
    results: list[GenerateUpcomingResultItem] = []


class TrainingPlanResponse(BaseModel):
    id: str
    athlete_id: str
    name: str
    start_date: date
    end_date: Optional[date] = None
    goal: Optional[str] = None
    weeks: Optional[int] = None
    status: str
    created_at: datetime
    workouts: list[PlannedWorkoutResponse] = []
    config: Optional[dict] = None
    generation_method: Optional[str] = None

    model_config = {"from_attributes": True}
