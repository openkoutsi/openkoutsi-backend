from datetime import date, datetime
from typing import Any, Optional
from pydantic import BaseModel, model_validator

from openkoutsi.plan_schema import DayConfig, PlanConfig  # noqa: F401 — re-exported for API layer


class PlannedWorkoutResponse(BaseModel):
    id: str
    plan_id: str
    week_number: int
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_load: Optional[int] = None
    # All activities linked to this workout (may be several when one session was
    # recorded as multiple activities). ``completed_activity_id`` is derived from
    # the first of these and kept for backward compatibility.
    linked_activity_ids: list[str] = []
    completed_activity_id: Optional[str] = None
    skip_reason: Optional[str] = None
    # Derived per-workout adherence match score (0–100). Null until the workout
    # is completed or past — set by the API layer, which has the plan's dates.
    match_score: Optional[float] = None

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _derive_links(cls, data: Any) -> Any:
        """Populate the link fields from the ORM ``linked_activities`` relationship."""
        if isinstance(data, dict):
            return data
        linked = getattr(data, "linked_activities", None)
        if linked is None:
            return data
        ids = [a.id for a in linked]
        # Build a plain dict so the derived fields are always consistent.
        return {
            "id": data.id,
            "plan_id": data.plan_id,
            "week_number": data.week_number,
            "day_of_week": data.day_of_week,
            "workout_type": data.workout_type,
            "description": data.description,
            "duration_min": data.duration_min,
            "target_load": data.target_load,
            "linked_activity_ids": ids,
            "completed_activity_id": ids[0] if ids else None,
            "skip_reason": data.skip_reason,
        }


class PlanAdherenceSummary(BaseModel):
    """Small breakdown of a plan's adherence, alongside the current score."""
    completed: int = 0
    missed: int = 0
    skipped: int = 0
    pending: int = 0
    # Sessions still to do from today onward: future workouts + today's un-acted
    # workout (the ``pending`` grace bucket).
    remaining: int = 0


class PlanAdherencePoint(BaseModel):
    """One persisted daily adherence snapshot, for charting the trend."""
    date: date
    score: Optional[float] = None
    completed: int = 0
    missed: int = 0
    skipped: int = 0
    pending: int = 0

    model_config = {"from_attributes": True}


class SkipWorkoutRequest(BaseModel):
    reason: str


class WorkoutCreate(BaseModel):
    """A single workout day as returned by the frontend LLM."""
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_load: Optional[int] = None


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
    target_load: Optional[int] = None
    day_of_week: Optional[int] = None
    week_number: Optional[int] = None


class PlannedWorkoutCreate(BaseModel):
    """A new planned workout added to an existing plan."""
    week_number: int
    day_of_week: int
    workout_type: str
    description: Optional[str] = None
    duration_min: Optional[int] = None
    target_load: Optional[int] = None


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
    # Current "so far" adherence score (0–100) and its breakdown. Null when the
    # plan has nothing contributing yet (empty / just-started).
    adherence_score: Optional[float] = None
    adherence_summary: Optional[PlanAdherenceSummary] = None

    model_config = {"from_attributes": True}
