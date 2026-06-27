from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class ActivityUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    workout_category: Optional[str] = None
    labels: Optional[list[str]] = None
    notes: Optional[str] = Field(None, max_length=5000)


class FrontendAnalysisBody(BaseModel):
    analysis: str = Field(..., min_length=1)


class AnalyzeBody(BaseModel):
    locale: Optional[str] = None


class ManualActivityCreate(BaseModel):
    sport_type: str
    start_time: datetime
    duration_s: int = Field(..., gt=0)
    name: Optional[str] = None
    # TSS resolution (in priority order): explicit tss > rpe > avg_hr
    tss: Optional[float] = Field(None, ge=0)
    rpe: Optional[int] = Field(None, ge=1, le=10)
    avg_hr: Optional[float] = Field(None, gt=0)
    distance_m: Optional[float] = None
    elevation_m: Optional[float] = None


class IntervalResponse(BaseModel):
    interval_number: int
    start_offset_s: int
    duration_s: int
    distance_m: Optional[float] = None
    avg_hr: Optional[float] = None
    avg_power: Optional[float] = None
    avg_speed_ms: Optional[float] = None
    avg_cadence: Optional[float] = None
    is_auto_split: bool

    model_config = {"from_attributes": True}


class ActivityResponse(BaseModel):
    id: str
    athlete_id: str
    # List of provider names that contributed data to this activity,
    # e.g. ["wahoo", "strava"] or ["upload"].
    sources: list[str] = []
    name: Optional[str] = None
    sport_type: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_s: Optional[int] = None
    distance_m: Optional[float] = None
    elevation_m: Optional[float] = None
    avg_power: Optional[float] = None
    normalized_power: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    tss: Optional[float] = None
    intensity_factor: Optional[float] = None
    workout_category: Optional[str] = None
    labels: list[str] = []
    notes: Optional[str] = None
    has_fit_file: bool = False
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def _extract_sources(cls, data: Any) -> Any:
        """Populate `sources` from the ORM relationship when validating from an ORM object."""
        if hasattr(data, "sources"):
            return {
                "id": data.id,
                "athlete_id": data.athlete_id,
                "sources": [s.provider for s in (data.sources or [])],
                "name": data.name,
                "sport_type": data.sport_type,
                "start_time": data.start_time,
                "duration_s": data.duration_s,
                "distance_m": data.distance_m,
                "elevation_m": data.elevation_m,
                "avg_power": data.avg_power,
                "normalized_power": data.normalized_power,
                "avg_hr": data.avg_hr,
                "max_hr": data.max_hr,
                "tss": data.tss,
                "intensity_factor": data.intensity_factor,
                "workout_category": data.workout_category,
                "labels": data.labels or [],
                "notes": data.notes,
                "has_fit_file": data.has_fit_file,
                "status": data.status,
                "created_at": data.created_at,
            }
        return data


class ActivityListResponse(BaseModel):
    items: list[ActivityResponse]
    total: int
    page: int
    page_size: int


class ActivityStreamsResponse(BaseModel):
    streams: dict[str, list[Any]] = {}


class ActivityDetailResponse(ActivityResponse):
    streams: dict[str, list[Any]] = {}
    power_bests: dict[int, float] = {}
    distance_bests: dict[int, int] = {}
    power_pr_badges: dict[int, dict[str, str]] = {}
    distance_pr_badges: dict[int, dict[str, str]] = {}
    intervals: list[IntervalResponse] = []
    analysis_status: Optional[str] = None
    analysis: Optional[str] = None

    @classmethod
    def from_orm_and_streams(
        cls,
        activity,
        streams: dict[str, list],
        power_bests: dict[int, float] | None = None,
        distance_bests: dict[int, int] | None = None,
        intervals: list[IntervalResponse] | None = None,
        power_pr_badges: dict[int, dict[str, str]] | None = None,
        distance_pr_badges: dict[int, dict[str, str]] | None = None,
    ) -> "ActivityDetailResponse":
        return cls(
            id=activity.id,
            athlete_id=activity.athlete_id,
            sources=[s.provider for s in (activity.sources or [])],
            name=activity.name,
            sport_type=activity.sport_type,
            start_time=activity.start_time,
            duration_s=activity.duration_s,
            distance_m=activity.distance_m,
            elevation_m=activity.elevation_m,
            avg_power=activity.avg_power,
            normalized_power=activity.normalized_power,
            avg_hr=activity.avg_hr,
            max_hr=activity.max_hr,
            tss=activity.tss,
            intensity_factor=activity.intensity_factor,
            workout_category=activity.workout_category,
            labels=activity.labels or [],
            notes=activity.notes,
            has_fit_file=activity.has_fit_file,
            status=activity.status,
            created_at=activity.created_at,
            streams=streams,
            power_bests=power_bests or {},
            distance_bests=distance_bests or {},
            power_pr_badges=power_pr_badges or {},
            distance_pr_badges=distance_pr_badges or {},
            intervals=intervals or [],
            analysis_status=activity.analysis_status,
            analysis=activity.analysis,
        )
