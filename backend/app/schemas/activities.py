from datetime import datetime
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, model_validator

from openkoutsi.training_math import efficiency_factor, variability_index


def _aerobic_ratios(activity) -> dict[str, float | None]:
    """Efficiency factor and variability index, derived from stored columns.

    Both are pure ratios of ``weighted_power`` / ``avg_hr`` / ``avg_power``, so
    they are computed on read rather than persisted: nothing can drift out of
    sync with its operands, and activities processed before these metrics
    existed carry them without needing a reprocess.
    """
    ef = efficiency_factor(activity.weighted_power, activity.avg_hr)
    vi = variability_index(activity.weighted_power, activity.avg_power)
    return {
        "efficiency_factor": round(ef, 3) if ef is not None else None,
        "variability_index": round(vi, 3) if vi is not None else None,
    }


class ActivityUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    workout_category: Optional[str] = None
    labels: Optional[list[str]] = None
    notes: Optional[str] = Field(None, max_length=5000)
    rpe: Optional[int] = Field(None, ge=1, le=10)


class FrontendAnalysisBody(BaseModel):
    analysis: str = Field(..., min_length=1)


class AnalyzeBody(BaseModel):
    locale: Optional[str] = None


class ManualActivityCreate(BaseModel):
    """Payload for logging a workout by hand (no device recording / FIT file).

    Every field is optional so users can jot down as much or as little as they
    remember, but a completely empty submission is rejected.
    """

    sport_type: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_s: Optional[int] = Field(None, gt=0)
    name: Optional[str] = None
    # Load resolution (in priority order): explicit load > rpe > avg_hr.
    # rpe/avg_hr derivation additionally require duration_s to be set.
    load: Optional[float] = Field(None, ge=0)
    rpe: Optional[int] = Field(None, ge=1, le=10)
    avg_hr: Optional[float] = Field(None, gt=0)
    max_hr: Optional[float] = Field(None, gt=0)
    avg_power: Optional[float] = Field(None, gt=0)
    avg_cadence: Optional[float] = Field(None, ge=0)
    distance_m: Optional[float] = Field(None, ge=0)
    elevation_m: Optional[float] = None

    @model_validator(mode="after")
    def _require_at_least_one_field(self) -> "ManualActivityCreate":
        if not any(
            getattr(self, field) is not None for field in type(self).model_fields
        ):
            raise ValueError("At least one field must be provided")
        return self


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
    weighted_power: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    avg_cadence: Optional[float] = None
    load: Optional[float] = None
    intensity: Optional[float] = None
    # Aerobic response metrics (issue #37). `efficiency_factor` (weighted power
    # per heartbeat) and `variability_index` (weighted / average power) are
    # derived on read from the columns above. `decoupling_pct` is the stored
    # power:HR drift over the ride; when it is null `decoupling_reason` says why
    # a figure would be misleading — one of `too_short`, `no_power`, `no_hr`,
    # `degenerate_hr`, `variable_effort`.
    efficiency_factor: Optional[float] = None
    variability_index: Optional[float] = None
    decoupling_pct: Optional[float] = None
    decoupling_reason: Optional[str] = None
    workout_category: Optional[str] = None
    labels: list[str] = []
    notes: Optional[str] = None
    rpe: Optional[int] = None
    has_fit_file: bool = False
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}

    # Field names this schema shares with an ORM relationship of the same name.
    # `_orm_values` skips them so the caller's own value wins instead of a lazy
    # relationship load — see `ActivityDetailResponse`.
    _ORM_OVERRIDES: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _orm_values(cls, activity) -> dict:
        """This schema's fields read off an ORM ``Activity``.

        Every field except the three derived ones is a straight column read, so
        they are pulled by name from the schema's own field list rather than
        restated here: declaring a field is enough to have it populated.
        """
        values = {
            name: getattr(activity, name)
            for name in cls.model_fields
            if name not in cls._ORM_OVERRIDES and hasattr(activity, name)
        }
        return {
            **values,
            "sources": [s.provider for s in (activity.sources or [])],
            "labels": activity.labels or [],
            **_aerobic_ratios(activity),
        }

    @model_validator(mode="before")
    @classmethod
    def _extract_sources(cls, data: Any) -> Any:
        """Derive `sources` and the aerobic ratios when validating from an ORM object."""
        if hasattr(data, "sources"):
            return cls._orm_values(data)
        return data


class ActivityListResponse(BaseModel):
    items: list[ActivityResponse]
    total: int
    page: int
    page_size: int


class RpeQueueResponse(BaseModel):
    """Pending RPE-rating queue for the dashboard/post-upload prompt (issue #28).

    ``items`` are qualifying cycling activities ingested after the athlete's
    ``rpe_head`` cursor that still lack an RPE, oldest-first. ``rpe_head`` is the
    server-side cursor (an activity ``created_at`` ISO timestamp) marking the
    boundary between already-handled and new activities.
    """

    items: list[ActivityResponse] = []
    rpe_head: Optional[str] = None


class ActivityStreamsResponse(BaseModel):
    streams: dict[str, list[Any]] = {}


class ActivityDetailResponse(ActivityResponse):
    streams: dict[str, list[Any]] = {}
    power_bests: dict[int, float] = {}
    distance_bests: dict[int, int] = {}
    power_pr_badges: dict[int, dict[str, str]] = {}
    distance_pr_badges: dict[int, dict[str, str]] = {}
    intervals: list[IntervalResponse] = []
    # CP (watts) and W' (joules) the `w_bal` stream was integrated with, frozen
    # at processing time from the athlete's power bests as of this activity's
    # date. Both null — and no `w_bal` in `streams` — when CP couldn't be fit.
    cp_w: Optional[float] = None
    w_prime_j: Optional[float] = None
    # How many duration bests the CP fit used. Low values mean the fit was made
    # against a thin power profile — typically an old ride processed early in a
    # provider backlog import, before the surrounding history existed.
    cp_fit_points: Optional[int] = None
    analysis_status: Optional[str] = None
    analysis: Optional[str] = None

    # `Activity` carries relationships under these names too; the caller passes
    # the already-shaped values, so never read them off the ORM object.
    _ORM_OVERRIDES: ClassVar[frozenset[str]] = frozenset(
        {"streams", "power_bests", "distance_bests", "intervals"}
    )

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
        return cls.model_validate({
            **cls._orm_values(activity),
            "streams": streams,
            "power_bests": power_bests or {},
            "distance_bests": distance_bests or {},
            "power_pr_badges": power_pr_badges or {},
            "distance_pr_badges": distance_pr_badges or {},
            "intervals": intervals or [],
        })
