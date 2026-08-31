from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class CourseSegmentResponse(BaseModel):
    segment_index: int
    start_distance_m: float
    end_distance_m: float
    length_m: float
    avg_gradient: float
    elevation_change_m: float
    segment_type: str
    power_w: Optional[float] = None
    speed_ms: Optional[float] = None
    duration_s: Optional[float] = None
    start_offset_s: Optional[float] = None
    speed_capped: bool = False
    # Surface classification (issue #56). All optional: a course analysed
    # before this landed, or on an instance with no matcher, reads exactly as
    # it always did rather than as a course with something missing.
    surface: Optional[str] = None
    #: "confirmed" when only an explicit OSM tag could have produced this
    #: class; "inferred" when openkoutsi could not confirm one. Carried
    #: separately, never folded into `surface`, because a guess and a fact
    #: shown at equal weight is worse than no answer at all.
    surface_confidence: Optional[str] = None
    surface_raw: Optional[str] = None
    crr_used: Optional[float] = None

    model_config = {"from_attributes": True}


class CourseSummaryResponse(BaseModel):
    id: str
    name: str
    goal_id: Optional[str] = None
    bike_id: Optional[str] = None
    status: str
    distance_m: float
    elevation_gain_m: Optional[float] = None
    # At most one of these is ever set: a course is paced to a finish time or
    # to an average power, never to both.
    target_time_s: Optional[int] = None
    target_power_w: Optional[int] = None
    start_time: Optional[datetime] = None
    predicted_time_s: Optional[float] = None
    feasible: Optional[bool] = None
    refusal_reason: Optional[str] = None
    plan_status: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CourseDetailResponse(CourseSummaryResponse):
    error: Optional[str] = None
    ftp_w_used: Optional[float] = None
    weight_kg_used: Optional[float] = None
    elevation_loss_m: Optional[float] = None
    min_elevation_m: Optional[float] = None
    max_elevation_m: Optional[float] = None
    intensity: Optional[float] = None
    required_intensity: Optional[float] = None
    # ≤400 × [distance_m, elevation_m, gradient] — the chart payload. The
    # course's *track* (coordinates) is deliberately absent from every
    # response shape in this module.
    profile: Optional[list] = None
    segments: list[CourseSegmentResponse] = []
    # ── Surface classification (issue #56) ──────────────────────────────────
    #: None = never matched (an instance with no sidecar, or a course from
    #: before the feature); "pending" while the background match runs; "done";
    #: "unavailable" when the matcher could not answer. None of these is an
    #: error — the course itself is complete either way.
    surface_status: Optional[str] = None
    surface_updated_at: Optional[datetime] = None
    #: Whether this instance could match this course if asked. Published here
    #: rather than on the public instance info because it is deployment
    #: topology, and an unauthenticated caller has no business learning it.
    surface_matching_available: bool = False
    #: The surface at full run resolution, run-length encoded as
    #: ``[start_m, end_m, class, confidence, severity_step]``.
    #:
    #: Separate from `segments` because the two have different jobs: the
    #: segment table is pacing-shaped and has a minimum row length, and this
    #: has none. A 130 m sector of mud inside 40 km of asphalt is the most
    #: important thing on that course and has to stay drawable and nameable
    #: even where the pacing rows fold it into a longer one.
    surface_ribbon: Optional[list] = None
    #: Stretches worth saying out loud, as
    #: ``[start_m, length_m, class, confidence, severity_step]`` — including
    #: ones too short to earn their own pacing row.
    rough_sectors: Optional[list] = None


class CourseReanalyzeBody(BaseModel):
    """Partial update for re-analysis. A field left unset keeps the stored
    value; an explicit null clears it (goal, target, start time).

    The two targets are alternatives, so setting one to a value clears the
    other — that is what "switch this course to a power target" means, and
    making the caller send the null as well would only be a way to get it
    wrong. Sending both as values is a 422: it has no meaning to guess at.
    """

    bike_id: Optional[str] = None
    goal_id: Optional[str] = None
    target_time_s: Optional[int] = Field(default=None, gt=0)
    target_power_w: Optional[int] = Field(default=None, gt=0)
    start_time: Optional[datetime] = None


class CoursePlanBody(BaseModel):
    locale: Optional[str] = None


class CoursePlanResponse(BaseModel):
    status: Optional[str] = None
    mood: Optional[str] = None
    plan: Optional[str] = None
    updated_at: Optional[datetime] = None

    # Issue #41 AI disclosure — same derivation as GoalGuidanceResponse.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def plan_ai_generated(self) -> bool:
        """True when `plan` was generated by a language model."""
        return bool(self.plan)
