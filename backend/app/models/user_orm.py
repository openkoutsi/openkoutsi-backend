import uuid
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import UserBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Athlete(UserBase):
    __tablename__ = "athletes"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # References registry users.id — no FK constraint (cross-DB boundary)
    global_user_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ftp: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_hr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    resting_hr: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    hr_zones: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    power_zones: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    availability: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ftp_tests: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, default=list)
    app_settings: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    avatar_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    training_status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_status_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    training_status_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    training_status_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # What the agentic coach is doing right now, as a progress *code* from a
    # fixed vocabulary (issue #43) — `thinking`, `tool.get_power_profile`, … The
    # agent loop spends its first few round trips calling tools and emitting no
    # prose at all, so without this the card would show a bare spinner for a
    # long time and then dump a finished answer. Deliberately a separate column
    # rather than an envelope inside `training_status`: the frontend's
    # `parseMoodAndParagraphs` reads that column as raw prose, and three
    # surfaces share the parser. Cleared the moment the prose starts, so a
    # finished card looks exactly as it did before this existed.
    training_status_progress: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="athlete"
    )
    goals: Mapped[list["Goal"]] = relationship("Goal", back_populates="athlete")
    daily_metrics: Mapped[list["DailyMetric"]] = relationship(
        "DailyMetric", back_populates="athlete"
    )
    training_plans: Mapped[list["TrainingPlan"]] = relationship(
        "TrainingPlan", back_populates="athlete"
    )
    weight_log: Mapped[list["WeightLog"]] = relationship(
        "WeightLog", back_populates="athlete", cascade="all, delete-orphan"
    )
    workout_definitions: Mapped[list["WorkoutDefinition"]] = relationship(
        "WorkoutDefinition", back_populates="athlete", cascade="all, delete-orphan"
    )


class WeightLog(UserBase):
    __tablename__ = "weight_log"
    __table_args__ = (UniqueConstraint("athlete_id", "effective_date"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE")
    )
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="weight_log")


class Activity(UserBase):
    __tablename__ = "activities"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE")
    )
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sport_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weighted_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_speed_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    load: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    workout_category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Accumulated time-in-zone captured at processing time, using the athlete's
    # zone definitions in effect then. Shape: {"hr": {"Z1": secs, ...},
    # "power": {...}}. Frozen once set — editing zones later never rewrites it,
    # so historical weekly zone distributions stay stable (issue #27).
    zone_times: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Aerobic decoupling (issue #37): how far the power:HR ratio drifted between
    # the first and second half of the ride, as a percentage. NULL when the ride
    # doesn't support a meaningful figure (too short, no HR, interval session);
    # `decoupling_reason` then carries a stable reason code — see
    # `openkoutsi.training_math.decoupling_unavailable_reason`. Exactly one of
    # the two is set.
    decoupling_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    decoupling_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # CP (watts) and W' (joules) the `w_bal` stream was integrated with, fit from
    # the athlete's power bests as they stood *on this activity's date*. Frozen
    # like `zone_times`: a ride's W' story shouldn't silently change months later
    # as the athlete's power curve moves. Both NULL when CP couldn't be fit, in
    # which case no `w_bal` stream is stored either.
    cp_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    w_prime_j: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # How many duration bests the CP fit had available. A provider backlog import
    # walks newest-first, so an old ride can be processed while almost nothing on
    # or before its date exists yet — this records that, so those rides stay
    # findable for a future re-fit instead of being indistinguishable from a
    # well-supported one. Set even when the fit was rejected.
    cp_fit_points: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    labels: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Rate of Perceived Exertion (RPE): athlete's subjective 1–10 effort score
    # for the ride. Nullable until the athlete rates it (issue #28).
    rpe: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    analysis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The agentic coach's current step, as a progress code (issue #43). Mirrors
    # `Athlete.training_status_progress` — see the note there for why the codes
    # live in their own column instead of inside the prose.
    analysis_progress: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # When the analysis last moved (issue #91). Written when the run is
    # triggered and touched on every progress commit, so it is an *inactivity*
    # clock rather than a start time — the same role
    # `Athlete.training_status_updated_at` and `Goal.guidance_updated_at` play
    # for the other two LLM surfaces. Without it a `pending` row left behind by
    # a killed process could never be aged out, and `trigger_analysis`
    # early-returns on `pending`, so that activity could never be analysed again.
    analysis_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="activities")
    sources: Mapped[list["ActivitySource"]] = relationship(
        "ActivitySource", back_populates="activity",
        cascade="all, delete-orphan", lazy="selectin",
    )
    streams: Mapped[list["ActivityStream"]] = relationship(
        "ActivityStream", back_populates="activity", cascade="all, delete-orphan"
    )
    power_bests: Mapped[list["ActivityPowerBest"]] = relationship(
        "ActivityPowerBest", back_populates="activity", cascade="all, delete-orphan"
    )
    distance_bests: Mapped[list["ActivityDistanceBest"]] = relationship(
        "ActivityDistanceBest", back_populates="activity", cascade="all, delete-orphan"
    )
    intervals: Mapped[list["ActivityInterval"]] = relationship(
        "ActivityInterval", back_populates="activity",
        cascade="all, delete-orphan", order_by="ActivityInterval.interval_number",
        lazy="selectin",
    )

    @property
    def has_fit_file(self) -> bool:
        return any(s.fit_file_path for s in self.sources)


class ActivitySource(UserBase):
    """Tracks which providers have contributed data to a single Activity."""

    __tablename__ = "activity_sources"
    __table_args__ = (
        UniqueConstraint("activity_id", "provider", name="uq_activity_sources_activity_provider"),
        Index("ix_activity_sources_provider_external_id", "provider", "external_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    fit_file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    fit_file_encrypted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="sources", lazy="selectin")


class ActivityStream(UserBase):
    __tablename__ = "activity_streams"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE")
    )
    stream_type: Mapped[str] = mapped_column(String)
    data: Mapped[list] = mapped_column(JSON)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="streams")


class ActivityPowerBest(UserBase):
    __tablename__ = "activity_power_bests"
    __table_args__ = (UniqueConstraint("activity_id", "duration_s"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE")
    )
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True
    )
    duration_s: Mapped[int] = mapped_column(Integer)
    power_w: Mapped[float] = mapped_column(Float)
    activity_start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Effective bodyweight at the time of the activity and the resulting W/kg.
    # NULL when no weight was logged on or before the activity date.
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    w_per_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="power_bests")


class ActivityDistanceBest(UserBase):
    __tablename__ = "activity_distance_bests"
    __table_args__ = (UniqueConstraint("activity_id", "distance_m"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE")
    )
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True
    )
    distance_m: Mapped[int] = mapped_column(Integer)
    time_s: Mapped[int] = mapped_column(Integer)
    activity_start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    activity: Mapped["Activity"] = relationship("Activity", back_populates="distance_bests")


class ActivityInterval(UserBase):
    __tablename__ = "activity_intervals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE")
    )
    interval_number: Mapped[int] = mapped_column(Integer)
    start_offset_s: Mapped[int] = mapped_column(Integer)
    duration_s: Mapped[int] = mapped_column(Integer)
    distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_speed_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_auto_split: Mapped[bool] = mapped_column(Boolean, default=False)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="intervals")


class DailyMetric(UserBase):
    __tablename__ = "daily_metrics"

    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    fitness: Mapped[float] = mapped_column(Float, default=0.0)
    fatigue: Mapped[float] = mapped_column(Float, default=0.0)
    form: Mapped[float] = mapped_column(Float, default=0.0)
    load_day: Mapped[float] = mapped_column(Float, default=0.0)

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="daily_metrics")


class PlanAdherenceDaily(UserBase):
    """Daily snapshot of a training plan's adherence score (issue #26).

    One row per active plan per day, mirroring the shape/pattern of
    ``DailyMetric`` (Fitness/Fatigue/Form). ``score`` is the "so far" Load-
    weighted adherence percentage as of ``date``; the counts are denormalised
    for cheap charting/summaries.
    """

    __tablename__ = "plan_adherence_daily"

    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), primary_key=True
    )
    plan_id: Mapped[str] = mapped_column(
        String, ForeignKey("training_plans.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    missed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    pending: Mapped[int] = mapped_column(Integer, default=0)


class AchievementUnlock(UserBase):
    """One earned achievement tier (issue #33).

    Unlocks are *derived state*, like ``DailyMetric`` and ``PlanAdherenceDaily``:
    the recompute in ``backend.app.services.achievements`` treats this table as a
    projection of the athlete's current data and rewrites it in place, so a
    deleted activity can revoke a tier rather than leaving a badge the data no
    longer supports.

    The two timestamps do different jobs. ``achieved_on`` is derived from the
    history — the day the criterion was actually first met — so back-filling an
    old ride moves it *earlier* instead of re-dating the unlock to today.
    ``created_at`` is wall-clock and only says when we first noticed, which is
    what the "new" marker and the inbox notification key off.

    The catalogue itself lives in code (``openkoutsi.achievements``), not in
    rows: ``achievement_id`` is a stable machine key whose display name is an
    i18n string in the web app, so no user-facing prose is stored here.
    """

    __tablename__ = "achievement_unlocks"

    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), primary_key=True
    )
    achievement_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Float so a future fractional tier (a 3.0 W/kg badge) fits without a
    # migration. The reconcile matches stored rows to computed ones on tier
    # *equality*, so every catalogue tier must be exactly representable as a
    # float — a value like 2.5 is fine, 0.1 is not. A non-representable tier
    # would never match, so every recompute would delete and re-insert the row,
    # announcing the badge again on every single upload.
    # ``test_achievements_math`` asserts the catalogue holds to this.
    tier: Mapped[float] = mapped_column(Float, primary_key=True)
    achieved_on: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Whether the athlete has looked at this badge in the UI. There is no
    # matching "notified" flag: the inbox message is driven by which rows a
    # reconcile *inserted*, and a reconcile over unchanged data inserts nothing,
    # so a stored flag would have had nothing left to dedupe.
    seen: Mapped[bool] = mapped_column(Boolean, default=False)
    # Optional deep-link payload for this specific tier, e.g. {"activity_id": …}
    # or {"plan_id": …}. Re-derived on every reconcile, so a link to a deleted
    # activity doesn't linger.
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class Goal(UserBase):
    __tablename__ = "goals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    metric: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    outcome_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # On-demand AI guidance (issue #17): mirrors Athlete.training_status* — the
    # streamed coach prose, its parsed REALISM verdict, the pending/done/error
    # state, and a timestamp for pending-timeout recovery.
    guidance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    guidance_verdict: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    guidance_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    guidance_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="goals")


class TrainingPlan(UserBase):
    __tablename__ = "training_plans"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String)
    start_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    goal: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    weeks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    generation_method: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Per-week metadata (build vs recovery week, focus note, target weekly
    # Load/hours, base Load) — a list of dicts, populated at generation time.
    week_meta: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="training_plans")
    workouts: Mapped[list["PlannedWorkout"]] = relationship(
        "PlannedWorkout", back_populates="plan", cascade="all, delete-orphan"
    )


class PlannedWorkout(UserBase):
    __tablename__ = "planned_workouts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(
        String, ForeignKey("training_plans.id", ondelete="CASCADE")
    )
    week_number: Mapped[int] = mapped_column(Integer, default=1)
    day_of_week: Mapped[int] = mapped_column(Integer, default=1)  # 1=Mon, 7=Sun
    workout_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_load: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    workout_definition_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("workout_definitions.id", ondelete="SET NULL"), nullable=True
    )
    skip_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    plan: Mapped["TrainingPlan"] = relationship("TrainingPlan", back_populates="workouts")
    # A single real-world session can be recorded as several activities (an
    # accidental stop, a break, back-to-back rides). A planned workout may
    # therefore link to many activities; each activity links to at most one
    # planned workout (enforced by the UniqueConstraint on the join table).
    linked_activities: Mapped[list["Activity"]] = relationship(
        "Activity",
        secondary="planned_workout_activities",
        order_by="Activity.start_time",
        lazy="selectin",
    )

    @property
    def is_completed(self) -> bool:
        """A planned workout is completed once at least one activity is linked."""
        return bool(self.linked_activities)


class PlannedWorkoutActivity(UserBase):
    """Join table linking a planned workout to one or more completing activities.

    The ``UniqueConstraint`` on ``activity_id`` keeps an activity linked to at
    most one planned workout, while a planned workout may hold many activities.
    """

    __tablename__ = "planned_workout_activities"

    planned_workout_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("planned_workouts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    activity_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("activities.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (
        UniqueConstraint("activity_id", name="uq_planned_workout_activities_activity_id"),
    )


class WorkoutDefinition(UserBase):
    __tablename__ = "workout_definitions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sport_type: Mapped[str] = mapped_column(String, nullable=False, default="Ride")
    steps: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    estimated_duration_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_load: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    athlete: Mapped["Athlete"] = relationship(
        "Athlete", back_populates="workout_definitions"
    )


class WahooWorkoutUpload(UserBase):
    """Tracks structured workouts pushed to Wahoo so re-pushes update in place.

    The ``external_id`` is deterministic per workout definition, letting Wahoo
    de-duplicate the plan record; the returned plan/workout ids are stored so we
    can issue PUT updates instead of creating duplicates.
    """

    __tablename__ = "wahoo_workout_uploads"
    __table_args__ = (
        UniqueConstraint("athlete_id", "external_id", name="uq_wahoo_upload_external"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    workout_definition_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("workout_definitions.id", ondelete="SET NULL"), nullable=True
    )
    external_id: Mapped[str] = mapped_column(String, nullable=False)
    wahoo_plan_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    wahoo_workout_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    starts: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provider_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
