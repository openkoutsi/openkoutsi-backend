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

from backend.app.db.base import TeamBase


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Athlete(TeamBase):
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


class WeightLog(TeamBase):
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


class Activity(TeamBase):
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
    normalized_power: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_hr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_speed_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_cadence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    workout_category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    labels: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    analysis_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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


class ActivitySource(TeamBase):
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


class ActivityStream(TeamBase):
    __tablename__ = "activity_streams"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    activity_id: Mapped[str] = mapped_column(
        String, ForeignKey("activities.id", ondelete="CASCADE")
    )
    stream_type: Mapped[str] = mapped_column(String)
    data: Mapped[list] = mapped_column(JSON)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="streams")


class ActivityPowerBest(TeamBase):
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

    activity: Mapped["Activity"] = relationship("Activity", back_populates="power_bests")


class ActivityDistanceBest(TeamBase):
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


class ActivityInterval(TeamBase):
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


class DailyMetric(TeamBase):
    __tablename__ = "daily_metrics"

    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    ctl: Mapped[float] = mapped_column(Float, default=0.0)
    atl: Mapped[float] = mapped_column(Float, default=0.0)
    tsb: Mapped[float] = mapped_column(Float, default=0.0)
    tss_day: Mapped[float] = mapped_column(Float, default=0.0)

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="daily_metrics")


class Goal(TeamBase):
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

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="goals")


class TrainingPlan(TeamBase):
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

    athlete: Mapped["Athlete"] = relationship("Athlete", back_populates="training_plans")
    workouts: Mapped[list["PlannedWorkout"]] = relationship(
        "PlannedWorkout", back_populates="plan", cascade="all, delete-orphan"
    )


class PlannedWorkout(TeamBase):
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
    target_tss: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completed_activity_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("activities.id", ondelete="SET NULL"), nullable=True
    )
    workout_definition_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("workout_definitions.id", ondelete="SET NULL"), nullable=True
    )
    skip_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    plan: Mapped["TrainingPlan"] = relationship("TrainingPlan", back_populates="workouts")


class WorkoutDefinition(TeamBase):
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
    estimated_tss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    athlete: Mapped["Athlete"] = relationship(
        "Athlete", back_populates="workout_definitions"
    )


class WahooWorkoutUpload(TeamBase):
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
