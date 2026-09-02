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
from backend.app.db.leases import LeaseMixin


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
    # The run that owns the columns above: the heartbeat says whether a run is
    # *alive*, this says whether its writes are still *wanted*. A slow run whose
    # token no longer matches discards its own writes rather than committing
    # over the run that replaced it. Mirrors `Course.plan_run_id` (issue #50).
    training_status_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Set by every write path that can change what the athlete has earned, and
    # cleared by the reconcile that settles it (issue #69). Achievements are the
    # one piece of derived state with no incremental path — `recompute_achievements`
    # re-reads the entire activity history and every plan — so doing it inline per
    # ingest event made a season-long import quadratic. Writes now mark; reads and
    # the daily sweep settle. NULL means nothing is owed.
    achievements_dirty_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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
    # Which bike the ride was done on, and who decided that (issue #64). NULL
    # `bike_source` means unassigned; otherwise "auto" (a `default_sports`
    # match) or "manual" (the athlete picked it).
    #
    # `bike_source` is the load-bearing half. Automapping must never overwrite
    # a choice made by hand, and without a persisted marker there is no way to
    # tell a bike the athlete picked from one a rule guessed — so every
    # reprocess, re-import or edit to what a bike claims would quietly stomp
    # the correction. It cannot be inferred at read time; it has to be written
    # down. `services.garage.assign_bike` writes only where this is NULL or
    # "auto", never over "manual".
    bike_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("bikes.id", ondelete="SET NULL"), index=True, nullable=True
    )
    bike_source: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    labels: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Labels openkoutsi *thinks* apply, kept strictly apart from the ones the
    # athlete has actually applied above (issue #63). Shape:
    # {"commute": {"state": "pending"|"accepted"|"dismissed", "source": ..., "at": ...}}
    #
    # A separate column rather than an early write into `labels`, for two
    # reasons that are not just tidiness. The `commuter` badge counts *labelled*
    # activities (`services.achievements`), so applying a guess mints tiers the
    # athlete never claimed; and the RPE queue excludes commute-labelled rides,
    # so writing the label early would delete the ride from the very prompt
    # where the athlete would have confirmed it.
    #
    # Persisted rather than derived on read because a dismissal has to be
    # durable: a suggestion recomputed on every read and re-offered after the
    # athlete said no is worse than not having the feature at all.
    label_suggestions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
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
    # The run that owns the columns above: the heartbeat says whether a run is
    # *alive*, this says whether its writes are still *wanted*. A slow run whose
    # token no longer matches discards its own writes rather than committing
    # over the run that replaced it. Mirrors `Course.plan_run_id` (issue #50).
    analysis_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
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
        """Is there an original activity file stored for this activity?

        Named for the days when the only answer was a FIT. Since issue #36 the
        stored original may be a GPX or a TCX, and the flag still means "there
        is a file behind ``GET /activities/{id}/fit`` to download" —
        ``original_format`` says which kind it is.
        """
        return any(s.fit_file_path for s in self.sources)

    @property
    def original_format(self) -> Optional[str]:
        """Format of the original file that would be served, or None if there is none."""
        with_files = [s for s in self.sources if s.fit_file_path]
        if not with_files:
            return None
        # Same ranking the download endpoint uses to choose which source's file
        # to serve, imported here rather than at module scope because
        # `provider_sync` imports this module.
        from backend.app.services.provider_sync import _source_priority

        best = min(with_files, key=lambda s: _source_priority(s.provider, True))
        return best.file_format


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
    # Which format the stored original is in: ``fit``, ``gpx`` or ``tcx``
    # (issue #36). NULL means either there is no file or the row predates bulk
    # import, in which case it is a FIT — nothing else could have been stored.
    #
    # Originals are kept in the format they arrived in rather than converted to
    # FIT on ingest. Converting would keep `has_fit_file`, the download and
    # reprocess on one path, but it is lossy and it means the file an athlete
    # downloads is not the file they uploaded. Everything that reads the
    # original — the download endpoint, reprocess — dispatches on this instead.
    format: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    activity: Mapped["Activity"] = relationship("Activity", back_populates="sources", lazy="selectin")

    @property
    def file_format(self) -> str:
        """The stored original's format, defaulting to FIT for pre-#36 rows."""
        return self.format or "fit"


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
    # The run that owns the columns above: the heartbeat says whether a run is
    # *alive*, this says whether its writes are still *wanted*. A slow run whose
    # token no longer matches discards its own writes rather than committing
    # over the run that replaced it. Mirrors `Course.plan_run_id` (issue #50).
    guidance_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

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


class ImportJob(UserBase):
    """One bulk import of activity files — an archive, or a pile of them (issue #36).

    Bulk import is a different interaction from the single-file upload, not a
    louder version of it. A Strava export is thousands of files and tens of
    minutes of parsing, which is not a request a browser can hold open, so the
    job is a resource: the endpoint creates one, hands back its id, and the
    client polls it.

    ``results`` is the part that makes a finished job useful. "847 of 900
    imported" says nothing an athlete can act on; the per-file list says *which*
    53 did not and why, so a corrupt export or an unsupported format is a thing
    they can see rather than a number they have to trust.
    """

    __tablename__ = "import_jobs"
    __table_args__ = (
        Index("ix_import_jobs_athlete_created", "athlete_id", "created_at"),
    )

    # Status values. A job that raised before finishing is `failed`; a job that
    # finished with some files rejected is still `completed` — one unreadable
    # file out of nine hundred is a result, not a failed import.
    STATUSES = ("pending", "running", "completed", "failed")

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    # What the athlete handed over, for a list of past imports that reads like
    # something they did: "strava_export_12345.zip", or "37 files".
    source_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Files found once archives were walked. 0 until the walk finishes, which is
    # why progress is reported as processed/total rather than as a percentage
    # while the job is still `pending`.
    total_files: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    imported: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped_duplicate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # [{filename, outcome, reason, activity_id, format}, ...] — one per file,
    # in the order they were processed.
    results: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Why the *job* died, as opposed to why a file did. Set only for `failed`.
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def processed(self) -> int:
        return self.imported + self.skipped_duplicate + self.failed


class SyncLease(UserBase, LeaseMixin):
    """Cross-process mutual exclusion for this user's write paths (issue #50).

    One row per named section. Today the only name in use is
    ``activity-create:{athlete_id}``, which serialises the ±5-minute duplicate
    check against the insert that follows it: without something at this level,
    the only guard is an ``asyncio.Lock``, and an ``asyncio.Lock`` is a statement
    about one event loop rather than about the database it is protecting.

    It lives in the per-user DB rather than the registry because that is the file
    the writes it guards land in — a lease is only meaningful to writers holding
    the same database open.
    """

    __tablename__ = "sync_leases"


class Bike(UserBase):
    """A bike the athlete owns, rides and maintains (issues #55, #64).

    Tyre width selects a rolling-resistance coefficient and riding position an
    aerodynamic drag area (both tables live in ``openkoutsi.course``). A row
    per bike rather than fields on the athlete because the inputs change per
    event — the gravel bike for one course, the TT bike for another — and a
    course keeps a reference to the bike it was solved for.

    Issue #64 promoted the row rather than adding a second table beside it: the
    garage edits the same rows the course bike selector reads, which is what
    makes "bikes in the garage are entries in the route-analysis picker" true
    by construction instead of a synchronisation problem.
    """

    __tablename__ = "bikes"

    RIDING_POSITIONS = ("tops", "hoods", "drops", "aero")

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    tyre_width_mm: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    riding_position: Mapped[str] = mapped_column(String, default="hoods", nullable=False)
    # Kilometres the bike had ridden *before openkoutsi ever saw it* (issue
    # #64). Almost every bike added to a garage has history behind it, and
    # without a baseline every wear figure reads low and every maintenance
    # interval is wrong. Set by the athlete; never written by the importer, so
    # a re-import can never move it.
    odometer_base_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Canonical cycling ``sport_type`` values this bike claims, as a JSON list
    # — the road / gravel / e-bike split that makes automapping possible. A
    # sport may be claimed by at most one bike per athlete; the API enforces
    # that, because two bikes claiming `GravelRide` has no correct resolution.
    default_sports: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # A sold or scrapped bike. It drops out of the pickers but keeps its rides,
    # its distance and its maintenance history — deleting would silently
    # rewrite the athlete's past totals, which is never what "I sold it" means.
    retired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    maintenance: Mapped[list["BikeMaintenance"]] = relationship(
        "BikeMaintenance", cascade="all, delete-orphan", lazy="selectin"
    )
    accessories: Mapped[list["BikeAccessory"]] = relationship(
        "BikeAccessory", cascade="all, delete-orphan", lazy="selectin"
    )


class BikeMaintenance(UserBase):
    """One thing done to a bike, on a date, at an odometer reading (issue #64).

    ``component`` is what makes "how long did these tyres last?" answerable:
    that question is about *two events of the same kind*, and with free-text
    notes alone nothing can compute the span. With a component key, component
    life falls out as the delta in ``odometer_km`` between consecutive entries
    sharing it. Stored as a plain string rather than an enum so the vocabulary
    stays open — the same treatment ``Activity.labels`` gets.

    ``odometer_km`` is an **absolute reading**, not an offset from anything. It
    must not move when history is re-imported, a baseline is corrected or a
    ride is reassigned to another bike: a maintenance log that rewrites itself
    is worse than no log at all.
    """

    __tablename__ = "bike_maintenance"

    #: Suggested vocabulary. Advisory, not validated — see the class docstring.
    COMPONENTS = (
        "tyres",
        "chain",
        "cassette",
        "chainrings",
        "brake_pads",
        "bottom_bracket",
        "bearings",
        "cables",
        "bar_tape",
        "service",
        "other",
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    bike_id: Mapped[str] = mapped_column(
        String, ForeignKey("bikes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    performed_on: Mapped[date] = mapped_column(Date, nullable=False)
    odometer_km: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    component: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class BikeAccessory(UserBase):
    """Something bolted to a bike — a child trailer, a rack, lights (issue #64).

    Deliberately a plain record: no weight, no drag, no coupling to the pacing
    model. A fitted trailer genuinely changes both mass and CdA, but modelling
    that means touching ``BikeParams`` in ``openkoutsi.course`` and deciding
    what happens to already-analysed courses — a separate piece of work with
    its own correctness questions. Noting that the trailer exists is what this
    is for.
    """

    __tablename__ = "bike_accessories"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    bike_id: Mapped[str] = mapped_column(
        String, ForeignKey("bikes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Course(UserBase):
    """An uploaded course and everything derived from it (issue #55).

    The sanctioned persistence of route data, per the Stage 0 decision in
    issue #54: the raw GPX sits encrypted on disk under an *opaque storage
    key* (a bare filename resolved against the user's upload directory at read
    time — never an absolute path, see issue #51), the thinned track lives in
    ``course_tracks``, and this row carries only coordinate-free metadata,
    inputs, the chart profile and the pacing outcome.

    The ``plan_*`` columns copy ``Goal.guidance*`` exactly — streamed coach
    prose, a status, and a timestamp for pending-timeout recovery — so the
    same stranded-run settlement applies at boot.
    """

    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    athlete_id: Mapped[str] = mapped_column(
        String, ForeignKey("athletes.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    # SET NULL, not CASCADE: deleting a goal or a bike must never destroy a
    # course. Re-analysis simply requires picking a bike again.
    goal_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("goals.id", ondelete="SET NULL"), nullable=True
    )
    bike_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("bikes.id", ondelete="SET NULL"), nullable=True
    )

    # Opaque storage key for the encrypted original — a bare filename, never a
    # path. Resolution and containment live in services/course_analysis.py.
    gpx_file_key: Mapped[str] = mapped_column(String, nullable=False)
    gpx_file_encrypted: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # `ready` | `error`. Processing is synchronous, so `pending` is never
    # persisted; `error` exists for re-analysis failures.
    status: Mapped[str] = mapped_column(String, default="ready", nullable=False)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Athlete-facing inputs. The two targets are alternatives — a course is
    # paced to a finish time or to a number of watts, never to both — so the
    # API clears one when the other is set and refuses a request carrying both.
    target_time_s: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # The *average* power asked of the whole ride, not a per-segment target:
    # the effort model still spends on the climbs and eases on the descents.
    target_power_w: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # The rider numbers the last analysis was solved from, snapshotted for
    # reproducibility — profile edits change future analyses, not this one.
    ftp_w_used: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_kg_used: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Coordinate-free route metadata.
    distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_gain_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_loss_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # ≤400 × [distance_m, elevation_m, gradient] — the chart payload.
    profile: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Pacing outcome. An infeasible target is a result, not an error: feasible
    # False with a refusal_reason and the required intensity on record.
    predicted_time_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    required_intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feasible: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    refusal_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # The written plan — Goal.guidance*'s shape, renamed.
    plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    plan_mood: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    plan_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Identifies the plan run that owns these columns. The generator runs on
    # its own session and commits after the request that started it, so
    # "clear the plan" cannot be expressed by nulling the columns alone — an
    # in-flight run would simply write them back. Re-analysis clears this
    # token, and a run whose token no longer matches has its writes discarded.
    plan_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    plan_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Surface classification (issue #56) ──────────────────────────────────
    # The status shape is plan_*'s, for the same reason: the match runs in the
    # background on its own session, so `stranded_runs` settles it at boot and
    # a run whose token no longer matches discards its own writes.
    # None throughout means "never matched" — which is the state of every
    # course on an instance with no sidecar, and is an absence rather than a
    # failure.
    surface_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    surface_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    surface_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # The surface at full run resolution: [[start_m, end_m, class, confidence,
    # severity_step], …], run-length encoded so it stays small.
    #
    # Kept here rather than folded into the segment table because the two have
    # different jobs. `course_segments` is pacing-shaped and has a minimum row
    # length; this has none, so a 130 m sector of mud in the middle of 40 km of
    # asphalt is still drawn and still named even when the pacing rows quite
    # reasonably fold it into a longer one. Coordinate-free, like everything
    # else on this table.
    surface_ribbon: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    segments: Mapped[list["CourseSegment"]] = relationship(
        "CourseSegment",
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="CourseSegment.segment_index",
        lazy="selectin",
    )
    track: Mapped[Optional["CourseTrack"]] = relationship(
        "CourseTrack", back_populates="course", cascade="all, delete-orphan", uselist=False
    )


class CourseTrack(UserBase):
    """The thinned track of a course — the one table that holds coordinates.

    One row per course, the points as a JSON series (the ``ActivityStream``
    pattern) rather than a row per point: ``[[lat, lon, elevation_m,
    distance_m], …]`` at ~8 m spacing. Loaded only by re-analysis and by the
    surface matcher (issue #56) — never serialized into an API
    response, an MCP result, or an LLM prompt. Deliberately its own table so
    that reading a course, listing courses and building the plan prompt touch
    rows with nothing location-shaped in them.
    """

    __tablename__ = "course_tracks"

    course_id: Mapped[str] = mapped_column(
        String, ForeignKey("courses.id", ondelete="CASCADE"), primary_key=True
    )
    points: Mapped[list] = mapped_column(JSON, nullable=False)
    # What the matcher said about each of those points (issue #56):
    # ``[[raw_value, confidence], …]``, aligned to ``points``.
    #
    # Only the raw value and its confidence are stored — the class, and every
    # dissolving decision made from it, are re-derived on read. That way tuning
    # a threshold later re-reads correctly from what is already on disk instead
    # of needing every stored course re-matched. It lives on this table because
    # it is the same length as the track and, like the track, is loaded only by
    # re-analysis: listing courses and building the plan prompt must keep
    # touching rows with nothing per-point in them.
    surfaces: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    surface_matched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    course: Mapped["Course"] = relationship("Course", back_populates="track")


class CourseSegment(UserBase):
    """One gradient segment of an analysed course (issue #55).

    The ``ActivityInterval`` of a course: a numbered, ordered child row with
    the physics outputs for the course's last analysis. Replaced wholesale on
    re-analysis.
    """

    __tablename__ = "course_segments"
    __table_args__ = (UniqueConstraint("course_id", "segment_index"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    course_id: Mapped[str] = mapped_column(
        String, ForeignKey("courses.id", ondelete="CASCADE"), nullable=False
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    end_distance_m: Mapped[float] = mapped_column(Float, nullable=False)
    length_m: Mapped[float] = mapped_column(Float, nullable=False)
    avg_gradient: Mapped[float] = mapped_column(Float, nullable=False)
    elevation_change_m: Mapped[float] = mapped_column(Float, nullable=False)
    segment_type: Mapped[str] = mapped_column(String, nullable=False)
    power_w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_offset_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed_capped: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Surface classification (issue #56). All nullable: a course analysed
    # before this landed, or on an instance with no sidecar, reads exactly the
    # same as one whose match has not run.
    surface: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # "confirmed" when only an explicit OSM tag could have produced this class;
    # "inferred" when openkoutsi could not confirm one. Never flattened away —
    # a guess shown beside a fact at equal weight is worse than no answer.
    surface_confidence: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Exactly what the matcher said, preserved rather than discarded.
    surface_raw: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # The rolling-resistance coefficient this row was solved with, so a number
    # the athlete is asked to trust can be inspected rather than taken on faith.
    crr_used: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    course: Mapped["Course"] = relationship("Course", back_populates="segments")
