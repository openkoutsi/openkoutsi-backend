from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class FitnessMetricResponse(BaseModel):
    date: date
    fitness: float
    fatigue: float
    form: float
    load_day: float = 0.0      # DB column name
    daily_load: float = 0.0    # frontend-facing alias

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def _sync_aliases(self) -> "FitnessMetricResponse":
        # Ensure both names are populated, whichever side has a value
        if self.daily_load == 0.0 and self.load_day != 0.0:
            self.daily_load = self.load_day
        elif self.load_day == 0.0 and self.daily_load != 0.0:
            self.load_day = self.daily_load
        return self


FormLabel = Literal["peak", "fresh", "neutral", "tired", "overreached"]


def _form_to_label(form: float) -> FormLabel:
    if form > 25:
        return "peak"
    if form > 5:
        return "fresh"
    if form > -10:
        return "neutral"
    if form > -30:
        return "tired"
    return "overreached"


class FitnessCurrentResponse(FitnessMetricResponse):
    form_label: FormLabel = "neutral"

    @model_validator(mode="after")
    def _compute_form(self) -> "FitnessCurrentResponse":
        self.daily_load = self.load_day if self.daily_load == 0.0 else self.daily_load
        self.form_label = _form_to_label(self.form)
        return self


class FitnessForecastResponse(FitnessMetricResponse):
    """One projected day of the Fitness/Fatigue/Form forecast (issue #34).

    Deliberately shares ``FitnessMetricResponse``'s shape so the frontend can
    concatenate the projected series straight onto the historical one. The
    ``projected`` marker is what distinguishes a modeled day from a measured
    one; it is always ``True`` here.
    """

    projected: bool = True
    form_label: FormLabel = "neutral"

    @model_validator(mode="after")
    def _compute_form(self) -> "FitnessForecastResponse":
        self.form_label = _form_to_label(self.form)
        return self


class ActivitySummaryResponse(BaseModel):
    """Totals for cycling activities over a selected time period."""

    num_activities: int = 0
    total_duration_s: int = 0
    total_distance_m: float = 0.0


class EfficiencyPoint(BaseModel):
    """One steady endurance ride in the aerobic efficiency trend (issue #37).

    Efficiency factor is weighted power per heartbeat; rising over time at a
    constant training load is aerobic progress the Fitness/Fatigue model can't
    see. ``decoupling_pct`` rides along where the activity has one, so the same
    chart can show durability alongside efficiency.
    """

    activity_id: str
    date: date
    duration_s: Optional[int] = None
    efficiency_factor: float
    decoupling_pct: Optional[float] = None


class WeeklyZoneBucket(BaseModel):
    """Accumulated time-in-zone for one ISO week (Monday-based).

    ``hr`` and ``power`` map zone name → seconds, summed across all of that
    week's activities. Either may be empty when no matching data exists.
    """

    week_start: date
    hr: dict[str, int] = {}
    power: dict[str, int] = {}


IntensityBasis = Literal["power", "hr"]
IntensityMethod = Literal["time", "session"]
IntensityShape = Literal["polarized", "pyramidal", "threshold", "predominantly_low"]


class IntensityBand(BaseModel):
    """One of the three intensity bands over a block (issue #38).

    ``band`` is 1 (below LT1), 2 (between LT1 and LT2) or 3 (above LT2).
    ``pct`` is the band's share **in the unit the method counts in**: seconds
    for ``method=time``, sessions for ``method=session``. ``seconds`` is
    reported either way, but for the session method it is context rather than
    the basis of the percentage — a VO2max session counts as one hard session
    whatever fraction of it was spent coasting.
    """

    band: int
    seconds: int = 0
    pct: float = 0.0
    sessions: Optional[int] = None


class IntensityCoverage(BaseModel):
    """How much of the window actually reached the distribution.

    A shape computed from 6 of 40 rides is not wrong so much as unfounded, so
    the numbers travel with the result rather than being implied by it.
    """

    activities_total: int = 0
    activities_used: int = 0
    seconds_total: int = 0


class IntensityDistributionResponse(BaseModel):
    """Intensity distribution over a training block (issue #38).

    ``basis`` is ``None`` for ``method=session``: session counting works off
    each activity's workout category, so the power/HR distinction does not
    apply. ``classification`` is ``None`` when the window has no usable data.
    """

    start: Optional[date] = None
    end: Optional[date] = None
    basis: Optional[IntensityBasis] = None
    method: IntensityMethod
    bands: list[IntensityBand] = []
    classification: Optional[IntensityShape] = None
    coverage: IntensityCoverage = IntensityCoverage()
    zone_definitions_changed: bool = False
