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


class ActivitySummaryResponse(BaseModel):
    """Totals for cycling activities over a selected time period."""

    num_activities: int = 0
    total_duration_s: int = 0
    total_distance_m: float = 0.0


class WeeklyZoneBucket(BaseModel):
    """Accumulated time-in-zone for one ISO week (Monday-based).

    ``hr`` and ``power`` map zone name → seconds, summed across all of that
    week's activities. Either may be empty when no matching data exists.
    """

    week_start: date
    hr: dict[str, int] = {}
    power: dict[str, int] = {}
