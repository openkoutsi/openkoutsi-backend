from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class FitnessMetricResponse(BaseModel):
    date: date
    ctl: float
    atl: float
    tsb: float
    tss_day: float = 0.0      # DB column name
    daily_tss: float = 0.0    # frontend-facing alias

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def _sync_aliases(self) -> "FitnessMetricResponse":
        # Ensure both names are populated, whichever side has a value
        if self.daily_tss == 0.0 and self.tss_day != 0.0:
            self.daily_tss = self.tss_day
        elif self.tss_day == 0.0 and self.daily_tss != 0.0:
            self.tss_day = self.daily_tss
        return self


FormLabel = Literal["peak", "fresh", "neutral", "tired", "overreached"]


def _tsb_to_form(tsb: float) -> FormLabel:
    if tsb > 25:
        return "peak"
    if tsb > 5:
        return "fresh"
    if tsb > -10:
        return "neutral"
    if tsb > -30:
        return "tired"
    return "overreached"


class FitnessCurrentResponse(FitnessMetricResponse):
    form: FormLabel = "neutral"

    @model_validator(mode="after")
    def _compute_form(self) -> "FitnessCurrentResponse":
        self.daily_tss = self.tss_day if self.daily_tss == 0.0 else self.daily_tss
        self.form = _tsb_to_form(self.tsb)
        return self


class ActivitySummaryResponse(BaseModel):
    """Totals for cycling activities over a selected time period."""

    num_activities: int = 0
    total_duration_s: int = 0
    total_distance_m: float = 0.0
