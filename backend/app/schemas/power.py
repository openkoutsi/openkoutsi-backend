from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class PowerBestEntry(BaseModel):
    duration_s: int
    rank: int
    power_w: float
    activity_id: str
    activity_name: Optional[str]
    activity_start_time: Optional[datetime]
    weight_kg: Optional[float] = None
    w_per_kg: Optional[float] = None


class AllTimePowerBestsResponse(BaseModel):
    bests: list[PowerBestEntry]


class FtpEstimateResponse(BaseModel):
    twenty_min_power: Optional[float] = None  # rank-1 1200s best, watts
    ftp_simple: Optional[int] = None          # round(0.95 * twenty_min_power)
    simple_available: bool = False
    cp: Optional[float] = None                # critical power, watts
    w_prime: Optional[float] = None           # anaerobic work capacity, joules
    ftp_cp: Optional[int] = None              # round(cp) — FTP = CP directly
    cp_available: bool = False


class PowerModelPoint(BaseModel):
    duration_s: int
    power_w: float


class PowerModelFit(BaseModel):
    """A single fitted power–duration model.

    ``model`` is a stable key (``cp2``, ``cp3``, ``exp``, ``power_law``).
    Parameter fields are populated only for the models that define them.
    ``curve`` is a dense sampled curve for plotting; ``predictions`` are the
    modeled potential values at the profile durations (5/60/300/1200 s).
    """
    model: str
    available: bool = False
    cp: Optional[float] = None           # critical power / asymptote, watts
    w_prime: Optional[float] = None      # anaerobic work capacity, joules (cp2/cp3)
    k: Optional[float] = None            # time offset, seconds (cp3)
    pmax: Optional[float] = None         # maximal instantaneous power, watts (cp3/exp)
    tau: Optional[float] = None          # decay time constant, seconds (exp)
    a: Optional[float] = None            # scale coefficient (power_law)
    b: Optional[float] = None            # exponent (power_law)
    rmse: Optional[float] = None         # fit error vs actual bests, watts
    curve: list[PowerModelPoint] = []
    predictions: list[PowerModelPoint] = []


class PowerModelsResponse(BaseModel):
    models: list[PowerModelFit] = []
    days: Optional[int] = None
