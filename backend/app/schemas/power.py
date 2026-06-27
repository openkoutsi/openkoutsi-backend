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
