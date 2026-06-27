"""Banister model (CTL/ATL/TSB) calculations."""

from __future__ import annotations

import math
from datetime import date, timedelta


def compute_daily_metrics(
    tss_by_date: dict[date, float],
    from_date: date,
    to_date: date,
    initial_ctl: float = 0.0,
    initial_atl: float = 0.0,
) -> list[dict]:
    """
    Apply the Banister impulse-response model day-by-day from from_date to to_date.

    Returns a list of dicts — one per day — with keys:
        date, ctl, atl, tsb, tss_day
    in chronological order.
    """
    k42 = 1 - math.exp(-1 / 42)
    k7 = 1 - math.exp(-1 / 7)

    ctl, atl = initial_ctl, initial_atl
    results = []
    current = from_date
    while current <= to_date:
        tss_day = tss_by_date.get(current, 0.0)
        tsb = ctl - atl
        ctl = ctl + (tss_day - ctl) * k42
        atl = atl + (tss_day - atl) * k7
        results.append({"date": current, "ctl": ctl, "atl": atl, "tsb": tsb, "tss_day": tss_day})
        current += timedelta(days=1)

    return results
