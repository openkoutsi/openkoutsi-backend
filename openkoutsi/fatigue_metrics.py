"""Banister model (Fitness/Fatigue/Form) calculations."""

from __future__ import annotations

import math
from datetime import date, timedelta


def compute_daily_metrics(
    load_by_date: dict[date, float],
    from_date: date,
    to_date: date,
    initial_fitness: float = 0.0,
    initial_fatigue: float = 0.0,
) -> list[dict]:
    """
    Apply the Banister impulse-response model day-by-day from from_date to to_date.

    Returns a list of dicts — one per day — with keys:
        date, fitness, fatigue, form, load_day
    in chronological order.
    """
    k42 = 1 - math.exp(-1 / 42)
    k7 = 1 - math.exp(-1 / 7)

    fitness, fatigue = initial_fitness, initial_fatigue
    results = []
    current = from_date
    while current <= to_date:
        load_day = load_by_date.get(current, 0.0)
        form = fitness - fatigue
        fitness = fitness + (load_day - fitness) * k42
        fatigue = fatigue + (load_day - fatigue) * k7
        results.append({"date": current, "fitness": fitness, "fatigue": fatigue, "form": form, "load_day": load_day})
        current += timedelta(days=1)

    return results
