"""Unit tests for the body-weight helpers used to derive W/kg power bests."""
from datetime import date, datetime, timezone

import pytest

from backend.app.models.user_orm import Activity, ActivityPowerBest, Athlete, WeightLog
from backend.app.services.weight import (
    effective_weight_for,
    recompute_power_best_weights,
    w_per_kg,
)


def test_effective_weight_picks_most_recent_on_or_before():
    log = [
        (date(2025, 1, 1), 80.0),
        (date(2025, 3, 1), 75.0),
        (date(2025, 6, 1), 70.0),
    ]
    assert effective_weight_for(log, date(2025, 2, 1)) == 80.0   # between 1st and 2nd
    assert effective_weight_for(log, date(2025, 3, 1)) == 75.0   # exactly on an entry
    assert effective_weight_for(log, date(2025, 12, 1)) == 70.0  # after the last entry


def test_effective_weight_none_before_first_entry_or_empty():
    log = [(date(2025, 6, 1), 70.0)]
    assert effective_weight_for(log, date(2025, 1, 1)) is None  # predates first entry
    assert effective_weight_for(log, None) is None
    assert effective_weight_for([], date(2025, 1, 1)) is None


def test_w_per_kg_guards_missing_and_nonpositive_weight():
    assert w_per_kg(300.0, 75.0) == pytest.approx(4.0)
    assert w_per_kg(300.0, None) is None
    assert w_per_kg(300.0, 0.0) is None
    assert w_per_kg(None, 75.0) is None


async def _seed_best(session, athlete_id, start, power_w, weight_kg=None):
    activity = Activity(athlete_id=athlete_id, start_time=start, status="processed")
    session.add(activity)
    await session.flush()
    best = ActivityPowerBest(
        activity_id=activity.id,
        athlete_id=athlete_id,
        duration_s=60,
        power_w=power_w,
        activity_start_time=start,
        weight_kg=weight_kg,
        w_per_kg=w_per_kg(power_w, weight_kg),
    )
    session.add(best)
    await session.flush()
    return best


async def test_recompute_reattributes_weight_after_history_change(session):
    athlete = Athlete(id="a1", global_user_id="u1")
    session.add(athlete)
    await session.flush()

    older = datetime(2025, 2, 1, tzinfo=timezone.utc)
    newer = datetime(2025, 6, 15, tzinfo=timezone.utc)
    # Bests initially stored with no weight (e.g. imported before any weigh-in).
    b_old = await _seed_best(session, athlete.id, older, 300.0)
    b_new = await _seed_best(session, athlete.id, newer, 280.0)
    assert b_old.w_per_kg is None and b_new.w_per_kg is None

    # Add dated weight history, then recompute.
    session.add(WeightLog(athlete_id=athlete.id, effective_date=date(2025, 1, 1), weight_kg=75.0))
    session.add(WeightLog(athlete_id=athlete.id, effective_date=date(2025, 6, 1), weight_kg=70.0))
    await session.flush()

    await recompute_power_best_weights(athlete.id, session)

    assert b_old.weight_kg == 75.0
    assert b_old.w_per_kg == pytest.approx(300.0 / 75.0)
    assert b_new.weight_kg == 70.0
    assert b_new.w_per_kg == pytest.approx(280.0 / 70.0)


async def test_recompute_clears_weight_when_no_entry_predates_activity(session):
    athlete = Athlete(id="a2", global_user_id="u2")
    session.add(athlete)
    await session.flush()

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    best = await _seed_best(session, athlete.id, start, 300.0, weight_kg=99.0)

    # Only weight entries that are *after* the activity exist → no effective weight.
    session.add(WeightLog(athlete_id=athlete.id, effective_date=date(2025, 6, 1), weight_kg=70.0))
    await session.flush()

    await recompute_power_best_weights(athlete.id, session)

    assert best.weight_kg is None
    assert best.w_per_kg is None
