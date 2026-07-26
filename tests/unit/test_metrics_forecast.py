"""Tests for the Fitness/Fatigue/Form forward projection (issue #34).

Exercised against the in-memory per-user session, with ``today`` injected so the
assertions don't depend on the wall clock.
"""
from datetime import date, timedelta

import pytest

from backend.app.models.user_orm import DailyMetric, PlannedWorkout, TrainingPlan
from backend.app.services.metrics_forecast import (
    _MAX_BRIDGE_DAYS,
    forecast_fitness,
    planned_load_by_date,
)

_TODAY = date(2025, 6, 1)          # A Sunday
_PLAN_START = date(2025, 6, 2)     # The Monday after — week 1 / day 1


async def _make_plan(
    session,
    athlete_id,
    workouts,
    start=_PLAN_START,
    status="active",
    plan_id="p1",
    weeks=4,
):
    plan = TrainingPlan(
        id=plan_id,
        athlete_id=athlete_id,
        name="P",
        start_date=start,
        end_date=(start + timedelta(weeks=weeks) - timedelta(days=1)) if start else None,
        weeks=weeks,
        status=status,
    )
    session.add(plan)
    for i, (week, day, load) in enumerate(workouts):
        session.add(
            PlannedWorkout(
                id=f"{plan_id}-w{i}",
                plan_id=plan_id,
                week_number=week,
                day_of_week=day,
                workout_type="endurance" if load else "rest",
                target_load=load,
            )
        )
    await session.commit()
    return plan


def _by_date(rows):
    return {row["date"]: row for row in rows}


class TestPlannedLoadByDate:
    async def test_week_and_day_map_to_calendar_dates(self, session, seeded_athlete):
        # Week 1 / day 1 is the plan's start date; week 4 / day 7 is the last day.
        await _make_plan(
            session, seeded_athlete.id, [(1, 1, 50), (4, 7, 90)], weeks=4
        )

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        assert loads[_PLAN_START] == 50.0
        assert loads[_PLAN_START + timedelta(days=27)] == 90.0

    async def test_mid_week_day_offsets(self, session, seeded_athlete):
        # Week 2 / day 3 (Wednesday) is start + 7 + 2 days.
        await _make_plan(session, seeded_athlete.id, [(2, 3, 70)])

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        assert loads == {_PLAN_START + timedelta(days=9): 70.0}

    async def test_rest_days_and_missing_target_load_contribute_nothing(
        self, session, seeded_athlete
    ):
        await _make_plan(
            session, seeded_athlete.id, [(1, 1, None), (1, 2, 0), (1, 3, 60)]
        )

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        # Only the day with a real target_load appears; the others are absent and
        # so are treated as 0.0 (and therefore decay) by the model.
        assert loads == {_PLAN_START + timedelta(days=2): 60.0}

    async def test_two_active_plans_sum_on_a_shared_date(self, session, seeded_athlete):
        await _make_plan(session, seeded_athlete.id, [(1, 1, 40)], plan_id="p1")
        await _make_plan(session, seeded_athlete.id, [(1, 1, 25)], plan_id="p2")

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        assert loads[_PLAN_START] == 65.0

    async def test_archived_plans_are_ignored(self, session, seeded_athlete):
        await _make_plan(
            session, seeded_athlete.id, [(1, 1, 80)], status="archived", plan_id="p1"
        )

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        assert loads == {}

    async def test_plans_without_start_date_are_skipped(self, session, seeded_athlete):
        await _make_plan(session, seeded_athlete.id, [(1, 1, 80)], start=None)

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _TODAY + timedelta(days=90), session
        )

        assert loads == {}

    async def test_workouts_outside_the_window_are_excluded(
        self, session, seeded_athlete
    ):
        await _make_plan(session, seeded_athlete.id, [(1, 1, 50), (3, 1, 50)])

        loads = await planned_load_by_date(
            seeded_athlete.id, _TODAY, _PLAN_START + timedelta(days=6), session
        )

        assert list(loads) == [_PLAN_START]


class TestForecastFitness:
    async def test_starts_tomorrow_and_runs_to_the_horizon(
        self, session, seeded_athlete
    ):
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY,
                fitness=50.0, fatigue=40.0, form=10.0, load_day=60.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=14, today=_TODAY)

        assert len(rows) == 14
        assert rows[0]["date"] == _TODAY + timedelta(days=1)
        assert rows[-1]["date"] == _TODAY + timedelta(days=14)

    async def test_seeds_from_the_last_daily_metric(self, session, seeded_athlete):
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY,
                fitness=50.0, fatigue=40.0, form=10.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=1, today=_TODAY)

        # First projected day carries the seed's Form (fitness - fatigue) and has
        # decayed both series toward a zero load.
        assert rows[0]["form"] == pytest.approx(10.0)
        assert rows[0]["fitness"] < 50.0
        assert rows[0]["fatigue"] < 40.0

    async def test_empty_history_seeds_zero(self, session, seeded_athlete):
        rows = await forecast_fitness(seeded_athlete.id, session, days=3, today=_TODAY)

        assert all(row["fitness"] == 0.0 for row in rows)
        assert all(row["fatigue"] == 0.0 for row in rows)
        assert all(row["form"] == 0.0 for row in rows)

    async def test_stale_seed_bridges_across_the_plan_not_through_rest(
        self, session, seeded_athlete
    ):
        # Regression: the bridge across a stale seed used to fetch planned load
        # only from today onwards, so the gap decayed as pure rest even when the
        # plan prescribed work there — always in the "you're fresher than you
        # are" direction. The gap must carry the plan's prescribed load.
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY - timedelta(days=7),
                fitness=50.0, fatigue=40.0, form=10.0, load_day=0.0,
            )
        )
        # A plan starting a week before today, prescribing 100 Load every day —
        # so it covers the whole bridge as well as the days ahead.
        await _make_plan(
            session, seeded_athlete.id,
            [(week, day, 100) for week in range(1, 4) for day in range(1, 8)],
            start=_TODAY - timedelta(days=7),
            weeks=3,
        )

        rows = await forecast_fitness(seeded_athlete.id, session, days=5, today=_TODAY)

        # Training through the gap keeps fatigue up; a rest bridge would have all
        # but erased it (7-day constant over 8 days) and made Form look positive.
        assert rows[0]["fatigue"] > 40.0
        assert rows[0]["form"] < 0.0

    async def test_stale_seed_with_no_plan_decays_through_the_gap(
        self, session, seeded_athlete
    ):
        # Last stored metric is a week old (catch-up hasn't run). With no plan
        # covering the gap there is nothing to project, so it decays — but the
        # days must be decayed through, not skipped, and the series must still
        # start tomorrow.
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY - timedelta(days=7),
                fitness=50.0, fatigue=40.0, form=10.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=5, today=_TODAY)

        assert rows[0]["date"] == _TODAY + timedelta(days=1)
        assert len(rows) == 5
        # Eight days of zero-load decay: fatigue (7-day constant) has fallen much
        # further than fitness (42-day constant).
        assert rows[0]["fitness"] < 50.0
        assert rows[0]["fatigue"] < 20.0

    async def test_seed_older_than_the_bridge_bound_is_ignored(
        self, session, seeded_athlete
    ):
        # A returning athlete's years-old row must not seed the recurrence: past
        # the bound the honest seed is 0/0, and running the bridge from it would
        # be thousands of iterations of numerically spent state.
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id,
                date=_TODAY - timedelta(days=_MAX_BRIDGE_DAYS + 1),
                fitness=80.0, fatigue=70.0, form=10.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=3, today=_TODAY)

        assert rows[0]["date"] == _TODAY + timedelta(days=1)
        assert all(row["fitness"] == 0.0 for row in rows)
        assert all(row["fatigue"] == 0.0 for row in rows)

    async def test_seed_just_inside_the_bridge_bound_is_used(
        self, session, seeded_athlete
    ):
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id,
                date=_TODAY - timedelta(days=_MAX_BRIDGE_DAYS),
                fitness=80.0, fatigue=70.0, form=10.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=3, today=_TODAY)

        # Carried forward and decayed over the bridge — Fitness is well down from
        # 80 after 180 days of rest, but has not reached zero.
        assert 0.0 < rows[0]["fitness"] < 80.0

    async def test_future_metric_rows_do_not_seed_the_forecast(
        self, session, seeded_athlete
    ):
        # A stray row dated after today must not be picked as the seed.
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY,
                fitness=50.0, fatigue=40.0, form=10.0, load_day=0.0,
            )
        )
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY + timedelta(days=3),
                fitness=999.0, fatigue=999.0, form=0.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=5, today=_TODAY)

        assert rows[0]["form"] == pytest.approx(10.0)
        assert all(row["fitness"] < 50.0 for row in rows)

    async def test_planned_load_raises_fitness(self, session, seeded_athlete):
        await _make_plan(
            session, seeded_athlete.id,
            [(week, day, 80) for week in range(1, 5) for day in range(1, 8)],
        )

        rows = await forecast_fitness(seeded_athlete.id, session, days=28, today=_TODAY)
        by_date = _by_date(rows)

        # Riding 80 Load every day from a zero base drives Fitness up monotonically.
        assert by_date[_PLAN_START]["load_day"] == 80.0
        assert by_date[_PLAN_START + timedelta(days=20)]["fitness"] > by_date[
            _PLAN_START
        ]["fitness"]

    async def test_rest_days_decay_rather_than_being_skipped(
        self, session, seeded_athlete
    ):
        # A single hard day followed by rest: the days after it must appear and
        # must show falling fatigue.
        await _make_plan(session, seeded_athlete.id, [(1, 1, 100)])

        rows = await forecast_fitness(seeded_athlete.id, session, days=10, today=_TODAY)
        by_date = _by_date(rows)

        day_after = by_date[_PLAN_START + timedelta(days=1)]
        two_days_after = by_date[_PLAN_START + timedelta(days=2)]
        assert day_after["load_day"] == 0.0
        assert two_days_after["fatigue"] < day_after["fatigue"]

    async def test_horizon_past_plan_end_decays_toward_zero(
        self, session, seeded_athlete
    ):
        await _make_plan(
            session, seeded_athlete.id,
            [(week, day, 80) for week in range(1, 3) for day in range(1, 8)],
            weeks=2,
        )

        rows = await forecast_fitness(seeded_athlete.id, session, days=120, today=_TODAY)
        by_date = _by_date(rows)

        plan_end = _PLAN_START + timedelta(days=13)
        # The projection keeps going past the plan, with everything decaying.
        assert rows[-1]["date"] == _TODAY + timedelta(days=120)
        assert rows[-1]["load_day"] == 0.0
        assert rows[-1]["fitness"] < by_date[plan_end]["fitness"]
        assert rows[-1]["fatigue"] == pytest.approx(0.0, abs=0.5)

    async def test_no_active_plan_is_pure_decay(self, session, seeded_athlete):
        session.add(
            DailyMetric(
                athlete_id=seeded_athlete.id, date=_TODAY,
                fitness=60.0, fatigue=30.0, form=30.0, load_day=0.0,
            )
        )
        await session.commit()

        rows = await forecast_fitness(seeded_athlete.id, session, days=30, today=_TODAY)

        assert all(row["load_day"] == 0.0 for row in rows)
        fitness_series = [row["fitness"] for row in rows]
        assert fitness_series == sorted(fitness_series, reverse=True)

    async def test_nothing_is_persisted(self, session, seeded_athlete):
        from sqlalchemy import select

        await _make_plan(session, seeded_athlete.id, [(1, 1, 80)])
        await forecast_fitness(seeded_athlete.id, session, days=30, today=_TODAY)

        stored = (await session.execute(select(DailyMetric))).scalars().all()
        assert stored == []
