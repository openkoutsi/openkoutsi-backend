"""Tests for the plan-adherence scoring service (issue #26).

``score_plan`` is exercised over transient ORM objects (no DB); the persistence
and catch-up backfill are exercised against the in-memory per-user session.
"""
from datetime import date, datetime, timedelta, timezone

import pytest

from backend.app.models.user_orm import (
    Activity, PlanAdherenceDaily, PlannedWorkout, PlannedWorkoutActivity, TrainingPlan,
)
from sqlalchemy import select
from backend.app.services.plan_adherence import (
    catch_up_adherence, score_plan,
)

_START = date(2025, 6, 2)  # A Monday


def _activity(load=None, duration_s=None):
    return Activity(
        athlete_id="a", sport_type="Ride", load=load, duration_s=duration_s,
        start_time=datetime(2025, 6, 2, 10, tzinfo=timezone.utc),
    )


def _workout(week=1, day=1, wtype="threshold", load=None, dur=None,
             activities=None, skip_reason=None, wid=None):
    w = PlannedWorkout(
        id=wid or f"w{week}-{day}",
        plan_id="p1", week_number=week, day_of_week=day,
        workout_type=wtype, target_load=load, duration_min=dur,
        skip_reason=skip_reason,
    )
    w.linked_activities = activities or []
    return w


def _plan(workouts, start=_START):
    p = TrainingPlan(id="p1", athlete_id="a", name="P", start_date=start, status="active")
    p.workouts = workouts
    return p


class TestScorePlan:
    def test_cycling_on_target_is_100(self):
        w = _workout(load=100, dur=60, activities=[_activity(load=100, duration_s=3600)])
        ps = score_plan(_plan([w]), today=_START)
        assert ps.score == pytest.approx(100.0)
        assert ps.completed == 1
        assert ps.match_scores[w.id] == pytest.approx(100.0)

    def test_multi_activity_sum_satisfies_workout(self):
        # A ride split in two: 60 + 40 load sums to the 100 target → 100.
        acts = [_activity(load=60, duration_s=1800), _activity(load=40, duration_s=1800)]
        w = _workout(load=100, dur=60, activities=acts)
        ps = score_plan(_plan([w]), today=_START)
        assert ps.match_scores[w.id] == pytest.approx(100.0)

    def test_completed_workout_floored_at_50_when_wildly_off(self):
        # Planned 85-min endurance ride, ridden as a 4-hour Z1/Z2 spin: both
        # Load and duration overshoot enough that the raw score is 0, but having
        # actually done the session floors the match score at 50.
        acts = [_activity(load=260, duration_s=4 * 3600)]
        w = _workout(load=75, dur=85, activities=acts)
        ps = score_plan(_plan([w]), today=_START)
        assert ps.completed == 1
        assert ps.match_scores[w.id] == pytest.approx(50.0)
        assert ps.score == pytest.approx(50.0)

    def test_missed_past_workout_is_zero_full_weight(self):
        w = _workout(load=100, dur=60)  # no activities
        # today is a week later → the workout date is in the past
        ps = score_plan(_plan([w]), today=_START + timedelta(days=7))
        assert ps.missed == 1
        assert ps.score == pytest.approx(0.0)
        assert ps.match_scores[w.id] == 0.0

    def test_future_workout_excluded(self):
        w = _workout(week=2, day=1, load=100, dur=60)
        ps = score_plan(_plan([w]), today=_START)  # week 2 is in the future
        assert ps.score is None  # nothing contributes yet
        assert ps.match_scores[w.id] is None

    def test_rest_day_excluded(self):
        w = _workout(wtype="rest")
        ps = score_plan(_plan([w]), today=_START + timedelta(days=7))
        assert ps.match_scores[w.id] is None
        assert ps.completed == ps.missed == ps.skipped == 0
        assert ps.score is None

    def test_skip_illness_barely_dents(self):
        done = _workout(day=1, load=100, dur=60,
                        activities=[_activity(load=100, duration_s=3600)])
        skipped = _workout(day=2, load=100, dur=60, skip_reason="illness")
        ps = score_plan(_plan([done, skipped]), today=_START + timedelta(days=7))
        assert ps.skipped == 1
        # illness f=0.9 → skip weight 0.1*100=10; done weight 100 score 100
        # adherence = 100*(100*1 + 10*0)/(110) ≈ 90.9
        assert ps.score == pytest.approx(100 * 100 / 110, abs=0.1)

    def test_skip_freeform_near_full_miss(self):
        done = _workout(day=1, load=100, dur=60,
                        activities=[_activity(load=100, duration_s=3600)])
        skipped = _workout(day=2, load=100, dur=60, skip_reason="couldn't be bothered")
        ps = score_plan(_plan([done, skipped]), today=_START + timedelta(days=7))
        # discretionary f=0.1 → skip weight 0.9*100=90
        assert ps.score == pytest.approx(100 * 100 / 190, abs=0.1)

    def test_provisional_today_empty_in_grace(self):
        w = _workout(load=100, dur=60)  # today, no activity yet
        ps = score_plan(_plan([w]), today=_START)
        assert ps.pending == 1
        assert ps.score is None  # not counted as a miss yet
        assert ps.match_scores[w.id] is None

    def test_provisional_today_scored_once_activity_links(self):
        w = _workout(load=100, dur=60, activities=[_activity(load=80, duration_s=3600)])
        ps = score_plan(_plan([w]), today=_START)
        assert ps.completed == 1
        # load 80/100 → 0.8; dur on target → 1.0; 0.7*0.8 + 0.3*1 = 0.86
        assert ps.match_scores[w.id] == pytest.approx(86.0)

    def test_supplemental_done_missed_no_partial(self):
        done = _workout(day=1, wtype="strength", load=100, dur=60,
                        activities=[_activity(load=10, duration_s=600)])
        ps = score_plan(_plan([done]), today=_START)
        # supplemental ignores load/duration grading → 100 for done
        assert ps.match_scores[done.id] == 100.0

    def test_empty_plan_score_none(self):
        ps = score_plan(_plan([]), today=_START)
        assert ps.score is None


class TestCatchUpAdherence:
    async def _make_plan(self, session, athlete_id):
        plan = TrainingPlan(
            athlete_id=athlete_id, name="P", start_date=_START,
            end_date=_START + timedelta(weeks=1), status="active",
        )
        session.add(plan)
        await session.flush()
        # Two past cycling workouts on day 1 and day 2 of week 1.
        session.add(PlannedWorkout(
            plan_id=plan.id, week_number=1, day_of_week=1,
            workout_type="threshold", target_load=100, duration_min=60,
        ))
        session.add(PlannedWorkout(
            plan_id=plan.id, week_number=1, day_of_week=2,
            workout_type="threshold", target_load=100, duration_min=60,
        ))
        await session.commit()
        return plan

    async def test_writes_snapshots_and_is_idempotent(self, session, seeded_athlete):
        import backend.app.services.plan_adherence as svc

        plan = await self._make_plan(session, seeded_athlete.id)

        # Freeze "today" to a week after the plan start so all workouts are past.
        frozen = _START + timedelta(days=7)
        orig = svc.date

        class _D:
            @staticmethod
            def today():
                return frozen
        svc.date = _D
        try:
            changed = await catch_up_adherence(seeded_athlete.id, session)
            assert changed is True
            rows = (await session.execute(
                PlanAdherenceDaily.__table__.select()
            )).fetchall()
            # A snapshot per day from plan start through frozen today (8 days).
            assert len(rows) == 8
            # Both workouts missed → score 0 on the final day.
            today_row = (await session.execute(
                PlanAdherenceDaily.__table__.select().where(
                    PlanAdherenceDaily.date == frozen
                )
            )).fetchone()
            assert today_row.score == pytest.approx(0.0)
            assert today_row.missed == 2

            # Re-running is idempotent: no new rows, same values.
            await catch_up_adherence(seeded_athlete.id, session)
            rows2 = (await session.execute(
                PlanAdherenceDaily.__table__.select()
            )).fetchall()
            assert len(rows2) == 8
        finally:
            svc.date = orig

    async def test_self_heals_stale_snapshot_after_retroactive_link(
        self, session, seeded_athlete
    ):
        """A stored day is rewritten when the underlying data changes after the
        fact — here an activity linked to a previously-missed past workout."""
        import backend.app.services.plan_adherence as svc

        plan = await self._make_plan(session, seeded_athlete.id)
        frozen = _START + timedelta(days=7)
        orig = svc.date

        class _D:
            @staticmethod
            def today():
                return frozen
        svc.date = _D
        try:
            await catch_up_adherence(seeded_athlete.id, session)
            before = (await session.execute(
                PlanAdherenceDaily.__table__.select().where(
                    PlanAdherenceDaily.date == frozen
                )
            )).fetchone()
            assert before.missed == 2 and before.score == pytest.approx(0.0)

            # Retroactively link an on-target activity to the day-1 workout.
            day1 = (await session.execute(
                select(PlannedWorkout).where(
                    PlannedWorkout.plan_id == plan.id,
                    PlannedWorkout.day_of_week == 1,
                )
            )).scalar_one()
            act = Activity(
                athlete_id=seeded_athlete.id, sport_type="Ride",
                load=100, duration_s=3600,
                start_time=datetime(2025, 6, 2, 10, tzinfo=timezone.utc),
            )
            session.add(act)
            await session.flush()
            session.add(PlannedWorkoutActivity(
                planned_workout_id=day1.id, activity_id=act.id,
            ))
            await session.commit()

            # Self-heal: no new rows, but the stale day is rewritten.
            changed = await catch_up_adherence(seeded_athlete.id, session)
            assert changed is True
            rows = (await session.execute(
                PlanAdherenceDaily.__table__.select()
            )).fetchall()
            assert len(rows) == 8  # still one row per day, none added

            after = (await session.execute(
                PlanAdherenceDaily.__table__.select().where(
                    PlanAdherenceDaily.date == frozen
                )
            )).fetchone()
            # One completed on-target (100) + one missed (0), equal weight → 50.
            assert after.completed == 1
            assert after.missed == 1
            assert after.score == pytest.approx(50.0)
        finally:
            svc.date = orig
