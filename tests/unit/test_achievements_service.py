"""Tests for the achievement recompute service (issue #33).

Exercised against the in-memory per-user session, mirroring
``test_plan_adherence_service.py``. The emphasis is on the properties that make
derived state safe: idempotence, self-healing, and stable unlock dates when
history is back-filled.
"""
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.models.user_orm import (
    AchievementUnlock,
    Activity,
    Goal,
    PlannedWorkout,
    PlannedWorkoutActivity,
    TrainingPlan,
)
from backend.app.services.achievements import (
    compute_achievements,
    gamification_enabled,
    mark_achievements_dirty,
    recompute_achievements,
)

_TODAY = date(2026, 7, 22)  # A Wednesday


async def _recompute(session, athlete, today=_TODAY):
    """Run a reconcile and hand back just the newly created rows.

    ``recompute_achievements`` returns ``(created, computation)`` so a read
    endpoint can reuse the computation; the tests only care about the rows.
    """
    created, _ = await recompute_achievements(athlete.id, session, today=today)
    return created


def _activity(session, athlete, *, day, **kwargs):
    act = Activity(
        athlete_id=athlete.id,
        sport_type=kwargs.pop("sport_type", "Ride"),
        start_time=datetime(day.year, day.month, day.day, 10, tzinfo=timezone.utc),
        status="processed",
        **kwargs,
    )
    session.add(act)
    return act


async def _unlocks(session, athlete) -> dict[tuple[str, float], AchievementUnlock]:
    rows = (
        await session.execute(
            select(AchievementUnlock).where(AchievementUnlock.athlete_id == athlete.id)
        )
    ).scalars()
    return {(r.achievement_id, r.tier): r for r in rows}


class TestBasicUnlocks:
    async def test_first_activity_unlocks_tier_one(self, session, seeded_athlete):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        created = await _recompute(session, seeded_athlete)

        ids = {(u.achievement_id, u.tier) for u in created}
        assert ("activity_count", 1.0) in ids

        stored = await _unlocks(session, seeded_athlete)
        assert stored[("activity_count", 1.0)].achieved_on == _TODAY

    async def test_unlock_date_is_the_day_the_criterion_was_met(
        self, session, seeded_athlete
    ):
        earned_on = _TODAY - timedelta(days=40)
        _activity(session, seeded_athlete, day=earned_on, duration_s=3600)
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        # Not today — the day the ride actually happened.
        assert stored[("activity_count", 1.0)].achieved_on == earned_on

    async def test_long_ride_needs_one_long_activity_not_several_short_ones(
        self, session, seeded_athlete
    ):
        for offset in range(3):
            _activity(
                session, seeded_athlete, day=_TODAY - timedelta(days=offset),
                duration_s=3600,
            )
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("long_activity", 2.0) not in stored

    async def test_everesting_unlocks_on_a_single_huge_climb(
        self, session, seeded_athlete
    ):
        _activity(
            session, seeded_athlete, day=_TODAY, duration_s=40000, elevation_m=8900,
        )
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("everesting", 8848.0) in stored
        # …and it links back to the ride that earned it.
        assert stored[("everesting", 8848.0)].context["activity_id"]

    async def test_labels_drive_the_race_and_commute_badges(
        self, session, seeded_athlete
    ):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600, labels=["race"])
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("race_day", 1.0) in stored
        assert ("commuter", 10.0) not in stored


class TestSelfHealing:
    async def test_recompute_is_idempotent(self, session, seeded_athlete):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        first = await _recompute(session, seeded_athlete)
        assert first

        second = await _recompute(session, seeded_athlete)
        assert second == []  # nothing new the second time round

    async def test_idempotent_run_does_not_move_achieved_on(
        self, session, seeded_athlete
    ):
        _activity(session, seeded_athlete, day=_TODAY - timedelta(days=5), duration_s=3600)
        await session.commit()

        await _recompute(session, seeded_athlete)
        before = (await _unlocks(session, seeded_athlete))[("activity_count", 1.0)].achieved_on

        await _recompute(session, seeded_athlete)
        after = (await _unlocks(session, seeded_athlete))[("activity_count", 1.0)].achieved_on

        assert before == after

    async def test_deleting_the_only_activity_revokes_the_unlock(
        self, session, seeded_athlete
    ):
        act = _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()
        await _recompute(session, seeded_athlete)
        assert ("activity_count", 1.0) in await _unlocks(session, seeded_athlete)

        await session.delete(act)
        await session.commit()
        await _recompute(session, seeded_athlete)

        assert await _unlocks(session, seeded_athlete) == {}

    async def test_backfilled_history_moves_the_unlock_earlier(
        self, session, seeded_athlete
    ):
        """Importing an old ride should age a badge, not re-date it to today."""
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()
        await _recompute(session, seeded_athlete)
        assert (await _unlocks(session, seeded_athlete))[
            ("activity_count", 1.0)
        ].achieved_on == _TODAY

        older = _TODAY - timedelta(days=365)
        _activity(session, seeded_athlete, day=older, duration_s=3600)
        await session.commit()
        await _recompute(session, seeded_athlete)

        assert (await _unlocks(session, seeded_athlete))[
            ("activity_count", 1.0)
        ].achieved_on == older


class TestFailureIsolation:
    async def test_a_failed_recompute_leaves_the_session_usable(
        self, session, seeded_athlete, monkeypatch
    ):
        """The "can never fail an upload" guarantee needs a rollback, not just a
        swallowed exception — the caller keeps using this same session."""
        from backend.app.services import achievements as svc

        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        async def boom(*args, **kwargs):
            raise RuntimeError("computation exploded")

        monkeypatch.setattr(svc, "compute_achievements", boom)
        await svc.recompute_achievements_safe(seeded_athlete.id, session)

        # The caller's next statement must not raise PendingRollbackError.
        rows = (await session.execute(select(Activity))).scalars().all()
        assert len(rows) == 1

    async def test_a_failed_recompute_does_not_leak_partial_rows(
        self, session, seeded_athlete, monkeypatch
    ):
        """Rows added before the failure must not ride along on the caller's commit."""
        from backend.app.services import achievements as svc

        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        athlete_id = seeded_athlete.id
        real_commit = session.commit
        calls = {"n": 0}

        async def fail_on_reconcile_commit():
            calls["n"] += 1
            raise RuntimeError("commit exploded")

        monkeypatch.setattr(session, "commit", fail_on_reconcile_commit)
        await svc.recompute_achievements_safe(athlete_id, session)
        monkeypatch.setattr(session, "commit", real_commit)

        assert calls["n"] == 1
        # The reconcile's pending inserts were rolled back, not left to be
        # committed by whatever the caller does next.
        await session.commit()
        rows = (
            await session.execute(
                select(AchievementUnlock).where(
                    AchievementUnlock.athlete_id == athlete_id
                )
            )
        ).scalars().all()
        assert rows == []


class TestContext:
    async def test_each_tier_links_to_the_activity_that_earned_it(
        self, session, seeded_athlete
    ):
        """A 5-hour first ride earns tiers 2–5; tier 2 must not claim the 5h ride
        was the first 2-hour one only because it was the biggest."""
        short = _activity(session, seeded_athlete, day=_TODAY - timedelta(days=30), duration_s=2 * 3600)
        await session.flush()
        long = _activity(session, seeded_athlete, day=_TODAY, duration_s=5 * 3600)
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert stored[("long_activity", 2.0)].context["activity_id"] == short.id
        assert stored[("long_activity", 5.0)].context["activity_id"] == long.id

    async def test_context_is_refreshed_rather_than_left_dangling(
        self, session, seeded_athlete
    ):
        """A deep link to a deleted activity must be re-derived, not kept."""
        first = _activity(
            session, seeded_athlete, day=_TODAY - timedelta(days=30), duration_s=3 * 3600,
        )
        await session.flush()
        second = _activity(session, seeded_athlete, day=_TODAY, duration_s=3 * 3600)
        await session.commit()
        await _recompute(session, seeded_athlete)
        assert (await _unlocks(session, seeded_athlete))[
            ("long_activity", 2.0)
        ].context["activity_id"] == first.id

        await session.delete(first)
        await session.commit()
        await _recompute(session, seeded_athlete)

        assert (await _unlocks(session, seeded_athlete))[
            ("long_activity", 2.0)
        ].context["activity_id"] == second.id


class TestSportCounting:
    async def test_unmapped_sports_still_count_as_distinct_sports(
        self, session, seeded_athlete
    ):
        """The category map covers the common types, not the long tail — an
        athlete doing cycling plus two unmapped sports is still multisport."""
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600, sport_type="Ride")
        _activity(
            session, seeded_athlete, day=_TODAY, duration_s=3600, sport_type="RockClimbing",
        )
        _activity(
            session, seeded_athlete, day=_TODAY, duration_s=3600, sport_type="StandUpPaddling",
        )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["multisport"] == 3

    async def test_variants_of_one_sport_still_fold_together(
        self, session, seeded_athlete
    ):
        for sport in ("Ride", "GravelRide", "MountainBikeRide"):
            _activity(
                session, seeded_athlete, day=_TODAY, duration_s=3600, sport_type=sport,
            )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["multisport"] == 1


class TestMalformedSettings:
    @pytest.mark.parametrize("tz", [3, ["Europe/Helsinki"], "Not/AZone", {"a": 1}])
    async def test_an_unusable_timezone_falls_back_to_utc_instead_of_raising(
        self, session, seeded_athlete, tz
    ):
        """app_settings is free-form, so `timezone` can hold anything a client
        sent. The achievements page must not 500 because of it."""
        seeded_athlete.app_settings = {"timezone": tz}
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["activity_count"] == 1


class TestOptOut:
    async def test_disabled_skips_recompute(self, session, seeded_athlete):
        seeded_athlete.app_settings = {"gamification": False}
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        created = await _recompute(session, seeded_athlete)

        assert created == []
        assert await _unlocks(session, seeded_athlete) == {}

    async def test_re_enabling_restores_the_same_unlocks(self, session, seeded_athlete):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()
        await _recompute(session, seeded_athlete)
        before = set(await _unlocks(session, seeded_athlete))

        seeded_athlete.app_settings = {"gamification": False}
        await session.commit()
        await _recompute(session, seeded_athlete)

        seeded_athlete.app_settings = {"gamification": True}
        await session.commit()
        await _recompute(session, seeded_athlete)

        assert set(await _unlocks(session, seeded_athlete)) == before

    @pytest.mark.parametrize(
        "settings,expected",
        [
            (None, True),
            ({}, True),
            ({"gamification": True}, True),
            ({"gamification": False}, False),
        ],
    )
    def test_default_is_on(self, seeded_athlete, settings, expected):
        seeded_athlete.app_settings = settings
        assert gamification_enabled(seeded_athlete) is expected


class TestAvailability:
    async def test_elevation_badges_hidden_without_elevation_data(
        self, session, seeded_athlete
    ):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600, distance_m=60_000)
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert "distance" in comp.available
        assert "elevation" not in comp.available
        assert comp.is_available("total_distance") is True
        assert comp.is_available("total_elevation") is False

    async def test_plan_requirement_appears_once_a_plan_exists(
        self, session, seeded_athlete
    ):
        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)
        assert comp.is_available("plans_completed") is False

        session.add(
            TrainingPlan(
                id="p1", athlete_id=seeded_athlete.id, name="P",
                start_date=_TODAY - timedelta(days=28), end_date=_TODAY - timedelta(days=1),
                status="archived",
            )
        )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)
        assert comp.is_available("plans_completed") is True


class TestStreaks:
    async def test_weekly_streak_counts_consecutive_weeks(self, session, seeded_athlete):
        for weeks_ago in range(4):
            _activity(
                session, seeded_athlete,
                day=_TODAY - timedelta(weeks=weeks_ago), duration_s=3600,
            )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.streaks["streak_active_weeks"].current == 4

    async def test_current_week_without_a_ride_keeps_the_streak_in_progress(
        self, session, seeded_athlete
    ):
        for weeks_ago in range(1, 4):
            _activity(
                session, seeded_athlete,
                day=_TODAY - timedelta(weeks=weeks_ago), duration_s=3600,
            )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        state = comp.streaks["streak_active_weeks"]
        assert state.current == 3
        assert state.in_progress is True

    async def test_timezone_decides_which_week_a_late_ride_lands_in(
        self, session, seeded_athlete
    ):
        """A Sunday 23:00 UTC ride is Monday in Helsinki — a different week."""
        sunday = date(2026, 7, 19)
        act = Activity(
            athlete_id=seeded_athlete.id,
            sport_type="Ride",
            duration_s=3600,
            start_time=datetime(sunday.year, sunday.month, sunday.day, 23, tzinfo=timezone.utc),
            status="processed",
        )
        session.add(act)

        seeded_athlete.app_settings = {"timezone": "UTC"}
        await session.commit()
        utc = await compute_achievements(seeded_athlete, session, today=_TODAY)

        seeded_athlete.app_settings = {"timezone": "Europe/Helsinki"}
        await session.commit()
        helsinki = await compute_achievements(seeded_athlete, session, today=_TODAY)

        # In UTC the ride sits in the previous week (streak already broken by
        # today); in Helsinki it lands in the current week and is live.
        assert utc.streaks["streak_active_weeks"].current == 1
        assert utc.streaks["streak_active_weeks"].in_progress is True
        assert helsinki.streaks["streak_active_weeks"].current == 1
        assert helsinki.streaks["streak_active_weeks"].in_progress is False

    async def test_activity_without_start_time_is_ignored(self, session, seeded_athlete):
        session.add(
            Activity(athlete_id=seeded_athlete.id, sport_type="Ride", duration_s=3600)
        )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["activity_count"] == 0


class TestPlanAndGoalRules:
    async def _finished_plan(self, session, athlete, *, linked: bool, skip=None):
        start = _TODAY - timedelta(days=14)
        plan = TrainingPlan(
            id="p1", athlete_id=athlete.id, name="P",
            start_date=start, end_date=_TODAY - timedelta(days=8),
            status="archived",
        )
        session.add(plan)
        workout = PlannedWorkout(
            id="w1", plan_id="p1", week_number=1, day_of_week=1,
            workout_type="endurance", target_load=100, duration_min=60,
            skip_reason=skip,
        )
        session.add(workout)
        await session.flush()

        if linked:
            act = _activity(
                session, athlete, day=start, duration_s=3600, load=100,
            )
            await session.flush()
            session.add(
                PlannedWorkoutActivity(planned_workout_id="w1", activity_id=act.id)
            )
        await session.commit()
        return plan

    async def test_finished_plan_with_a_completed_workout_counts(
        self, session, seeded_athlete
    ):
        await self._finished_plan(session, seeded_athlete, linked=True)

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("plans_completed", 1.0) in stored
        assert ("plan_flawless", 1.0) in stored

    async def test_elapsed_but_empty_plan_is_not_an_achievement(
        self, session, seeded_athlete
    ):
        await self._finished_plan(session, seeded_athlete, linked=False)

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("plans_completed", 1.0) not in stored
        assert ("plan_flawless", 1.0) not in stored

    async def test_running_plan_does_not_count_yet(self, session, seeded_athlete):
        session.add(
            TrainingPlan(
                id="p2", athlete_id=seeded_athlete.id, name="Live",
                start_date=_TODAY - timedelta(days=7),
                end_date=_TODAY + timedelta(days=7),
                status="active",
            )
        )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["plans_completed"] == 0

    async def test_achieved_goal_unlocks(self, session, seeded_athlete):
        session.add(
            Goal(
                id="g1", athlete_id=seeded_athlete.id, title="FTP 300",
                status="achieved", target_date=_TODAY - timedelta(days=3),
            )
        )
        await session.commit()

        await _recompute(session, seeded_athlete)

        stored = await _unlocks(session, seeded_athlete)
        assert ("goals_reached", 1.0) in stored
        assert stored[("goals_reached", 1.0)].achieved_on == _TODAY - timedelta(days=3)

    async def test_active_goal_does_not_unlock(self, session, seeded_athlete):
        session.add(
            Goal(id="g2", athlete_id=seeded_athlete.id, title="Later", status="active")
        )
        await session.commit()

        comp = await compute_achievements(seeded_athlete, session, today=_TODAY)

        assert comp.progress["goals_reached"] == 0


# ── The dirty flag (issue #69) ───────────────────────────────────────────────

class TestDirtyFlag:
    """Writes mark, reads settle.

    The reconcile has no incremental path — it re-reads the athlete's whole
    history every time — so running it inline per ingest event made importing a
    season quadratic. These pin the two halves of the split that replaced it.
    """

    async def test_marking_records_the_debt_without_paying_it(
        self, session, seeded_athlete
    ):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        await mark_achievements_dirty(seeded_athlete.id, session)

        assert seeded_athlete.achievements_dirty_at is not None
        # The whole point: no reconcile ran, so no rows were written.
        assert await _unlocks(session, seeded_athlete) == {}

    async def test_a_settle_clears_the_mark(self, session, seeded_athlete):
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()
        await mark_achievements_dirty(seeded_athlete.id, session)

        created = await _recompute(session, seeded_athlete)

        assert created
        assert seeded_athlete.achievements_dirty_at is None

    async def test_a_settle_clears_the_mark_even_when_nothing_changed(
        self, session, seeded_athlete
    ):
        """A no-op reconcile still has to drop the flag, or it pins forever."""
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()
        await _recompute(session, seeded_athlete)

        await mark_achievements_dirty(seeded_athlete.id, session)
        created = await _recompute(session, seeded_athlete)

        assert created == []
        assert seeded_athlete.achievements_dirty_at is None

    async def test_the_mark_is_cleared_for_an_opted_out_athlete(
        self, session, seeded_athlete
    ):
        """The mark deliberately doesn't check the preference, so the settle must.

        Otherwise an athlete who turned gamification off stays in the sweep's
        work list forever, for a recompute that can never produce anything.
        """
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        seeded_athlete.app_settings = {"gamification": False}
        await session.commit()
        await mark_achievements_dirty(seeded_athlete.id, session)

        created, comp = await recompute_achievements(
            seeded_athlete.id, session, today=_TODAY
        )

        assert created == []
        assert comp is None
        assert seeded_athlete.achievements_dirty_at is None

    async def test_deferring_converges_on_the_same_rows(self, session, seeded_athlete):
        """The property the whole change rests on.

        Unlocks are a pure function of the data, so N writes followed by one
        reconcile must produce exactly what reconciling after every write would.
        """
        for offset in range(5):
            _activity(
                session, seeded_athlete,
                day=_TODAY - timedelta(days=offset),
                duration_s=3600 * (offset + 1),
            )
            await session.commit()
            await mark_achievements_dirty(seeded_athlete.id, session)
        deferred = await _recompute(session, seeded_athlete)

        # Same data, reconciled from scratch after every single write.
        for row in list((await session.execute(select(AchievementUnlock))).scalars()):
            await session.delete(row)
        await session.commit()
        eager = []
        for _ in range(5):
            eager.extend(await _recompute(session, seeded_athlete))

        def key(rows):
            return sorted((r.achievement_id, r.tier, r.achieved_on) for r in rows)

        assert key(deferred) == key(eager)
        assert deferred  # the comparison would be vacuous on two empty lists

    async def test_a_failed_mark_leaves_the_session_usable(
        self, session, seeded_athlete, monkeypatch
    ):
        """Same guarantee as the reconcile it replaced: never fail an upload.

        A swallowed exception is not enough — the caller keeps using this session,
        so a failure has to be rolled back or its next statement raises
        ``PendingRollbackError``.
        """
        _activity(session, seeded_athlete, day=_TODAY, duration_s=3600)
        await session.commit()

        async def boom():
            raise RuntimeError("commit exploded")

        real_commit = session.commit
        monkeypatch.setattr(session, "commit", boom)
        await mark_achievements_dirty(seeded_athlete.id, session)
        monkeypatch.setattr(session, "commit", real_commit)

        rows = (await session.execute(select(Activity))).scalars().all()
        assert len(rows) == 1
