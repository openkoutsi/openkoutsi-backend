"""The clock a `pending` LLM run is judged by, and the startup sweep (issue #91).

Two failures live here, one root: "is this run still alive?" was answered by a
clock nothing kept wound.

* A run whose *process* died left `pending` behind forever. For the activity
  analysis that was terminal — ``trigger_analysis`` early-returns on `pending`,
  so the activity could never be analysed again by any route — and the trigger
  was an ordinary redeploy, not a crash.
* The age check counted from the trigger rather than from the last sign of life,
  so a healthy run that streamed past the budget was declared dead underneath
  itself.

The sweep tests run against real per-user SQLite files (the ``isolate_user_dbs``
fixture points the data dir at a tmp dir), because what is being checked is
precisely that it finds the databases and settles rows in them.
"""
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.user_orm import Activity, Athlete, Goal
from backend.app.services.stranded_runs import (
    PENDING_TIMEOUT_MINUTES,
    pending_timed_out,
    settle_activity_analysis,
    settle_activity_analysis_if_timed_out,
    settle_goal_guidance,
    settle_stranded_runs,
    settle_training_status,
    user_ids_with_a_database,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


# ── The age check ───────────────────────────────────────────────────────────


class TestPendingTimedOut:
    def test_a_run_that_just_reported_progress_is_alive(self):
        assert pending_timed_out(NOW - timedelta(seconds=30), NOW) is False

    def test_a_run_silent_for_the_whole_budget_is_not(self):
        assert pending_timed_out(
            NOW - timedelta(minutes=PENDING_TIMEOUT_MINUTES + 1), NOW
        ) is True

    def test_the_boundary_belongs_to_the_run(self):
        assert pending_timed_out(
            NOW - timedelta(minutes=PENDING_TIMEOUT_MINUTES), NOW
        ) is False

    def test_no_timestamp_at_all_counts_as_timed_out(self):
        """A pre-migration row, or one that never recorded a step.

        There is no evidence anything is alive, and treating it as alive is what
        made an activity permanently un-analysable.
        """
        assert pending_timed_out(None, NOW) is True

    def test_a_naive_timestamp_is_read_as_utc(self):
        # SQLite hands back naive datetimes; reading one as local time would put
        # a fresh run hours into the past for anyone east of UTC.
        naive = (NOW - timedelta(minutes=1)).replace(tzinfo=None)
        assert pending_timed_out(naive, NOW) is False

    def test_the_budget_is_shared_by_every_surface(self):
        """One constant, not a copy per router — they used to drift by omission.

        Tight enough to be a usable recovery time, and comfortably above the
        transport's 300 s read timeout, which fails a genuinely silent stream
        first.
        """
        from backend.app.api import athlete as athlete_api
        from backend.app.api import goals as goals_api

        assert athlete_api.pending_timed_out is pending_timed_out
        assert goals_api.pending_timed_out is pending_timed_out
        assert 5 < PENDING_TIMEOUT_MINUTES <= 30


# ── Settling one row ────────────────────────────────────────────────────────


class TestSettleOneRow:
    def test_training_status_goes_to_error_with_no_step_left_under_it(self):
        athlete = Athlete(
            id="a1",
            global_user_id="u1",
            training_status_status="pending",
            training_status_progress="tool.get_power_profile",
        )
        assert settle_training_status(athlete, NOW) is True
        assert athlete.training_status_status == "error"
        assert athlete.training_status_progress is None
        assert athlete.training_status_updated_at == NOW

    def test_activity_analysis_goes_to_error_the_same_way(self):
        activity = Activity(
            id="act-1",
            athlete_id="a1",
            analysis_status="pending",
            analysis_progress="thinking",
        )
        assert settle_activity_analysis(activity, NOW) is True
        assert activity.analysis_status == "error"
        assert activity.analysis_progress is None
        assert activity.analysis_updated_at == NOW

    def test_goal_guidance_goes_to_error_the_same_way(self):
        goal = Goal(id="g1", athlete_id="a1", title="Sub-4 gran fondo",
                    guidance_status="pending")
        assert settle_goal_guidance(goal, NOW) is True
        assert goal.guidance_status == "error"
        assert goal.guidance_updated_at == NOW

    @pytest.mark.parametrize("status", ["done", "error", None])
    def test_a_row_that_is_not_pending_is_left_alone(self, status):
        """Nothing here may clobber a finished answer."""
        activity = Activity(
            id="act-1", athlete_id="a1", analysis="Nice ride.", analysis_status=status
        )
        assert settle_activity_analysis(activity, NOW) is False
        assert activity.analysis_status == status
        assert activity.analysis == "Nice ride."

    def test_the_age_check_and_the_settle_compose(self):
        fresh = Activity(
            id="act-1", athlete_id="a1", analysis_status="pending",
            analysis_updated_at=NOW - timedelta(seconds=5),
        )
        stale = Activity(
            id="act-2", athlete_id="a1", analysis_status="pending",
            analysis_updated_at=NOW - timedelta(hours=3),
        )
        assert settle_activity_analysis_if_timed_out(fresh, NOW) is False
        assert fresh.analysis_status == "pending"
        assert settle_activity_analysis_if_timed_out(stale, NOW) is True
        assert stale.analysis_status == "error"


# ── The startup sweep ───────────────────────────────────────────────────────


async def _seed(user_id: str, *, athlete_id: str = "ath") -> None:
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        session.add(
            Athlete(
                id=athlete_id,
                global_user_id=user_id,
                training_status_status="pending",
                training_status_progress="tool.get_training_status",
                training_status_date=date(2026, 8, 9),
                training_status_updated_at=NOW,
            )
        )
        session.add(
            Goal(
                id=f"{user_id}-goal",
                athlete_id=athlete_id,
                title="Ride 200 km",
                guidance_status="pending",
                guidance_updated_at=NOW,
            )
        )
        session.add_all(
            [
                Activity(
                    id=f"{user_id}-act-stuck",
                    athlete_id=athlete_id,
                    sport_type="Ride",
                    analysis_status="pending",
                    analysis_progress="thinking",
                    analysis_updated_at=NOW,
                ),
                Activity(
                    id=f"{user_id}-act-done",
                    athlete_id=athlete_id,
                    sport_type="Ride",
                    analysis="Strong tempo work.",
                    analysis_status="done",
                ),
            ]
        )
        await session.commit()


async def _read(user_id: str):
    async with get_user_session_factory(user_id)() as session:
        athlete = (await session.execute(select(Athlete))).scalars().first()
        goal = (await session.execute(select(Goal))).scalars().first()
        activities = {
            a.id: a for a in (await session.execute(select(Activity))).scalars()
        }
    return athlete, goal, activities


@pytest.mark.replica_unsafe  # settles every pending row, whoever owns it
class TestStartupSweep:
    async def test_it_settles_every_surface_in_every_users_database(self):
        """A `pending` row at boot is from a process that no longer exists.

        Nothing that writes one survives a restart: the auto-analyse paths run
        under ``asyncio.create_task``, the explicit triggers under
        ``BackgroundTasks``. So an ordinary redeploy is enough to strand them,
        and `failure_recovery` cannot help — its ``except Exception`` never sees
        the ``CancelledError`` that kills those tasks, and once the process is
        gone nothing runs at all.
        """
        await _seed("user-a")
        await _seed("user-b")

        settled = await settle_stranded_runs()

        # Three rows per user; the finished analysis is not one of them.
        assert settled == 6
        for user_id in ("user-a", "user-b"):
            athlete, goal, activities = await _read(user_id)
            assert athlete.training_status_status == "error"
            assert athlete.training_status_progress is None
            assert goal.guidance_status == "error"
            assert activities[f"{user_id}-act-stuck"].analysis_status == "error"
            assert activities[f"{user_id}-act-stuck"].analysis_progress is None
            assert activities[f"{user_id}-act-done"].analysis_status == "done"
            assert activities[f"{user_id}-act-done"].analysis == "Strong tempo work."

    async def test_the_stranded_activity_can_be_analysed_again(self):
        """The point of the whole thing, stated in the terms the user feels it.

        ``trigger_analysis`` refuses while the row says `pending`, so until this
        clears there is no route back to an analysis for that ride.
        """
        await _seed("user-a")

        await settle_stranded_runs()

        _athlete, _goal, activities = await _read("user-a")
        assert activities["user-a-act-stuck"].analysis_status != "pending"

    async def test_the_training_status_date_is_left_alone_so_it_regenerates(self):
        """The router stamps the date when *it* times a run out, to stop the
        auto-refresh re-firing a run that just failed. Here the run didn't fail
        — a restart killed it — so the athlete should get a fresh status on the
        next read rather than an error they have to clear by hand.
        """
        await _seed("user-a")

        await settle_stranded_runs()

        athlete, _goal, _activities = await _read("user-a")
        assert athlete.training_status_date == date(2026, 8, 9)

    async def test_a_second_sweep_settles_nothing(self):
        await _seed("user-a")
        assert await settle_stranded_runs() == 3
        assert await settle_stranded_runs() == 0

    async def test_no_databases_is_not_an_error(self):
        assert user_ids_with_a_database() == []
        assert await settle_stranded_runs() == 0

    async def test_it_never_creates_a_database_for_a_user_that_has_none(self, tmp_path):
        """Reading the user set from disk rather than the registry is deliberate:
        opening a session for an unknown id would *create* an empty database.
        """
        await _seed("user-a")
        (tmp_path / "users" / "no-db-here").mkdir(parents=True)

        await settle_stranded_runs()

        assert user_ids_with_a_database() == ["user-a"]
        assert not (tmp_path / "users" / "no-db-here" / "user.db").exists()

    async def test_one_unreadable_database_does_not_cost_everyone_else(self, tmp_path):
        """A database mid-migration is one user's lost recovery, not the boot."""
        await _seed("user-a")
        broken = tmp_path / "users" / "user-broken"
        broken.mkdir(parents=True)
        (broken / "user.db").write_bytes(b"this is not a sqlite file")

        settled = await settle_stranded_runs()

        assert settled == 3
        athlete, _goal, _activities = await _read("user-a")
        assert athlete.training_status_status == "error"


# ── The wiring ──────────────────────────────────────────────────────────────


@pytest.mark.replica_unsafe  # a boot is not evidence the other process died
class TestTheSweepRunsAtStartup:
    async def _lifespan(self, sweep):
        from unittest.mock import AsyncMock, patch

        from backend.main import lifespan

        with (
            patch("backend.main.init_registry_db", new=AsyncMock()),
            patch("backend.main.init_usage_db", new=AsyncMock()),
            patch("backend.app.api.strava.strava_bridge_poller", new=AsyncMock()),
            patch("backend.app.api.wahoo.wahoo_bridge_poller", new=AsyncMock()),
            patch("backend.app.services.pat_expiry.pat_expiry_sweeper", new=AsyncMock()),
            patch(
                "backend.app.services.stranded_runs.settle_stranded_runs", new=sweep
            ),
        ):
            async with lifespan(object()):
                pass

    async def test_it_is_awaited_before_the_app_serves_a_request(self):
        """In ``lifespan``, before the yield, on purpose.

        Uvicorn accepts no connection until startup returns, so the sweep cannot
        race a run triggered by an early request and settle a row that is
        genuinely alive.
        """
        from unittest.mock import AsyncMock

        sweep = AsyncMock(return_value=2)
        await self._lifespan(sweep)
        sweep.assert_awaited_once()

    async def test_a_failing_sweep_does_not_stop_the_app_from_starting(self):
        """Recovering yesterday's stranded rows is not worth refusing to boot."""
        from unittest.mock import AsyncMock

        sweep = AsyncMock(side_effect=RuntimeError("data volume is not mounted yet"))
        await self._lifespan(sweep)  # must not raise
        sweep.assert_awaited_once()
