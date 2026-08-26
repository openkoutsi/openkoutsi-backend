"""The clock a streaming LLM run is judged alive by (issue #91, bug 2).

All three coaching surfaces write a ``pending`` status and let a reader age it
out: the training status, goal guidance, and — since this issue — the activity
analysis. The timestamp that age is measured from used to be written when the
run was *triggered* and then not again until it settled, which made the budget a
duration budget. A single completion streaming steadily past it was therefore
declared dead underneath itself: the card flipped to ``error`` while the run was
healthy, the run kept spending, and the settle then overwrote the error with
``done``.

The fix is to touch the timestamp on every progress commit, turning it into an
inactivity budget. What that requires is an assertion made *mid-stream* — the
state a poll would actually see — because a run that has finished has touched
the timestamp anyway and would pass a naive check trivially. So the fake model
here reads the row back through a separate session between chunks, which is
exactly what the frontend's poll does.

Runs against a real per-user SQLite file with the real analyzers; only the model
is fake. The blob path is used throughout: it is the one bug 2 still applied to
after #90, and it stays the answer for every provider that can't call tools.
"""
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.user_orm import Activity, Athlete, DailyMetric, Goal
from backend.app.services.llm_client import ResolvedLlm
from backend.app.services.llm_streaming import StreamSetup, TextDelta
from backend.app.services.stranded_runs import pending_timed_out

USER_ID = "clock-user"
ATHLETE_ID = "clock-athlete"
ACTIVITY_ID = "clock-act"
GOAL_ID = "clock-goal"

#: When the run was triggered — long enough ago that the old duration budget
#: would have expired mid-stream.
TRIGGERED_AT = datetime.now(timezone.utc) - timedelta(hours=2)

ANSWER = ["MOOD:knowing\n\n", "You held tempo well. ", "Recover tomorrow."]
GUIDANCE = ["REALISM:ambitious\n\n", "It is a stretch. ", "Keep the volume up."]


@pytest.fixture
async def athlete_db(isolate_user_dbs):
    """One athlete, one activity and one goal, each mid-run."""
    await init_user_db(USER_ID)
    today = date.today()
    async with get_user_session_factory(USER_ID)() as session:
        session.add(
            Athlete(
                id=ATHLETE_ID,
                global_user_id=USER_ID,
                ftp=250,
                max_hr=185,
                ftp_tests=[],
                # No `agentic_koutsi`: the blob path, deliberately.
                app_settings={},
                training_status_status="pending",
                training_status_updated_at=TRIGGERED_AT,
            )
        )
        session.add(
            Activity(
                id=ACTIVITY_ID,
                athlete_id=ATHLETE_ID,
                sport_type="Ride",
                start_time=datetime.combine(today, time(9, 0), tzinfo=timezone.utc),
                duration_s=3600,
                distance_m=30000.0,
                avg_power=210.0,
                status="processed",
                analysis_status="pending",
                analysis_updated_at=TRIGGERED_AT,
            )
        )
        session.add(
            Goal(
                id=GOAL_ID,
                athlete_id=ATHLETE_ID,
                title="Sub-4 gran fondo",
                target_date=today + timedelta(days=60),
                guidance_status="pending",
                guidance_updated_at=TRIGGERED_AT,
            )
        )
        session.add(
            DailyMetric(
                athlete_id=ATHLETE_ID, date=today, fitness=55.0, fatigue=45.0,
                form=10.0, load_day=60.0,
            )
        )
        await session.commit()
    return USER_ID


def _utc(stamp: datetime | None) -> datetime | None:
    """SQLite hands timestamps back naive; read them as UTC, as the app does."""
    if stamp is None:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


async def _poll(model, row_id: str):
    """Read a row the way the frontend does: a separate session, mid-run."""
    async with get_user_session_factory(USER_ID)() as session:
        return (
            await session.execute(select(model).where(model.id == row_id))
        ).scalar_one()


@pytest.fixture
def fake_model(monkeypatch):
    """Serve canned prose, and record what a poll would see between chunks."""

    def _apply(chunks, *, watch):
        seen: list[datetime | None] = []

        async def _resolve(athlete, user_id, *, usage_out=None):
            cfg = ResolvedLlm(
                base_url="http://llm.invalid/v1", model="test-model",
                api_key=None, source="instance",
            )
            if usage_out is not None:
                usage_out["cfg"] = cfg
            return StreamSetup(cfg=cfg)

        async def _events(cfg, messages, *, tools=None, tool_choice=None, usage_out=None):
            assert tools is None, "these surfaces must take the single-shot path here"
            for chunk in chunks:
                yield TextDelta(chunk)
                seen.append(await watch())

        monkeypatch.setattr(
            "backend.app.services.llm_streaming.resolve_stream_setup", _resolve
        )
        monkeypatch.setattr(
            "backend.app.services.llm_streaming.stream_completion_events", _events
        )
        # Commit every chunk rather than every 500 ms, so "what a poll sees"
        # is observable without sleeping through the real cadence.
        monkeypatch.setattr("backend.app.services.llm_streaming._FLUSH_INTERVAL_S", 0)
        monkeypatch.setattr(
            "backend.app.services.llm_streaming.record_llm_usage", AsyncMock()
        )
        return seen

    return _apply


class TestTheClockKeepsMovingWhileTheRunStreams:
    async def test_activity_analysis(self, athlete_db, fake_model):
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg

        async def _watch():
            row = await _poll(Activity, ACTIVITY_ID)
            assert row.analysis_status == "pending"
            return _utc(row.analysis_updated_at)

        seen = fake_model(ANSWER, watch=_watch)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID)

        assert seen, "the stream produced no observable progress"
        assert all(stamp > TRIGGERED_AT for stamp in seen)
        assert not any(pending_timed_out(stamp) for stamp in seen)
        assert (await _poll(Activity, ACTIVITY_ID)).analysis_status == "done"

    async def test_training_status(self, athlete_db, fake_model):
        from backend.app.services.llm_training_status_analyzer import (
            analyze_training_status_bg,
        )

        async def _watch():
            row = await _poll(Athlete, ATHLETE_ID)
            assert row.training_status_status == "pending"
            return _utc(row.training_status_updated_at)

        seen = fake_model(ANSWER, watch=_watch)
        await analyze_training_status_bg(ATHLETE_ID, USER_ID)

        assert seen
        assert all(stamp > TRIGGERED_AT for stamp in seen)
        assert not any(pending_timed_out(stamp) for stamp in seen)
        assert (await _poll(Athlete, ATHLETE_ID)).training_status_status == "done"

    async def test_goal_guidance(self, athlete_db, fake_model):
        from backend.app.services.llm_goal_guidance import generate_goal_guidance_bg

        async def _watch():
            row = await _poll(Goal, GOAL_ID)
            assert row.guidance_status == "pending"
            return _utc(row.guidance_updated_at)

        seen = fake_model(GUIDANCE, watch=_watch)
        await generate_goal_guidance_bg(ATHLETE_ID, GOAL_ID, USER_ID)

        assert seen
        assert all(stamp > TRIGGERED_AT for stamp in seen)
        assert not any(pending_timed_out(stamp) for stamp in seen)
        assert (await _poll(Goal, GOAL_ID)).guidance_status == "done"


class TestARollingRestartDoesNotKillALiveRun:
    """The failure mode in the terms it actually occurs in (issue #50).

    ``TestTheClockKeepsMovingWhileTheRunStreams`` shows the heartbeat is written.
    This shows the sweep reads it: a second process booting mid-stream — which is
    what a rolling redeploy behind a proxy is — runs the same startup sweep, and
    it must leave the run alone rather than error it underneath the process that
    is still serving.

    Before the sweep consulted the heartbeat this settled the row to ``error``
    while the stream was still writing to it, and the stream then finished into a
    row a reader had already been shown as failed.
    """

    async def test_the_sweep_leaves_a_streaming_analysis_alone(
        self, athlete_db, fake_model
    ):
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg
        from backend.app.services.stranded_runs import settle_stranded_runs

        after_each_sweep: list[str] = []

        async def _boot_a_second_process():
            """Stand in for the replica starting while this run is in flight."""
            row = await _poll(Activity, ACTIVITY_ID)
            assert row.analysis_status == "pending", "set-up: the run should be live"
            await settle_stranded_runs()
            # Read back through a fresh session: the question is what the sweep
            # committed, not what this one already had loaded.
            after_each_sweep.append(
                (await _poll(Activity, ACTIVITY_ID)).analysis_status
            )
            return _utc(row.analysis_updated_at)

        seen = fake_model(ANSWER, watch=_boot_a_second_process)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID)

        assert seen, "the stream produced no observable progress"
        # The athlete and goal rows in the fixture are stale by two hours and are
        # settled by these sweeps, correctly — this is about the one row that is
        # being written to.
        assert after_each_sweep and all(
            status == "pending" for status in after_each_sweep
        ), f"a sweep settled a run that was still streaming: {after_each_sweep}"

        row = await _poll(Activity, ACTIVITY_ID)
        assert row.analysis_status == "done", (
            "the run finished into a row the sweep had left alone"
        )
        assert row.analysis == "".join(ANSWER), "and the whole stream landed in it"


class TestASupersededRunDiscardsItsWrites:
    """The token's guarantee, at the point it has to hold (issue #50).

    The heartbeat cannot answer this one. A previous run's process can be alive
    and merely slow, so "is it still running?" says *yes* about a run whose
    answer nobody wants any more — the athlete re-triggered, or a sweep settled
    the row and the re-trigger that unblocked came in behind it.

    Without the token the slow run commits its finished answer over the new one.
    """

    async def test_a_superseded_run_does_not_touch_the_live_run_that_replaced_it(
        self, athlete_db, fake_model
    ):
        """Run B is not a token — it is a run, and its state has to survive.

        The earlier version of this test only *stamped* B's token and never ran
        B, so blanking every column looked correct: there was nothing of B's to
        destroy. That is precisely the state the bug needed to hide in, and it
        is why CI passed over it.

        Here B writes a real answer. A then finishes and finds itself superseded.
        Whatever A does next, B's work must still be on the row.
        """
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg
        from backend.app.services.stranded_runs import begin_activity_analysis_run

        # Run A claims the row.
        async with get_user_session_factory(USER_ID)() as session:
            activity = (
                await session.execute(select(Activity).where(Activity.id == ACTIVITY_ID))
            ).scalar_one()
            run_a = begin_activity_analysis_run(activity)
            await session.commit()

        b_answer = "MOOD:direct\n\nRun B got there first."

        async def _run_b_takes_over_and_finishes():
            """The athlete re-triggers, and that run completes before A does."""
            async with get_user_session_factory(USER_ID)() as other:
                activity = (
                    await other.execute(
                        select(Activity).where(Activity.id == ACTIVITY_ID)
                    )
                ).scalar_one()
                begin_activity_analysis_run(activity)
                # B streams and settles, the way a real second run would.
                activity.analysis = b_answer
                activity.analysis_status = "done"
                activity.analysis_updated_at = datetime.now(timezone.utc)
                await other.commit()
            return datetime.now(timezone.utc)

        fake_model(ANSWER, watch=_run_b_takes_over_and_finishes)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID, run_id=run_a)

        row = await _poll(Activity, ACTIVITY_ID)
        assert row.analysis == b_answer, (
            "the superseded run destroyed the answer of the run that replaced it"
        )
        assert row.analysis_status == "done", (
            "and nulled the status, which disarms the re-trigger guard and lets "
            "repeated clicks stack concurrent agentic runs on one row"
        )

    async def test_a_run_superseded_by_a_settle_does_take_its_writes_back_out(
        self, athlete_db, fake_model
    ):
        """The other reason the check fails, where clearing *is* correct.

        A settle clears the token, so nobody owns the row. A's answer is
        unwanted and leaving it would show the athlete prose from a run that was
        already declared dead — so the row goes back to un-analysed, which is
        also what makes it re-triggerable.
        """
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg
        from backend.app.services.stranded_runs import (
            begin_activity_analysis_run,
            settle_activity_analysis,
        )

        async with get_user_session_factory(USER_ID)() as session:
            activity = (
                await session.execute(select(Activity).where(Activity.id == ACTIVITY_ID))
            ).scalar_one()
            run_a = begin_activity_analysis_run(activity)
            await session.commit()

        async def _the_sweep_settles_it():
            async with get_user_session_factory(USER_ID)() as other:
                activity = (
                    await other.execute(
                        select(Activity).where(Activity.id == ACTIVITY_ID)
                    )
                ).scalar_one()
                settle_activity_analysis(activity)
                await other.commit()
            return datetime.now(timezone.utc)

        fake_model(ANSWER, watch=_the_sweep_settles_it)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID, run_id=run_a)

        row = await _poll(Activity, ACTIVITY_ID)
        assert row.analysis is None, (
            "a run declared dead left its answer on the row"
        )
        assert row.analysis_status is None, "and left a status behind it"

    async def test_a_run_that_still_owns_its_row_writes_normally(
        self, athlete_db, fake_model
    ):
        """The other side: holding the token must not cost anything."""
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg
        from backend.app.services.stranded_runs import begin_activity_analysis_run

        async with get_user_session_factory(USER_ID)() as session:
            activity = (
                await session.execute(select(Activity).where(Activity.id == ACTIVITY_ID))
            ).scalar_one()
            mine = begin_activity_analysis_run(activity)
            await session.commit()

        async def _look_but_change_nothing():
            return datetime.now(timezone.utc)

        fake_model(ANSWER, watch=_look_but_change_nothing)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID, run_id=mine)

        row = await _poll(Activity, ACTIVITY_ID)
        assert row.analysis_status == "done"
        assert row.analysis == "".join(ANSWER)


class TestAFailedRunStillSettles:
    async def test_the_activity_clock_is_stamped_on_error_too(
        self, athlete_db, fake_model
    ):
        """Otherwise a failure could leave a `pending`-era timestamp behind on a
        row that has already settled, and the next reader would be judging a
        stale clock.
        """
        from backend.app.services.llm_activity_analyzer import analyze_activity_bg

        async def _watch():
            raise RuntimeError("the provider fell over")

        fake_model(ANSWER, watch=_watch)
        await analyze_activity_bg(ACTIVITY_ID, ATHLETE_ID, USER_ID)

        row = await _poll(Activity, ACTIVITY_ID)
        assert row.analysis_status == "error"
        assert row.analysis_progress is None
        assert _utc(row.analysis_updated_at) > TRIGGERED_AT
