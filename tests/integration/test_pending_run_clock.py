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
