"""The agentic coaching surfaces, end to end (issue #43).

Where ``tests/unit/test_llm_agent.py`` drives the loop against stubs, this file
runs the real thing: a real per-user SQLite file with real training data, the
real tool layer from issue #42, the real ``analyze_*_bg`` background tasks, and
the real columns the frontend polls. Only the model is fake — a scripted
provider standing in for the HTTP call.

That is the point. The properties worth checking here cannot be seen from a unit
test, because they are properties of the *seam*:

* an answer built from tool results actually contains the athlete's own numbers,
  and still starts with the ``MOOD:`` line the avatar parser needs;
* a poll landing mid-run sees a progress code, and a poll landing after it sees
  none — the card ends up exactly as it looked before any of this existed;
* the pending status settles on success, on failure, and on hitting the cap;
* a failure *before* the drain loop owns the row still clears pending, via
  ``failure_recovery``;
* the tools a run reaches are the caller's own data and nothing else.

Consent is stubbed rather than seeded. The background analyzers hold no registry
session, so ``call_tool`` opens its own, which in a test would reach the real
registry file rather than the in-memory fixture. Consent enforcement itself is
covered where it belongs, in ``test_mcp_tools.py``.
"""
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.models.user_orm import (
    Activity,
    Athlete,
    DailyMetric,
    PlannedWorkout,
    TrainingPlan,
)
from backend.app.services import llm_agent
from backend.app.services.llm_client import ResolvedLlm
from backend.app.services.llm_streaming import StreamSetup, TextDelta, ToolCallDelta

USER_ID = "agentic-user"
ATHLETE_ID = "agentic-athlete"
HOUSE_STYLE = "Always answer in metric units."

ANSWER = "MOOD:cheer\n\n" + "You are riding well and the numbers back it up. " * 6


# ── A scripted provider ─────────────────────────────────────────────────────


class ScriptedProvider:
    """Serves canned turns in place of the HTTP call, recording what it was sent.

    ``before`` is awaited just before a turn is served — the hook that lets a
    test observe the database *while* a run is in flight, which is the only way
    to check what a poll would actually see.
    """

    def __init__(self, *turns, before=None):
        self.turns = list(turns)
        self.sent: list[dict] = []
        self.before = before

    async def __call__(self, cfg, messages, *, tools=None, tool_choice=None, usage_out=None):
        self.sent.append({"messages": [dict(m) for m in messages], "tools": tools})
        if self.before is not None:
            await self.before(len(self.sent))
        assert self.turns, "the provider was called more times than the script allows"
        events, usage, error = self.turns.pop(0)
        if usage_out is not None and usage is not None:
            usage_out["usage"] = usage
        for event in events:
            yield event
        if error is not None:
            raise error


def say(text: str, *, usage: dict | None = None):
    return ([TextDelta(text)], usage, None)


def ask(name: str, arguments: str = "{}", call_id: str = "c1", *, usage: dict | None = None):
    return (
        [
            ToolCallDelta(index=0, id=call_id, name=name),
            ToolCallDelta(index=0, arguments=arguments),
        ],
        usage,
        None,
    )


def boom(error: Exception):
    return ([], None, error)


def _rejects_tools() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
    return httpx.HTTPStatusError(
        "rejected",
        request=request,
        response=httpx.Response(400, text="Unknown parameter: tools", request=request),
    )


def _setup(**cfg_overrides) -> StreamSetup:
    defaults: dict[str, Any] = dict(
        base_url="http://llm.invalid/v1",
        model="test-model",
        api_key=None,
        source="instance",
    )
    defaults.update(cfg_overrides)
    return StreamSetup(cfg=ResolvedLlm(**defaults), analysis_context=HOUSE_STYLE)


@pytest.fixture
def wire(monkeypatch):
    """Patch out the model and the consent round-trip; leave everything else real."""

    def _apply(
        agent_provider,
        *,
        blob_chunks=("MOOD:knowing\n\n", "Blob answer."),
        setup=None,
        record_usage=False,
    ):
        resolved = setup or _setup()

        async def _resolve(athlete, user_id, *, usage_out=None):
            if usage_out is not None:
                usage_out["cfg"] = resolved.cfg
            return resolved

        async def _blob_provider(cfg, messages, *, tools=None, tool_choice=None, usage_out=None):
            assert tools is None, "the single-shot path must never send a tools array"
            for chunk in blob_chunks:
                yield TextDelta(chunk)

        monkeypatch.setattr(llm_agent, "resolve_stream_setup", _resolve)
        monkeypatch.setattr(llm_agent, "stream_completion_events", agent_provider)
        monkeypatch.setattr(
            "backend.app.services.llm_streaming.resolve_stream_setup", _resolve
        )
        monkeypatch.setattr(
            "backend.app.services.llm_streaming.stream_completion_events", _blob_provider
        )
        monkeypatch.setattr(
            "backend.app.mcp.dispatch._consent_ok", AsyncMock(return_value=True)
        )
        # The in-flight count is process-wide; start every run from zero.
        monkeypatch.setattr(llm_agent, "_active_runs", 0)
        if not record_usage:
            # Most tests here have nothing to say about accounting, and the real
            # recorder would open the usage DB on every run.
            monkeypatch.setattr(
                "backend.app.services.llm_streaming.record_llm_usage", AsyncMock()
            )
        return resolved

    return _apply


# ── A real athlete in a real database ───────────────────────────────────────


@pytest.fixture
async def athlete_db(isolate_user_dbs):
    """A per-user SQLite file with enough history for the tools to answer."""
    from backend.app.db.user_session import get_user_session_factory, init_user_db

    await init_user_db(USER_ID)
    today = date.today()

    async with get_user_session_factory(USER_ID)() as session:
        athlete = Athlete(
            id=ATHLETE_ID,
            global_user_id=USER_ID,
            ftp=263,
            max_hr=185,
            weight_kg=71.0,
            ftp_tests=[],
            app_settings={"agentic_koutsi": True, "experience_level": "intermediate"},
            power_zones=[
                {"name": f"Z{i}", "low": low, "high": high}
                for i, (low, high) in enumerate(
                    [(0, 144), (144, 197), (197, 228), (228, 250), (250, 279), (279, 9999)],
                    start=1,
                )
            ],
        )
        session.add(athlete)
        session.add_all(
            [
                Activity(
                    id=f"act-{n}",
                    athlete_id=ATHLETE_ID,
                    name=f"Ride {n}",
                    sport_type="Ride",
                    workout_category="endurance" if n % 2 else "threshold",
                    start_time=datetime.combine(
                        today - timedelta(days=n), time(9, 0), tzinfo=timezone.utc
                    ),
                    duration_s=3600 + n * 60,
                    distance_m=30000.0 + n * 100,
                    avg_power=190.0,
                    weighted_power=205.0,
                    avg_hr=142.0,
                    load=70.0 + n,
                    status="processed",
                    zone_times={"power": {"Z1": 1200, "Z2": 2400}},
                )
                for n in range(1, 6)
            ]
        )
        for offset in range(0, 30):
            session.add(
                DailyMetric(
                    athlete_id=ATHLETE_ID,
                    date=today - timedelta(days=offset),
                    fitness=61.5,
                    fatigue=48.0,
                    form=13.5,
                    load_day=70.0,
                )
            )
        plan = TrainingPlan(
            id="plan-1",
            athlete_id=ATHLETE_ID,
            name="Spring base",
            start_date=today - timedelta(days=7),
            end_date=today + timedelta(days=21),
            weeks=4,
            status="active",
        )
        session.add(plan)
        session.add(
            PlannedWorkout(
                id="pw-1", plan_id="plan-1", week_number=2, day_of_week=1,
                workout_type="threshold", duration_min=75, target_load=95,
            )
        )
        await session.commit()

    return USER_ID


async def read_athlete() -> Athlete:
    """The athlete as a *separate* reader sees it — what a poll would get."""
    from sqlalchemy import select

    from backend.app.db.user_session import get_user_session_factory

    async with get_user_session_factory(USER_ID)() as session:
        return (
            await session.execute(select(Athlete).where(Athlete.id == ATHLETE_ID))
        ).scalar_one()


async def read_activity(activity_id: str) -> Activity:
    from sqlalchemy import select

    from backend.app.db.user_session import get_user_session_factory

    async with get_user_session_factory(USER_ID)() as session:
        return (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()


async def mark_pending() -> None:
    """Put the rows into the state the API writes just before spawning the task.

    The analyzers never set ``pending`` themselves — the request handler does,
    so that a second request can see one is already in flight. Mirroring that
    here is what makes "a poll mid-run sees pending" a real assertion rather
    than an artefact of the fixture.
    """
    from sqlalchemy import select

    from backend.app.db.user_session import get_user_session_factory

    async with get_user_session_factory(USER_ID)() as session:
        athlete = (
            await session.execute(select(Athlete).where(Athlete.id == ATHLETE_ID))
        ).scalar_one()
        athlete.training_status_status = "pending"
        athlete.training_status = None
        athlete.training_status_progress = None
        athlete.training_status_updated_at = datetime.now(timezone.utc)
        await session.commit()


async def run_status(**kwargs) -> Athlete:
    from backend.app.services.llm_training_status_analyzer import (
        analyze_training_status_bg,
    )

    await mark_pending()
    await analyze_training_status_bg(ATHLETE_ID, USER_ID, **kwargs)
    return await read_athlete()


async def run_activity(activity_id: str = "act-1", **kwargs) -> Activity:
    from sqlalchemy import select

    from backend.app.db.user_session import get_user_session_factory
    from backend.app.services.llm_activity_analyzer import analyze_activity_bg

    async with get_user_session_factory(USER_ID)() as session:
        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()
        activity.analysis_status = "pending"
        await session.commit()

    await analyze_activity_bg(activity_id, ATHLETE_ID, USER_ID, **kwargs)
    return await read_activity(activity_id)


def parse_mood(text: str) -> Optional[str]:
    """The backend's copy of the frontend's ``parseMoodAndParagraphs`` rule.

    Deliberately duplicated rather than imported: the contract this asserts is
    the *format of the prose*, and a shared helper would let both sides drift
    together without the test noticing.
    """
    first, _, _rest = text.partition("\n")
    if not first.startswith("MOOD:"):
        return None
    return first[5:].strip().lower()


# ── The training-status surface ─────────────────────────────────────────────


class TestTrainingStatusAgentically:
    async def test_the_answer_is_built_from_real_tool_results(self, athlete_db, wire):
        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1"),
            ask("list_recent_activities", '{"limit": 5}', "c2"),
            say(ANSWER),
        )
        wire(provider)

        athlete = await run_status()

        assert athlete.training_status_status == "done"
        assert athlete.training_status == ANSWER
        assert parse_mood(athlete.training_status) == "cheer"

        # The tools ran against this athlete's own database, and their output
        # reached the model — the whole reason the loop exists.
        results = [m["content"] for m in provider.sent[2]["messages"] if m.get("role") == "tool"]
        assert len(results) == 2
        assert "263" in results[0]      # the athlete's own FTP, via get_training_status
        assert "Ride 1" in results[1]   # their own rides, via list_recent_activities
        # And nothing that is not theirs: the tools open one user's database.
        assert all("Ride 9" not in r for r in results)

    async def test_a_poll_mid_run_sees_a_progress_code_and_a_finished_one_sees_none(
        self, athlete_db, wire
    ):
        observed: list[tuple[str | None, str | None]] = []

        async def peek(turn_number: int) -> None:
            # Reading through a fresh session is the point: it only sees what has
            # actually been committed, which is exactly what the poll sees.
            athlete = await read_athlete()
            observed.append((athlete.training_status_status, athlete.training_status_progress))

        provider = ScriptedProvider(
            ask("get_power_profile", "{}", "c1"),
            say(ANSWER),
            before=peek,
        )
        wire(provider)

        athlete = await run_status()

        # Before the first turn: pending, and already saying it is thinking.
        assert observed[0] == ("pending", "thinking")
        # Before the answering turn — the long one — the card still names the
        # tool whose result that turn is reading, rather than falling back to a
        # generic line for the slowest part of the run.
        assert observed[1] == ("pending", "tool.get_power_profile")
        # Settled: no leftover step under the finished answer.
        assert athlete.training_status_status == "done"
        assert athlete.training_status_progress is None

    async def test_the_api_only_reports_progress_while_pending(
        self, athlete_db, wire, client, auth_headers, session, seeded_athlete
    ):
        # A run killed between its last progress commit and settling would leave
        # a code behind. The endpoint gates on the status so it can never surface.
        seeded_athlete.training_status_status = "done"
        seeded_athlete.training_status = ANSWER
        seeded_athlete.training_status_progress = "tool.get_power_profile"
        seeded_athlete.training_status_date = date.today()
        # A fresh timestamp, or the endpoint's stuck-pending recovery would fire
        # below and settle the row before it could report anything.
        seeded_athlete.training_status_updated_at = datetime.now(timezone.utc)
        await session.commit()

        response = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["progress"] is None

        seeded_athlete.training_status_status = "pending"
        await session.commit()
        response = await client.get("/api/athlete/training-status", headers=auth_headers)
        assert response.json()["progress"] == "tool.get_power_profile"

    async def test_hitting_the_cap_still_settles(self, athlete_db, wire, monkeypatch):
        monkeypatch.setattr(llm_agent, "MAX_ROUNDS_STATUS", 2, raising=False)
        from backend.app.services import llm_training_status_analyzer as analyzer

        monkeypatch.setattr(analyzer, "MAX_ROUNDS_STATUS", 2)

        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1"),
            ask("get_plan_status", "{}", "c2"),
            say(ANSWER),          # the forced final turn
            ask("get_zone_totals", "{}", "c3"),  # never reached
        )
        wire(provider)

        athlete = await run_status()

        assert len(provider.sent) == 3
        assert provider.sent[2]["tools"] is None
        assert athlete.training_status_status == "done"
        assert athlete.training_status_progress is None

    async def test_a_provider_failure_settles_as_an_error_with_no_leftover_step(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1"),
            boom(RuntimeError("the model server died")),
        )
        wire(provider)

        athlete = await run_status()

        assert athlete.training_status_status == "error"
        assert athlete.training_status_progress is None

    async def test_a_failure_before_the_drain_loop_still_clears_pending(
        self, athlete_db, wire, monkeypatch
    ):
        # `stream_into_db` settles the status itself, but only once it is
        # running. Blowing up in the context-gathering ahead of it is exactly the
        # case `failure_recovery` exists for — and `run_status` leaves the row
        # `pending` first, so a failure to clear it would be a stuck spinner.
        wire(ScriptedProvider(say(ANSWER)))
        monkeypatch.setattr(
            "backend.app.services.llm_training_status_analyzer.compute_intensity_distribution",
            AsyncMock(side_effect=RuntimeError("metrics engine exploded")),
        )

        athlete = await run_status()

        assert athlete.training_status_status == "error"
        assert athlete.training_status_progress is None


class TestTrainingStatusFallsBackToTheBlob:
    async def test_a_provider_rejecting_tools_produces_a_blob_answer(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider(boom(_rejects_tools()))
        wire(provider)

        athlete = await run_status()

        assert athlete.training_status_status == "done"
        assert athlete.training_status == "MOOD:knowing\n\nBlob answer."
        assert athlete.training_status_progress is None

    async def test_a_provider_that_calls_no_tool_produces_a_blob_answer(
        self, athlete_db, wire
    ):
        # An answer written with no data at all is worse than the blob, which at
        # least had the numbers in front of it.
        wire(ScriptedProvider(say("MOOD:cheer\n\nEverything looks great!")))

        athlete = await run_status()

        assert athlete.training_status == "MOOD:knowing\n\nBlob answer."

    async def test_a_preset_flagged_unsupported_never_reaches_the_loop(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider()  # scripted with nothing: a call would fail
        wire(provider, setup=_setup(tools_supported=False))

        athlete = await run_status()

        assert athlete.training_status == "MOOD:knowing\n\nBlob answer."
        assert provider.sent == []

    async def test_an_athlete_who_has_not_opted_in_takes_the_blob_path(
        self, athlete_db, wire
    ):
        from sqlalchemy import select

        from backend.app.db.user_session import get_user_session_factory

        async with get_user_session_factory(USER_ID)() as session:
            athlete = (
                await session.execute(select(Athlete).where(Athlete.id == ATHLETE_ID))
            ).scalar_one()
            athlete.app_settings = {"experience_level": "intermediate"}
            await session.commit()

        provider = ScriptedProvider()
        wire(provider)

        result = await run_status()

        assert result.training_status == "MOOD:knowing\n\nBlob answer."
        assert provider.sent == []

    async def test_a_bulk_import_stays_on_the_blob_path_even_when_opted_in(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider()
        wire(provider)

        athlete = await run_status(allow_agentic=False)

        assert athlete.training_status == "MOOD:knowing\n\nBlob answer."
        assert provider.sent == []


# ── The activity surface ────────────────────────────────────────────────────


class TestActivityAnalysisAgentically:
    async def test_the_analysis_is_built_from_the_activity_detail_tool(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider(
            ask("get_activity_detail", '{"activity_id": "act-1"}', "c1"),
            say(ANSWER),
        )
        wire(provider)

        activity = await run_activity("act-1")

        assert activity.analysis_status == "done"
        assert parse_mood(activity.analysis) == "cheer"
        assert activity.analysis_progress is None
        assert "Ride 1" in str(provider.sent[1]["messages"])

    async def test_the_brief_names_the_activity_the_task_was_given(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider(
            ask("get_activity_detail", '{"activity_id": "act-3"}', "c1"),
            say(ANSWER),
        )
        wire(provider)
        await run_activity("act-3")

        user_turn = [m for m in provider.sent[0]["messages"] if m["role"] == "user"][0]
        assert "act-3" in user_turn["content"]

    async def test_a_missing_activity_id_is_a_sentence_the_run_recovers_from(
        self, athlete_db, wire
    ):
        # The tool layer answers a bad id with prose naming what is nearby
        # (issue #42), and the loop feeds that back rather than dying.
        provider = ScriptedProvider(
            ask("get_activity_detail", '{"activity_id": "act-does-not-exist"}', "c1"),
            ask("get_activity_detail", '{"activity_id": "act-1"}', "c2"),
            say(ANSWER),
        )
        wire(provider)

        activity = await run_activity("act-1")

        assert activity.analysis_status == "done"
        tool_reply = [m for m in provider.sent[1]["messages"] if m.get("role") == "tool"][0]
        assert "act-does-not-exist" in tool_reply["content"]

    async def test_a_bulk_import_analysis_stays_on_the_blob_path(self, athlete_db, wire):
        provider = ScriptedProvider()
        wire(provider)

        activity = await run_activity("act-1", allow_agentic=False)

        assert activity.analysis == "MOOD:knowing\n\nBlob answer."
        assert provider.sent == []


# ── The format contract ─────────────────────────────────────────────────────


class TestTheMoodContractSurvives:
    async def test_the_house_style_reaches_every_turn_including_the_answer(
        self, athlete_db, wire
    ):
        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1"),
            ask("get_plan_status", "{}", "c2"),
            say(ANSWER),
        )
        wire(provider)
        await run_status()

        for turn in provider.sent:
            systems = [m["content"] for m in turn["messages"] if m["role"] == "system"]
            assert HOUSE_STYLE in systems

    async def test_the_mood_rule_is_restated_on_the_forced_final_turn(
        self, athlete_db, wire, monkeypatch
    ):
        from backend.app.services import llm_training_status_analyzer as analyzer

        monkeypatch.setattr(analyzer, "MAX_ROUNDS_STATUS", 1)
        provider = ScriptedProvider(ask("get_training_status", "{}", "c1"), say(ANSWER))
        wire(provider)
        await run_status()

        final_system = provider.sent[1]["messages"][-1]
        assert final_system["role"] == "system"
        assert "MOOD:<mood>" in final_system["content"]

    async def test_an_answer_without_a_mood_line_is_stored_as_written(
        self, athlete_db, wire
    ):
        # The parser tolerates a missing MOOD line by defaulting the avatar. What
        # must never happen is the backend "helpfully" prepending one, or the
        # literal token leaking into the first paragraph.
        plain = "Your form is fine. " * 20
        provider = ScriptedProvider(ask("get_training_status", "{}", "c1"), say(plain))
        wire(provider)

        athlete = await run_status()

        assert athlete.training_status == plain
        assert parse_mood(athlete.training_status) is None
        assert "MOOD:" not in athlete.training_status


# ── Accounting ──────────────────────────────────────────────────────────────


class TestUsageAccounting:
    async def test_an_instance_paid_run_records_the_whole_run_not_the_last_turn(
        self, athlete_db, wire, usage_db, monkeypatch
    ):
        from sqlalchemy import select

        from backend.app.models.usage_orm import LlmUsage

        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1", usage={"prompt_tokens": 900, "completion_tokens": 30}),
            ask("get_plan_status", "{}", "c2", usage={"prompt_tokens": 1400, "completion_tokens": 25}),
            say(ANSWER, usage={"prompt_tokens": 2000, "completion_tokens": 350}),
        )
        # The real recorder, writing into this test's temp usage DB.
        wire(provider, record_usage=True)

        await run_status()

        async with usage_db() as session:
            rows = (await session.execute(select(LlmUsage))).scalars().all()
        assert len(rows) == 1
        # Recording only the final turn would report 2000/350 for a run that
        # actually cost 4300/405 — a 2.1× under-report on this three-turn run.
        assert rows[0].prompt_tokens == 4300
        assert rows[0].completion_tokens == 405
        assert rows[0].feature == "training_status"

    async def test_a_byok_run_records_nothing(self, athlete_db, wire, usage_db):
        from sqlalchemy import select

        from backend.app.models.usage_orm import LlmUsage

        provider = ScriptedProvider(
            ask("get_training_status", "{}", "c1", usage={"prompt_tokens": 900}),
            say(ANSWER, usage={"prompt_tokens": 2000, "completion_tokens": 350}),
        )
        wire(
            provider,
            setup=_setup(source="user", key_source="user"),
            record_usage=True,
        )

        await run_status()

        async with usage_db() as session:
            rows = (await session.execute(select(LlmUsage))).scalars().all()
        # The athlete paid their own provider; the hoster has no cost to account
        # for. Unchanged from the single-shot path.
        assert rows == []
