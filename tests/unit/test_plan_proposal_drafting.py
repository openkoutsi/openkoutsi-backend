"""The nested model call behind a drafted plan (issue #72).

Issue #42 kept ``/api/llm/*`` out of the tool surface because *"a tool that calls
an LLM from inside an LLM loop is a recursion waiting to happen"*. The proposal
tools call one anyway, and the answer to that worry is asserted here rather than
asserted in prose: the nested call goes through
``call_llm_with_optional_schema``, which sends **one schema-constrained
completion with no ``tools`` array at all**, so it cannot re-enter the tool
layer. It is a nested *call*, not a nested *loop*.

The rest is what has to hold around it: the fallback when it fails, the usage
ledger, and whose calendar the draft is written against.
"""
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from backend.app.mcp.dispatch import ToolCaller, call_tool
from backend.app.mcp.limits import tool_limiter
from backend.app.models.user_orm import PlanProposal
from backend.app.services.llm_client import ResolvedLlm

_TEST_USER_ID = "test-user-00000000"

_ARGS = {
    "name": "October fondo",
    "start_date": (date.today() + timedelta(days=7)).isoformat(),
    "weeks": 2,
    "goal": "Hilly gran fondo",
}


@pytest.fixture(autouse=True)
def _clean_rate_limiter():
    tool_limiter.reset()
    yield
    tool_limiter.reset()


def _cfg() -> ResolvedLlm:
    return ResolvedLlm(
        base_url="http://localhost:11434/v1", model="test-model", api_key=None
    )


def _week_json(weeks: int) -> str:
    import json

    return json.dumps(
        {
            "weeks": [
                {
                    "week_number": w,
                    "workouts": [
                        {
                            "day_of_week": d,
                            "workout_type": "endurance" if d in (2, 6) else "rest",
                            "description": "Steady Zone 2" if d in (2, 6) else None,
                            "duration_min": 90 if d in (2, 6) else None,
                            "target_load": 80 if d in (2, 6) else None,
                        }
                        for d in range(1, 8)
                    ],
                }
                for w in range(1, weeks + 1)
            ]
        }
    )


def _http(content: str) -> AsyncMock:
    resp = MagicMock()
    resp.is_error = False
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 340},
    }
    http = AsyncMock()
    http.post = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)
    return http


async def _propose(session, athlete, registry_session, *, llm, http=None, **overrides):
    args = dict(_ARGS)
    args.update(overrides)
    caller = ToolCaller(user_id=_TEST_USER_ID, scopes=None, kind="session")
    if http is None:
        return await call_tool(
            caller, "propose_training_plan", args, session=session, athlete=athlete,
            registry_session=registry_session, llm=llm,
        )
    with patch("httpx.AsyncClient", return_value=http):
        return await call_tool(
            caller, "propose_training_plan", args, session=session, athlete=athlete,
            registry_session=registry_session, llm=llm,
        )


async def _proposal(session):
    return (await session.execute(select(PlanProposal))).scalar_one()


# ── The recursion guard ─────────────────────────────────────────────────────


async def test_the_nested_call_carries_no_tools_array(
    session, seeded_athlete, registry_session
):
    """**Asserted, not assumed.**

    A ``tools`` array here would let the nested completion call back into the
    tool layer, and the loop that dispatched this tool is still on the stack.
    """
    http = _http(_week_json(2))
    result = await _propose(
        session, seeded_athlete, registry_session, llm=_cfg(), http=http
    )
    assert result.ok, result.error

    payloads = [call.kwargs["json"] for call in http.post.call_args_list]
    assert payloads, "the nested call was never made"
    for payload in payloads:
        assert "tools" not in payload
        assert "tool_choice" not in payload
    # One completion, not two: the retry is deliberately skipped on this path.
    assert len(payloads) == 1


async def test_a_drafted_plan_records_that_a_model_wrote_it(
    session, seeded_athlete, registry_session
):
    result = await _propose(
        session, seeded_athlete, registry_session, llm=_cfg(), http=_http(_week_json(2))
    )
    assert result.data["summary"]["built_by"] == "llm"
    assert result.data["summary"]["fallback_reason"] is None
    assert (await _proposal(session)).built_by == "llm"
    # And the weeks in the payload are the ones the model produced.
    payload = (await _proposal(session)).payload
    assert payload["generation_method"] == "llm"
    assert [d["duration_min"] for d in payload["weeks_data"][0]] == [
        None, 90, None, None, None, 90, None
    ]


# ── Degrade, never fail ─────────────────────────────────────────────────────


async def test_an_unparseable_draft_falls_back_to_the_builder(
    session, seeded_athlete, registry_session
):
    """Same posture as ``coaching_stream``: a worse draft now beats no draft.

    And the fallback is *visible* — ``built_by`` says which builder wrote the
    weeks, so a rule-built plan is never passed off as a written one.
    """
    result = await _propose(
        session, seeded_athlete, registry_session, llm=_cfg(),
        http=_http("this is not JSON"),
    )
    assert result.ok, result.error
    summary = result.data["summary"]
    assert summary["built_by"] == "rule_based"
    assert summary["fallback_reason"]
    assert len(summary["weekly"]) == 2
    assert (await _proposal(session)).built_by == "rule_based"
    # The model is told, so its prose does not claim authorship it does not have.
    assert "openkoutsi's own builder" in result.data["note"]


async def test_a_provider_that_cannot_be_reached_still_produces_a_proposal(
    session, seeded_athlete, registry_session
):
    import httpx

    http = AsyncMock()
    http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    result = await _propose(
        session, seeded_athlete, registry_session, llm=_cfg(), http=http
    )
    assert result.ok, result.error
    assert result.data["summary"]["built_by"] == "rule_based"


async def test_a_caller_with_no_model_gets_the_deterministic_builder(
    session, seeded_athlete, registry_session
):
    """The shape the MCP door will arrive in when it opens.

    An external caller has no entitled loop in front of it, so it arrives with
    ``llm=None`` — and gets a rule-built proposal rather than a refusal.
    """
    result = await _propose(session, seeded_athlete, registry_session, llm=None)
    assert result.ok, result.error
    assert result.data["summary"]["built_by"] == "rule_based"
    assert "no model was available" in result.data["summary"]["fallback_reason"]


# ── The ledger and the calendar ─────────────────────────────────────────────


async def test_the_spend_lands_under_plan_generation(
    session, seeded_athlete, registry_session
):
    """A proposal costs tokens *outside* the chat turn budget, which counts
    turns rather than completions — so it has to land somewhere countable."""
    recorded = []

    async def _record(*, user_id, feature, cfg, usage):
        recorded.append((feature, usage))

    with patch(
        "backend.app.services.llm_plan_generator.record_llm_usage", new=_record
    ):
        result = await _propose(
            session, seeded_athlete, registry_session, llm=_cfg(),
            http=_http(_week_json(2)),
        )
    assert result.ok, result.error
    assert [feature for feature, _ in recorded] == ["plan_generate"]
    assert recorded[0][1]["completion_tokens"] == 340


async def test_the_recent_training_window_is_the_athletes_day_not_the_servers(
    session, seeded_athlete, registry_session
):
    """``generate_plan_weeks_llm`` measured its 12-week window back from
    ``date.today()`` — the *server's* date. ``ToolRun.today`` is the athlete's,
    which is the whole reason issue #43 threaded it through."""
    seen = {}

    async def _distribution(athlete, session, *, start, end):
        seen["start"] = start
        seen["end"] = end
        return MagicMock()

    athlete_today = date.today() + timedelta(days=2)
    caller = ToolCaller(user_id=_TEST_USER_ID, scopes=None, kind="session")
    with patch(
        "backend.app.services.llm_plan_generator.compute_intensity_distribution",
        new=_distribution,
    ), patch("httpx.AsyncClient", return_value=_http(_week_json(2))):
        result = await call_tool(
            caller, "propose_training_plan",
            dict(_ARGS, start_date=(athlete_today + timedelta(days=7)).isoformat()),
            session=session, athlete=seeded_athlete,
            registry_session=registry_session, llm=_cfg(), today=athlete_today,
        )
    assert result.ok, result.error
    assert seen["end"] == athlete_today
