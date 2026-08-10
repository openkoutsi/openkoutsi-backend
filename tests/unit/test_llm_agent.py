"""The agentic coaching loop, against a scripted provider (issue #43).

No network and no real model: every test here drives
:func:`~backend.app.services.llm_agent.agentic_stream` with a
:class:`FakeProvider` whose turns are written out in advance, and asserts on two
things — what the loop yielded downstream, and what it *sent* on each turn. The
second matters as much as the first. The conversation the provider receives is
the contract with the dialect: one ``role: "tool"`` message per ``tool_call_id``,
the instance's house style present on every turn rather than only the first, and
the ``MOOD:`` rule restated where the model actually answers.

The behaviours worth naming, because they are decisions rather than mechanics:

* **A turn-zero answer is discarded.** The agentic prompt carries no data, so
  prose written before any tool call is guesswork, and the run falls back.
* **Prose is buffered before it is committed.** A preamble ("let me look at…")
  must not reach the column the answer lives in.
* **Nothing raises out of a tool.** Bad JSON, a refusal, a timeout — each becomes
  a sentence in the conversation and the run continues.
* **Usage sums across turns**, because the alternative silently under-reports the
  admin usage summary by however many turns a run took.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import pytest

from backend.app.core.config import settings
from backend.app.mcp.dispatch import ToolResult
from backend.app.services import llm_agent
from backend.app.services.llm_agent import (
    MAX_TOOL_RESULT_CHARS,
    PROGRESS_THINKING,
    AgenticUnavailable,
    AgentRequest,
    agentic_enabled,
    agentic_stream,
    coaching_stream,
    progress_code_for_tool,
    progress_vocabulary,
    tool_definitions,
)
from backend.app.services.llm_client import ResolvedLlm
from backend.app.services.llm_streaming import (
    AgentProgress,
    StreamSetup,
    TextDelta,
    ToolCallDelta,
)

HOUSE_STYLE = "This instance coaches in metric units and never mentions weight."
MOOD_RULE = "The MOOD line must be the very first line."


# ── A scripted provider ─────────────────────────────────────────────────────


@dataclass
class Turn:
    """One completion the fake provider will serve."""

    events: list[Any] = field(default_factory=list)
    usage: Optional[dict] = None
    #: Raised instead of yielding, to script a provider-side failure.
    error: Optional[Exception] = None


def text(*parts: str, usage: dict | None = None) -> Turn:
    return Turn(events=[TextDelta(p) for p in parts], usage=usage)


def calls(*specs: tuple, usage: dict | None = None, preamble: str = "") -> Turn:
    """A tool-calling turn.

    Each spec is ``(index, call_id, name, arguments_json)``. Arguments are split
    across two deltas so the reassembly is exercised the way a real provider
    streams it, rather than arriving whole.
    """
    events: list[Any] = [TextDelta(preamble)] if preamble else []
    for index, call_id, name, arguments in specs:
        events.append(ToolCallDelta(index=index, id=call_id, name=name))
        midpoint = len(arguments) // 2
        events.append(ToolCallDelta(index=index, arguments=arguments[:midpoint]))
        events.append(ToolCallDelta(index=index, arguments=arguments[midpoint:]))
    return Turn(events=events, usage=usage)


class FakeProvider:
    """Serves scripted turns and records what each one was sent."""

    def __init__(self, *turns: Turn):
        self.turns = list(turns)
        self.sent: list[dict] = []

    async def __call__(self, cfg, messages, *, tools=None, tool_choice=None, usage_out=None):
        self.sent.append({"messages": [dict(m) for m in messages], "tools": tools})
        assert self.turns, "the provider was called more times than the script allows"
        turn = self.turns.pop(0)
        if usage_out is not None and turn.usage is not None:
            usage_out["usage"] = turn.usage
        for event in turn.events:
            yield event
        # After the events, so a turn can be scripted to fail *mid-stream* —
        # which is the only way prose reaches the DB before the provider gives
        # up, and therefore the only way to exercise the no-late-fallback rule.
        if turn.error is not None:
            raise turn.error

    @property
    def turn_count(self) -> int:
        return len(self.sent)

    def tool_messages(self, turn_index: int) -> list[dict]:
        return [m for m in self.sent[turn_index]["messages"] if m.get("role") == "tool"]


@dataclass
class FakeTool:
    """A stand-in for one registry tool, returning a canned result."""

    name: str
    result: Any = None
    error: Optional[str] = None
    delay_s: float = 0.0


class FakeDispatch:
    """Stands in for ``call_tool``, recording every invocation."""

    def __init__(self, *tools: FakeTool):
        self.tools = {t.name: t for t in tools}
        self.invocations: list[tuple[str, dict]] = []

    async def __call__(self, caller, name, arguments=None, **kwargs):
        self.invocations.append((name, dict(arguments or {})))
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(
                tool=name, ok=False, error=f"No tool named '{name}'.", duration_ms=1.0
            )
        if tool.delay_s:
            await asyncio.sleep(tool.delay_s)
        if tool.error is not None:
            return ToolResult(tool=name, ok=False, error=tool.error, duration_ms=1.0)
        return ToolResult(tool=name, ok=True, data=tool.result, duration_ms=1.0)


def _setup(*, tools_supported: bool = True, house_style: str | None = HOUSE_STYLE) -> StreamSetup:
    return StreamSetup(
        cfg=ResolvedLlm(
            base_url="http://llm.invalid/v1",
            model="test-model",
            api_key=None,
            source="instance",
            tools_supported=tools_supported,
        ),
        analysis_context=house_style,
    )


class _Athlete:
    def __init__(self, app_settings: dict | None = None):
        self.id = "ath-1"
        self.app_settings = app_settings if app_settings is not None else {}


def _request(**overrides) -> AgentRequest:
    defaults: dict = dict(
        athlete=_Athlete(),
        user_id="user-1",
        system_prompt="You are Koutsi.",
        user_prompt="Give the athlete their daily feedback.",
        feature="training_status",
        max_rounds=3,
        format_rule=MOOD_RULE,
    )
    defaults.update(overrides)
    return AgentRequest(**defaults)


@pytest.fixture(autouse=True)
def _fresh_slots(monkeypatch):
    """The in-flight count is process-wide; each test starts from zero."""
    monkeypatch.setattr(llm_agent, "_active_runs", 0)
    yield
    monkeypatch.setattr(llm_agent, "_active_runs", 0)


async def drive(
    provider: FakeProvider,
    dispatch: FakeDispatch | None = None,
    *,
    request: AgentRequest | None = None,
    setup: StreamSetup | None = None,
    monkeypatch=None,
    usage_out: dict | None = None,
) -> tuple[str, list[Optional[str]]]:
    """Run the loop, returning ``(prose, progress_codes)``."""
    resolved_setup = setup or _setup()

    async def _resolve(athlete, user_id, *, usage_out=None):
        if usage_out is not None:
            usage_out["cfg"] = resolved_setup.cfg
        return resolved_setup

    monkeypatch.setattr(llm_agent, "stream_completion_events", provider)
    monkeypatch.setattr(llm_agent, "resolve_stream_setup", _resolve)
    monkeypatch.setattr(llm_agent, "call_tool", dispatch or FakeDispatch())

    prose: list[str] = []
    steps: list[Optional[str]] = []
    async for item in agentic_stream(request or _request(), usage_out if usage_out is not None else {}):
        if isinstance(item, AgentProgress):
            steps.append(item.code)
        else:
            prose.append(item)
    return "".join(prose), steps


ANSWER = "MOOD:knowing\n\n" + "Your form is holding up well after that block. " * 8


# ── The happy path ──────────────────────────────────────────────────────────


class TestChainedToolCalls:
    async def test_two_chained_calls_complete_and_the_answer_uses_them(self, monkeypatch):
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": -4.2}),
            FakeTool("list_recent_activities", {"items": [{"name": "Long Sunday"}]}),
        )
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            calls((0, "c2", "list_recent_activities", '{"limit": 5}')),
            text(ANSWER),
        )
        prose, steps = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        assert [name for name, _ in dispatch.invocations] == [
            "get_training_status",
            "list_recent_activities",
        ]
        # Arguments survived being split across two deltas.
        assert dispatch.invocations[1][1] == {"limit": 5}
        # The second turn saw the first tool's result.
        assert "-4.2" in json.dumps(provider.sent[1]["messages"])

    async def test_progress_codes_narrate_the_run_and_clear_for_the_answer(
        self, monkeypatch
    ):
        dispatch = FakeDispatch(FakeTool("get_power_profile", {"ftp_w": 250}))
        provider = FakeProvider(
            calls((0, "c1", "get_power_profile", "{}")),
            text(ANSWER),
        )
        _, steps = await drive(provider, dispatch, monkeypatch=monkeypatch)

        # `thinking` only until the first tool is named; then the tool's own code
        # stays up through the turn that reads its result — the slow part — and
        # is cleared the instant real prose starts.
        assert steps == [
            PROGRESS_THINKING,
            progress_code_for_tool("get_power_profile"),
            None,
        ]
        # A cleared step is what makes the finished card look like it always did.
        assert steps[-1] is None

    async def test_a_short_answer_below_the_commit_threshold_still_arrives(
        self, monkeypatch
    ):
        # A supplemental-sport acknowledgement is one or two sentences — well
        # under `_COMMIT_AFTER_CHARS` — so it is only released when the turn ends.
        short = "MOOD:cheer\n\nNice swim."
        provider = FakeProvider(
            calls((0, "c1", "get_activity_detail", '{"activity_id": "a1"}')),
            text(short),
        )
        dispatch = FakeDispatch(FakeTool("get_activity_detail", {"sport": "Swim"}))
        prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)
        assert prose == short


class TestPreambleIsNotTheAnswer:
    async def test_narration_before_a_tool_call_never_reaches_the_column(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            calls(
                (0, "c2", "list_recent_activities", "{}"),
                preamble="Let me also look at the last four weeks. ",
            ),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("list_recent_activities", {"items": []}),
        )
        prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        assert "Let me also look" not in prose


# ── Recovering from a badly-behaved provider ────────────────────────────────


class TestMalformedToolCalls:
    async def test_unparseable_json_is_fed_back_not_raised(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{not json at all")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        # The tool was never run, and the model was told why in a sentence.
        assert dispatch.invocations == []
        tool_msgs = provider.tool_messages(1)
        assert len(tool_msgs) == 1
        assert "not valid JSON" in tool_msgs[0]["content"]

    async def test_arguments_that_are_json_but_not_an_object_are_refused(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "[1, 2, 3]")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {}))
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert dispatch.invocations == []
        assert "must be a JSON object" in provider.tool_messages(1)[0]["content"]

    async def test_empty_arguments_mean_no_arguments(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        await drive(provider, dispatch, monkeypatch=monkeypatch)
        assert dispatch.invocations == [("get_training_status", {})]

    async def test_a_call_with_no_function_name_is_dropped(self, monkeypatch):
        # Nothing to dispatch and nothing the model could act on if told, so the
        # call simply never happens — and crucially the run does not die.
        provider = FakeProvider(
            Turn(events=[ToolCallDelta(index=0, id="c1", arguments="{}")]),
            text(ANSWER),
        )
        with pytest.raises(AgenticUnavailable):
            # No usable call and no prose on turn zero: nothing to work with.
            await drive(provider, monkeypatch=monkeypatch)


class TestToolFailuresAreContent:
    async def test_a_failing_tool_returns_its_message_and_the_run_continues(
        self, monkeypatch
    ):
        message = (
            "No activity on 2026-07-14. Nearest rides: 2026-07-13 (endurance, 2 h 04)."
        )
        provider = FakeProvider(
            calls((0, "c1", "find_activity", '{"on_date": "2026-07-14"}')),
            calls((0, "c2", "find_activity", '{"on_date": "2026-07-13"}')),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("find_activity", error=message))
        prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        assert provider.tool_messages(1)[0]["content"] == message

    async def test_an_unknown_tool_names_the_real_ones(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_the_weather", "{}")),
            text(ANSWER),
        )
        prose, _ = await drive(provider, FakeDispatch(), monkeypatch=monkeypatch)
        assert prose == ANSWER
        assert "No tool named" in provider.tool_messages(1)[0]["content"]

    async def test_a_hanging_tool_is_stopped_and_the_run_carries_on(
        self, monkeypatch
    ):
        monkeypatch.setattr(llm_agent, "TOOL_TIMEOUT_S", 0.02)
        provider = FakeProvider(
            calls((0, "c1", "get_zone_totals", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_zone_totals", {}, delay_s=0.5))
        prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        assert "took longer than" in provider.tool_messages(1)[0]["content"]

    async def test_the_tools_open_their_own_session(self, monkeypatch):
        # The reason a timeout is survivable at all. `wait_for` cancels
        # `call_tool` wherever it happens to be; on a *shared* session that
        # invalidates the connection (every later use raises
        # PendingRollbackError) and the `rollback()` repairing it expires every
        # ORM instance, so later attribute reads raise MissingGreenlet — which
        # broke the blob fallback the timeout exists to protect. A session
        # nobody else holds has neither problem, for no inputs rather than for
        # the ones we thought to handle.
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")), text(ANSWER)
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))

        seen: list[dict] = []
        original = dispatch.__call__

        async def record(caller, name, arguments=None, **kwargs):
            seen.append(kwargs)
            return await original(caller, name, arguments, **kwargs)

        await drive(provider, record, monkeypatch=monkeypatch)

        assert seen, "no tool was dispatched"
        for kwargs in seen:
            assert "session" not in kwargs, "a shared session is the hazard"
            assert "athlete" not in kwargs, "the athlete comes from that session too"
            assert kwargs["today"] is not None


class TestOneResultPerCall:
    async def test_every_call_in_a_parallel_turn_gets_exactly_one_result(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls(
                (0, "c1", "get_training_status", "{}"),
                (1, "c2", "get_plan_status", "{}"),
                (2, "c3", "get_goal_progress", "{}"),
            ),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
            FakeTool("get_goal_progress", {"goals": []}),
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        second_turn = provider.sent[1]["messages"]
        assistant = [m for m in second_turn if m.get("role") == "assistant"]
        assert len(assistant) == 1
        announced = [c["id"] for c in assistant[0]["tool_calls"]]
        answered = [m["tool_call_id"] for m in provider.tool_messages(1)]
        # The classic 400 on the next request is a mismatch here, in either
        # direction: an unanswered call or a result with no call.
        assert announced == answered == ["c1", "c2", "c3"]

    async def test_repeated_ids_are_made_unique_so_the_pairing_still_holds(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls(
                (0, "dup", "get_training_status", "{}"),
                (1, "dup", "get_plan_status", "{}"),
            ),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        assistant = [m for m in provider.sent[1]["messages"] if m.get("role") == "assistant"][0]
        announced = [c["id"] for c in assistant["tool_calls"]]
        answered = [m["tool_call_id"] for m in provider.tool_messages(1)]
        assert len(set(announced)) == 2
        assert announced == answered


# ── Budgets ─────────────────────────────────────────────────────────────────


class TestIterationCap:
    async def test_the_cap_is_never_exceeded_and_a_forced_turn_settles_the_run(
        self, monkeypatch
    ):
        # Four tool turns scripted against a cap of three: the fourth is never
        # reached, and the loop forces an answer instead.
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            calls((0, "c2", "get_training_status", "{}")),
            calls((0, "c3", "get_training_status", "{}")),
            text(ANSWER),
            calls((0, "c4", "get_training_status", "{}")),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        prose, steps = await drive(
            provider, dispatch, request=_request(max_rounds=3), monkeypatch=monkeypatch
        )

        assert prose == ANSWER
        assert provider.turn_count == 4  # three tool rounds + the forced answer
        assert steps[-1] is None

    async def test_the_forced_turn_is_offered_no_tools_at_all(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        await drive(
            provider, dispatch, request=_request(max_rounds=1), monkeypatch=monkeypatch
        )
        # `tool_choice: "none"` is advisory and some servers ignore it; removing
        # the array is not.
        assert provider.sent[0]["tools"]
        assert provider.sent[1]["tools"] is None

    async def test_a_forced_turn_that_says_nothing_falls_back(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            text(""),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        with pytest.raises(AgenticUnavailable):
            await drive(
                provider, dispatch, request=_request(max_rounds=1), monkeypatch=monkeypatch
            )


class TestBreadthAndBudget:
    async def test_a_turn_asking_for_everything_runs_only_the_first_few(
        self, monkeypatch
    ):
        # The round cap counts round *trips*; without this a shotgunning model
        # does 6 x 9 calls and replays all of it into every later turn.
        names = [
            "get_training_status", "list_recent_activities", "get_plan_status",
            "get_goal_progress", "get_zone_totals", "get_power_profile",
        ]
        provider = FakeProvider(
            calls(*[(i, f"c{i}", n, "{}") for i, n in enumerate(names)]),
            text(ANSWER),
        )
        dispatch = FakeDispatch(*[FakeTool(n, {"ok": True}) for n in names])
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert len(dispatch.invocations) == llm_agent.MAX_CALLS_PER_TURN
        # Every announced call still gets exactly one result — the pairing the
        # dialect checks — and the refused ones say why rather than vanishing.
        assistant = [m for m in provider.sent[1]["messages"] if m.get("role") == "assistant"][0]
        answered = provider.tool_messages(1)
        assert len(answered) == len(assistant["tool_calls"]) == len(names)
        refusals = [m for m in answered if "was not run" in m["content"]]
        assert len(refusals) == len(names) - llm_agent.MAX_CALLS_PER_TURN
        assert "a few at a time" in refusals[0]["content"]

    async def test_a_turn_within_the_breadth_cap_is_untouched(self, monkeypatch):
        provider = FakeProvider(
            calls(
                (0, "c1", "get_training_status", "{}"),
                (1, "c2", "get_plan_status", "{}"),
            ),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)
        assert len(dispatch.invocations) == 2
        assert not [m for m in provider.tool_messages(1) if "was not run" in m["content"]]

    async def test_a_run_that_eats_its_character_budget_is_forced_to_answer(
        self, monkeypatch
    ):
        # Bounds the thing that actually blows a small context window: the *sum*
        # of tool results replayed into every subsequent turn.
        monkeypatch.setattr(llm_agent, "MAX_RUN_RESULT_CHARS", 500)
        provider = FakeProvider(
            calls((0, "c1", "list_recent_activities", "{}")),
            text(ANSWER),
            calls((0, "c2", "list_recent_activities", "{}")),  # never reached
        )
        dispatch = FakeDispatch(FakeTool("list_recent_activities", {"items": ["x" * 900]}))
        prose, steps = await drive(
            provider, dispatch, request=_request(max_rounds=6), monkeypatch=monkeypatch
        )

        assert prose == ANSWER
        # Two turns, not six: the budget stopped the gathering and the forced
        # final turn — no tools offered — produced the answer.
        assert provider.turn_count == 2
        assert provider.sent[1]["tools"] is None
        assert steps[-1] is None


class TestOversizedResults:
    async def test_a_long_result_is_truncated_with_the_marker_present(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "list_recent_activities", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("list_recent_activities", {"items": ["x" * 20_000]})
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        content = provider.tool_messages(1)[0]["content"]
        assert len(content) < 20_000
        assert content.startswith('{"items"')
        # The marker is the point: silently shortening a list makes the model
        # report a confident wrong answer instead of narrowing its query.
        assert "truncated" in content
        assert "NOT seeing the whole result" in content

    async def test_a_result_at_the_bound_is_left_alone(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_zone_totals", "{}")),
            text(ANSWER),
        )
        payload = {"note": "y" * (MAX_TOOL_RESULT_CHARS - 100)}
        dispatch = FakeDispatch(FakeTool("get_zone_totals", payload))
        await drive(provider, dispatch, monkeypatch=monkeypatch)
        assert "truncated" not in provider.tool_messages(1)[0]["content"]


# ── Usage accounting ────────────────────────────────────────────────────────


class TestUsageSumsAcrossTheRun:
    async def test_every_call_in_the_run_is_counted(self, monkeypatch):
        provider = FakeProvider(
            calls(
                (0, "c1", "get_training_status", "{}"),
                usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            ),
            calls(
                (0, "c2", "get_plan_status", "{}"),
                usage={"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320},
            ),
            text(
                ANSWER,
                usage={"prompt_tokens": 500, "completion_tokens": 400, "total_tokens": 900},
            ),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        usage_out: dict = {}
        await drive(provider, dispatch, monkeypatch=monkeypatch, usage_out=usage_out)

        # Recording only the last turn would report 900 for a run that cost 1330.
        assert usage_out["usage"] == {
            "prompt_tokens": 900,
            "completion_tokens": 430,
            "total_tokens": 1330,
        }

    async def test_a_turn_reporting_nothing_does_not_zero_the_total(self, monkeypatch):
        provider = FakeProvider(
            calls(
                (0, "c1", "get_training_status", "{}"),
                usage={"prompt_tokens": 100, "completion_tokens": 10},
            ),
            text(ANSWER),  # an Ollama-family turn with no usage chunk
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        usage_out: dict = {}
        await drive(provider, dispatch, monkeypatch=monkeypatch, usage_out=usage_out)
        assert usage_out["usage"]["prompt_tokens"] == 100


# ── Degrading to the blob prompt ────────────────────────────────────────────


class TestProviderCannotDoTools:
    def _rejection(self, status: int, body: str) -> httpx.HTTPStatusError:
        request = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
        response = httpx.Response(status, text=body, request=request)
        return httpx.HTTPStatusError("rejected", request=request, response=response)

    def _error_turn(self, status: int, body: str) -> Turn:
        return Turn(error=self._rejection(status, body))

    async def test_a_400_naming_the_param_asks_for_the_blob_path(self, monkeypatch):
        provider = FakeProvider(
            Turn(error=self._rejection(400, '{"error": "Unknown parameter: tools"}'))
        )
        with pytest.raises(AgenticUnavailable):
            await drive(provider, monkeypatch=monkeypatch)

    async def test_a_context_length_400_falls_back_rather_than_failing(
        self, monkeypatch
    ):
        # The failure this loop *creates*: it accumulates tool results a
        # single-shot prompt never would, and the small windows on self-hosted
        # models are exactly the population the fallback exists for. Failing
        # here would show an error card on a provider where the blob prompt
        # would have fit comfortably.
        provider = FakeProvider(
            self._error_turn(
                400, '{"error": "This model\'s maximum context length is 4096 tokens"}'
            )
        )
        with pytest.raises(AgenticUnavailable, match="upstream error"):
            await drive(provider, monkeypatch=monkeypatch)

    @pytest.mark.parametrize(
        "status,body",
        [
            (429, '{"error": "rate limit exceeded"}'),
            (500, '{"error": "internal server error"}'),
            (503, "upstream is restarting"),
        ],
    )
    async def test_any_other_upstream_status_falls_back(
        self, monkeypatch, status, body
    ):
        # No prose has been written at this point, so the blob prompt is still
        # available and is a better answer than an error card.
        provider = FakeProvider(self._error_turn(status, body))
        with pytest.raises(AgenticUnavailable, match="upstream error"):
            await drive(provider, monkeypatch=monkeypatch)

    async def test_a_network_failure_falls_back(self, monkeypatch):
        provider = FakeProvider(
            Turn(
                error=httpx.ConnectError(
                    "connection reset",
                    request=httpx.Request("POST", "http://llm.invalid/v1/chat/completions"),
                )
            )
        )
        with pytest.raises(AgenticUnavailable, match="upstream request failed"):
            await drive(provider, monkeypatch=monkeypatch)

    async def test_a_broken_function_schema_is_our_bug_and_surfaces(self, monkeypatch):
        # Tool schemas come from the registry's own pydantic models, so this is a
        # regression in one of them. Swallowing it would drop every athlete to
        # the blob path with the suite still green.
        provider = FakeProvider(
            Turn(
                error=self._rejection(
                    400, '{"error": "Invalid schema for function \'find_activity\'"}'
                )
            )
        )
        with pytest.raises(httpx.HTTPStatusError):
            await drive(provider, monkeypatch=monkeypatch)

    async def test_a_preset_flagged_unsupported_is_never_even_tried(self, monkeypatch):
        provider = FakeProvider()  # no turns scripted: calling it would assert
        with pytest.raises(AgenticUnavailable):
            await drive(
                provider, setup=_setup(tools_supported=False), monkeypatch=monkeypatch
            )
        assert provider.turn_count == 0


class TestProviderAcceptsToolsButCallsNone:
    async def test_an_answer_with_no_tool_call_falls_back_rather_than_shipping(
        self, monkeypatch
    ):
        # The agentic prompt carries no data, so this answer was written from the
        # question alone. Better a blob answer than a confident empty one.
        provider = FakeProvider(text(ANSWER))
        with pytest.raises(AgenticUnavailable):
            await drive(provider, monkeypatch=monkeypatch)

    async def test_a_turn_producing_absolutely_nothing_falls_back(self, monkeypatch):
        provider = FakeProvider(Turn(events=[]))
        with pytest.raises(AgenticUnavailable):
            await drive(provider, monkeypatch=monkeypatch)


class TestConcurrencyGuard:
    async def test_a_run_that_cannot_get_a_slot_falls_back_instead_of_waiting(
        self, monkeypatch
    ):
        # Waiting would push the run towards the 30-minute pending timeout with
        # the athlete watching a spinner. The cheaper answer, now, is better.
        monkeypatch.setattr(settings, "agent_max_concurrent_runs", 1)
        with llm_agent._run_slot():
            provider = FakeProvider()
            with pytest.raises(AgenticUnavailable):
                await drive(provider, monkeypatch=monkeypatch)
            assert provider.turn_count == 0

    async def test_the_slot_is_released_when_the_run_ends(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_max_concurrent_runs", 1)
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        for _ in range(2):
            provider = FakeProvider(
                calls((0, "c1", "get_training_status", "{}")), text(ANSWER)
            )
            prose, _ = await drive(provider, dispatch, monkeypatch=monkeypatch)
            assert prose == ANSWER
        assert llm_agent._active_runs == 0

    async def test_a_failed_run_does_not_leak_its_slot(self, monkeypatch):
        monkeypatch.setattr(settings, "agent_max_concurrent_runs", 1)
        with pytest.raises(AgenticUnavailable):
            await drive(FakeProvider(text(ANSWER)), monkeypatch=monkeypatch)
        assert llm_agent._active_runs == 0

    async def test_a_second_run_is_refused_while_the_first_is_mid_flight(
        self, monkeypatch
    ):
        # The property the guard actually promises, pinned against a run that is
        # genuinely suspended inside the loop rather than one holding the slot
        # synchronously: the second run is *refused*, not queued behind it.
        #
        # This is where an `asyncio.Semaphore` was the wrong primitive. Checking
        # `.locked()` and then acquiring was correct only because CPython's
        # uncontended `acquire()` happens not to suspend — an undocumented fast
        # path, in a class whose `locked()` semantics have already changed once.
        # A counter claimed before the first `await` needs no such guarantee.
        monkeypatch.setattr(settings, "agent_max_concurrent_runs", 1)
        released = asyncio.Event()
        entered = asyncio.Event()

        async def hold(cfg, messages, *, tools=None, tool_choice=None, usage_out=None):
            entered.set()
            await released.wait()
            yield TextDelta(ANSWER)

        async def _resolve(athlete, user_id, *, usage_out=None):
            return _setup()

        monkeypatch.setattr(llm_agent, "resolve_stream_setup", _resolve)
        monkeypatch.setattr(llm_agent, "call_tool", FakeDispatch())

        async def first() -> None:
            monkeypatch.setattr(llm_agent, "stream_completion_events", hold)
            with pytest.raises(AgenticUnavailable):
                # Turn zero prose → falls back, but only after holding the slot.
                async for _ in agentic_stream(_request(), {}):
                    pass

        holder = asyncio.create_task(first())
        await entered.wait()
        assert llm_agent._active_runs == 1

        second = FakeProvider()
        with pytest.raises(AgenticUnavailable, match="slots are in use"):
            await drive(second, monkeypatch=monkeypatch)
        # Refused outright rather than queued behind the run in flight.
        assert second.turn_count == 0

        released.set()
        await holder
        assert llm_agent._active_runs == 0


class TestLateToolCallsAndLateFallback:
    async def test_a_tool_call_arriving_after_the_answer_started_is_ignored(
        self, monkeypatch
    ):
        # The answer has already been streamed into the column and cannot be
        # unsaid. A complete answer that ignored a late tool call beats a
        # truncated one, so the run ends here.
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            Turn(
                events=[
                    TextDelta(ANSWER),
                    ToolCallDelta(index=0, id="c2", name="get_plan_status"),
                    ToolCallDelta(index=0, arguments="{}"),
                ]
            ),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        prose, steps = await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert prose == ANSWER
        assert steps[-1] is None
        # `get_plan_status` was announced but never run: the run was over.
        assert [name for name, _ in dispatch.invocations] == ["get_training_status"]

    async def test_a_fallback_signal_after_prose_fails_loudly(self, monkeypatch):
        # The provider streams a full answer and *then* rejects `tools`
        # mid-stream. Normally that means "use the blob prompt", but the prose is
        # already committed: falling back would staple a second answer onto the
        # first. The run fails instead, and `stream_into_db` marks it.
        request = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
        rejection = httpx.HTTPStatusError(
            "rejected",
            request=request,
            response=httpx.Response(400, text="Unknown parameter: tools", request=request),
        )
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            Turn(events=[TextDelta(ANSWER)], error=rejection),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))

        with pytest.raises(RuntimeError, match="already produced output"):
            await drive(provider, dispatch, monkeypatch=monkeypatch)

    async def test_the_same_signal_before_any_prose_is_an_ordinary_fallback(
        self, monkeypatch
    ):
        request = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
        rejection = httpx.HTTPStatusError(
            "rejected",
            request=request,
            response=httpx.Response(400, text="Unknown parameter: tools", request=request),
        )
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            Turn(events=[], error=rejection),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        with pytest.raises(AgenticUnavailable):
            await drive(provider, dispatch, monkeypatch=monkeypatch)


# ── What the provider actually sees ─────────────────────────────────────────


class TestConversationShape:
    async def test_the_house_style_is_a_system_message_on_every_turn(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            calls((0, "c2", "get_plan_status", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        assert provider.turn_count == 3
        for turn in provider.sent:
            systems = [m["content"] for m in turn["messages"] if m["role"] == "system"]
            # The hoster's rules are not something three tool results are allowed
            # to push out of the model's attention.
            assert HOUSE_STYLE in systems

    async def test_no_house_style_configured_sends_one_system_message(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")), text(ANSWER)
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        await drive(
            provider, dispatch, setup=_setup(house_style=None), monkeypatch=monkeypatch
        )
        systems = [m for m in provider.sent[0]["messages"] if m["role"] == "system"]
        assert len(systems) == 1

    async def test_the_format_rule_is_restated_on_every_post_tool_turn(
        self, monkeypatch
    ):
        # The reason for restating it — models obey a leading-format rule less
        # reliably after tool results — applies to whichever turn answers, and
        # the common shape is answering after one or two rounds rather than
        # hitting the cap. Applying it only to the forced turn would put the
        # mitigation on the rare path and skip the usual one.
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            calls((0, "c2", "get_plan_status", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(
            FakeTool("get_training_status", {"form": 1}),
            FakeTool("get_plan_status", {"plans": []}),
        )
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        # Turn zero has no tool results behind it yet, so nothing to restate.
        assert MOOD_RULE not in str(provider.sent[0]["messages"])
        for turn in provider.sent[1:]:
            last = turn["messages"][-1]
            # A *user* turn, not a system one: several llama.cpp / Ollama chat
            # templates render only the leading system message and silently drop
            # later ones, which would make this reminder a no-op precisely on the
            # models most likely to need it — and leave nothing in the logs.
            assert last["role"] == "user"
            assert last["content"] == MOOD_RULE
        # And it does not carry the forced turn's "stop calling tools", which
        # would end the loop after round one.
        assert "do not call any more tools" not in str(provider.sent[1]["messages"])
        assert provider.sent[1]["tools"]

    async def test_the_forced_final_turn_adds_the_stop_instruction(
        self, monkeypatch
    ):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")),
            text(ANSWER),
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        await drive(
            provider, dispatch, request=_request(max_rounds=1), monkeypatch=monkeypatch
        )

        final = provider.sent[1]["messages"]
        assert MOOD_RULE in final[-1]["content"]
        assert final[-1]["role"] == "user"  # see the note on the reminder above
        assert "do not call any more tools" in final[-1]["content"]

    async def test_the_tool_array_carries_the_registry_schemas(self, monkeypatch):
        provider = FakeProvider(
            calls((0, "c1", "get_training_status", "{}")), text(ANSWER)
        )
        dispatch = FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        await drive(provider, dispatch, monkeypatch=monkeypatch)

        sent_tools = provider.sent[0]["tools"]
        assert {t["function"]["name"] for t in sent_tools} == {
            t["function"]["name"] for t in tool_definitions(llm_agent.all_tools())
        }
        assert all(t["type"] == "function" for t in sent_tools)


# ── The seam between the two paths ──────────────────────────────────────────


class TestCoachingStream:
    async def _blob(self, chunks):
        async def factory(usage_out):
            usage_out["cfg"] = ResolvedLlm(
                base_url="http://llm.invalid/v1", model="m", api_key=None, source="instance"
            )
            usage_out["usage"] = {"prompt_tokens": 40, "completion_tokens": 60}
            for chunk in chunks:
                yield chunk

        return factory

    async def test_no_request_goes_straight_to_the_blob_prompt(self):
        usage_out: dict = {}
        blob = await self._blob(["MOOD:knowing\n\n", "Solid week."])
        out = [
            item
            async for item in coaching_stream(request=None, blob=blob, usage_out=usage_out)
        ]
        assert "".join(i for i in out if isinstance(i, str)) == "MOOD:knowing\n\nSolid week."
        assert usage_out["usage"]["prompt_tokens"] == 40

    async def test_a_fallback_clears_progress_and_keeps_both_paths_tokens(
        self, monkeypatch
    ):
        # Two tool rounds were paid for before the provider gave up. Dropping
        # them would under-report exactly the runs that cost the most.
        resolved = _setup()

        async def _resolve(athlete, user_id, *, usage_out=None):
            if usage_out is not None:
                usage_out["cfg"] = resolved.cfg
            return resolved

        request_obj = httpx.Request("POST", "http://llm.invalid/v1/chat/completions")
        provider = FakeProvider(
            calls(
                (0, "c1", "get_training_status", "{}"),
                usage={"prompt_tokens": 200, "completion_tokens": 15},
            ),
            Turn(
                error=httpx.HTTPStatusError(
                    "rejected",
                    request=request_obj,
                    response=httpx.Response(
                        400, text="tool_calls is not supported", request=request_obj
                    ),
                )
            ),
        )
        monkeypatch.setattr(llm_agent, "stream_completion_events", provider)
        monkeypatch.setattr(llm_agent, "resolve_stream_setup", _resolve)
        monkeypatch.setattr(
            llm_agent, "call_tool", FakeDispatch(FakeTool("get_training_status", {"form": 1}))
        )

        usage_out: dict = {}
        blob = await self._blob(["MOOD:stern\n\n", "Blob answer."])
        out = [
            item
            async for item in coaching_stream(
                request=_request(), blob=blob, usage_out=usage_out
            )
        ]

        prose = "".join(i for i in out if isinstance(i, str))
        steps = [i.code for i in out if isinstance(i, AgentProgress)]
        assert prose == "MOOD:stern\n\nBlob answer."
        # The tool-round step is wiped before prose from the other path lands.
        assert steps[-1] is None
        assert usage_out["usage"] == {
            "prompt_tokens": 240,
            "completion_tokens": 75,
            "total_tokens": 315,
        }


# ── Small surfaces ──────────────────────────────────────────────────────────


class TestOptIn:
    def test_off_by_default(self):
        assert agentic_enabled(_Athlete()) is False

    def test_on_when_the_athlete_asked_for_it(self):
        assert agentic_enabled(_Athlete({"agentic_koutsi": True})) is True

    def test_a_missing_athlete_is_not_a_crash(self):
        assert agentic_enabled(None) is False


class TestProgressVocabulary:
    def test_every_registered_tool_has_a_code(self):
        vocabulary = progress_vocabulary()
        assert PROGRESS_THINKING in vocabulary
        for tool in llm_agent.all_tools():
            assert progress_code_for_tool(tool.name) in vocabulary

    def test_codes_are_closed_at_the_top_level(self):
        # A client switches on `thinking` or the `tool.` prefix and falls back to
        # generic copy for a suffix it does not know. A third shape would break
        # that contract silently.
        for code in progress_vocabulary():
            assert code == PROGRESS_THINKING or code.startswith("tool.")
