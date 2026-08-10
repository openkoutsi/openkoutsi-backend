"""The drain loop's handling of agentic progress markers (issue #43).

``stream_into_db`` is what stands between a many-turn agent run and the two
columns the frontend polls. Three properties matter enough to pin:

* a stream that yields only strings behaves exactly as it did before this
  existed — the goal-guidance surface and the whole non-agentic path depend on
  that;
* a progress marker commits **immediately** rather than waiting for the 500 ms
  text cadence, because a step nobody's poll can see is a step that did not
  happen; and
* usage is still recorded whatever the outcome, including for a run that failed
  after spending tokens.
"""
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.services.llm_client import ResolvedLlm
from backend.app.services.llm_streaming import AgentProgress, stream_into_db


class Recorder:
    """Stands in for the analyzer's four persistence callbacks."""

    def __init__(self):
        self.text: str | None = None
        self.steps: list[str | None] = []
        self.done: str | None = None
        self.failed = False
        #: What had been written at each commit — the state a poll would see.
        self.commits: list[tuple[str | None, str | None]] = []

    def on_progress(self, text: str) -> None:
        self.text = text

    def on_step(self, code: str | None) -> None:
        self.steps.append(code)

    def on_done(self, text: str) -> None:
        self.done = text

    def on_error(self) -> None:
        self.failed = True


async def run(items, recorder: Recorder, *, flush_interval: float = 0.0):
    session = AsyncMock()

    async def _commit():
        recorder.commits.append((recorder.text, recorder.steps[-1] if recorder.steps else None))

    session.commit = AsyncMock(side_effect=_commit)

    async def _stream(usage_out):
        # A resolved config is what makes the run recordable at all; without one
        # `stream_into_db` has nothing to attribute the tokens to.
        usage_out["cfg"] = ResolvedLlm(
            base_url="http://llm.invalid/v1",
            model="test-model",
            api_key=None,
            source="instance",
        )
        for item in items:
            if isinstance(item, Exception):
                raise item
            yield item

    with (
        patch("backend.app.services.llm_streaming._FLUSH_INTERVAL_S", flush_interval),
        patch(
            "backend.app.services.llm_streaming.record_llm_usage", new=AsyncMock()
        ) as record,
    ):
        await stream_into_db(
            session,
            _stream,
            on_progress=recorder.on_progress,
            on_done=recorder.on_done,
            on_error=recorder.on_error,
            on_step=recorder.on_step,
            user_id="u1",
            feature="training_status",
            label="test run",
        )
    return record


class TestBackwardsCompatibility:
    async def test_a_plain_text_stream_is_unchanged_and_needs_no_on_step(self):
        session = AsyncMock()

        async def _stream(usage_out):
            for chunk in ["MOOD:knowing\n\n", "Solid week."]:
                yield chunk

        recorder = Recorder()
        with patch(
            "backend.app.services.llm_streaming.record_llm_usage", new=AsyncMock()
        ):
            await stream_into_db(
                session,
                _stream,
                on_progress=recorder.on_progress,
                on_done=recorder.on_done,
                on_error=recorder.on_error,
                # No on_step at all: the three existing callers pass none.
                user_id="u1",
                feature="goal_guidance",
                label="test run",
            )
        assert recorder.done == "MOOD:knowing\n\nSolid week."
        assert recorder.steps == []


class TestProgressMarkers:
    async def test_a_step_is_committed_as_it_arrives(self):
        recorder = Recorder()
        await run(
            [AgentProgress("thinking"), AgentProgress("tool.get_plan_status")], recorder
        )
        # Two step commits plus the final on_done commit. A step batched behind
        # the text cadence would be invisible to the poll it exists for.
        assert recorder.steps == ["thinking", "tool.get_plan_status"]
        assert [step for _text, step in recorder.commits] == [
            "thinking",
            "tool.get_plan_status",
            "tool.get_plan_status",
        ]

    async def test_steps_and_text_interleave_in_order(self):
        recorder = Recorder()
        await run(
            [
                AgentProgress("thinking"),
                AgentProgress("tool.get_training_status"),
                AgentProgress(None),
                "MOOD:cheer\n\n",
                "Strong block.",
            ],
            recorder,
        )
        assert recorder.steps == ["thinking", "tool.get_training_status", None]
        assert recorder.done == "MOOD:cheer\n\nStrong block."

    async def test_a_progress_only_stream_still_settles(self):
        # A run that emitted steps and then died before any prose must not leave
        # the row pending — the drain loop settles it either way.
        recorder = Recorder()
        await run([AgentProgress("thinking"), RuntimeError("provider died")], recorder)
        assert recorder.failed is True
        assert recorder.done is None

    async def test_a_marker_does_not_pollute_the_prose(self):
        recorder = Recorder()
        await run([AgentProgress("tool.find_activity"), "Just this."], recorder)
        assert recorder.done == "Just this."


class TestUsageIsRecordedWhateverHappens:
    async def test_a_failed_run_still_records_what_it_spent(self):
        recorder = Recorder()
        record = await run(
            [AgentProgress("thinking"), RuntimeError("provider died")], recorder
        )
        # The tokens were spent whether or not an answer came out of them.
        assert record.await_count == 1

    @pytest.mark.parametrize("items", [[], ["text"], [AgentProgress("thinking")]])
    async def test_every_shape_of_stream_records_once(self, items):
        record = await run(items, Recorder())
        assert record.await_count == 1
