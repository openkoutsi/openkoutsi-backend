"""The FIT parse must not run on the event loop (issue #101 §2.1/2.2, #102 F-05).

Parsing is pure-Python iteration over the whole file. The finding measured it:

       records    file size   parse time
         3,600       0.09 MB       0.21 s
        36,000       0.86 MB       2.12 s
       200,000       4.77 MB      11.22 s

    baseline event-loop heartbeat gap :  5.3 ms
    heartbeat gap during the parse    : 11.59 s   ← loop frozen

4.77 MB is under a tenth of the 50 MB upload limit, and an authenticated user
may upload thirty files an hour. Starlette runs ``BackgroundTasks`` on the event
loop after the response is sent — not in a threadpool — so the background half
stalls the process just as thoroughly as the inline half.

These tests reproduce that measurement rather than the file size: a heartbeat
coroutine ticks while the code under test runs, and the parse is replaced by a
function that holds the CPU for a known interval. A parse left on the loop shows
up as a heartbeat gap the length of the block; one moved to a worker thread does
not. Using a fake parse instead of a 200k-record fixture keeps the test both
deterministic and fast — what is being tested is which thread the call lands on,
not how quickly fitdecode reads a file.
"""
import asyncio
import time
from unittest.mock import patch

import pytest

# How long the stand-in parse holds the CPU. Long enough to dwarf scheduler
# noise, short enough that four of these stay under a second in total.
_BLOCK = 0.4

# A parse on the loop produces a gap of ~_BLOCK. A parse in a thread produces
# one of a few milliseconds. Half the block sits far from both, so the test
# neither flakes on a loaded machine nor passes with the fix reverted.
_MAX_GAP = _BLOCK / 2


class _Done(Exception):
    """Raised by the stand-in parse: the work after it is not under test."""


def _blocking_parse(*args, **kwargs):
    """Hold the CPU exactly as a real parse does, then stop the caller."""
    time.sleep(_BLOCK)
    raise _Done()


class _Heartbeat:
    """Ticks on the event loop and records the largest gap between ticks.

    The gap is the thing a blocked loop shows up in: nothing else can run
    while a synchronous parse holds it, so the tick that should have happened
    5 ms ago happens when the parse finishes instead.
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.gaps: list[float] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(self.interval)
            now = time.perf_counter()
            self.gaps.append(now - last)
            last = now

    async def __aenter__(self) -> "_Heartbeat":
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(self.interval * 5)  # let it settle
        self.gaps.clear()
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._task is not None
        # Let an overdue tick land before stopping. Code that blocks and then
        # returns without awaiting anything leaves the heartbeat's `sleep`
        # pending: cancelling here would discard the very gap the block caused,
        # and every test in this module would pass against a blocked loop.
        await asyncio.sleep(self.interval * 2)
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    @property
    def worst(self) -> float:
        return max(self.gaps, default=0.0)


async def test_the_harness_detects_a_blocked_loop():
    """The control. Without this, every test below could pass for free.

    If a synchronous sleep on the loop did *not* register as a heartbeat gap,
    the assertions in this module would be measuring nothing.
    """
    async with _Heartbeat() as hb:
        time.sleep(_BLOCK)
        await asyncio.sleep(0)

    assert hb.worst >= _BLOCK, (
        f"harness failed to see a {_BLOCK}s block (worst gap {hb.worst:.3f}s)"
    )


class TestUploadProcessing:
    """§2.2 — `process_activity_file`, awaited from a BackgroundTasks callback."""

    async def test_parse_runs_off_the_loop(self):
        from backend.app.services import fit_processor

        with patch.object(fit_processor, "parse_activity_file", _blocking_parse):
            async with _Heartbeat() as hb:
                with pytest.raises(_Done):
                    # athlete/activity/session are untouched before the parse,
                    # which is the only part of this function under test.
                    await fit_processor.process_activity_file(
                        "irrelevant.fit", None, None, None
                    )

        assert hb.worst < _MAX_GAP, f"loop stalled for {hb.worst:.3f}s"

    async def test_a_prepared_parse_is_not_redone(self):
        """The bulk-import path parses in its own thread and passes the result.

        It must keep skipping the parse entirely — moving the call to a thread
        must not turn one parse into two.
        """
        from backend.app.services import fit_processor

        with patch.object(fit_processor, "parse_activity_file", _blocking_parse):
            with pytest.raises(AttributeError):
                # Reaches past the parse and fails on the None athlete, which is
                # proof enough that _blocking_parse was never called.
                await fit_processor.process_activity_file(
                    "irrelevant.fit", None, None, None, parsed=(object(), [])
                )


class TestUploadRequestHandler:
    """§2.1 — `read_fit_start_time`, inline in the request handler."""

    async def test_start_time_read_runs_off_the_loop(self, client, auth_headers, tmp_path):
        from backend.app.api import activities

        fit = tmp_path / "ride.fit"
        fit.write_bytes(b"\x0e\x10\x00\x00" + b"\x00" * 4 + b".FIT" + b"\x00" * 64)

        def _slow_start_time(_path):
            time.sleep(_BLOCK)
            return None

        with patch.object(activities, "read_fit_start_time", _slow_start_time):
            async with _Heartbeat() as hb:
                with open(fit, "rb") as f:
                    await client.post(
                        "/api/activities/upload",
                        files={"file": ("ride.fit", f, "application/octet-stream")},
                        headers=auth_headers,
                    )

        assert hb.worst < _MAX_GAP, f"loop stalled for {hb.worst:.3f}s"


class TestProviderSync:
    """The same pattern on the provider paths the finding also names."""

    async def test_interval_rebuild_runs_off_the_loop(self):
        from backend.app.services import provider_sync

        class _SlowParser:
            @staticmethod
            def extractIntervals(_fileish):
                time.sleep(_BLOCK)
                raise _Done()

        with patch.object(provider_sync, "parser_for", lambda _fmt: _SlowParser):
            async with _Heartbeat() as hb:
                with pytest.raises(_Done):
                    await provider_sync.rebuild_intervals(
                        activity=None, session=None, fileish="ride.fit", stream_map={}
                    )

        assert hb.worst < _MAX_GAP, f"loop stalled for {hb.worst:.3f}s"

    async def test_wahoo_stream_parse_runs_off_the_loop(self):
        from backend.app.services.providers import wahoo

        provider = wahoo.WahooClient()

        async def _fake_download(*_args, **_kwargs):
            return b"fit-bytes"

        with (
            patch.object(wahoo, "_parse_fit_streams", _blocking_parse),
            patch.object(provider, "download_fit_file", _fake_download),
        ):
            async with _Heartbeat() as hb:
                with pytest.raises(_Done):
                    await provider.get_activity_streams("token", "workout-1")

        assert hb.worst < _MAX_GAP, f"loop stalled for {hb.worst:.3f}s"
