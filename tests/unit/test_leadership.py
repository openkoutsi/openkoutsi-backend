"""One process at a time runs the background work (issue #50).

The three `lifespan` pollers were started by whichever process booted, with
nothing arbitrating between them. Two processes therefore drained the same
bridge queues, and each drain is a real import and a real LLM bill — the
concrete reason `DEPLOY.md` says to run exactly one process.

These run against a **file-based** registry DB. An in-memory URL gives
SQLAlchemy a `StaticPool`, so every session would share one connection and the
claim would appear to work whether or not the conditional UPDATE is doing
anything. `test_sync_concurrency.py` takes the same approach for the same
reason.
"""
import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models.registry_orm  # noqa: F401 — populate the metadata
from backend.app.db.base import RegistryBase, _set_wal_mode
from backend.app.models.registry_orm import RegistryLease
from backend.app.services import leadership


@pytest.fixture
async def registry(tmp_path, monkeypatch):
    """A real registry file, wired in as the module's session source."""
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}",
        echo=False, pool_size=3, max_overflow=2, connect_args={"timeout": 30},
    )
    event.listen(engine.sync_engine, "connect", _set_wal_mode)
    async with engine.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _session():
        async with factory() as session:
            yield session

    monkeypatch.setattr(leadership, "registry_session", _session)
    yield factory
    await engine.dispose()


@pytest.fixture(autouse=True)
def _fast_timings(monkeypatch):
    """Real cadences are 15–120 s; the properties do not depend on the units."""
    monkeypatch.setattr(leadership, "STANDBY_POLL_S", 0.02)
    monkeypatch.setattr(leadership, "STANDBY_JITTER_S", 0.0)
    monkeypatch.setattr(leadership, "RENEW_EVERY_S", 0.02)
    monkeypatch.setattr(leadership, "LEASE_TTL", timedelta(seconds=30))


async def _lease_row(factory):
    async with factory() as session:
        return (
            await session.execute(
                select(RegistryLease).where(
                    RegistryLease.name == leadership.BACKGROUND_WORK
                )
            )
        ).scalar_one_or_none()


@asynccontextmanager
async def registry_forced_steal(factory):
    """Hand the lease to somebody else, as an expiry-and-steal would."""
    async with factory() as session:
        await session.execute(
            update(RegistryLease)
            .where(RegistryLease.name == leadership.BACKGROUND_WORK)
            .values(
                holder="another-process",
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=30),
            )
        )
        await session.commit()
    yield


class TestOnlyOneHolderAtATime:
    async def test_a_second_process_does_not_get_in(self, registry):
        """The whole point: two processes, one claim."""
        inside: list[str] = []

        async def _process(tag: str, hold_for: float):
            async with leadership.hold_background_work():
                inside.append(tag)
                await asyncio.sleep(hold_for)
                inside.append(f"{tag}-done")

        first = asyncio.create_task(_process("a", 0.3))
        await asyncio.sleep(0.05)  # let A win
        second = asyncio.create_task(_process("b", 0.01))

        await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

        # B may only enter after A has left — never between A's two marks.
        assert inside == ["a", "a-done", "b", "b-done"], inside

    async def test_the_loser_takes_over_when_the_holder_leaves(self, registry):
        """A redeploy hands over in milliseconds, not after a full TTL.

        `hold_background_work` releases on the way out rather than letting the
        deadline do it — without that, every restart leaves the background work
        dead for the length of the TTL.
        """
        async with leadership.hold_background_work():
            row = await _lease_row(registry)
            assert row is not None and row.holder is not None

        row = await _lease_row(registry)
        assert row.holder is None, "the lease was not released on the way out"

        # And it is immediately takeable.
        async with leadership.hold_background_work():
            pass


class TestRenewal:
    async def test_a_held_lease_outlives_its_ttl(self, registry, monkeypatch):
        """A cycle can be longer than the deadline, because renewal moves it."""
        monkeypatch.setattr(leadership, "LEASE_TTL", timedelta(seconds=0.15))
        monkeypatch.setattr(leadership, "RENEW_EVERY_S", 0.02)

        async with leadership.hold_background_work() as lost:
            await asyncio.sleep(0.4)  # well past the TTL
            assert not lost.is_set(), "renewal did not keep the claim alive"
            row = await _lease_row(registry)
            assert row.expires_at is not None

    async def test_losing_the_lease_sets_the_event(self, registry, monkeypatch):
        """Stolen out from under the holder — it must find out, not assume."""
        monkeypatch.setattr(leadership, "LEASE_TTL", timedelta(seconds=30))

        async with leadership.hold_background_work() as lost:
            # A second process takes it: only possible if we stop renewing, so
            # forge it directly, which is what an expiry-and-steal looks like.
            async with registry_forced_steal(registry):
                await asyncio.wait_for(lost.wait(), timeout=5)

            assert lost.is_set()


class TestRunUntilLost:
    async def test_the_work_repeats_on_its_interval(self, registry):
        calls = 0

        async def _work():
            nonlocal calls
            calls += 1

        lost = asyncio.Event()
        task = asyncio.create_task(
            leadership.run_until_lost(lost, _work, 0.02, label="test")
        )
        await asyncio.sleep(0.15)
        lost.set()
        await asyncio.wait_for(task, timeout=5)

        assert calls > 2, f"the work ran {calls} time(s)"

    async def test_a_failing_cycle_does_not_end_the_loop(self, registry):
        """One bad poll must not stop every later one."""
        calls = 0

        async def _work():
            nonlocal calls
            calls += 1
            raise RuntimeError("bridge unreachable")

        lost = asyncio.Event()
        task = asyncio.create_task(
            leadership.run_until_lost(lost, _work, 0.02, label="test")
        )
        await asyncio.sleep(0.15)
        lost.set()
        await asyncio.wait_for(task, timeout=5)

        assert calls > 2

    async def test_losing_the_claim_cancels_the_cycle_in_flight(self, registry):
        """Finishing the cycle first would be the duplicate this prevents.

        A lapsed deadline means another process may already be draining the same
        queue, so the right move is to stop where we stand.
        """
        started = asyncio.Event()
        finished = False

        async def _slow_work():
            nonlocal finished
            started.set()
            await asyncio.sleep(5)
            finished = True

        lost = asyncio.Event()
        task = asyncio.create_task(
            leadership.run_until_lost(lost, _slow_work, 0.02, label="test")
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        lost.set()
        await asyncio.wait_for(task, timeout=5)

        assert not finished, "the cycle ran to completion after the claim was lost"


class TestTheSupervisorSurvivesARegistryError:
    """One registry error must not end all background work for the process.

    Before the guard, the path from `hold_background_work()` through
    `registry_session()` to `leases.acquire` was unprotected — the per-poller
    `try/except` had moved into `_guarded`, which wraps `work()` only. A pool
    `TimeoutError` from the 5-connection registry therefore killed the
    supervisor task outright: both bridge pollers and the expiry sweep stopped
    until the container restarted, with nothing in the log, because nobody
    awaits that task until shutdown.

    It is a failure this design makes *more* likely rather than less: the
    standby and renewal writes are new registry load that did not exist before.
    """

    async def test_a_failing_acquire_is_retried_rather_than_fatal(
        self, registry, monkeypatch
    ):
        from backend.main import _background_work

        monkeypatch.setattr(leadership, "STANDBY_POLL_S", 0.01)

        calls = {"n": 0}
        real_acquire = leadership.leases.acquire

        async def _fail_twice_then_work(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise TimeoutError("QueuePool limit reached")
            return await real_acquire(*args, **kwargs)

        monkeypatch.setattr(leadership.leases, "acquire", _fail_twice_then_work)
        # No bridges configured, so the only job is the 24 h sweep — it runs
        # once per claim and then waits, which keeps this test quick.
        monkeypatch.setattr(
            "backend.app.services.pat_expiry.pat_expiry_sweep_once",
            AsyncMock(),
        )

        task = asyncio.create_task(_background_work())
        try:
            # Long enough for two failures, two backoffs, and a success.
            await asyncio.sleep(0.3)
            assert not task.done(), (
                "a registry error killed the supervisor: "
                f"{task.exception() if task.done() else ''}"
            )
            assert calls["n"] > 2, "the supervisor gave up instead of retrying"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def test_shutdown_does_not_re_raise_a_supervisor_that_already_died(
        self, registry, monkeypatch
    ):
        """`cancel()` on a failed task is a no-op, so the await re-raises.

        Without the handler that surfaced hours-old registry errors out of
        `lifespan`'s shutdown path as ASGI lifespan errors.
        """
        from backend.main import lifespan

        async def _dies_immediately():
            raise RuntimeError("registry is gone")

        with (
            patch("backend.main.init_registry_db", new=AsyncMock()),
            patch("backend.main.init_usage_db", new=AsyncMock()),
            patch("backend.main._background_work", new=_dies_immediately),
            patch(
                "backend.app.services.stranded_runs.settle_stranded_runs",
                new=AsyncMock(return_value=0),
            ),
        ):
            # Must not raise on the way out.
            async with lifespan(object()):
                await asyncio.sleep(0.01)
