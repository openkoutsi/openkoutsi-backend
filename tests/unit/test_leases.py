"""The lease primitive itself (issue #50).

``test_sync_concurrency.py`` proves the activity-creation path is guarded;
these prove the thing doing the guarding behaves, including the cases that path
does not reach — a lease taken twice, a holder that never released, a release
attempted by someone who no longer holds it.

File-based SQLite throughout, for the reason given in ``test_sync_concurrency``:
an in-memory URL would quietly serialise every session onto one connection.
"""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models.user_orm  # noqa: F401 — populate UserBase.metadata
from backend.app.db import leases
from backend.app.db.base import UserBase, _set_wal_mode
from backend.app.models.user_orm import SyncLease

_TTL = timedelta(seconds=60)


@pytest.fixture
async def factory(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'leases.db'}",
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(engine.sync_engine, "connect", _set_wal_mode)
    async with engine.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class TestAcquire:
    async def test_first_acquire_creates_the_row_and_wins(self, factory):
        async with factory() as session:
            token = await leases.acquire(
                session, SyncLease, "job", ttl=_TTL, wait=1.0
            )
            assert token is not None

            row = (
                await session.execute(select(SyncLease).where(SyncLease.name == "job"))
            ).scalar_one()
            assert row.holder == token
            assert row.expires_at is not None

    async def test_a_second_caller_cannot_take_a_held_lease(self, factory):
        async with factory() as first, factory() as second:
            held = await leases.acquire(first, SyncLease, "job", ttl=_TTL, wait=1.0)
            assert held is not None

            # A short wait, because the point is that it never succeeds.
            refused = await leases.acquire(
                second, SyncLease, "job", ttl=_TTL, wait=0.1
            )
            assert refused is None

    async def test_release_lets_the_next_caller_in(self, factory):
        async with factory() as first, factory() as second:
            token = await leases.acquire(first, SyncLease, "job", ttl=_TTL, wait=1.0)
            await leases.release(first, SyncLease, "job", token)

            assert await leases.acquire(
                second, SyncLease, "job", ttl=_TTL, wait=1.0
            ) is not None

    async def test_an_expired_lease_is_taken_over(self, factory):
        """A holder that died cannot release; the deadline is what frees it."""
        async with factory() as first, factory() as second:
            await leases.acquire(first, SyncLease, "job", ttl=_TTL, wait=1.0)

            # Stand in for "that process is gone" by ageing the deadline.
            await first.execute(
                update(SyncLease)
                .where(SyncLease.name == "job")
                .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
                .execution_options(synchronize_session=False)
            )
            await first.commit()

            assert await leases.acquire(
                second, SyncLease, "job", ttl=_TTL, wait=1.0
            ) is not None

    async def test_a_waiter_gets_in_once_the_holder_releases(self, factory):
        async with factory() as first, factory() as second:
            token = await leases.acquire(first, SyncLease, "job", ttl=_TTL, wait=1.0)

            async def release_shortly():
                await asyncio.sleep(0.05)
                await leases.release(first, SyncLease, "job", token)

            waiter, _ = await asyncio.gather(
                leases.acquire(second, SyncLease, "job", ttl=_TTL, wait=5.0),
                release_shortly(),
            )
            assert waiter is not None

    async def test_only_one_of_many_racing_callers_wins(self, factory):
        """Including the first-ever acquire, where the row does not exist yet."""
        sessions = [factory() for _ in range(6)]
        opened = [await s.__aenter__() for s in sessions]
        try:
            tokens = await asyncio.gather(
                *[
                    leases.acquire(s, SyncLease, "job", ttl=_TTL, wait=0.1)
                    for s in opened
                ]
            )
        finally:
            for s in sessions:
                await s.__aexit__(None, None, None)

        assert len([t for t in tokens if t is not None]) == 1

    async def test_leases_with_different_names_do_not_block_each_other(self, factory):
        async with factory() as first, factory() as second:
            assert await leases.acquire(
                first, SyncLease, "athlete-a", ttl=_TTL, wait=1.0
            ) is not None
            assert await leases.acquire(
                second, SyncLease, "athlete-b", ttl=_TTL, wait=1.0
            ) is not None


class TestRelease:
    async def test_releasing_a_lease_you_lost_is_a_no_op(self, factory):
        """A holder whose lease expired must not free its successor's."""
        async with factory() as first, factory() as second:
            stale_token = await leases.acquire(
                first, SyncLease, "job", ttl=_TTL, wait=1.0
            )
            await first.execute(
                update(SyncLease)
                .where(SyncLease.name == "job")
                .values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
                .execution_options(synchronize_session=False)
            )
            await first.commit()
            successor = await leases.acquire(
                second, SyncLease, "job", ttl=_TTL, wait=1.0
            )

            # The late finisher tidies up, and must not take the lease with it.
            await leases.release(first, SyncLease, "job", stale_token)

            async with factory() as reader:
                row = (
                    await reader.execute(
                        select(SyncLease).where(SyncLease.name == "job")
                    )
                ).scalar_one()
            assert row.holder == successor


class TestHold:
    async def test_hold_releases_on_the_way_out(self, factory):
        async with factory() as session:
            async with leases.hold(
                session, SyncLease, "job", ttl=_TTL, wait=1.0
            ) as token:
                assert token is not None

            row = (
                await session.execute(select(SyncLease).where(SyncLease.name == "job"))
            ).scalar_one()
            assert row.holder is None

    async def test_hold_releases_when_the_body_raises(self, factory):
        async with factory() as session:
            with pytest.raises(RuntimeError):
                async with leases.hold(session, SyncLease, "job", ttl=_TTL, wait=1.0):
                    raise RuntimeError("boom")

            row = (
                await session.execute(select(SyncLease).where(SyncLease.name == "job"))
            ).scalar_one()
            assert row.holder is None

    async def test_hold_yields_none_rather_than_blocking_forever(self, factory):
        """Losing the lease degrades the guard; it must not refuse the work."""
        async with factory() as blocker, factory() as session:
            await leases.acquire(blocker, SyncLease, "job", ttl=_TTL, wait=1.0)

            ran = False
            async with leases.hold(
                session, SyncLease, "job", ttl=_TTL, wait=0.1
            ) as token:
                ran = True
                assert token is None
            assert ran
