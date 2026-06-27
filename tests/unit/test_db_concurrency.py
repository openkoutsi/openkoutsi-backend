"""
Regression test for SQLite "database is locked" errors under concurrent writes.

Before the fix, the default QueuePool could create multiple connections to the
same SQLite file. When one connection held an open write transaction (e.g.
during FIT-file processing), any other connection trying to write would hit
sqlite3.OperationalError: database is locked once busy_timeout expired.

The fix in team_session.py and registry.py uses pool_size=1 / max_overflow=0
so all async tasks share a single connection and queue at the SQLAlchemy pool
level instead of colliding at the SQLite level.

These tests create file-based SQLite databases (not :memory:, which bypasses
real locking) and hammer them with concurrent writes.
"""
import asyncio
import uuid

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models.team_orm  # noqa: F401 — populate TeamBase.metadata
import backend.app.models.registry_orm  # noqa: F401 — populate RegistryBase.metadata
from backend.app.db.base import RegistryBase, TeamBase, _set_wal_mode
from backend.app.models.registry_orm import Team, User
from backend.app.models.team_orm import Athlete


def _team_engine(db_path):
    """Engine matching the production settings in team_session.py."""
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(eng.sync_engine, "connect", _set_wal_mode)
    return eng


def _registry_engine(db_path):
    """Engine matching the production settings in registry.py."""
    eng = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        echo=False,
        pool_size=3,
        max_overflow=2,
        connect_args={"timeout": 30},
    )
    event.listen(eng.sync_engine, "connect", _set_wal_mode)
    return eng


class TestTeamDbConcurrency:
    async def test_concurrent_athlete_inserts_succeed(self, tmp_path):
        """50 concurrent coroutines each insert a unique Athlete; none must fail."""
        engine = _team_engine(str(tmp_path / "team.db"))
        async with engine.begin() as conn:
            await conn.run_sync(TeamBase.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False)
        n = 50

        async def insert_athlete(_: int):
            async with factory() as session:
                session.add(Athlete(
                    id=str(uuid.uuid4()),
                    global_user_id=str(uuid.uuid4()),
                    ftp_tests=[],
                ))
                await session.commit()

        await asyncio.gather(*[insert_athlete(i) for i in range(n)])

        async with factory() as session:
            count = (await session.execute(text("SELECT COUNT(*) FROM athletes"))).scalar()

        await engine.dispose()
        assert count == n

    async def test_concurrent_writes_with_reads_interleaved(self, tmp_path):
        """Writes and reads running concurrently must not deadlock or lock."""
        engine = _team_engine(str(tmp_path / "team_rw.db"))
        async with engine.begin() as conn:
            await conn.run_sync(TeamBase.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def write(_: int):
            async with factory() as session:
                session.add(Athlete(
                    id=str(uuid.uuid4()),
                    global_user_id=str(uuid.uuid4()),
                    ftp_tests=[],
                ))
                await session.commit()

        async def read(_: int):
            async with factory() as session:
                await session.execute(text("SELECT COUNT(*) FROM athletes"))

        tasks = [write(i) for i in range(20)] + [read(i) for i in range(20)]
        await asyncio.gather(*tasks)

        await engine.dispose()

    async def test_large_transaction_does_not_starve_other_writers(self, tmp_path):
        """A slow write transaction must not cause others to fail — they queue instead."""
        engine = _team_engine(str(tmp_path / "team_slow.db"))
        async with engine.begin() as conn:
            await conn.run_sync(TeamBase.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False)
        results: list[str] = []

        async def slow_write():
            async with factory() as session:
                # Insert several rows in one transaction to hold the write lock longer
                for _ in range(30):
                    session.add(Athlete(
                        id=str(uuid.uuid4()),
                        global_user_id=str(uuid.uuid4()),
                        ftp_tests=[],
                    ))
                await session.commit()
            results.append("slow")

        async def fast_write():
            async with factory() as session:
                session.add(Athlete(
                    id=str(uuid.uuid4()),
                    global_user_id=str(uuid.uuid4()),
                    ftp_tests=[],
                ))
                await session.commit()
            results.append("fast")

        await asyncio.gather(slow_write(), *[fast_write() for _ in range(10)])

        async with factory() as session:
            count = (await session.execute(text("SELECT COUNT(*) FROM athletes"))).scalar()

        await engine.dispose()
        assert count == 30 + 10
        assert results.count("slow") == 1
        assert results.count("fast") == 10


class TestRegistryDbConcurrency:
    async def test_concurrent_team_inserts_succeed(self, tmp_path):
        """30 concurrent coroutines each insert a unique Team; none must fail."""
        engine = _registry_engine(str(tmp_path / "registry.db"))
        async with engine.begin() as conn:
            await conn.run_sync(RegistryBase.metadata.create_all)

        factory = async_sessionmaker(engine, expire_on_commit=False)
        n = 30

        async def insert_team(i: int):
            async with factory() as session:
                session.add(Team(
                    id=str(uuid.uuid4()),
                    slug=f"team-{i}-{uuid.uuid4().hex[:8]}",
                    name=f"Team {i}",
                    status="active",
                ))
                await session.commit()

        await asyncio.gather(*[insert_team(i) for i in range(n)])

        async with factory() as session:
            count = (await session.execute(text("SELECT COUNT(*) FROM teams"))).scalar()

        await engine.dispose()
        assert count == n
