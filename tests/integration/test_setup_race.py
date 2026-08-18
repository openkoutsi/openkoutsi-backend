"""First-run setup admits exactly one administrator (issue #102, F-13).

`first_run_setup` counted users, then inserted, with an `await` between — and
the bcrypt hash sits in that gap, which on a production cost factor is roughly
a quarter of a second. Two requests arriving inside it both observed zero users
and both created an account with `["administrator", "user"]`.

The window is narrow and only open on an instance that has never been set up.
But the endpoint is unauthenticated by definition, and a freshly deployed
instance is exactly when an admin is about to make this request themselves —
so the race is against a request someone is *known* to be making.

These tests use a file-backed database rather than `:memory:`. SQLAlchemy hands
every session the same connection for an in-memory SQLite, which would serialise
the two attempts and hide the very interleaving under test.
"""
import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.app.db.base import RegistryBase
from backend.app.db.registry import get_registry_session
from backend.app.models.registry_orm import User

_PASSWORD = "Testpass1234"


@pytest.fixture
async def setup_client(app, tmp_path):
    """A client whose registry is a real file, with a fresh session per request.

    The shared `client` fixture yields one session to every request, which is
    not what two concurrent callers get in production and would make the race
    untestable.
    """
    from backend.app.core.limiter import limiter

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'registry.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_registry():
        async with factory() as s:
            yield s

    app.dependency_overrides[get_registry_session] = _override_registry
    limiter.enabled = False
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c, factory
    finally:
        limiter.enabled = True
        app.dependency_overrides.clear()
        await engine.dispose()


async def _setup(client, username: str):
    return await client.post(
        "/api/setup",
        json={"admin_username": username, "admin_password": _PASSWORD},
    )


async def _users(factory) -> list[User]:
    async with factory() as s:
        return (await s.execute(select(User).order_by(User.username))).scalars().all()


class TestConcurrentSetup:
    async def test_two_at_once_produce_one_administrator(self, setup_client):
        """The reproduction: two requests, different usernames, same instant."""
        client, factory = setup_client

        first, second = await asyncio.gather(
            _setup(client, "admin-a"), _setup(client, "admin-b")
        )

        statuses = sorted([first.status_code, second.status_code])
        assert statuses == [201, 409], f"got {statuses}"

        users = await _users(factory)
        assert len(users) == 1, [u.username for u in users]
        assert json.loads(users[0].roles) == ["administrator", "user"]

    async def test_five_at_once_still_produce_one(self, setup_client):
        """More contenders must not mean more administrators."""
        client, factory = setup_client

        responses = await asyncio.gather(
            *[_setup(client, f"admin-{i}") for i in range(5)]
        )

        assert sorted(r.status_code for r in responses) == [201, 409, 409, 409, 409]
        assert len(await _users(factory)) == 1

    async def test_the_winner_gets_a_usable_token(self, setup_client):
        """Losing the race must not leave the winner half-provisioned."""
        client, factory = setup_client

        responses = await asyncio.gather(
            _setup(client, "admin-a"), _setup(client, "admin-b")
        )
        winner = next(r for r in responses if r.status_code == 201)

        assert winner.json()["access_token"]
        users = await _users(factory)
        assert winner.json()["token_type"] == "bearer"
        assert len(users) == 1

    async def test_the_loser_creates_no_user_row(self, setup_client):
        """Not just 'one wins' — the loser must leave nothing behind."""
        client, factory = setup_client

        await asyncio.gather(_setup(client, "admin-a"), _setup(client, "admin-b"))

        async with factory() as s:
            count = await s.scalar(select(func.count()).select_from(User))
        assert count == 1


class TestSequentialSetup:
    """The ordinary paths, unchanged by the fix."""

    async def test_first_setup_succeeds(self, setup_client):
        client, factory = setup_client
        resp = await _setup(client, "admin")
        assert resp.status_code == 201
        assert resp.json()["access_token"]

        users = await _users(factory)
        assert len(users) == 1
        assert users[0].username == "admin"
        assert json.loads(users[0].roles) == ["administrator", "user"]

    async def test_second_setup_is_refused(self, setup_client):
        client, _factory = setup_client
        assert (await _setup(client, "admin")).status_code == 201

        resp = await _setup(client, "second-admin")
        assert resp.status_code == 409
        assert "already completed" in resp.json()["detail"]

    async def test_status_flips_once_set_up(self, setup_client):
        client, _factory = setup_client
        assert (await client.get("/api/setup/status")).json()["needs_setup"] is True

        await _setup(client, "admin")

        assert (await client.get("/api/setup/status")).json()["needs_setup"] is False

    async def test_the_admin_row_carries_what_it_should(self, setup_client):
        """The atomic insert names its columns explicitly — check none were lost."""
        client, factory = setup_client
        await _setup(client, "admin")

        user = (await _users(factory))[0]
        assert user.id
        assert user.username == "admin"
        assert user.password_hash and user.password_hash != _PASSWORD
        assert user.created_at is not None
        assert user.token_version == 0
        assert user.deleted_at is None
