"""
Shared fixtures for the test suite.

DB strategy: every test function gets fresh in-memory SQLite engines:
  - registry_engine/registry_session: global identity + team registry
  - team_engine/team_session: per-team athletic data

The `client` fixture overrides both FastAPI session dependencies so all
routes hit the in-memory DBs. A seeded Athlete row is created automatically.

Background tasks are suppressed via mock so they never touch real storage.
Rate limiting is disabled so tests are not throttled.
"""
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.auth import TeamContext, create_access_token
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.base import RegistryBase, TeamBase
from backend.app.db.registry import get_registry_session
from backend.main import create_app

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"

# Fixed IDs used across all test fixtures
_TEST_TEAM_ID = "test-team-00000000"
_TEST_USER_ID = "test-user-00000000"
_TEST_ATHLETE_ID = "test-athlete-0000"
_TEST_ROLES = ["administrator", "user"]


# ── Per-user DB isolation ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_user_dbs(tmp_path, monkeypatch):
    """Point per-user DB files at a temp dir and reset the engine cache per test.

    Per-user mailbox DBs (backend.app.db.user_session) are file-based, so each
    test gets its own data_dir and a cleared engine cache for isolation.
    """
    from backend.app.core.config import settings
    from backend.app.db import user_session

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    user_session._get_user_engine.cache_clear()
    yield
    user_session._get_user_engine.cache_clear()


# ── DB fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
async def registry_engine():
    """Fresh in-memory registry SQLite engine per test."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(RegistryBase.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def team_engine():
    """Fresh in-memory team SQLite engine per test."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(TeamBase.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def registry_session(registry_engine):
    """Async session backed by the in-memory registry engine, with seeded User, Team, and TeamMembership."""
    from backend.app.core.auth import hash_password
    from backend.app.models.registry_orm import Team, TeamMembership, User
    factory = async_sessionmaker(registry_engine, expire_on_commit=False)
    async with factory() as s:
        user = User(
            id=_TEST_USER_ID,
            username="test-user",
            password_hash=hash_password("Testpass1234"),
        )
        s.add(user)
        team = Team(id=_TEST_TEAM_ID, slug="test-team", name="Test Team")
        s.add(team)
        await s.flush()
        membership = TeamMembership(
            team_id=_TEST_TEAM_ID,
            user_id=_TEST_USER_ID,
            roles='["administrator","user"]',
        )
        s.add(membership)
        await s.commit()
        yield s


@pytest.fixture
async def session(team_engine):
    """Async session backed by the in-memory team engine (team data only)."""
    factory = async_sessionmaker(team_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


# ── Seeded athlete ─────────────────────────────────────────────────────────

@pytest.fixture
async def seeded_athlete(session):
    """Insert a minimal Athlete row into the team session and return it."""
    from backend.app.models.team_orm import Athlete
    athlete = Athlete(
        id=_TEST_ATHLETE_ID,
        global_user_id=_TEST_USER_ID,
        ftp_tests=[],
    )
    session.add(athlete)
    await session.commit()
    return athlete


# ── HTTP client with DI overrides ─────────────────────────────────────────

@pytest.fixture
async def client(session, registry_session, seeded_athlete):
    """
    HTTP test client wired to in-memory test DBs.

    - `get_ctx_and_session` is overridden: yields a fixed TeamContext +
      the in-memory team session. No JWT validation occurs.
    - `get_registry_session` is overridden to use the in-memory registry.
    - Background tasks are suppressed.
    - Rate limiting is disabled.
    """
    from backend.app.core.limiter import limiter

    app = create_app()

    test_ctx = TeamContext(
        user_id=_TEST_USER_ID,
        team_id=_TEST_TEAM_ID,
        roles=_TEST_ROLES,
    )

    async def _override_ctx_session(request: Request):
        if not request.headers.get("Authorization", "").startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        yield test_ctx, session

    async def _override_registry():
        yield registry_session

    app.dependency_overrides[get_ctx_and_session] = _override_ctx_session
    app.dependency_overrides[get_registry_session] = _override_registry

    limiter.enabled = False
    try:
        with patch("starlette.background.BackgroundTasks.add_task"):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                yield c
    finally:
        limiter.enabled = True
        app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict:
    """Bearer token headers for the seeded test athlete (no real auth needed)."""
    token = create_access_token(_TEST_USER_ID, _TEST_TEAM_ID, _TEST_ROLES)
    return {"Authorization": f"Bearer {token}"}
