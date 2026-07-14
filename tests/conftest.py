"""
Shared fixtures for the test suite.

DB strategy: every test function gets fresh in-memory SQLite engines:
  - registry_engine/registry_session: global identity registry
  - user_engine/session: the per-user DB (athlete + all training data + inbox)

The `client` fixture overrides both FastAPI session dependencies so all
routes hit the in-memory DBs. A seeded Athlete row is created automatically.

Background tasks are suppressed via mock so they never touch real storage.
Rate limiting is disabled so tests are not throttled.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.auth import UserContext, create_access_token
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.base import RegistryBase, UserBase
from backend.app.db.registry import get_registry_session
from backend.main import create_app

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"

# Fixed IDs used across all test fixtures
_TEST_USER_ID = "test-user-00000000"
_TEST_ATHLETE_ID = "test-athlete-0000"
_TEST_ROLES = ["administrator", "user"]
_TEST_PASSWORD = "Testpass1234"


# ── Faster password hashing for the test session ────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _cheap_bcrypt():
    """Lower bcrypt's work factor for the whole test session.

    Production uses bcrypt's default cost (12 rounds), which is deliberately
    slow (~0.27s per hash). Tests hash and verify passwords constantly, so this
    dominates the runtime. We patch ``bcrypt.gensalt`` to use a cheap cost for
    the duration of the test session only — ``backend.app.core.auth`` is left
    untouched, so production behaviour is unchanged. bcrypt records the cost in
    the hash itself, so cheaper hashes still verify correctly.
    """
    import bcrypt

    original = bcrypt.gensalt
    bcrypt.gensalt = lambda rounds=4, prefix=b"2b": original(rounds=rounds, prefix=prefix)
    try:
        yield
    finally:
        bcrypt.gensalt = original


@pytest.fixture(scope="session")
def _test_password_hash(_cheap_bcrypt):
    """Hash the shared test password once for the whole session and reuse it."""
    from backend.app.core.auth import hash_password

    return hash_password(_TEST_PASSWORD)


# ── Per-user DB isolation ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_user_dbs(tmp_path, monkeypatch):
    """Point per-user DB files at a temp dir and reset the engine cache per test.

    Per-user DBs (backend.app.db.user_session) are file-based, so each test gets
    its own data_dir and a cleared engine cache for isolation.
    """
    from backend.app.core.config import settings
    from backend.app.db import usage, user_session

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    user_session._get_user_engine.cache_clear()
    usage._get_usage_engine.cache_clear()
    yield
    user_session._get_user_engine.cache_clear()
    usage._get_usage_engine.cache_clear()


@pytest.fixture
async def usage_db(isolate_user_dbs):
    """Initialise the dedicated LLM-usage DB in this test's temp data dir.

    Returns a session factory for asserting on recorded ``llm_usage`` rows.
    """
    from backend.app.db.usage import init_usage_db, usage_session_factory

    await init_usage_db()
    return usage_session_factory()


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
async def user_engine():
    """Fresh in-memory per-user SQLite engine per test."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(UserBase.metadata.create_all)
    yield eng
    await eng.dispose()


# Backwards-compatible alias for tests that still request `team_engine`.
@pytest.fixture
async def team_engine(user_engine):
    yield user_engine


@pytest.fixture
async def registry_session(registry_engine, _test_password_hash):
    """Async session backed by the in-memory registry engine, with a seeded admin User."""
    import json
    from datetime import datetime, timezone

    from backend.app.api.consent import CURRENT_CONSENT_VERSION
    from backend.app.models.registry_orm import User
    factory = async_sessionmaker(registry_engine, expire_on_commit=False)
    async with factory() as s:
        user = User(
            id=_TEST_USER_ID,
            username="test-user",
            password_hash=_test_password_hash,
            roles=json.dumps(_TEST_ROLES),
            # The default test user is a normal, onboarded user who has accepted
            # the current privacy policy, so consent-gated routes (upload,
            # provider connect) work. Tests for the un-consented case clear this.
            consented_at=datetime.now(timezone.utc),
            consent_version=CURRENT_CONSENT_VERSION,
        )
        s.add(user)
        await s.commit()
        yield s


@pytest.fixture
async def session(user_engine):
    """Async session backed by the in-memory per-user engine (training data)."""
    factory = async_sessionmaker(user_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


# ── Seeded athlete ─────────────────────────────────────────────────────────

@pytest.fixture
async def seeded_athlete(session):
    """Insert a minimal Athlete row into the per-user session and return it."""
    from backend.app.models.user_orm import Athlete
    athlete = Athlete(
        id=_TEST_ATHLETE_ID,
        global_user_id=_TEST_USER_ID,
        ftp_tests=[],
    )
    session.add(athlete)
    await session.commit()
    return athlete


# ── HTTP client with DI overrides ─────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """Build the FastAPI app once per test module.

    ``create_app()`` is relatively expensive (imports and wires every router),
    so we construct it once per module rather than once per test. The app is
    stateless between tests except for ``dependency_overrides``, which the
    per-test ``client`` fixture sets and clears; tests within a module run
    sequentially, so there is no cross-test bleed. The lifespan (which starts
    the poller tasks) is never triggered because ``ASGITransport`` does not run
    lifespan events.
    """
    return create_app()


@pytest.fixture
async def client(app, session, registry_session, seeded_athlete):
    """
    HTTP test client wired to in-memory test DBs.

    - `get_ctx_and_session` is overridden: yields a fixed UserContext +
      the in-memory per-user session. No JWT validation occurs.
    - `get_registry_session` is overridden to use the in-memory registry.
    - Background tasks are suppressed.
    - Rate limiting is disabled.

    Overrides are set on the shared (module-scoped) app for this test and
    cleared afterwards so each test wires the app to its own fresh sessions.
    """
    from backend.app.core.limiter import limiter

    test_ctx = UserContext(
        user_id=_TEST_USER_ID,
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
    token = create_access_token(_TEST_USER_ID, _TEST_ROLES)
    return {"Authorization": f"Bearer {token}"}
