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
import functools
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Set before anything imports `backend.app.core.config`, which builds its
# Settings singleton at import time and now refuses an empty ENCRYPTION_KEY
# (issue #102, F-08). A real Fernet key rather than ALLOW_PLAINTEXT_SECRETS so
# the suite exercises the encrypted column path — encrypt on write, decrypt on
# read — which nothing covered while the empty key quietly disabled it.
# `setdefault` so a caller can still override either one.
os.environ.setdefault("ENCRYPTION_KEY", "0xQFqFc2TsglXsh-2Nn0DAclf8Gn6VgGJm1gpMJ53cw=")

# The two bridges are separate services with their own Settings, built at import
# time, and both now refuse the "changeme" placeholder (issue #102, F-10). The
# bridge test modules import those apps, so the secrets have to exist before
# collection reaches them. Length matters — the validators enforce a 32-character
# floor, the same bar the main app's SECRET_KEY uses.
os.environ.setdefault("BRIDGE_SECRET", "ci-test-strava-bridge-secret-not-for-production")
os.environ.setdefault("WAHOO_BRIDGE_SECRET", "ci-test-wahoo-bridge-secret-not-for-production")

import pytest
from fastapi import HTTPException, Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.auth import UserContext, create_access_token
from backend.app.core.deps import get_ctx_and_session
from backend.app.db.base import RegistryBase, UserBase
# Imported for their side effect: the two metadata objects are only complete once
# every model module has been imported, and the schema each test gets is built from
# whatever is in them at that moment. Without this the tables present in a test would
# depend on which fixture happened to import which module first.
import backend.app.models.chat_orm  # noqa: F401,E402
import backend.app.models.message_orm  # noqa: F401,E402
import backend.app.models.registry_orm  # noqa: F401,E402
import backend.app.models.user_orm  # noqa: F401,E402
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


@pytest.fixture(scope="session", autouse=True)
def _allow_private_llm_hosts():
    """Let the suite point at localhost as a stand-in for a model server.

    ``LLM_ALLOW_PRIVATE_NETWORKS`` is off in production: a user-supplied base
    URL that may resolve into the private address space is issue #102's F-02.
    The LLM fixtures here use ``http://localhost:11434`` the way a self-hosted
    instance running Ollama does, which is the case the opt-out exists for.

    Tests that assert the *default-deny* behaviour patch the flag themselves
    (``tests/unit/test_llm_ssrf.py``) rather than depending on this default.
    """
    from backend.app.core.config import settings

    original = settings.llm_allow_private_networks
    settings.llm_allow_private_networks = True
    yield
    settings.llm_allow_private_networks = original


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

@functools.lru_cache(maxsize=None)
def _schema_script(base) -> str:
    """The CREATE statements ``create_all`` emits for *base*, as one SQL script.

    Every test builds its databases from scratch, and ``metadata.create_all`` is
    dominated by compiling DDL out of the metadata rather than by running it:
    ~47 ms for the 26 per-user tables and ~19 ms for the 10 registry ones, of
    which SQLite spends under 2 ms executing the statements. Multiplied by the
    ~1450 tests that take a database, that compilation was around a quarter of
    the suite's runtime.

    So it is done once and the result replayed. The statements are read back out
    of ``sqlite_master`` *after* ``create_all`` ran, which means they are exactly
    what SQLAlchemy emitted — there is no second description of the schema here
    that could drift from the models. Rows with a NULL ``sql`` are SQLite's own
    implicit indexes, which it recreates from the UNIQUE/PRIMARY KEY clauses.
    """
    from sqlalchemy import create_engine

    eng = create_engine("sqlite://")
    try:
        with eng.begin() as conn:
            base.metadata.create_all(conn)
            statements = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
            ).scalars().all()
    finally:
        eng.dispose()
    return ";\n".join(statements) + ";"


async def _engine_with_schema(base):
    """A fresh in-memory engine carrying *base*'s schema.

    The schema outlives the connection that created it because SQLAlchemy gives
    a ``:memory:`` database a ``StaticPool``: closing the connection returns it
    to the pool rather than dropping the database underneath it.
    """
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.connect() as conn:
        raw = await conn.get_raw_connection()
        await raw.driver_connection.executescript(_schema_script(base))
    return eng


@pytest.fixture
async def registry_engine():
    """Fresh in-memory registry SQLite engine per test."""
    eng = await _engine_with_schema(RegistryBase)
    yield eng
    await eng.dispose()


@pytest.fixture
async def user_engine():
    """Fresh in-memory per-user SQLite engine per test."""
    eng = await _engine_with_schema(UserBase)
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

@pytest.fixture(scope="session")
def app():
    """Build the FastAPI app once for the whole test session.

    ``create_app()`` is expensive — it wires two dozen routers into 152 routes,
    and building their Pydantic models costs ~0.3 s. It used to be built once
    per module, which meant paying that for each of the 56 modules that take a
    client; once per session is the same thing done once.

    The app is stateless between tests except for ``dependency_overrides``,
    which the per-test ``client`` fixture sets and clears in a ``finally`` (as
    does ``test_public_rate_limit``'s own client), so nothing carries over. The
    lifespan (which starts the poller tasks) is never triggered because
    ``ASGITransport`` does not run lifespan events.
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
    token = create_access_token(_TEST_USER_ID, _TEST_ROLES, token_version=0)
    return {"Authorization": f"Bearer {token}"}
