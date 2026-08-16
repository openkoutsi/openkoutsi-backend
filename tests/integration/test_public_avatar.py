"""Integration tests for GET /api/public/users/{user_id}/avatar.

Issue #102, F-03: the route takes no credential and no rate limit, and passed
its path segment straight to the per-user engine factory — which used to
``mkdir`` the user's directory on a cache miss. Opening the session then created
``user.db`` plus its ``-wal`` and ``-shm`` companions, the missing-table
``SELECT`` failed, and the ``except Exception`` handler returned a tidy 404 with
four filesystem objects left behind. Roughly 3.7 MB and 400 inodes per second
from one anonymous client, against a service whose whole dataset is SQLite files
on the same volume.

The secondary effect was cache eviction: ``_get_user_engine`` is an
``lru_cache(maxsize=256)``, so 256 junk ids evicted every real user's engine and
its connection pool.
"""
import json
import uuid
from pathlib import Path

import pytest

from backend.app.core.config import settings
from backend.app.core.deps import open_user_session
from backend.app.db.user_session import _get_user_engine, init_user_db
from backend.app.models.registry_orm import User
from backend.app.models.user_orm import Athlete

_JPEG = b"\xff\xd8\xff\xe0"


def _users_dir() -> Path:
    """The directory the attack created entries in (data_dir is per-test)."""
    return Path(settings.data_dir) / "users"


def _disk_entries() -> set:
    d = _users_dir()
    return set(d.rglob("*")) if d.exists() else set()


class TestUnknownIdsTouchNothing:
    @pytest.mark.parametrize("user_id", [
        "not-a-uuid",
        "../../../etc",
        "%2e%2e%2f",
        "",
        "x" * 300,
        "{12345678-1234-5678-1234-567812345678}",   # non-canonical UUID spelling
        "urn:uuid:12345678-1234-5678-1234-567812345678",
        "12345678123456781234567812345678",         # undashed
    ])
    async def test_malformed_id_is_404_and_creates_nothing(self, client, user_id):
        before = _disk_entries()
        resp = await client.get(f"/api/public/users/{user_id}/avatar")
        assert resp.status_code == 404
        assert _disk_entries() == before

    async def test_wellformed_but_unknown_uuid_creates_nothing(self, client):
        """A syntactically perfect id for an account that does not exist."""
        before = _disk_entries()
        resp = await client.get(f"/api/public/users/{uuid.uuid4()}/avatar")
        assert resp.status_code == 404
        assert _disk_entries() == before

    async def test_repeated_requests_do_not_accumulate(self, client):
        """The reported PoC, scaled down: 50 anonymous requests, 0 bytes gained."""
        before = _disk_entries()
        statuses = set()
        for _ in range(50):
            resp = await client.get(f"/api/public/users/{uuid.uuid4()}/avatar")
            statuses.add(resp.status_code)

        assert statuses == {404}
        assert _disk_entries() == before

    async def test_engine_cache_not_polluted(self, client):
        """The eviction half: junk ids must never reach the cached factory.

        With maxsize=256, 256 junk requests used to evict every live user's
        engine and connection pool.
        """
        _get_user_engine.cache_clear()
        for _ in range(20):
            await client.get(f"/api/public/users/{uuid.uuid4()}/avatar")

        assert _get_user_engine.cache_info().currsize == 0


class TestKnownUserStillServed:
    """The route has to keep working — the fix is a gate, not a wall."""

    @pytest.fixture
    async def real_user(self, registry_session, tmp_path):
        """A registry user with a canonical UUID id and a real per-user DB."""
        uid = str(uuid.uuid4())
        registry_session.add(
            User(id=uid, username=f"u-{uid[:8]}", password_hash="x",
                 roles=json.dumps(["user"]))
        )
        await registry_session.commit()

        await init_user_db(uid)

        avatar = tmp_path / "avatar.jpg"
        avatar.write_bytes(_JPEG + b"body")
        async with open_user_session(uid) as s:
            s.add(Athlete(id=f"a-{uid[:8]}", global_user_id=uid,
                          avatar_path=str(avatar)))
            await s.commit()
        return uid, avatar

    async def test_avatar_served_for_known_user(self, client, real_user):
        uid, avatar = real_user
        resp = await client.get(f"/api/public/users/{uid}/avatar")
        assert resp.status_code == 200
        assert resp.content == avatar.read_bytes()

    async def test_known_user_without_avatar_is_404(self, client, registry_session):
        uid = str(uuid.uuid4())
        registry_session.add(
            User(id=uid, username=f"u-{uid[:8]}", password_hash="x",
                 roles=json.dumps(["user"]))
        )
        await registry_session.commit()
        await init_user_db(uid)
        async with open_user_session(uid) as s:
            s.add(Athlete(id=f"a-{uid[:8]}", global_user_id=uid))
            await s.commit()

        resp = await client.get(f"/api/public/users/{uid}/avatar")
        assert resp.status_code == 404

    async def test_soft_deleted_user_is_404(self, client, registry_session, real_user):
        """A deleted account is not a public identity any more."""
        from datetime import datetime, timezone

        uid, _ = real_user
        user = await registry_session.get(User, uid)
        user.deleted_at = datetime.now(timezone.utc)
        await registry_session.commit()

        resp = await client.get(f"/api/public/users/{uid}/avatar")
        assert resp.status_code == 404


class TestEngineConstructionIsSideEffectFree:
    """The other half of the fix, independent of the route that exposed it."""

    def test_getting_an_engine_creates_nothing(self):
        uid = str(uuid.uuid4())
        before = _disk_entries()

        _get_user_engine(uid)

        assert _disk_entries() == before, "building an engine must not touch disk"

    async def test_init_user_db_does_create_the_directory(self):
        """Creating a user's database stays an explicit act, and still works."""
        uid = str(uuid.uuid4())
        assert not (_users_dir() / uid).exists()

        await init_user_db(uid)

        assert (_users_dir() / uid / "user.db").exists()

    async def test_session_for_unknown_user_fails_rather_than_creating(self):
        """No directory means no database — the caller sees an error, not a new file."""
        from backend.app.db.user_session import get_user_session_factory

        uid = str(uuid.uuid4())
        before = _disk_entries()

        with pytest.raises(Exception):
            async with get_user_session_factory(uid)() as s:
                await s.execute(Athlete.__table__.select())

        assert _disk_entries() == before
