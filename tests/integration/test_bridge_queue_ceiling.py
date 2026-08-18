"""The bridge event queue is bounded (issue #102, F-11).

Every accepted event is a row retained for seven days, and the main app drains
100 a minute (`api/strava.py`). Nothing capped the queue, so a burst that
outpaced the drain grew the SQLite file until the disk ran out — and because
the events sit in front of the drain, a large enough backlog also delays
legitimate webhook processing indefinitely.

Fixing F-01 (#103) closed the injection route that made this trivially
reachable, which is why this is a Low. It does not bound anything: a genuine
burst, or a poller that has stopped, still grows the queue without limit.

Refusing rather than evicting is the deliberate half. The events already queued
are the ones the main app is about to process; dropping the oldest to make room
for the newest would lose exactly those, silently.
"""
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import strava_bridge.main as strava_bridge
import wahoo_bridge.main as wahoo_bridge

_SECRET = "b" * 32


def _bridges():
    return [
        pytest.param(strava_bridge, id="strava"),
        pytest.param(wahoo_bridge, id="wahoo"),
    ]


@pytest.fixture
def bridge_db(request, monkeypatch):
    """In-memory queue for whichever bridge module the test is parametrised on."""
    module = request.getfixturevalue("module")

    async def _setup():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(module.Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(module, "AsyncSessionLocal", factory)
        return engine, factory

    return _setup


async def _fill_queue(module, factory, count: int, *, claimed: bool = False) -> None:
    now = datetime.now(timezone.utc)
    async with factory() as s:
        for i in range(count):
            kwargs = {
                "id": f"e{i}",
                "payload": {},
                "received_at": now,
                "claimed_at": now if claimed else None,
            }
            if module is strava_bridge:
                kwargs |= {"strava_event_type": "create", "strava_owner_id": "1"}
            else:
                kwargs |= {"wahoo_event_type": "workout_summary", "wahoo_owner_id": "1"}
            s.add(module.WebhookEvent(**kwargs))
        await s.commit()


@pytest.mark.parametrize("module", _bridges())
class TestPendingCount:
    async def test_counts_only_unclaimed(self, module, bridge_db, monkeypatch):
        """A busy week already drained must not close the queue to a quiet one."""
        _engine, factory = await bridge_db()
        await _fill_queue(module, factory, 5, claimed=True)

        async with factory() as s:
            assert await module._pending_count(s) == 0

        await _fill_queue(module, factory, 0)
        async with factory() as s:
            total = await s.scalar(
                select(func.count()).select_from(module.WebhookEvent)
            )
        assert total == 5, "the claimed rows are still there, just not counted"

    async def test_counts_unclaimed(self, module, bridge_db):
        _engine, factory = await bridge_db()
        await _fill_queue(module, factory, 3)
        async with factory() as s:
            assert await module._pending_count(s) == 3


class TestStravaCeiling:
    @pytest.fixture
    async def client(self, monkeypatch):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(strava_bridge.Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(strava_bridge, "AsyncSessionLocal", factory)
        monkeypatch.setattr(strava_bridge.settings, "strava_client_secret", _SECRET)
        async with AsyncClient(
            transport=ASGITransport(app=strava_bridge.app), base_url="http://bridge"
        ) as c:
            yield c, factory
        await engine.dispose()

    async def _post(self, client, object_id: int = 1):
        import hashlib
        import hmac
        import json

        payload = {
            "object_type": "activity",
            "aspect_type": "create",
            "object_id": object_id,
            "owner_id": 12345678,
        }
        body = json.dumps(payload).encode()
        sig = "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
        return await client.post(
            "/webhook",
            content=body,
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig},
        )

    async def test_accepts_below_the_ceiling(self, client, monkeypatch):
        c, factory = client
        monkeypatch.setattr(strava_bridge.settings, "max_queue_events", 3)
        await _fill_queue(strava_bridge, factory, 2)

        assert (await self._post(c)).status_code == 200

    async def test_refuses_at_the_ceiling(self, client, monkeypatch):
        c, factory = client
        monkeypatch.setattr(strava_bridge.settings, "max_queue_events", 3)
        await _fill_queue(strava_bridge, factory, 3)

        resp = await self._post(c)
        assert resp.status_code == 503
        assert "full" in resp.json()["detail"].lower()

    async def test_nothing_is_written_when_refused(self, client, monkeypatch):
        """The point of the ceiling: the file stops growing."""
        c, factory = client
        monkeypatch.setattr(strava_bridge.settings, "max_queue_events", 3)
        await _fill_queue(strava_bridge, factory, 3)

        for i in range(10):
            assert (await self._post(c, object_id=i)).status_code == 503

        async with factory() as s:
            total = await s.scalar(
                select(func.count()).select_from(strava_bridge.WebhookEvent)
            )
        assert total == 3

    async def test_queued_events_are_kept_not_evicted(self, client, monkeypatch):
        """Refusing the newest, not dropping the oldest."""
        c, factory = client
        monkeypatch.setattr(strava_bridge.settings, "max_queue_events", 2)
        await _fill_queue(strava_bridge, factory, 2)

        await self._post(c)

        async with factory() as s:
            ids = (await s.execute(
                select(strava_bridge.WebhookEvent.id).order_by(
                    strava_bridge.WebhookEvent.id
                )
            )).scalars().all()
        assert ids == ["e0", "e1"]

    async def test_draining_reopens_the_queue(self, client, monkeypatch):
        """A full queue is a backlog, not a permanent state."""
        c, factory = client
        monkeypatch.setattr(strava_bridge.settings, "max_queue_events", 2)
        await _fill_queue(strava_bridge, factory, 2)
        assert (await self._post(c)).status_code == 503

        async with factory() as s:
            event = await s.get(strava_bridge.WebhookEvent, "e0")
            event.claimed_at = datetime.now(timezone.utc)
            await s.commit()

        assert (await self._post(c)).status_code == 200


class TestWahooCeiling:
    @pytest.fixture
    async def client(self, monkeypatch):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(wahoo_bridge.Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        monkeypatch.setattr(wahoo_bridge, "AsyncSessionLocal", factory)
        monkeypatch.setattr(wahoo_bridge.settings, "wahoo_webhook_token", _SECRET)
        async with AsyncClient(
            transport=ASGITransport(app=wahoo_bridge.app), base_url="http://bridge"
        ) as c:
            yield c, factory
        await engine.dispose()

    async def _post(self, client):
        return await client.post(
            "/webhook",
            json={
                "webhook_token": _SECRET,
                "event_type": "workout_summary",
                "user": {"id": 42},
            },
        )

    async def test_accepts_below_the_ceiling(self, client, monkeypatch):
        c, factory = client
        monkeypatch.setattr(wahoo_bridge.settings, "max_queue_events", 3)
        await _fill_queue(wahoo_bridge, factory, 2)
        assert (await self._post(c)).status_code == 200

    async def test_refuses_at_the_ceiling(self, client, monkeypatch):
        c, factory = client
        monkeypatch.setattr(wahoo_bridge.settings, "max_queue_events", 3)
        await _fill_queue(wahoo_bridge, factory, 3)
        assert (await self._post(c)).status_code == 503
