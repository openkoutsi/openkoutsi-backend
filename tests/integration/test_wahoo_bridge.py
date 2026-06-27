"""
Integration tests for the Wahoo Bridge round-trip.

Both apps run in-process via ASGITransport — no real network or server.
The bridge uses a fresh in-memory SQLite database per test.

Scenarios covered:
  1. Happy path: webhook → bridge → poller → process_wahoo_webhook called
  2. Invalid token rejected at the bridge
  3. Non-workout events not queued
  4. Backend offline: events accumulate, then all processed when poller runs
  5. Claimed events not reprocessed on subsequent poll
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import ASGITransport, AsyncClient

# Captured before any patching so _BridgeAsgiClient can always reach the real class.
_real_AsyncClient = httpx.AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import select

import wahoo_bridge.main as bridge_module
from wahoo_bridge.main import app as bridge_app, Base as BridgeBase, WebhookEvent
from backend.app.api.wahoo import _poll_bridge_once

BRIDGE_SECRET = "test-bridge-secret"
WEBHOOK_TOKEN = "test-webhook-token"
BRIDGE_BASE_URL = "http://wahoo-bridge"

_WORKOUT_PAYLOAD = {
    "event_type": "workout_summary",
    "webhook_token": WEBHOOK_TOKEN,
    "user": {"id": 9876543},
    "workout_summary": {
        "id": 12341234,
        "started_at": "2026-04-25T09:36:48.000Z",
        "ascent_accum": "103.0",
        "cadence_avg": "75.0",
        "distance_accum": "27441.58",
        "duration_active_accum": "4184.0",
        "duration_paused_accum": "337.0",
        "duration_total_accum": "4521.0",
        "heart_rate_avg": "141.0",
        "power_avg": "175.0",
        "workout_type_id": 0,
        "file": {"url": "https://cdn.example.com/fit/12341234.fit"},
        "workout": {
            "id": 12341234,
            "name": "Afternoon Ride",
            "workout_type_id": 0,
            "starts": "2026-04-25T09:36:48.000Z",
        },
    },
}


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def bridge_db():
    """Fresh in-memory SQLite for the bridge, tables pre-created."""
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(BridgeBase.metadata.create_all)
    sessions = async_sessionmaker(eng, expire_on_commit=False)
    yield eng, sessions
    await eng.dispose()


@pytest.fixture
def patched_bridge(bridge_db):
    """
    Patch the bridge's module-level engine, session factory, and settings
    to use the in-memory DB and test credentials.
    """
    eng, sessions = bridge_db
    with (
        patch.object(bridge_module, "engine", eng),
        patch.object(bridge_module, "AsyncSessionLocal", sessions),
        patch.object(bridge_module.settings, "wahoo_webhook_token", WEBHOOK_TOKEN),
        patch.object(bridge_module.settings, "bridge_secret", BRIDGE_SECRET),
    ):
        yield eng, sessions


@pytest.fixture
async def bridge_client(patched_bridge):
    """AsyncClient wired to the (patched) bridge ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=bridge_app), base_url=BRIDGE_BASE_URL
    ) as c:
        yield c


def _make_polling_client_class():
    """
    Return a drop-in replacement for httpx.AsyncClient that routes all
    requests to the bridge ASGI app instead of the real network.
    """
    class _BridgeAsgiClient:
        def __init__(self, **kwargs):
            self._inner = _real_AsyncClient(
                transport=ASGITransport(app=bridge_app),
            )

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self._inner

        async def __aexit__(self, *args):
            await self._inner.__aexit__(*args)

    return _BridgeAsgiClient


async def _poll(mock_process: AsyncMock) -> None:
    """
    Call _poll_bridge_once() with the bridge ASGI transport in place of real
    HTTP and process_wahoo_webhook replaced by mock_process.
    """
    mock_settings = MagicMock()
    mock_settings.wahoo_bridge_url = BRIDGE_BASE_URL
    mock_settings.wahoo_bridge_secret = BRIDGE_SECRET

    BridgeAsgiClient = _make_polling_client_class()
    with (
        patch("backend.app.api.wahoo.httpx.AsyncClient", new=BridgeAsgiClient),
        patch("backend.app.api.wahoo.settings", mock_settings),
        patch("backend.app.api.wahoo.process_wahoo_webhook", mock_process),
    ):
        await _poll_bridge_once()


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBridgeWebhookEndpoint:
    async def test_valid_workout_event_queued(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge

        resp = await bridge_client.post("/webhook", json=_WORKOUT_PAYLOAD)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()

        assert len(events) == 1
        assert events[0].wahoo_event_type == "workout_summary"
        assert events[0].wahoo_owner_id == "9876543"
        assert events[0].claimed_at is None

    async def test_invalid_token_rejected(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge

        bad_payload = {**_WORKOUT_PAYLOAD, "webhook_token": "wrong-token"}
        resp = await bridge_client.post("/webhook", json=bad_payload)
        assert resp.status_code == 403

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_non_workout_event_not_queued(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge

        other_payload = {**_WORKOUT_PAYLOAD, "event_type": "user_update"}
        resp = await bridge_client.post("/webhook", json=other_payload)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []


class TestPollerHappyPath:
    async def test_webhook_processed_and_claimed(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge
        mock_process = AsyncMock()

        # Deliver a webhook to the bridge
        resp = await bridge_client.post("/webhook", json=_WORKOUT_PAYLOAD)
        assert resp.status_code == 200

        # Run the poller once
        await _poll(mock_process)

        # process_wahoo_webhook called with the full payload
        mock_process.assert_called_once()
        called_payload = mock_process.call_args[0][0]
        assert called_payload["workout_summary"]["id"] == 12341234

        # Event is now claimed
        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            event = result.scalars().first()
        assert event.claimed_at is not None


class TestBackendOfflineRecovery:
    async def test_queued_events_all_processed_when_backend_comes_online(
        self, bridge_client, patched_bridge
    ):
        """
        Events accumulate in the bridge while the backend is offline.
        When the backend calls _poll_bridge_once() for the first time,
        all queued events are processed and claimed.
        """
        _, sessions = patched_bridge
        mock_process = AsyncMock()

        # Three events arrive while backend is offline (poller not running)
        payloads = [
            {**_WORKOUT_PAYLOAD, "workout_summary": {**_WORKOUT_PAYLOAD["workout_summary"], "id": i}}
            for i in [1001, 1002, 1003]
        ]
        for p in payloads:
            resp = await bridge_client.post("/webhook", json=p)
            assert resp.status_code == 200

        # Verify all three are pending
        async with sessions() as s:
            result = await s.execute(
                select(WebhookEvent).where(WebhookEvent.claimed_at.is_(None))
            )
            pending = result.scalars().all()
        assert len(pending) == 3

        # Backend comes online — single poll drains the queue
        await _poll(mock_process)

        assert mock_process.call_count == 3

        # All events are now claimed
        async with sessions() as s:
            result = await s.execute(
                select(WebhookEvent).where(WebhookEvent.claimed_at.is_(None))
            )
            still_pending = result.scalars().all()
        assert still_pending == []

    async def test_claimed_events_not_reprocessed(self, bridge_client, patched_bridge):
        """
        Running the poller a second time must not reprocess already-claimed events.
        """
        _, sessions = patched_bridge
        mock_process = AsyncMock()

        await bridge_client.post("/webhook", json=_WORKOUT_PAYLOAD)

        # First poll: processes and claims the event
        await _poll(mock_process)
        assert mock_process.call_count == 1

        # Second poll: event already claimed, nothing new to process
        await _poll(mock_process)
        assert mock_process.call_count == 1  # unchanged
