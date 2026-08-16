"""
Integration tests for the Strava Bridge round-trip.

Both apps run in-process via ASGITransport — no real network or server.
The bridge uses a fresh in-memory SQLite database per test.

Scenarios covered:
  1. Happy path: webhook → bridge → poller → process_webhook_event called
  2. Invalid HMAC signature rejected
  3. Missing HMAC signature rejected (F-01: the check used to fail open)
  4. Unconfigured client secret rejects everything (fail closed)
  5. Non-activity events not queued
  6. Hub challenge verification
  7. Backend offline: events accumulate, then all processed when poller runs
  8. Claimed events not reprocessed on subsequent poll
"""
import hashlib
import hmac as hmac_mod
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy import select

import strava_bridge.main as bridge_module
from strava_bridge.main import app as bridge_app, Base as BridgeBase, WebhookEvent
from backend.app.api.strava import _poll_bridge_once

BRIDGE_SECRET = "test-bridge-secret"
STRAVA_CLIENT_SECRET = "test-strava-secret"
BRIDGE_BASE_URL = "http://strava-bridge"

# Captured before any patching so _BridgeAsgiClient can always reach the real class.
_real_AsyncClient = httpx.AsyncClient

_ACTIVITY_PAYLOAD = {
    "object_type": "activity",
    "aspect_type": "create",
    "object_id": 99887766,
    "owner_id": 12345678,
    "subscription_id": 1,
    "event_time": 1745000000,
}


def _make_signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac_mod.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()


async def _post_signed(client: AsyncClient, payload: dict):
    """
    POST a payload to /webhook, signed like Strava signs it.

    The signature covers the exact bytes sent, so the body is serialised here
    rather than handed to httpx's `json=` — re-serialising would change it.
    """
    body = json.dumps(payload).encode()
    return await client.post(
        "/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": _make_signature(body, STRAVA_CLIENT_SECRET),
        },
    )


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
        patch.object(bridge_module.settings, "strava_client_secret", STRAVA_CLIENT_SECRET),
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
    HTTP and process_webhook_event replaced by mock_process.
    """
    mock_settings = MagicMock()
    mock_settings.bridge_url = BRIDGE_BASE_URL
    mock_settings.bridge_secret = BRIDGE_SECRET

    BridgeAsgiClient = _make_polling_client_class()
    with (
        patch("backend.app.api.strava.httpx.AsyncClient", new=BridgeAsgiClient),
        patch("backend.app.api.strava.settings", mock_settings),
        patch("backend.app.api.strava.process_webhook_event", mock_process),
    ):
        await _poll_bridge_once()


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBridgeWebhookEndpoint:
    async def test_valid_activity_event_queued_with_hmac(
        self, bridge_client, patched_bridge
    ):
        _, sessions = patched_bridge

        resp = await _post_signed(bridge_client, _ACTIVITY_PAYLOAD)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()

        assert len(events) == 1
        assert events[0].strava_event_type == "create"
        assert events[0].strava_owner_id == "12345678"
        assert events[0].claimed_at is None

    async def test_invalid_hmac_rejected(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge

        resp = await bridge_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": "sha256=badbadbadbad"},
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_missing_hmac_rejected(self, bridge_client, patched_bridge):
        """
        F-01: the check used to `return True` when the header was absent, so any
        unauthenticated caller could queue events simply by omitting it.
        """
        _, sessions = patched_bridge

        resp = await bridge_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_empty_hmac_header_rejected(self, bridge_client, patched_bridge):
        """An empty header is as unauthenticated as an absent one."""
        _, sessions = patched_bridge

        resp = await bridge_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": ""},
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_non_ascii_hmac_header_rejected(self, bridge_client, patched_bridge):
        """
        The header is attacker-supplied. compare_digest raises TypeError on a
        str holding non-ASCII, which would turn this into a 500 generator.

        Sent as raw bytes because httpx refuses to encode a non-ASCII str
        header; Starlette decodes it back to a non-ASCII str, which is exactly
        what a hand-rolled client puts in front of the comparison.
        """
        _, sessions = patched_bridge

        resp = await bridge_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": b"sha256=\xfc\xfc\xfc\xfc"},
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_signature_over_different_body_rejected(
        self, bridge_client, patched_bridge
    ):
        """A signature lifted from one request must not authenticate another."""
        _, sessions = patched_bridge

        signed_body = json.dumps(_ACTIVITY_PAYLOAD).encode()
        tampered = json.dumps({**_ACTIVITY_PAYLOAD, "aspect_type": "delete"}).encode()

        resp = await bridge_client.post(
            "/webhook",
            content=tampered,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _make_signature(
                    signed_body, STRAVA_CLIENT_SECRET
                ),
            },
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_unconfigured_secret_rejects_signed_and_unsigned(
        self, bridge_client, patched_bridge
    ):
        """
        With no client secret there is nothing to authenticate against, so the
        bridge fails closed rather than accepting everything. Matches the Wahoo
        bridge, which refuses outright when its token is unconfigured.
        """
        _, sessions = patched_bridge

        with patch.object(bridge_module.settings, "strava_client_secret", ""):
            unsigned = await bridge_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
            assert unsigned.status_code == 403

            signed = await _post_signed(bridge_client, _ACTIVITY_PAYLOAD)
            assert signed.status_code == 403

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_non_activity_event_not_queued(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge

        athlete_payload = {**_ACTIVITY_PAYLOAD, "object_type": "athlete"}
        resp = await _post_signed(bridge_client, athlete_payload)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_hub_challenge_verification(self, bridge_client, patched_bridge):
        resp = await bridge_client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": BRIDGE_SECRET,
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"hub.challenge": "abc123"}

    async def test_hub_challenge_wrong_token_rejected(
        self, bridge_client, patched_bridge
    ):
        resp = await bridge_client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong-token",
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 403


@pytest.mark.replica_unsafe  # assumes a single poller, claiming after it processes
class TestPollerHappyPath:
    async def test_webhook_processed_and_claimed(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge
        mock_process = AsyncMock()

        resp = await _post_signed(bridge_client, _ACTIVITY_PAYLOAD)
        assert resp.status_code == 200

        await _poll(mock_process)

        mock_process.assert_called_once()
        called_event = mock_process.call_args[0][0]
        assert called_event["strava_event_type"] == "create"
        assert called_event["strava_owner_id"] == "12345678"

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            event = result.scalars().first()
        assert event.claimed_at is not None


@pytest.mark.replica_unsafe  # assumes a single poller, claiming after it processes
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

        payloads = [
            {**_ACTIVITY_PAYLOAD, "object_id": oid}
            for oid in [1001, 1002, 1003]
        ]
        for p in payloads:
            resp = await _post_signed(bridge_client, p)
            assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(
                select(WebhookEvent).where(WebhookEvent.claimed_at.is_(None))
            )
            pending = result.scalars().all()
        assert len(pending) == 3

        # Backend comes online — single poll drains the queue
        await _poll(mock_process)

        assert mock_process.call_count == 3

        async with sessions() as s:
            result = await s.execute(
                select(WebhookEvent).where(WebhookEvent.claimed_at.is_(None))
            )
            still_pending = result.scalars().all()
        assert still_pending == []

    async def test_claimed_events_not_reprocessed(self, bridge_client, patched_bridge):
        """Running the poller a second time must not reprocess already-claimed events."""
        _, sessions = patched_bridge
        mock_process = AsyncMock()

        await _post_signed(bridge_client, _ACTIVITY_PAYLOAD)

        await _poll(mock_process)
        assert mock_process.call_count == 1

        await _poll(mock_process)
        assert mock_process.call_count == 1  # unchanged
