"""
Integration tests for the Inbound Email Bridge round-trip (issue #38).

Both apps run in-process via ASGITransport — no real network or server. The
bridge uses a fresh in-memory SQLite database per test.

The bridge *holds* verified mail in a queue; the backend *polls* it (like the
Strava/Wahoo bridges), so nothing is lost while the backend is down. Scenarios:

  1. Valid signature → message verified, parsed and queued
  2. Invalid / missing / stale signature rejected (401), nothing queued
  3. Valid signature but bad / incomplete payload rejected (400)
  4. Challenge handshake on GET /
  5. Polling endpoints require the bearer secret
  6. Happy path: webhook → bridge → poller → _deliver called, event claimed
  7. Backend offline: events accumulate, then all processed on first poll
  8. Claimed events not reprocessed on a subsequent poll
  9. _deliver fans a message out to every administrator's inbox (+ snippet cap,
     operator-address filter)
"""
import hashlib
import hmac as hmac_mod
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import inbound_bridge.main as bridge_module
from inbound_bridge.main import app as bridge_app, Base as BridgeBase, InboundEmailEvent
from backend.app.api.inbound import _deliver, _poll_inbound_bridge_once
from backend.app.core.config import settings as app_settings
from backend.app.db.user_session import get_user_session_factory, init_user_db
from backend.app.models.message_orm import Message
from backend.app.services import notifications

# Captured before any patching so the polling client can always reach the real class.
_real_AsyncClient = httpx.AsyncClient

WEBHOOK_SECRET = "test-euromail-webhook-secret"
BRIDGE_SECRET = "test-inbound-bridge-secret"
BRIDGE_BASE_URL = "http://inbound-bridge"
_TEST_USER_ID = "test-user-00000000"

_INBOUND_PAYLOAD = {
    "type": "email.inbound",
    "data": {
        "id": "msg_abc123",
        "from_address": "sender@example.com",
        "to_addresses": ["lassi@koutsi.dev"],
        "subject": "Hello operator",
        "text_body": "Please help me with my account.",
        "created_at": "2026-07-17T10:00:00Z",
    },
}


def _sign(secret: str, body: bytes, ts: int | None = None) -> str:
    ts = int(time.time()) if ts is None else ts
    signed = f"{ts}.{body.decode('utf-8')}".encode()
    sig = hmac_mod.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


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
    eng, sessions = bridge_db
    with (
        patch.object(bridge_module, "engine", eng),
        patch.object(bridge_module, "AsyncSessionLocal", sessions),
        patch.object(bridge_module.settings, "euromail_webhook_secret", WEBHOOK_SECRET),
        patch.object(bridge_module.settings, "inbound_bridge_secret", BRIDGE_SECRET),
    ):
        yield eng, sessions


@pytest.fixture
async def bridge_client(patched_bridge):
    async with AsyncClient(
        transport=ASGITransport(app=bridge_app), base_url=BRIDGE_BASE_URL
    ) as c:
        yield c


def _make_polling_client_class():
    """Drop-in httpx.AsyncClient that routes all requests to the bridge app."""

    class _BridgeAsgiClient:
        def __init__(self, **kwargs):
            self._inner = _real_AsyncClient(transport=ASGITransport(app=bridge_app))

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self._inner

        async def __aexit__(self, *args):
            await self._inner.__aexit__(*args)

    return _BridgeAsgiClient


async def _poll(mock_deliver: AsyncMock) -> None:
    """Run _poll_inbound_bridge_once() against the bridge ASGI app with _deliver mocked."""
    mock_settings = MagicMock()
    mock_settings.inbound_bridge_url = BRIDGE_BASE_URL
    mock_settings.inbound_bridge_secret = BRIDGE_SECRET

    BridgeAsgiClient = _make_polling_client_class()
    with (
        patch("backend.app.api.inbound.httpx.AsyncClient", new=BridgeAsgiClient),
        patch("backend.app.api.inbound.settings", mock_settings),
        patch("backend.app.api.inbound._deliver", mock_deliver),
    ):
        await _poll_inbound_bridge_once()


async def _post_signed(client: AsyncClient, payload: dict) -> httpx.Response:
    body = json.dumps(payload).encode()
    return await client.post(
        "/webhook", content=body, headers={"X-Euromail-Signature": _sign(WEBHOOK_SECRET, body)}
    )


# ── Bridge webhook endpoint ────────────────────────────────────────────────


class TestBridgeWebhookEndpoint:
    async def test_valid_signature_queued(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge
        resp = await _post_signed(bridge_client, _INBOUND_PAYLOAD)
        assert resp.status_code == 202

        async with sessions() as s:
            events = (await s.execute(select(InboundEmailEvent))).scalars().all()
        assert len(events) == 1
        assert events[0].from_addr == "sender@example.com"
        assert events[0].to_addr == "lassi@koutsi.dev"
        assert events[0].subject == "Hello operator"
        assert events[0].message_id == "msg_abc123"
        assert events[0].claimed_at is None

    async def test_invalid_signature_rejected(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge
        body = json.dumps(_INBOUND_PAYLOAD).encode()
        resp = await bridge_client.post(
            "/webhook",
            content=body,
            headers={"X-Euromail-Signature": f"t={int(time.time())},v1=deadbeef"},
        )
        assert resp.status_code == 401
        async with sessions() as s:
            assert (await s.execute(select(InboundEmailEvent))).scalars().all() == []

    async def test_missing_signature_rejected(self, bridge_client, patched_bridge):
        resp = await bridge_client.post(
            "/webhook", content=json.dumps(_INBOUND_PAYLOAD).encode()
        )
        assert resp.status_code == 401

    async def test_stale_timestamp_rejected(self, bridge_client, patched_bridge):
        body = json.dumps(_INBOUND_PAYLOAD).encode()
        sig = _sign(WEBHOOK_SECRET, body, ts=int(time.time()) - 3600)
        resp = await bridge_client.post(
            "/webhook", content=body, headers={"X-Euromail-Signature": sig}
        )
        assert resp.status_code == 401

    async def test_non_json_body_rejected(self, bridge_client, patched_bridge):
        body = b"this is not json"
        sig = _sign(WEBHOOK_SECRET, body)
        resp = await bridge_client.post(
            "/webhook", content=body, headers={"X-Euromail-Signature": sig}
        )
        assert resp.status_code == 400

    async def test_incomplete_payload_rejected(self, bridge_client, patched_bridge):
        resp = await _post_signed(bridge_client, {"data": {"subject": "no sender"}})
        assert resp.status_code == 400


class TestChallengeHandshake:
    async def test_challenge_echoed(self, bridge_client, patched_bridge):
        resp = await bridge_client.get("/", params={"challenge": "abc123"})
        assert resp.status_code == 200
        assert resp.json() == {"challenge": "abc123"}

    async def test_plain_health(self, bridge_client, patched_bridge):
        resp = await bridge_client.get("/")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPollingEndpointsAuth:
    async def test_pending_requires_bearer(self, bridge_client, patched_bridge):
        assert (await bridge_client.get("/events/pending")).status_code == 401
        ok = await bridge_client.get(
            "/events/pending", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
        )
        assert ok.status_code == 200

    async def test_claim_requires_bearer(self, bridge_client, patched_bridge):
        assert (await bridge_client.post("/events/x/claim")).status_code == 401


# ── Poller round-trip ──────────────────────────────────────────────────────


class TestPollerHappyPath:
    async def test_message_delivered_and_claimed(self, bridge_client, patched_bridge):
        _, sessions = patched_bridge
        await _post_signed(bridge_client, _INBOUND_PAYLOAD)

        mock_deliver = AsyncMock()
        await _poll(mock_deliver)

        mock_deliver.assert_awaited_once()
        delivered_event = mock_deliver.await_args[0][1]
        assert delivered_event["from_addr"] == "sender@example.com"
        assert delivered_event["message_id"] == "msg_abc123"

        async with sessions() as s:
            event = (await s.execute(select(InboundEmailEvent))).scalars().first()
        assert event.claimed_at is not None


class TestBackendOfflineRecovery:
    async def test_queued_messages_all_processed_when_backend_polls(
        self, bridge_client, patched_bridge
    ):
        _, sessions = patched_bridge
        for i in range(3):
            await _post_signed(
                bridge_client, {**_INBOUND_PAYLOAD, "data": {**_INBOUND_PAYLOAD["data"], "id": f"m{i}"}}
            )

        async with sessions() as s:
            pending = (
                await s.execute(
                    select(InboundEmailEvent).where(InboundEmailEvent.claimed_at.is_(None))
                )
            ).scalars().all()
        assert len(pending) == 3

        mock_deliver = AsyncMock()
        await _poll(mock_deliver)
        assert mock_deliver.await_count == 3

        async with sessions() as s:
            still_pending = (
                await s.execute(
                    select(InboundEmailEvent).where(InboundEmailEvent.claimed_at.is_(None))
                )
            ).scalars().all()
        assert still_pending == []

    async def test_claimed_events_not_reprocessed(self, bridge_client, patched_bridge):
        await _post_signed(bridge_client, _INBOUND_PAYLOAD)

        mock_deliver = AsyncMock()
        await _poll(mock_deliver)
        assert mock_deliver.await_count == 1

        await _poll(mock_deliver)
        assert mock_deliver.await_count == 1  # unchanged


# ── Fan-out to admin inboxes (_deliver) ────────────────────────────────────


async def _mailbox(user_id: str):
    await init_user_db(user_id)
    async with get_user_session_factory(user_id)() as session:
        return (await session.execute(select(Message))).scalars().all()


_EVENT = {
    "from_addr": "sender@example.com",
    "to_addr": "lassi@koutsi.dev",
    "subject": "Hello operator",
    "text": "Please help me with my account.",
    "message_id": "msg_abc123",
    "received_at": "2026-07-17T10:00:00Z",
}


class TestDeliverFanOut:
    async def test_delivered_to_admin_inbox(self, registry_session, monkeypatch):
        monkeypatch.setattr(app_settings, "inbound_email_address", "")
        await _deliver(registry_session, _EVENT)

        msgs = await _mailbox(_TEST_USER_ID)
        assert len(msgs) == 1
        assert msgs[0].type == notifications.INBOUND_EMAIL
        assert msgs[0].data["from"] == "sender@example.com"
        assert msgs[0].data["subject"] == "Hello operator"
        assert msgs[0].data["snippet"] == "Please help me with my account."
        assert msgs[0].data["message_id"] == "msg_abc123"

    async def test_snippet_truncated(self, registry_session, monkeypatch):
        monkeypatch.setattr(app_settings, "inbound_email_address", "")
        await _deliver(registry_session, {**_EVENT, "text": "x" * 1000})
        msgs = await _mailbox(_TEST_USER_ID)
        assert len(msgs[0].data["snippet"]) <= 501  # 500 chars + ellipsis

    async def test_non_operator_recipient_dropped(self, registry_session, monkeypatch):
        monkeypatch.setattr(app_settings, "inbound_email_address", "lassi@koutsi.dev")
        await _deliver(registry_session, {**_EVENT, "to_addr": "someone-else@koutsi.dev"})
        assert await _mailbox(_TEST_USER_ID) == []

    async def test_operator_match_case_insensitive(self, registry_session, monkeypatch):
        monkeypatch.setattr(app_settings, "inbound_email_address", "Lassi@Koutsi.dev")
        await _deliver(registry_session, _EVENT)
        assert len(await _mailbox(_TEST_USER_ID)) == 1
