"""
Integration tests for the Strava Bridge round-trip.

Both apps run in-process via ASGITransport — no real network or server.
The bridge uses a fresh in-memory SQLite database per test.

Scenarios covered:
  1. Happy path: webhook → bridge → poller → process_webhook_event called
  2. Signature checking off by default — unsigned events are accepted, because
     Strava does not document webhook signing and sends no signature
  3. Signature checking on (STRAVA_VERIFY_WEBHOOK_SIGNATURE=true) still rejects
     a missing, empty, wrong, or replayed signature, and fails closed with no
     secret configured (F-01: the check used to fail open)
  4. Non-activity events not queued
  5. Hub challenge verification
  6. Backend offline: events accumulate, then all processed when poller runs
  7. Claimed events not reprocessed on subsequent poll
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

    Signature verification is pinned off, the way it ships — a deployment that
    turns it on is the exception, and `verifying_bridge` below covers it.
    """
    eng, sessions = bridge_db
    with (
        patch.object(bridge_module, "engine", eng),
        patch.object(bridge_module, "AsyncSessionLocal", sessions),
        patch.object(bridge_module.settings, "strava_client_secret", STRAVA_CLIENT_SECRET),
        patch.object(bridge_module.settings, "bridge_secret", BRIDGE_SECRET),
        patch.object(bridge_module.settings, "strava_verify_webhook_signature", False),
    ):
        yield eng, sessions


@pytest.fixture
async def bridge_client(patched_bridge):
    """AsyncClient wired to the (patched) bridge ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=bridge_app), base_url=BRIDGE_BASE_URL
    ) as c:
        yield c


@pytest.fixture
def verifying_bridge(patched_bridge):
    """The same bridge with STRAVA_VERIFY_WEBHOOK_SIGNATURE turned on."""
    with patch.object(
        bridge_module.settings, "strava_verify_webhook_signature", True
    ):
        yield patched_bridge


@pytest.fixture
async def verifying_client(verifying_bridge):
    """AsyncClient wired to a bridge that requires X-Hub-Signature-256."""
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
    async def test_valid_activity_event_queued(self, bridge_client, patched_bridge):
        """A signed event is still accepted; the header is simply not checked."""
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


class TestWebhookAuthenticationDisabled:
    """
    The signature check ships off, so `POST /webhook` is unauthenticated.

    Strava documents the `hub.challenge` subscription handshake and nothing
    else — no signing secret, no header, no statement of which bytes are
    signed. Verifying an `X-Hub-Signature-256` that Strava never sends, with a
    check that fails closed, answered every real delivery with 401. These tests
    pin the accepting behaviour so it is not quietly re-tightened; turn it back
    on through `STRAVA_VERIFY_WEBHOOK_SIGNATURE` once Strava documents the
    sequence.
    """

    async def test_unsigned_event_is_queued(self, bridge_client, patched_bridge):
        """What Strava actually sends today."""
        _, sessions = patched_bridge

        resp = await bridge_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()

        assert len(events) == 1
        assert events[0].strava_event_type == "create"
        assert events[0].strava_owner_id == "12345678"

    @pytest.mark.parametrize(
        "header",
        ["sha256=badbadbadbad", "", "not-even-a-digest"],
        ids=["wrong", "empty", "malformed"],
    )
    async def test_a_bogus_signature_is_ignored_rather_than_rejected(
        self, bridge_client, patched_bridge, header
    ):
        """The header is not read at all, so its contents cannot fail."""
        _, sessions = patched_bridge

        resp = await bridge_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": header},
        )
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert len(events) == 1

    async def test_non_ascii_signature_header_does_not_error(
        self, bridge_client, patched_bridge
    ):
        """
        An unread header still must not crash the endpoint: the value is
        attacker-supplied, and `compare_digest` raises `TypeError` on a `str`
        holding non-ASCII. Sent as raw bytes because httpx refuses to encode a
        non-ASCII str header; Starlette decodes it back to the non-ASCII str a
        hand-rolled client would put in front of the comparison.
        """
        resp = await bridge_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": b"sha256=\xfc\xfc\xfc\xfc"},
        )
        assert resp.status_code == 200

    async def test_unconfigured_client_secret_still_accepts(
        self, bridge_client, patched_bridge
    ):
        """
        With verification off the secret is unused, so a bridge deployed
        without `STRAVA_CLIENT_SECRET` receives events instead of answering
        403 to all of them.
        """
        _, sessions = patched_bridge

        with patch.object(bridge_module.settings, "strava_client_secret", ""):
            resp = await bridge_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert len(events) == 1

    def test_verification_is_off_in_the_shipped_defaults(self):
        """Not just in the fixture — the setting itself defaults to off."""
        settings = bridge_module.Settings(bridge_secret="b" * 32)
        assert settings.strava_verify_webhook_signature is False


class TestWebhookSignatureOptIn:
    """
    `STRAVA_VERIFY_WEBHOOK_SIGNATURE=true` restores the fail-closed check.

    Kept exercised so the code is still working the day Strava documents
    webhook signing and this becomes the default again.
    """

    async def test_valid_signature_accepted(self, verifying_client, verifying_bridge):
        _, sessions = verifying_bridge

        resp = await _post_signed(verifying_client, _ACTIVITY_PAYLOAD)
        assert resp.status_code == 200

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert len(events) == 1

    async def test_invalid_hmac_rejected(self, verifying_client, verifying_bridge):
        _, sessions = verifying_bridge

        resp = await verifying_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": "sha256=badbadbadbad"},
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_missing_hmac_rejected(self, verifying_client, verifying_bridge):
        """
        F-01: the check used to `return True` when the header was absent, so any
        unauthenticated caller could queue events simply by omitting it. Opting
        in must mean the check is real, not decorative.
        """
        _, sessions = verifying_bridge

        resp = await verifying_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_empty_hmac_header_rejected(self, verifying_client, verifying_bridge):
        """An empty header is as unauthenticated as an absent one."""
        _, sessions = verifying_bridge

        resp = await verifying_client.post(
            "/webhook",
            json=_ACTIVITY_PAYLOAD,
            headers={"X-Hub-Signature-256": ""},
        )
        assert resp.status_code == 401

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []

    async def test_non_ascii_hmac_header_rejected(
        self, verifying_client, verifying_bridge
    ):
        """
        The header is attacker-supplied. compare_digest raises TypeError on a
        str holding non-ASCII, which would turn this into a 500 generator.

        Sent as raw bytes because httpx refuses to encode a non-ASCII str
        header; Starlette decodes it back to a non-ASCII str, which is exactly
        what a hand-rolled client puts in front of the comparison.
        """
        _, sessions = verifying_bridge

        resp = await verifying_client.post(
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
        self, verifying_client, verifying_bridge
    ):
        """A signature lifted from one request must not authenticate another."""
        _, sessions = verifying_bridge

        signed_body = json.dumps(_ACTIVITY_PAYLOAD).encode()
        tampered = json.dumps({**_ACTIVITY_PAYLOAD, "aspect_type": "delete"}).encode()

        resp = await verifying_client.post(
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
        self, verifying_client, verifying_bridge
    ):
        """
        Having opted in to verification with no client secret to verify
        against, there is nothing to authenticate requests with, so the bridge
        fails closed rather than accepting everything. Matches the Wahoo
        bridge, which refuses outright when its token is unconfigured.
        """
        _, sessions = verifying_bridge

        with patch.object(bridge_module.settings, "strava_client_secret", ""):
            unsigned = await verifying_client.post("/webhook", json=_ACTIVITY_PAYLOAD)
            assert unsigned.status_code == 403

            signed = await _post_signed(verifying_client, _ACTIVITY_PAYLOAD)
            assert signed.status_code == 403

        async with sessions() as s:
            result = await s.execute(select(WebhookEvent))
            events = result.scalars().all()
        assert events == []


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
