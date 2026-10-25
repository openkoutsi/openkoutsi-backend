"""
Integration tests for the Hammerhead Bridge (issue #69).

Both apps run in-process via ASGITransport — no real network or server. The
bridge uses a fresh in-memory SQLite database per test.

Covered:
  * signature: accepted (hex and base64), rejected (wrong, missing, tampered
    body), and refused outright when no webhook secret is configured;
  * the start-up refusal on a placeholder or short bridge secret;
  * queue → claim → ack round trip through the main app's poller, and nack on
    a processing failure.
"""
import base64
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_real_AsyncClient = httpx.AsyncClient

import hammerhead_bridge.main as bridge_module
from hammerhead_bridge.main import Base as BridgeBase
from hammerhead_bridge.main import WebhookEvent
from hammerhead_bridge.main import app as bridge_app
from backend.app.api.hammerhead import _poll_bridge_once

BRIDGE_SECRET = "test-hammerhead-bridge-secret-0123456789"
WEBHOOK_SECRET = "test-hammerhead-webhook-secret"
BRIDGE_BASE_URL = "http://hammerhead-bridge"

_PAYLOAD = {"activityId": "act-123", "userId": "hh-user-1"}


def _sign_hex(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _sign_b64(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


@pytest.fixture
async def bridge_db():
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
        patch.object(bridge_module.settings, "hammerhead_webhook_secret", WEBHOOK_SECRET),
        patch.object(bridge_module.settings, "hammerhead_bridge_secret", BRIDGE_SECRET),
    ):
        yield eng, sessions


@pytest.fixture
async def bridge_client(patched_bridge):
    async with AsyncClient(
        transport=ASGITransport(app=bridge_app), base_url=BRIDGE_BASE_URL
    ) as c:
        yield c


async def _deliver(client, payload=_PAYLOAD, *, signature=None, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    sig = _sign_hex(body) if signature is None else signature
    if sig:
        headers["X-Hmac-Signature"] = sig
    return await client.post("/webhook", content=body, headers=headers)


async def _queued(sessions) -> list[WebhookEvent]:
    async with sessions() as s:
        return (await s.execute(select(WebhookEvent))).scalars().all()


# ── Signature ─────────────────────────────────────────────────────────────


class TestSignature:
    async def test_valid_hex_signature_is_queued(self, bridge_client, patched_bridge):
        resp = await _deliver(bridge_client)
        assert resp.status_code == 200
        events = await _queued(patched_bridge[1])
        assert len(events) == 1
        assert events[0].owner_id == "hh-user-1"
        assert events[0].payload == _PAYLOAD

    async def test_valid_base64_signature_is_queued(self, bridge_client, patched_bridge):
        body = json.dumps(_PAYLOAD).encode()
        resp = await _deliver(bridge_client, signature=_sign_b64(body))
        assert resp.status_code == 200
        assert len(await _queued(patched_bridge[1])) == 1

    async def test_prefixed_signature_is_accepted(self, bridge_client, patched_bridge):
        body = json.dumps(_PAYLOAD).encode()
        resp = await _deliver(bridge_client, signature=f"sha256={_sign_hex(body)}")
        assert resp.status_code == 200

    async def test_wrong_secret_is_rejected(self, bridge_client, patched_bridge):
        body = json.dumps(_PAYLOAD).encode()
        resp = await _deliver(bridge_client, signature=_sign_hex(body, "nope"))
        assert resp.status_code == 401
        assert await _queued(patched_bridge[1]) == []

    async def test_missing_signature_is_rejected(self, bridge_client, patched_bridge):
        resp = await _deliver(bridge_client, signature="")
        assert resp.status_code == 401
        assert await _queued(patched_bridge[1]) == []

    async def test_signature_is_over_the_raw_bytes(self, bridge_client, patched_bridge):
        """Same JSON, different whitespace: the signature of one does not fit the other."""
        compact = json.dumps(_PAYLOAD, separators=(",", ":")).encode()
        spaced = json.dumps(_PAYLOAD, indent=2).encode()
        resp = await _deliver(bridge_client, raw=spaced, signature=_sign_hex(compact))
        assert resp.status_code == 401

    async def test_non_ascii_signature_is_a_401_not_a_500(self, bridge_client, patched_bridge):
        body = json.dumps(_PAYLOAD).encode()
        resp = await bridge_client.post(
            "/webhook", content=body, headers=[(b"X-Hmac-Signature", b"\xfc\xfc\xfc")]
        )
        assert resp.status_code == 401

    async def test_unconfigured_secret_refuses_everything(self, bridge_client, patched_bridge):
        with patch.object(bridge_module.settings, "hammerhead_webhook_secret", ""):
            resp = await _deliver(bridge_client)
        assert resp.status_code == 403
        assert await _queued(patched_bridge[1]) == []

    async def test_signed_event_without_ids_is_ignored(self, bridge_client, patched_bridge):
        resp = await _deliver(bridge_client, payload={"type": "ping"})
        assert resp.status_code == 200
        assert await _queued(patched_bridge[1]) == []


class TestBridgeSecretRequired:
    @pytest.mark.parametrize("value", ["changeme", "", "short", "b" * 31])
    def test_weak_secret_refuses_to_start(self, value):
        with pytest.raises(ValueError) as exc_info:
            bridge_module.Settings(hammerhead_bridge_secret=value)
        assert "HAMMERHEAD_BRIDGE_SECRET" in str(exc_info.value)

    def test_strong_secret_is_accepted(self):
        s = bridge_module.Settings(hammerhead_bridge_secret="b" * 32)
        assert s.hammerhead_bridge_secret == "b" * 32


# ── Queue / claim / ack through the poller ─────────────────────────────────


def _polling_client_class():
    class _BridgeAsgiClient:
        def __init__(self, **kwargs):
            self._inner = _real_AsyncClient(transport=ASGITransport(app=bridge_app))

        async def __aenter__(self):
            await self._inner.__aenter__()
            return self._inner

        async def __aexit__(self, *args):
            await self._inner.__aexit__(*args)

    return _BridgeAsgiClient


async def _poll(mock_process: AsyncMock) -> None:
    mock_settings = MagicMock()
    mock_settings.hammerhead_bridge_url = BRIDGE_BASE_URL
    mock_settings.hammerhead_bridge_secret = BRIDGE_SECRET
    with (
        patch("backend.app.api.hammerhead.httpx.AsyncClient", new=_polling_client_class()),
        patch("backend.app.api.hammerhead.settings", mock_settings),
        patch("backend.app.api.hammerhead.process_hammerhead_webhook", mock_process),
    ):
        await _poll_bridge_once()


class TestPollerRoundTrip:
    async def test_event_is_processed_and_acked(self, bridge_client, patched_bridge):
        await _deliver(bridge_client)
        process = AsyncMock()
        await _poll(process)
        process.assert_awaited_once_with(_PAYLOAD)

        events = await _queued(patched_bridge[1])
        assert events[0].claimed_at is not None  # acked: terminal

        # Nothing left to serve on the next poll.
        again = AsyncMock()
        await _poll(again)
        again.assert_not_awaited()

    async def test_failed_event_is_nacked_back_to_the_queue(
        self, bridge_client, patched_bridge
    ):
        await _deliver(bridge_client)
        await _poll(AsyncMock(side_effect=RuntimeError("provider down")))

        events = await _queued(patched_bridge[1])
        assert events[0].claimed_at is None  # not retired
        assert events[0].claim_token is None  # released, behind a backoff
        assert events[0].claim_expires_at is not None
        assert events[0].attempts == 1

    async def test_bearer_is_required_to_claim(self, bridge_client, patched_bridge):
        resp = await bridge_client.post("/events/claim")
        assert resp.status_code == 401
        resp = await bridge_client.post(
            "/events/claim", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
        )
        assert resp.status_code == 200

    async def test_stats_name_the_provider(self, bridge_client, patched_bridge):
        await _deliver(bridge_client)
        await _deliver(bridge_client, signature="bad")
        resp = await bridge_client.get(
            "/stats", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
        )
        body = resp.json()
        assert body["provider"] == "hammerhead"
        outcomes = {row["outcome"]: row["count"] for row in body["stats"]}
        assert outcomes == {"accepted": 1, "rejected": 1}


class TestRetryBudget:
    """Hammerhead never redelivers, so the queue's retries must outlast an incident."""

    def test_the_budget_spans_hours_not_minutes(self):
        attempts = bridge_module.Settings(hammerhead_bridge_secret="b" * 32).max_delivery_attempts
        delays = [
            min(bridge_module._NACK_BASE_BACKOFF_S * 2 ** (n - 1), bridge_module._NACK_MAX_BACKOFF_S)
            for n in range(1, attempts + 1)
        ]
        # Every delay at least one poll interval, so none is wasted between polls.
        assert min(delays) >= 60
        assert sum(delays) >= 2 * 3600

    async def test_a_nack_waits_at_least_a_poll(self, bridge_client, patched_bridge):
        await _deliver(bridge_client)
        await _poll(AsyncMock(side_effect=RuntimeError("down")))
        event = (await _queued(patched_bridge[1]))[0]
        wait = (
            event.claim_expires_at.replace(tzinfo=None)
            - event.received_at.replace(tzinfo=None)
        ).total_seconds()
        assert wait >= 59

    async def test_a_retired_event_is_named_in_the_log(
        self, bridge_client, patched_bridge, caplog
    ):
        from sqlalchemy import update as sa_update

        await _deliver(bridge_client)
        async with patched_bridge[1]() as s:
            await s.execute(sa_update(WebhookEvent).values(
                attempts=bridge_module.settings.max_delivery_attempts
            ))
            await s.commit()
        with caplog.at_level("ERROR", logger=bridge_module.log.name):
            resp = await bridge_client.post(
                "/events/claim", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
            )
        assert resp.json() == []
        assert "hh-user-1" in caplog.text and "act-123" in caplog.text

    async def test_the_last_attempt_is_actually_delivered(self, bridge_client, patched_bridge):
        """Retiring after the claim used to swallow the final try before anyone saw it."""
        from sqlalchemy import update as sa_update

        await _deliver(bridge_client)
        async with patched_bridge[1]() as s:
            await s.execute(sa_update(WebhookEvent).values(
                attempts=bridge_module.settings.max_delivery_attempts - 1
            ))
            await s.commit()
        resp = await bridge_client.post(
            "/events/claim", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
        )
        assert len(resp.json()) == 1
