"""Integration tests for the bridges' webhook delivery counters (issue #66).

The point of counting at the bridge rather than at the backend: deliveries never
reach the backend, and the two available numbers are not interchangeable. The
backend sees *events processed* — which counts nacked redeliveries repeatedly
and never sees events shed by the queue ceiling — while the bridge sees
*deliveries received*, which is what the issue asks for.

The counters are also the only per-month figure that can exist, because both
bridges delete events older than seven days.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import strava_bridge.main as strava_bridge
import wahoo_bridge.main as wahoo_bridge

BRIDGE_SECRET = "test-bridge-secret"
WAHOO_TOKEN = "test-wahoo-webhook-token"

_ACTIVITY = {
    "object_type": "activity",
    "aspect_type": "create",
    "object_id": 99887766,
    "owner_id": 12345678,
}
_ATHLETE = {"object_type": "athlete", "aspect_type": "update", "owner_id": 1}


async def _fresh(module):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(module.Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _stats(client, secret=BRIDGE_SECRET):
    resp = await client.get("/stats", headers={"Authorization": f"Bearer {secret}"})
    assert resp.status_code == 200
    return {(r["day"], r["outcome"]): r["count"] for r in resp.json()["stats"]}


def _counts(stats):
    """Collapse the day dimension — these tests care about outcome totals."""
    out: dict[str, int] = {}
    for (_day, outcome), count in stats.items():
        out[outcome] = out.get(outcome, 0) + count
    return out


# ── Strava bridge ──────────────────────────────────────────────────────────


@pytest.fixture
async def strava_client():
    engine, sessions = await _fresh(strava_bridge)
    with (
        patch.object(strava_bridge, "engine", engine),
        patch.object(strava_bridge, "AsyncSessionLocal", sessions),
        patch.object(strava_bridge.settings, "bridge_secret", BRIDGE_SECRET),
        patch.object(strava_bridge.settings, "strava_verify_webhook_signature", False),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=strava_bridge.app), base_url="http://bridge"
        ) as c:
            yield c, sessions
    await engine.dispose()


class TestStravaBridgeCounters:
    async def test_accepted_deliveries_are_counted(self, strava_client):
        client, _ = strava_client
        for _ in range(3):
            await client.post("/webhook", content=json.dumps(_ACTIVITY))
        assert _counts(await _stats(client)) == {"accepted": 3}

    async def test_ignored_is_kept_apart_from_rejected(self, strava_client):
        # Strava routinely sends athlete events we do not queue. Counting those
        # as rejections would read as a fault where there is none.
        client, _ = strava_client
        await client.post("/webhook", content=json.dumps(_ATHLETE))
        assert _counts(await _stats(client)) == {"ignored": 1}

    async def test_malformed_payload_is_rejected(self, strava_client):
        client, _ = strava_client
        resp = await client.post("/webhook", content="not json at all")
        assert resp.status_code == 400
        assert _counts(await _stats(client)) == {"rejected": 1}

    async def test_verification_handshake_is_counted_separately(self, strava_client):
        client, _ = strava_client
        resp = await client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": BRIDGE_SECRET,
                "hub.challenge": "abc123",
            },
        )
        assert resp.status_code == 200
        await client.post("/webhook", content=json.dumps(_ACTIVITY))
        # The handshake does not inflate the delivery count.
        assert _counts(await _stats(client)) == {"verification": 1, "accepted": 1}

    async def test_failed_handshake_is_a_rejection(self, strava_client):
        client, _ = strava_client
        resp = await client.get(
            "/webhook",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong",
                "hub.challenge": "abc",
            },
        )
        assert resp.status_code == 403
        assert _counts(await _stats(client)) == {"rejected": 1}

    async def test_queue_ceiling_refusal_is_counted(self, strava_client):
        client, _ = strava_client
        with patch.object(strava_bridge.settings, "max_queue_events", 1):
            await client.post("/webhook", content=json.dumps(_ACTIVITY))
            resp = await client.post("/webhook", content=json.dumps(_ACTIVITY))
        assert resp.status_code == 503
        assert _counts(await _stats(client)) == {"accepted": 1, "rejected": 1}

    async def test_counters_survive_the_seven_day_event_prune(self, strava_client):
        """The blocker the counters exist to solve.

        Both bridges delete events older than seven days, so a per-month figure
        cannot come from `webhook_events`. It can come from here.
        """
        client, sessions = strava_client
        await client.post("/webhook", content=json.dumps(_ACTIVITY))

        # Run the same delete the cleanup loop runs, against an aged event.
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        async with sessions() as session:
            event = (
                await session.execute(select(strava_bridge.WebhookEvent))
            ).scalars().one()
            event.received_at = cutoff - timedelta(days=1)
            await session.commit()
            await session.execute(
                delete(strava_bridge.WebhookEvent).where(
                    strava_bridge.WebhookEvent.received_at < cutoff
                )
            )
            await session.commit()
            remaining = (
                await session.execute(select(strava_bridge.WebhookEvent))
            ).scalars().all()

        assert remaining == [], "the event itself is gone"
        assert _counts(await _stats(client)) == {"accepted": 1}, "the count is not"

    async def test_stats_requires_the_bearer_secret(self, strava_client):
        client, _ = strava_client
        assert (await client.get("/stats")).status_code == 401
        resp = await client.get("/stats", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    async def test_day_range_filter(self, strava_client):
        client, sessions = strava_client
        async with sessions() as session:
            session.add_all([
                strava_bridge.WebhookStat(day="2026-08-01", outcome="accepted", count=4),
                strava_bridge.WebhookStat(day="2026-09-01", outcome="accepted", count=9),
            ])
            await session.commit()

        resp = await client.get(
            "/stats",
            params={"from": "2026-09-01"},
            headers={"Authorization": f"Bearer {BRIDGE_SECRET}"},
        )
        assert [r["count"] for r in resp.json()["stats"]] == [9]

    async def test_a_broken_counter_does_not_reject_the_delivery(self, strava_client):
        """A provider does not resend on our account.

        Losing the event to protect the statistic would be exactly backwards, so
        the counter swallows its own failure.
        """
        client, _ = strava_client
        with patch.object(
            strava_bridge, "sqlite_insert", side_effect=RuntimeError("disk full")
        ):
            resp = await client.post("/webhook", content=json.dumps(_ACTIVITY))
        assert resp.status_code == 200


# ── Wahoo bridge ───────────────────────────────────────────────────────────


@pytest.fixture
async def wahoo_client():
    engine, sessions = await _fresh(wahoo_bridge)
    with (
        patch.object(wahoo_bridge, "engine", engine),
        patch.object(wahoo_bridge, "AsyncSessionLocal", sessions),
        patch.object(wahoo_bridge.settings, "wahoo_bridge_secret", BRIDGE_SECRET),
        patch.object(wahoo_bridge.settings, "wahoo_webhook_token", WAHOO_TOKEN),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=wahoo_bridge.app), base_url="http://bridge"
        ) as c:
            yield c, sessions
    await engine.dispose()


def _wahoo_payload(event_type="workout_summary"):
    return json.dumps(
        {
            "event_type": event_type,
            "webhook_token": WAHOO_TOKEN,
            "user": {"id": 4242},
            "workout_summary": {"id": 1},
        }
    )


class TestWahooBridgeCounters:
    async def test_accepted_deliveries_are_counted(self, wahoo_client):
        client, _ = wahoo_client
        await client.post("/webhook", content=_wahoo_payload())
        await client.post("/webhook", content=_wahoo_payload())
        assert _counts(await _stats(client)) == {"accepted": 2}

    async def test_other_event_types_are_ignored_not_rejected(self, wahoo_client):
        client, _ = wahoo_client
        await client.post("/webhook", content=_wahoo_payload("workout_started"))
        assert _counts(await _stats(client)) == {"ignored": 1}

    async def test_a_bad_token_is_rejected(self, wahoo_client):
        client, _ = wahoo_client
        payload = json.dumps({"event_type": "workout_summary", "webhook_token": "wrong"})
        resp = await client.post("/webhook", content=payload)
        assert resp.status_code == 403
        assert _counts(await _stats(client)) == {"rejected": 1}

    async def test_malformed_payload_is_rejected(self, wahoo_client):
        client, _ = wahoo_client
        resp = await client.post("/webhook", content="not json")
        assert resp.status_code == 400
        assert _counts(await _stats(client)) == {"rejected": 1}

    async def test_stats_requires_the_bearer_secret(self, wahoo_client):
        client, _ = wahoo_client
        assert (await client.get("/stats")).status_code == 401

    async def test_provider_is_named_in_the_response(self, wahoo_client):
        client, _ = wahoo_client
        resp = await client.get(
            "/stats", headers={"Authorization": f"Bearer {BRIDGE_SECRET}"}
        )
        assert resp.json()["provider"] == "wahoo"
