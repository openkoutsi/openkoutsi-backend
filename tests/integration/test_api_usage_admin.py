"""Integration tests for the third-party API-usage admin endpoints (issue #66)."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.app.core.auth import create_access_token, hash_password
from backend.app.models.api_usage_orm import ApiUsage
from backend.app.models.registry_orm import User

_TEST_USER_ID = "test-user-00000000"
OBSERVED_AT = datetime(2026, 9, 10, 10, 14, 0, tzinfo=timezone.utc)


async def _add(factory, **overrides):
    fields = dict(
        created_at=OBSERVED_AT,
        service="strava",
        endpoint="/athlete",
        method="GET",
        status_code=200,
        outcome="ok",
        duration_ms=120,
        user_id=_TEST_USER_ID,
    )
    fields.update(overrides)
    async with factory() as session:
        session.add(ApiUsage(**fields))
        await session.commit()


async def _plain_user_token(registry_session) -> str:
    plain = User(
        id="plain-user-66",
        username="plain66",
        password_hash=hash_password("Testpass1234"),
        roles=json.dumps(["user"]),
    )
    registry_session.add(plain)
    await registry_session.commit()
    return create_access_token("plain-user-66", ["user"], token_version=0)


class TestApiUsageSummary:
    async def test_groups_by_service(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db)
        await _add(api_usage_db, service="wahoo", endpoint="/workouts")
        await _add(api_usage_db, service="wahoo", endpoint="/workouts")

        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=service", headers=auth_headers
        )
        assert resp.status_code == 200
        buckets = {b["key"]: b for b in resp.json()["buckets"]}
        assert buckets["strava"]["calls"] == 1
        assert buckets["wahoo"]["calls"] == 2

    async def test_groups_by_endpoint_template(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db, endpoint="/activities/{id}/streams")
        await _add(api_usage_db, endpoint="/activities/{id}/streams")

        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=endpoint", headers=auth_headers
        )
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == "/activities/{id}/streams"
        assert bucket["calls"] == 2

    async def test_outcome_breakdown_travels_with_the_count(
        self, client, auth_headers, api_usage_db
    ):
        await _add(api_usage_db)
        await _add(api_usage_db, status_code=429, outcome="rate_limited")
        await _add(api_usage_db, status_code=500, outcome="server_error")
        await _add(api_usage_db, status_code=None, outcome="transport_error")

        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=service", headers=auth_headers
        )
        (bucket,) = resp.json()["buckets"]
        assert bucket["calls"] == 4
        assert bucket["ok"] == 1
        assert bucket["rate_limited"] == 1
        assert bucket["server_error"] == 1
        assert bucket["transport_error"] == 1

    @pytest.mark.parametrize("group_by", ["day", "week", "month"])
    async def test_time_buckets(self, client, auth_headers, api_usage_db, group_by):
        await _add(api_usage_db)
        resp = await client.get(
            f"/api/admin/api-usage/summary?group_by={group_by}", headers=auth_headers
        )
        assert resp.status_code == 200
        (bucket,) = resp.json()["buckets"]
        assert bucket["calls"] == 1
        assert bucket["key"]

    async def test_day_bucket_matches_the_row_date(
        self, client, auth_headers, api_usage_db
    ):
        await _add(api_usage_db)
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=day", headers=auth_headers
        )
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == "2026-09-10"

    async def test_groups_by_user(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db)
        await _add(api_usage_db, user_id=None)
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=user", headers=auth_headers
        )
        buckets = {b["key"]: b["calls"] for b in resp.json()["buckets"]}
        assert buckets[_TEST_USER_ID] == 1
        assert buckets[None] == 1

    async def test_service_filter(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db)
        await _add(api_usage_db, service="lettermint", endpoint="send/verification")
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=endpoint&service=lettermint",
            headers=auth_headers,
        )
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == "send/verification"

    async def test_date_range_filter(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db)
        await _add(api_usage_db, created_at=OBSERVED_AT - timedelta(days=40))
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=service&from=2026-09-01",
            headers=auth_headers,
        )
        (bucket,) = resp.json()["buckets"]
        assert bucket["calls"] == 1

    async def test_bad_group_by_is_400(self, client, auth_headers, api_usage_db):
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=nonsense", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_bad_date_is_400(self, client, auth_headers, api_usage_db):
        resp = await client.get(
            "/api/admin/api-usage/summary?from=not-a-date", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_non_admin_is_403(self, client, registry_session, api_usage_db):
        token = await _plain_user_token(registry_session)
        resp = await client.get(
            "/api/admin/api-usage/summary",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestQuotaHeadroom:
    async def test_reports_usage_against_the_limit(
        self, client, auth_headers, api_usage_db
    ):
        await _add(
            api_usage_db,
            created_at=datetime.now(timezone.utc),
            ratelimit_usage_short=100,
            ratelimit_limit_short=600,
            ratelimit_usage_daily=2000,
            ratelimit_limit_daily=30000,
        )
        resp = await client.get("/api/admin/quota/headroom", headers=auth_headers)
        assert resp.status_code == 200
        services = {s["service"]: s for s in resp.json()["services"]}

        strava = services["strava"]
        assert strava["observed_at"] is not None
        assert strava["age_seconds"] < 60
        short = next(
            w for w in strava["windows"] if w["scope"] == "overall" and w["window"] == "short"
        )
        assert short["usage"] == 100
        assert short["limit"] == 600
        assert short["remaining"] == 500
        assert short["observed_in_window"] is True

    async def test_a_service_with_no_observation_says_so(
        self, client, auth_headers, api_usage_db
    ):
        resp = await client.get("/api/admin/quota/headroom", headers=auth_headers)
        services = {s["service"]: s for s in resp.json()["services"]}
        assert set(services) == {"strava", "wahoo"}
        assert services["wahoo"]["observed_at"] is None
        assert services["wahoo"]["windows"] == []

    async def test_a_reading_from_a_past_window_shows_as_reset(
        self, client, auth_headers, api_usage_db
    ):
        # An hour old: several short windows have passed since.
        await _add(
            api_usage_db,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            ratelimit_usage_short=580,
            ratelimit_limit_short=600,
        )
        resp = await client.get("/api/admin/quota/headroom", headers=auth_headers)
        strava = next(
            s for s in resp.json()["services"] if s["service"] == "strava"
        )
        short = next(w for w in strava["windows"] if w["window"] == "short")
        assert short["observed_in_window"] is False
        assert short["usage"] == 0
        assert short["remaining"] == 600

    async def test_last_429_is_surfaced(self, client, auth_headers, api_usage_db):
        await _add(
            api_usage_db,
            created_at=datetime.now(timezone.utc),
            status_code=429,
            outcome="rate_limited",
            ratelimit_usage_short=600,
            ratelimit_limit_short=600,
        )
        resp = await client.get("/api/admin/quota/headroom", headers=auth_headers)
        strava = next(s for s in resp.json()["services"] if s["service"] == "strava")
        assert strava["last_rate_limited_at"] is not None

    async def test_non_admin_is_403(self, client, registry_session, api_usage_db):
        token = await _plain_user_token(registry_session)
        resp = await client.get(
            "/api/admin/quota/headroom",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


class TestWebhookUsageSummary:
    """The bridges are stubbed: what is under test is the proxy and its bucketing."""

    @staticmethod
    def _bridge(stats):
        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=stats)
        return lambda *args, **kwargs: stub

    async def test_proxies_and_buckets_by_day(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        rows = [
            {"day": "2026-09-10", "outcome": "accepted", "count": 12},
            {"day": "2026-09-10", "outcome": "rejected", "count": 1},
        ]
        with patch("backend.app.api.admin.BridgeClient", self._bridge(rows)):
            resp = await client.get(
                "/api/admin/webhook-usage/summary?group_by=day", headers=auth_headers
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["unavailable"] == []
        counts = {(b["key"], b["outcome"]): b["count"] for b in body["buckets"]}
        assert counts[("2026-09-10", "accepted")] == 12
        assert counts[("2026-09-10", "rejected")] == 1

    async def test_month_bucketing_collapses_days(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        rows = [
            {"day": "2026-09-01", "outcome": "accepted", "count": 5},
            {"day": "2026-09-20", "outcome": "accepted", "count": 7},
        ]
        with patch("backend.app.api.admin.BridgeClient", self._bridge(rows)):
            resp = await client.get(
                "/api/admin/webhook-usage/summary?group_by=month", headers=auth_headers
            )
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == "2026-09"
        assert bucket["count"] == 12

    async def test_an_unreachable_bridge_is_named_not_silently_zero(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        with patch("backend.app.api.admin.BridgeClient", self._bridge(None)):
            resp = await client.get(
                "/api/admin/webhook-usage/summary", headers=auth_headers
            )
        body = resp.json()
        assert body["unavailable"] == ["strava"]
        assert body["buckets"] == []

    async def test_unconfigured_bridge_is_skipped(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")
        resp = await client.get(
            "/api/admin/webhook-usage/summary", headers=auth_headers
        )
        body = resp.json()
        assert body["buckets"] == []
        assert body["unavailable"] == []

    async def test_bad_group_by_is_400(self, client, auth_headers, api_usage_db):
        resp = await client.get(
            "/api/admin/webhook-usage/summary?group_by=nonsense", headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_non_admin_is_403(self, client, registry_session, api_usage_db):
        token = await _plain_user_token(registry_session)
        resp = await client.get(
            "/api/admin/webhook-usage/summary",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
