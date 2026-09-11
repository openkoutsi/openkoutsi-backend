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

    async def test_to_filter_excludes_later_rows(self, client, auth_headers, api_usage_db):
        await _add(api_usage_db)
        await _add(api_usage_db, created_at=OBSERVED_AT + timedelta(days=40))
        resp = await client.get(
            "/api/admin/api-usage/summary?group_by=service&to=2026-09-11",
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

    @pytest.mark.parametrize(
        "group_by,expected_key",
        [
            # Every group_by the endpoint offers; each takes a different branch
            # of the bucketing, and a wrong one silently mislabels the table.
            ("provider", "strava"),
            ("outcome", "accepted"),
            ("week", "2026-W35"),
            ("day", "2026-09-01"),
            ("month", "2026-09"),
        ],
    )
    async def test_every_group_by_buckets_correctly(
        self, client, auth_headers, monkeypatch, api_usage_db, group_by, expected_key
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        rows = [{"day": "2026-09-01", "outcome": "accepted", "count": 5}]
        with patch("backend.app.api.admin.BridgeClient", self._bridge(rows)):
            resp = await client.get(
                f"/api/admin/webhook-usage/summary?group_by={group_by}",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == expected_key
        assert bucket["count"] == 5

    async def _two_by_two_by_two(self, client, auth_headers, monkeypatch, group_by):
        """Two providers x two outcomes x two days, so aggregation is visible.

        A single row makes every ``group_by`` look like it aggregates, which is
        what let the collapse bug hide.
        """
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "https://wahoo.test")
        monkeypatch.setattr(settings, "wahoo_bridge_secret", "s3cret")

        rows = [
            {"day": "2026-09-01", "outcome": "accepted", "count": 1},
            {"day": "2026-09-01", "outcome": "rejected", "count": 2},
            {"day": "2026-09-02", "outcome": "accepted", "count": 4},
            {"day": "2026-09-02", "outcome": "rejected", "count": 8},
        ]
        # Both bridges answer the same shape, so every total is doubled.
        with patch("backend.app.api.admin.BridgeClient", self._bridge(rows)):
            resp = await client.get(
                f"/api/admin/webhook-usage/summary?group_by={group_by}",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        return resp.json()["buckets"]

    async def test_group_by_provider_collapses_outcomes(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """One row per provider, not one per (provider, outcome).

        Before this was literal, both providers emitted two buckets each sharing
        a key — a client rendering key -> count showed duplicate keys and a total
        that looked wrong.
        """
        buckets = await self._two_by_two_by_two(
            client, auth_headers, monkeypatch, "provider"
        )
        by_key = {b["key"]: b for b in buckets}
        assert sorted(by_key) == ["strava", "wahoo"], "one bucket per provider"
        assert by_key["strava"]["count"] == 15  # 1 + 2 + 4 + 8
        assert by_key["wahoo"]["count"] == 15
        assert all(b["outcome"] is None for b in buckets), "outcomes collapsed"

    async def test_group_by_outcome_collapses_providers(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        buckets = await self._two_by_two_by_two(
            client, auth_headers, monkeypatch, "outcome"
        )
        by_key = {b["key"]: b for b in buckets}
        assert sorted(by_key) == ["accepted", "rejected"]
        assert by_key["accepted"]["count"] == 10  # (1 + 4) from each of two bridges
        assert by_key["rejected"]["count"] == 20  # (2 + 8) from each of two bridges
        assert all(b["provider"] is None for b in buckets), "providers collapsed"

    async def test_provider_and_outcome_are_not_the_same_table(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        # They used to return identical row sets differing only in `key`.
        by_provider = await self._two_by_two_by_two(
            client, auth_headers, monkeypatch, "provider"
        )
        by_outcome = await self._two_by_two_by_two(
            client, auth_headers, monkeypatch, "outcome"
        )
        assert {b["key"] for b in by_provider} != {b["key"] for b in by_outcome}

    async def test_time_buckets_keep_the_outcome_breakdown(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """A daily total that hides accepted-vs-rejected is not worth reading."""
        buckets = await self._two_by_two_by_two(
            client, auth_headers, monkeypatch, "day"
        )
        seen = {(b["key"], b["provider"], b["outcome"]): b["count"] for b in buckets}
        assert seen[("2026-09-01", "strava", "accepted")] == 1
        assert seen[("2026-09-02", "wahoo", "rejected")] == 8
        assert len(seen) == 8, "2 days x 2 providers x 2 outcomes"

    @pytest.mark.parametrize("param", ["from", "to"])
    async def test_iso_datetimes_are_narrowed_to_a_day(
        self, client, auth_headers, monkeypatch, api_usage_db, param
    ):
        """The bridge compares day strings, so a time component must not survive.

        `"2026-09-10" >= "2026-09-10T12:00:00"` is False, so forwarding the raw
        value silently dropped a whole day at the `from` end and kept it at the
        `to` end — asymmetric, and neither what was asked for.
        """
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=[])
        with patch(
            "backend.app.api.admin.BridgeClient", lambda *a, **k: stub
        ):
            resp = await client.get(
                f"/api/admin/webhook-usage/summary?{param}=2026-09-10T12:00:00",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        forwarded = stub.stats.await_args.kwargs
        assert forwarded[f"{param}_day"] == "2026-09-10"

    async def test_a_day_the_bridge_wrote_that_we_cannot_parse_is_skipped(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """One unreadable row must not take the whole table down with it."""
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        rows = [
            {"day": "not-a-date", "outcome": "accepted", "count": 99},
            {"day": "2026-09-01", "outcome": "accepted", "count": 5},
        ]
        with patch("backend.app.api.admin.BridgeClient", self._bridge(rows)):
            resp = await client.get(
                "/api/admin/webhook-usage/summary?group_by=month", headers=auth_headers
            )
        assert resp.status_code == 200
        (bucket,) = resp.json()["buckets"]
        assert bucket["key"] == "2026-09"
        assert bucket["count"] == 5, "the readable row still counts"

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

    async def test_both_bridges_are_queried_concurrently(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """Sequentially, this request's worst case is the *sum* of the timeouts.

        Long enough for a reverse proxy to give up and turn a partially-degraded
        page into a 504 — on the panel an admin opens precisely when a bridge is
        down.
        """
        import asyncio
        import time

        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "https://wahoo.test")
        monkeypatch.setattr(settings, "wahoo_bridge_secret", "s3cret")

        async def slow_stats(**kwargs):
            await asyncio.sleep(0.3)
            return []

        stub = AsyncMock()
        stub.stats = slow_stats
        started = time.perf_counter()
        with patch("backend.app.api.admin.BridgeClient", lambda *a, **k: stub):
            resp = await client.get(
                "/api/admin/webhook-usage/summary", headers=auth_headers
            )
        elapsed = time.perf_counter() - started

        assert resp.status_code == 200
        assert elapsed < 0.55, (
            f"took {elapsed:.2f}s — two 0.3s bridges ran in series, not parallel"
        )

    async def test_an_unbounded_default_window_would_grow_forever(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """`webhook_stats` is never pruned, so the default load needs a bound."""
        from datetime import date

        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=[])
        with patch("backend.app.api.admin.BridgeClient", lambda *a, **k: stub):
            resp = await client.get(
                "/api/admin/webhook-usage/summary", headers=auth_headers
            )
        assert resp.status_code == 200

        asked_for = stub.stats.await_args.kwargs["from_day"]
        assert asked_for is not None, "no default window"
        days_back = (date.today() - date.fromisoformat(asked_for)).days
        assert days_back == 90

    async def test_the_response_reports_the_window_it_actually_used(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        """Not the caller's raw input, which is None when the default applies.

        Reporting an unbounded window over 90 days of data is the same class of
        mistake `observed_in_window` and `unavailable` exist to prevent.
        """
        from datetime import date

        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=[])
        with patch("backend.app.api.admin.BridgeClient", lambda *a, **k: stub):
            resp = await client.get(
                "/api/admin/webhook-usage/summary", headers=auth_headers
            )
        body = resp.json()
        assert body["from"] is not None, "reported an unbounded window"
        assert body["from"] == stub.stats.await_args.kwargs["from_day"]
        assert (date.today() - date.fromisoformat(body["from"])).days == 90

    async def test_an_iso_datetime_is_reported_as_the_day_it_became(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=[])
        with patch("backend.app.api.admin.BridgeClient", lambda *a, **k: stub):
            resp = await client.get(
                "/api/admin/webhook-usage/summary?from=2026-09-01T12:00:00",
                headers=auth_headers,
            )
        assert resp.json()["from"] == "2026-09-01"

    async def test_an_explicit_from_overrides_the_default_window(
        self, client, auth_headers, monkeypatch, api_usage_db
    ):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "bridge_url", "https://bridge.test")
        monkeypatch.setattr(settings, "bridge_secret", "s3cret")
        monkeypatch.setattr(settings, "wahoo_bridge_url", "")

        stub = AsyncMock()
        stub.stats = AsyncMock(return_value=[])
        with patch("backend.app.api.admin.BridgeClient", lambda *a, **k: stub):
            await client.get(
                "/api/admin/webhook-usage/summary?from=2024-01-01",
                headers=auth_headers,
            )
        assert stub.stats.await_args.kwargs["from_day"] == "2024-01-01"

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
