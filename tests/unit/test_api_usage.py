"""Unit tests for third-party API-usage recording (issue #66).

Covers the four behaviours that separate this from a plain counter: one row per
HTTP *request* (not per provider method), a transport failure still being a
recorded request, the rate-limit headers being read off every response, and —
the one that matters most — a recording failure never reaching the caller.
"""

import httpx
import pytest
from sqlalchemy import select

from backend.app.models.api_usage_orm import ApiUsage
from backend.app.services import api_usage as api_usage_service
from backend.app.services.api_usage import (
    OUTCOME_CLIENT_ERROR,
    OUTCOME_OK,
    OUTCOME_RATE_LIMITED,
    OUTCOME_SERVER_ERROR,
    OUTCOME_TRANSPORT_ERROR,
    attribute_to_user,
    outcome_for,
    parse_rate_limit,
    record_api_usage,
)
from backend.app.services.providers.http import (
    CountingTransport,
    normalise_endpoint,
    provider_client,
)


async def _rows(factory):
    async with factory() as session:
        return (
            (await session.execute(select(ApiUsage).order_by(ApiUsage.created_at)))
            .scalars()
            .all()
        )


def _client(handler, *, service="strava", user_id=None, **kwargs):
    return httpx.AsyncClient(
        transport=CountingTransport(
            httpx.MockTransport(handler), service=service, user_id=user_id
        ),
        **kwargs,
    )


# ── Endpoint normalisation (the privacy requirement) ───────────────────────


class TestNormaliseEndpoint:
    @pytest.mark.parametrize(
        "url,expected",
        [
            (
                "https://www.strava.com/api/v3/activities/12345/streams?keys=time",
                "/activities/{id}/streams",
            ),
            ("https://www.strava.com/api/v3/athlete/zones", "/athlete/zones"),
            (
                "https://www.strava.com/api/v3/athlete/activities?page=2&per_page=200",
                "/athlete/activities",
            ),
            ("https://www.strava.com/oauth/token", "/oauth/token"),
            (
                "https://api.wahooligan.com/v1/workouts/98765/fit_file",
                "/workouts/{id}/fit_file",
            ),
            ("https://api.wahooligan.com/v1/power_zones", "/power_zones"),
        ],
    )
    def test_templates_replace_ids_and_drop_queries(self, url, expected):
        assert normalise_endpoint(httpx.URL(url)) == expected

    def test_redirect_target_keeps_only_its_host(self):
        # A signed CDN URL: the path is an opaque blob and the query is a
        # credential. Neither may be stored.
        url = httpx.URL(
            "https://cdn.wahooligan.com/a1b2c3d4e5f6a7b8/ride.fit?sig=SECRETVALUE"
        )
        assert normalise_endpoint(url) == "cdn.wahooligan.com/*"

    @pytest.mark.parametrize(
        "url,expected",
        [
            # A trailing slash leaves an empty final segment, and a doubled one
            # an empty middle segment. Neither is an id, and neither may make
            # the template unrecognisable.
            ("https://www.strava.com/api/v3/athlete/", "/athlete/"),
            ("https://www.strava.com/api/v3//athlete", "//athlete"),
        ],
    )
    def test_empty_segments_are_left_alone(self, url, expected):
        assert normalise_endpoint(httpx.URL(url)) == expected

    @pytest.mark.parametrize(
        "segment",
        [
            "550e8400-e29b-41d4-a716-446655440000",  # uuid
            "a1b2c3d4",                              # short hex
            "deadbeefcafef00d",                      # long hex
            "9f8e7d6c5b4a39281706",                  # hex, digits and letters
        ],
    )
    def test_non_numeric_ids_are_replaced_too(self, segment):
        """Not every id is a run of digits.

        The numeric cases short-circuit on ``isdigit()``, so without this the
        uuid/hex branch is never exercised — and that branch is the one standing
        between a provider that switches to opaque ids and a table full of them.
        """
        url = httpx.URL(f"https://api.wahooligan.com/v1/workouts/{segment}/fit_file")
        stored = normalise_endpoint(url)
        assert stored == "/workouts/{id}/fit_file"
        assert segment not in stored

    def test_no_identifier_or_credential_survives(self):
        stored = normalise_endpoint(
            httpx.URL(
                "https://www.strava.com/api/v3/activities/987654321/streams"
                "?access_token=SECRETVALUE"
            )
        )
        assert "987654321" not in stored
        assert "SECRETVALUE" not in stored
        assert "access_token" not in stored


class TestOutcomeFor:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (200, OUTCOME_OK),
            (302, OUTCOME_OK),
            (404, OUTCOME_CLIENT_ERROR),
            (429, OUTCOME_RATE_LIMITED),
            (500, OUTCOME_SERVER_ERROR),
            (503, OUTCOME_SERVER_ERROR),
            (None, OUTCOME_TRANSPORT_ERROR),
        ],
    )
    def test_classification(self, status, expected):
        assert outcome_for(status) == expected


# ── Rate-limit header parsing ──────────────────────────────────────────────


class TestParseRateLimit:
    def test_reads_both_quotas(self):
        reading = parse_rate_limit(
            httpx.Headers(
                {
                    "X-RateLimit-Limit": "600,30000",
                    "X-RateLimit-Usage": "314,27536",
                    "X-ReadRateLimit-Limit": "300,15000",
                    "X-ReadRateLimit-Usage": "50,200",
                }
            )
        )
        assert (reading.usage_short, reading.limit_short) == (314, 600)
        assert (reading.usage_daily, reading.limit_daily) == (27536, 30000)
        assert (reading.read_usage_short, reading.read_limit_short) == (50, 300)
        assert (reading.read_usage_daily, reading.read_limit_daily) == (200, 15000)

    def test_absent_headers_are_empty_not_zero(self):
        # Zero would read as "the window is empty", which is the opposite of
        # "we have no idea".
        reading = parse_rate_limit(httpx.Headers({}))
        assert reading.is_empty
        assert reading.usage_short is None

    @pytest.mark.parametrize("raw", ["", "600", "not,numbers", "600,"])
    def test_unparseable_values_are_none(self, raw):
        reading = parse_rate_limit(httpx.Headers({"X-RateLimit-Usage": raw}))
        assert reading.usage_short is None
        assert reading.usage_daily is None


# ── The counting transport ─────────────────────────────────────────────────


class TestCountingTransport:
    async def test_records_one_row_per_request(self, api_usage_db):
        async with _client(lambda r: httpx.Response(200, json={})) as client:
            await client.get("https://www.strava.com/api/v3/athlete")
            await client.get("https://www.strava.com/api/v3/athlete/zones")

        rows = await _rows(api_usage_db)
        assert [r.endpoint for r in rows] == ["/athlete", "/athlete/zones"]
        assert {r.service for r in rows} == {"strava"}
        assert {r.outcome for r in rows} == {OUTCOME_OK}
        assert all(r.duration_ms is not None for r in rows)

    async def test_redirect_is_counted_as_two_requests(self, api_usage_db):
        # One method call, two requests against the quota. Counting at the
        # method layer would report half of what we actually spent.
        def handler(request):
            if request.url.path.endswith("/fit_file"):
                return httpx.Response(
                    302,
                    headers={"Location": "https://cdn.wahooligan.com/abcd1234efgh5678/r.fit"},
                )
            return httpx.Response(200, content=b"FIT")

        async with _client(handler, service="wahoo", follow_redirects=True) as client:
            await client.get("https://api.wahooligan.com/v1/workouts/7/fit_file")

        rows = await _rows(api_usage_db)
        assert [r.endpoint for r in rows] == [
            "/workouts/{id}/fit_file",
            "cdn.wahooligan.com/*",
        ]

    async def test_transport_failure_is_recorded_with_no_status(self, api_usage_db):
        def handler(request):
            raise httpx.ConnectError("name resolution failed", request=request)

        async with _client(handler) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("https://www.strava.com/api/v3/athlete")

        (row,) = await _rows(api_usage_db)
        assert row.status_code is None
        assert row.outcome == OUTCOME_TRANSPORT_ERROR

    async def test_429_is_recorded_distinguishably(self, api_usage_db):
        async with _client(lambda r: httpx.Response(429)) as client:
            await client.get("https://www.strava.com/api/v3/athlete")

        (row,) = await _rows(api_usage_db)
        assert row.status_code == 429
        assert row.outcome == OUTCOME_RATE_LIMITED

    async def test_rate_limit_headers_are_captured(self, api_usage_db):
        headers = {
            "X-RateLimit-Limit": "600,30000",
            "X-RateLimit-Usage": "100,2000",
            "X-ReadRateLimit-Limit": "300,15000",
            "X-ReadRateLimit-Usage": "40,900",
        }
        async with _client(lambda r: httpx.Response(200, headers=headers)) as client:
            await client.get("https://www.strava.com/api/v3/athlete")

        (row,) = await _rows(api_usage_db)
        assert (row.ratelimit_usage_short, row.ratelimit_limit_short) == (100, 600)
        assert (row.ratelimit_usage_daily, row.ratelimit_limit_daily) == (2000, 30000)
        assert (row.ratelimit_read_usage_short, row.ratelimit_read_limit_short) == (40, 300)

    async def test_no_raw_url_is_persisted(self, api_usage_db):
        async with _client(lambda r: httpx.Response(200)) as client:
            await client.get(
                "https://www.strava.com/api/v3/activities/987654321/streams"
                "?access_token=SECRETVALUE"
            )

        (row,) = await _rows(api_usage_db)
        assert row.endpoint == "/activities/{id}/streams"
        assert "987654321" not in row.endpoint
        assert "SECRETVALUE" not in row.endpoint


class TestAttribution:
    async def test_context_var_attributes_the_call(self, api_usage_db):
        async with _client(lambda r: httpx.Response(200)) as client:
            with attribute_to_user("athlete-7"):
                await client.get("https://www.strava.com/api/v3/athlete")
            await client.get("https://www.strava.com/api/v3/athlete")

        rows = await _rows(api_usage_db)
        assert [r.user_id for r in rows] == ["athlete-7", None]

    async def test_explicit_user_id_wins(self, api_usage_db):
        async with _client(lambda r: httpx.Response(200), user_id="explicit") as client:
            with attribute_to_user("contextual"):
                await client.get("https://www.strava.com/api/v3/athlete")

        (row,) = await _rows(api_usage_db)
        assert row.user_id == "explicit"


class TestRecordingNeverBreaksTheCaller:
    async def test_a_failed_write_does_not_fail_the_request(
        self, api_usage_db, monkeypatch
    ):
        """The single most important property: accounting must not fail a sync.

        Forced here by making the session factory itself raise, which stands in
        for the real cases — a locked database, a full disk.
        """
        def boom():
            raise RuntimeError("database is locked")

        monkeypatch.setattr(
            api_usage_service, "api_usage_session_factory", boom
        )

        async with _client(lambda r: httpx.Response(200, json={"ok": True})) as client:
            response = await client.get("https://www.strava.com/api/v3/athlete")

        assert response.status_code == 200
        assert response.json() == {"ok": True}
        assert await _rows(api_usage_db) == []

    async def test_record_api_usage_swallows_its_own_failure(
        self, api_usage_db, monkeypatch
    ):
        def boom():
            raise RuntimeError("disk full")

        monkeypatch.setattr(api_usage_service, "api_usage_session_factory", boom)
        # Returns normally rather than raising.
        await record_api_usage(
            service="strava", endpoint="/athlete", method="GET", status_code=200
        )


class TestDatabasePath:
    """``API_USAGE_DB`` overrides the path; otherwise it sits under the data dir.

    The override is what lets a hoster put this file on a different volume, or
    prune it on its own schedule — the whole reason it is a separate database.
    """

    def test_defaults_under_the_data_dir(self, monkeypatch):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "api_usage_db", "")
        monkeypatch.setattr(settings, "data_dir", "/srv/okdata")
        assert settings.api_usage_db_path == "/srv/okdata/api_usage.db"

    def test_explicit_override_wins(self, monkeypatch):
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "api_usage_db", "/mnt/slow/api_usage.db")
        monkeypatch.setattr(settings, "data_dir", "/srv/okdata")
        assert settings.api_usage_db_path == "/mnt/slow/api_usage.db"

    def test_it_is_a_different_file_from_the_llm_usage_db(self, monkeypatch):
        # Sibling databases, not one file with two tables: their retention needs
        # differ by an order of magnitude.
        from backend.app.core.config import settings

        monkeypatch.setattr(settings, "api_usage_db", "")
        monkeypatch.setattr(settings, "llm_usage_db", "")
        monkeypatch.setattr(settings, "data_dir", "/srv/okdata")
        assert settings.api_usage_db_path != settings.llm_usage_db_path


class TestProviderClientFactory:
    def test_builds_a_counting_client(self):
        client = provider_client("strava", timeout=httpx.Timeout(5.0))
        assert isinstance(client._transport, CountingTransport)
        assert client._transport._service == "strava"
