"""
Unit tests for HammerheadClient (issue #69).

Every request goes through an ``httpx.MockTransport`` patched in for
``provider_client``, so these exercise the real request shapes and response
handling: normalisation, pagination termination, throttling, partial-scope
403s, completion from a FIT, and the metrics/route pushes.
"""
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from backend.app.services.providers import hammerhead as hh
from backend.app.services.providers.hammerhead import (
    HammerheadClient,
    ScopeNotGranted,
    build_metrics_payload,
    sport_type_for,
)
from backend.app.services.providers.throttling import ProviderThrottled

FIT = (
    Path(__file__).parent.parent.parent
    / "testdata" / "fixtures" / "synthetic_bike_power_gps.fit"
)


class _Router:
    """A MockTransport handler that records requests and answers by route."""

    def __init__(self, routes):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        handler = self.routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": "not found"})
        return handler(request) if callable(handler) else handler


@pytest.fixture
def router(monkeypatch):
    holder = {}

    def install(routes):
        r = _Router(routes)
        holder["router"] = r

        def fake_provider_client(service, *, timeout=None, follow_redirects=False, **kw):
            assert service == "hammerhead"
            return httpx.AsyncClient(
                transport=httpx.MockTransport(r), follow_redirects=follow_redirects
            )

        monkeypatch.setattr(hh, "provider_client", fake_provider_client)
        return r

    return install


def _page(items, current, total):
    return httpx.Response(
        200,
        json={"data": items, "currentPage": current, "totalPages": total, "perPage": 100},
    )


_SUMMARY = {
    "id": "a1",
    "name": "Morning loop",
    "createdAt": "2026-04-25T11:00:00Z",
    "duration": 76765,
    "distance": 123.45,
}


# ── OAuth ────────────────────────────────────────────────────────────────────


class TestOAuth:
    def test_authorize_url_requests_every_scope(self, monkeypatch):
        monkeypatch.setattr(hh.settings, "hammerhead_client_id", "cid")
        url = HammerheadClient().get_oauth_url("st", "https://api/cb")
        parsed = urllib.parse.urlparse(url)
        assert parsed.netloc == "api.hammerhead.io"
        assert parsed.path == "/v1/auth/oauth/authorize"
        q = urllib.parse.parse_qs(parsed.query)
        assert q["client_id"] == ["cid"]
        assert q["state"] == ["st"]
        assert q["response_type"] == ["code"]
        assert set(q["scope"][0].split()) == {
            "activity:read", "route:read", "route:write", "workout:write", "metrics:write",
        }

    async def test_exchange_takes_user_id_from_the_token_response(self, router):
        r = router({
            ("POST", "/v1/auth/oauth/token"): httpx.Response(200, json={
                "access_token": "at", "refresh_token": "rt",
                "expires_in": 3600, "user_id": "hh-42",
            }),
        })
        tokens = await HammerheadClient.exchange_code("code", "https://api/cb")
        assert tokens["access_token"] == "at"
        assert tokens["refresh_token"] == "rt"
        assert tokens["provider_athlete_id"] == "hh-42"
        assert tokens["expires_at"] > datetime.now(timezone.utc).timestamp()
        # One call: no profile lookup, unlike Wahoo.
        assert len(r.requests) == 1
        form = urllib.parse.parse_qs(r.requests[0].content.decode())
        assert form["grant_type"] == ["authorization_code"]

    async def test_refresh_keeps_old_refresh_token_when_none_returned(self, router):
        router({
            ("POST", "/v1/auth/oauth/token"): httpx.Response(
                200, json={"access_token": "at2", "expires_in": 60}
            ),
        })
        tokens = await HammerheadClient.refresh_access_token("old-rt")
        assert tokens["access_token"] == "at2"
        assert tokens["refresh_token"] == "old-rt"


# ── list_activities ──────────────────────────────────────────────────────────


class TestListActivities:
    async def test_records_are_sparse(self, router):
        router({("GET", "/v1/api/activities"): _page([_SUMMARY], 1, 1)})
        [norm] = await HammerheadClient().list_activities("tok", 1)
        assert norm.external_id == "a1"
        assert norm.source == "hammerhead"
        assert norm.name == "Morning loop"
        # createdAt stands in until completion; nothing undocumented is guessed.
        assert norm.start_time == datetime(2026, 4, 25, 11, 0, tzinfo=timezone.utc)
        assert norm.duration_s is None
        assert norm.distance_m is None
        assert norm.sport_type is None

    async def test_requests_the_maximum_page_size(self, router):
        r = router({("GET", "/v1/api/activities"): _page([_SUMMARY], 1, 1)})
        await HammerheadClient().list_activities("tok", 1)
        params = r.requests[0].url.params
        assert params["perPage"] == "100"
        assert params["page"] == "1"
        assert "startDate" not in params
        assert r.requests[0].headers["Authorization"] == "Bearer tok"

    async def test_incremental_runs_send_start_date(self, router):
        from datetime import date

        r = router({("GET", "/v1/api/activities"): _page([], 1, 1)})
        client = HammerheadClient()
        client.start_date = date(2026, 4, 1)
        await client.list_activities("tok", 1)
        assert r.requests[0].url.params["startDate"] == "2026-04-01"

    async def test_pagination_ends_without_probing_an_empty_page(self, router):
        def page(request):
            n = int(request.url.params["page"])
            return _page([{**_SUMMARY, "id": f"a{n}"}], n, 2)

        r = router({("GET", "/v1/api/activities"): page})
        client = HammerheadClient()
        assert len(await client.list_activities("tok", 1)) == 1
        assert len(await client.list_activities("tok", 2)) == 1
        assert await client.list_activities("tok", 3) == []
        assert len(r.requests) == 2  # page 3 never asked for

    async def test_page_past_the_end_is_empty(self, router):
        router({("GET", "/v1/api/activities"): _page([], 5, 4)})
        assert await HammerheadClient().list_activities("tok", 5) == []

    async def test_429_raises_for_the_throttle_handling(self, router):
        router({("GET", "/v1/api/activities"): httpx.Response(429, headers={"Retry-After": "5"})})
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await HammerheadClient().list_activities("tok", 1)
        assert exc_info.value.response.status_code == 429

    async def test_403_names_the_missing_scope(self, router):
        router({("GET", "/v1/api/activities"): httpx.Response(403)})
        with pytest.raises(ScopeNotGranted) as exc_info:
            await HammerheadClient().list_activities("tok", 1)
        assert exc_info.value.scope == "activity:read"


# ── FIT download and completion ──────────────────────────────────────────────


class TestFitAndCompletion:
    async def test_missing_fit_is_none(self, router):
        router({})
        assert await HammerheadClient().download_fit_file("tok", "a1") is None

    async def test_throttled_fit_raises_rather_than_reading_as_absent(self, router):
        router({("GET", "/v1/api/activities/a1/file"): httpx.Response(503)})
        with pytest.raises(ProviderThrottled):
            await HammerheadClient().download_fit_file("tok", "a1")

    async def test_completion_dates_the_ride_from_its_fit(self, router):
        fit = FIT.read_bytes()
        r = router({
            ("GET", "/v1/api/activities/a1/file"): httpx.Response(200, content=fit),
            ("GET", "/v1/api/activities/a1"): httpx.Response(200, json={
                "id": "a1", "name": "Gravel day", "activityType": "GRAVEL",
            }),
        })
        client = HammerheadClient()
        [sparse] = [hh.NormalizedActivity(
            external_id="a1", source="hammerhead", name="Morning loop",
            sport_type=None, start_time=datetime(2030, 1, 1, tzinfo=timezone.utc),
            duration_s=None, distance_m=None, elevation_m=None, avg_power=None,
            avg_hr=None, max_hr=None, avg_speed_ms=None, avg_cadence=None,
        )]
        norm = await client.complete_activity("tok", sparse)

        from openkoutsi.fit import summarizeWorkout
        import io

        profile = summarizeWorkout(io.BytesIO(fit))
        assert norm.start_time == profile.start_time
        assert norm.start_time != sparse.start_time
        assert norm.sport_type == "GravelRide"
        assert norm.name == "Gravel day"
        assert norm.duration_s == profile.duration
        assert norm.avg_power == pytest.approx(profile.avgPower)

        # The import that follows is served from the cache.
        before = len(r.requests)
        assert await client.download_fit_file("tok", "a1") == fit
        assert len(r.requests) == before

    async def test_completion_survives_a_missing_detail(self, router):
        router({
            ("GET", "/v1/api/activities/a1/file"): httpx.Response(200, content=FIT.read_bytes()),
            ("GET", "/v1/api/activities/a1"): httpx.Response(404),
        })
        sparse = (await _sparse())
        norm = await HammerheadClient().complete_activity("tok", sparse)
        assert norm.sport_type  # from the FIT's own sport
        assert norm.name == "Morning loop"

    async def test_completion_without_a_fit_is_a_lookup_error(self, router):
        router({})
        with pytest.raises(LookupError):
            await HammerheadClient().complete_activity("tok", await _sparse())

    async def test_a_throttled_detail_raises(self, router):
        router({
            ("GET", "/v1/api/activities/a1/file"): httpx.Response(200, content=FIT.read_bytes()),
            ("GET", "/v1/api/activities/a1"): httpx.Response(429),
        })
        with pytest.raises(httpx.HTTPStatusError):
            await HammerheadClient().complete_activity("tok", await _sparse())

    async def test_streams_come_from_the_fit(self, router):
        router({
            ("GET", "/v1/api/activities/a1/file"): httpx.Response(200, content=FIT.read_bytes()),
        })
        streams = await HammerheadClient().get_activity_streams("tok", "a1")
        assert streams["power"]


async def _sparse():
    return hh.NormalizedActivity(
        external_id="a1", source="hammerhead", name="Morning loop", sport_type=None,
        start_time=datetime(2030, 1, 1, tzinfo=timezone.utc), duration_s=None,
        distance_m=None, elevation_m=None, avg_power=None, avg_hr=None, max_hr=None,
        avg_speed_ms=None, avg_cadence=None,
    )


@pytest.mark.parametrize("raw, expected", [
    ("RIDE", "Ride"),
    ("EBIKE", "EBikeRide"),
    ("MOUNTAIN_BIKE", "MountainBikeRide"),
    ("GRAVEL", "GravelRide"),
    ("EMOUNTAIN_BIKE", "EMountainBikeRide"),
    ("VELOMOBILE", "Velomobile"),
    ("SOMETHING_NEW", "Ride"),
    (None, None),
])
def test_sport_type_mapping(raw, expected):
    assert sport_type_for(raw) == expected


def test_every_mapped_sport_counts_as_cycling():
    from openkoutsi.sport_matching import CYCLING_SPORT_TYPES

    for raw in hh._SPORT_TYPES:
        assert sport_type_for(raw) in CYCLING_SPORT_TYPES


# ── Metrics push ─────────────────────────────────────────────────────────────


class _Athlete:
    weight_kg = 72.5
    ftp = 260
    max_hr = 188
    resting_hr = None
    hr_zones = [{"low": 0, "high": 120, "name": "Z1"}, {"low": 120, "high": 999, "name": "Z2"}]
    power_zones = None


class TestMetrics:
    def test_payload_renames_zones_and_omits_unset_fields(self):
        payload = build_metrics_payload(_Athlete())
        assert payload == {
            "weightInKilograms": 72.5,
            "powerFtp": 260,
            "maxHr": 188,
            "heartRateZones": [
                {"min": 0, "max": 120, "name": "Z1"},
                {"min": 120, "max": 999, "name": "Z2"},
            ],
        }
        assert "restingHr" not in payload
        assert "powerZones" not in payload

    async def test_push_posts_json(self, router):
        r = router({("POST", "/v1/api/metrics"): httpx.Response(200, json={})})
        await HammerheadClient().push_metrics("tok", {"powerFtp": 250})
        assert json.loads(r.requests[0].content) == {"powerFtp": 250}

    async def test_push_without_the_scope(self, router):
        router({("POST", "/v1/api/metrics"): httpx.Response(403)})
        with pytest.raises(ScopeNotGranted) as exc_info:
            await HammerheadClient().push_metrics("tok", {"powerFtp": 250})
        assert exc_info.value.scope == "metrics:write"


# ── Route push ───────────────────────────────────────────────────────────────


class TestRoutes:
    async def test_first_push_creates(self, router):
        r = router({("POST", "/v1/api/routes/file"): httpx.Response(200, json={"id": "r1"})})
        route_id = await HammerheadClient().upload_route(
            "tok", gpx_bytes=b"<gpx/>", filename="c.gpx"
        )
        assert route_id == "r1"
        assert b'filename="c.gpx"' in r.requests[0].content

    async def test_re_push_replaces(self, router):
        r = router({("PUT", "/v1/api/routes/r1/file"): httpx.Response(200, json={"id": "r1"})})
        route_id = await HammerheadClient().upload_route(
            "tok", gpx_bytes=b"<gpx/>", filename="c.gpx", existing_id="r1"
        )
        assert route_id == "r1"
        assert [q.method for q in r.requests] == ["PUT"]

    async def test_re_push_of_a_route_deleted_on_the_karoo_creates_anew(self, router):
        r = router({
            ("PUT", "/v1/api/routes/r1/file"): httpx.Response(404),
            ("POST", "/v1/api/routes/file"): httpx.Response(200, json={"id": "r2"}),
        })
        route_id = await HammerheadClient().upload_route(
            "tok", gpx_bytes=b"<gpx/>", filename="c.gpx", existing_id="r1"
        )
        assert route_id == "r2"
        assert [q.method for q in r.requests] == ["PUT", "POST"]

    async def test_push_without_the_scope(self, router):
        router({("POST", "/v1/api/routes/file"): httpx.Response(403)})
        with pytest.raises(ScopeNotGranted):
            await HammerheadClient().upload_route("tok", gpx_bytes=b"x", filename="c.gpx")

    async def test_delete_tolerates_an_already_deleted_route(self, router):
        router({("DELETE", "/v1/api/routes/r1"): httpx.Response(404)})
        await HammerheadClient().delete_route("tok", "r1")
