"""
Unit tests for WahooClient.list_activities.

Verifies that planned/structured workouts (ones scheduled onto a device but not
actually performed) are filtered out of the sync, so only performed activities
are imported (issue #10).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.services.providers.wahoo import WahooClient, _is_performed_workout


# ── Sample workout records ───────────────────────────────────────────────────

# A performed workout: carries a populated workout_summary with recorded data.
_PERFORMED = {
    "id": 111,
    "starts": "2026-04-25T09:36:48.000Z",
    "name": "Gravel cycling",
    "workout_type_id": 0,
    "plan_id": None,
    "workout_summary": {
        "id": 999,
        "duration_active_accum": "4184.0",
        "duration_total_accum": "4521.0",
        "distance_accum": "27441.58",
        "file": {"url": "https://example.com/fit_files/myworkout.fit"},
    },
}

# A planned workout pushed to a device: has a plan_id but no summary yet.
_PLANNED_NO_SUMMARY = {
    "id": 222,
    "starts": "2026-04-24T06:00:00.000Z",
    "name": "Planned VO2max intervals",
    "workout_type_id": 0,
    "plan_id": 555,
}

# A planned workout whose summary key is present but empty.
_PLANNED_EMPTY_SUMMARY = {
    "id": 333,
    "starts": "2026-04-23T06:00:00.000Z",
    "name": "Planned endurance ride",
    "workout_type_id": 0,
    "plan_id": 556,
    "workout_summary": None,
}


def _mock_httpx_context(response) -> MagicMock:
    mock_instance = MagicMock()
    mock_instance.get = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_instance)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _mock_response(json_data: dict) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# ── _is_performed_workout ────────────────────────────────────────────────────


def test_is_performed_true_for_populated_summary():
    assert _is_performed_workout(_PERFORMED) is True


def test_is_performed_false_for_missing_summary():
    assert _is_performed_workout(_PLANNED_NO_SUMMARY) is False


def test_is_performed_false_for_none_summary():
    assert _is_performed_workout(_PLANNED_EMPTY_SUMMARY) is False


def test_is_performed_false_for_empty_summary():
    assert _is_performed_workout({"id": 1, "workout_summary": {}}) is False


# ── list_activities ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_activities_filters_out_planned_workouts():
    """Only performed workouts (populated workout_summary) are returned."""
    client = WahooClient()
    payload = {
        "workouts": [_PERFORMED, _PLANNED_NO_SUMMARY, _PLANNED_EMPTY_SUMMARY]
    }

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(_mock_response(payload))):
        activities = await client.list_activities("access-tok", page=1)

    assert len(activities) == 1
    assert activities[0].external_id == "111"
    assert activities[0].name == "Gravel cycling"


@pytest.mark.asyncio
async def test_list_activities_only_caches_fit_urls_for_performed():
    """CDN FIT URLs are cached only for performed workouts, not planned ones."""
    client = WahooClient()
    payload = {
        "workouts": [_PERFORMED, _PLANNED_NO_SUMMARY, _PLANNED_EMPTY_SUMMARY]
    }

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(_mock_response(payload))):
        await client.list_activities("access-tok", page=1)

    assert client._fit_urls == {"111": "https://example.com/fit_files/myworkout.fit"}


@pytest.mark.asyncio
async def test_list_activities_all_planned_returns_empty():
    """A page containing only planned workouts yields no activities."""
    client = WahooClient()
    payload = {"workouts": [_PLANNED_NO_SUMMARY, _PLANNED_EMPTY_SUMMARY]}

    with patch("httpx.AsyncClient", return_value=_mock_httpx_context(_mock_response(payload))):
        activities = await client.list_activities("access-tok", page=1)

    assert activities == []


# ── download_fit_file: a throttle is not an absence (issue #67) ──────────────


class TestDownloadFitFileTellsThrottlingFromAbsence:
    """``None`` is an answer this client gives about the *workout*.

    It inspects statuses itself instead of raising for them, so before issue #67
    a rate-limited download reached the sync pipeline looking exactly like "this
    workout has no FIT" — and the workout was imported with no streams behind it.
    """

    def _client(self) -> WahooClient:
        client = WahooClient()
        client._fit_urls = {}
        return client

    async def test_a_404_still_means_there_is_no_file(self):
        response = httpx.Response(
            404, request=httpx.Request("GET", "https://api.wahooligan.test/fit")
        )
        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            return_value=_mock_httpx_context(response),
        ):
            assert await self._client().download_fit_file("tok", "1") is None

    async def test_a_429_raises_instead_of_reading_as_absence(self):
        from backend.app.services.providers.throttling import ProviderThrottled

        response = httpx.Response(
            429,
            headers={"Retry-After": "30"},
            request=httpx.Request("GET", "https://api.wahooligan.test/fit"),
        )
        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            return_value=_mock_httpx_context(response),
        ):
            with pytest.raises(ProviderThrottled) as exc:
                await self._client().download_fit_file("tok", "1")
        assert exc.value.status_code == 429
        assert exc.value.retry_after == 30.0

    async def test_a_5xx_raises_too(self):
        from backend.app.services.providers.throttling import ProviderThrottled

        response = httpx.Response(
            503, request=httpx.Request("GET", "https://api.wahooligan.test/fit")
        )
        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            return_value=_mock_httpx_context(response),
        ):
            with pytest.raises(ProviderThrottled):
                await self._client().download_fit_file("tok", "1")

    async def test_the_cdn_fallback_still_gets_its_turn(self):
        """A throttled API endpoint must not skip the copy we can still fetch."""
        throttled = httpx.Response(
            429, request=httpx.Request("GET", "https://api.wahooligan.test/fit")
        )
        from_cdn = httpx.Response(
            200,
            content=b"FIT-BYTES",
            request=httpx.Request("GET", "https://cdn.wahooligan.test/x.fit"),
        )
        client = WahooClient()
        client._fit_urls = {"1": "https://cdn.wahooligan.test/x.fit"}

        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            side_effect=[
                _mock_httpx_context(throttled),
                _mock_httpx_context(from_cdn),
            ],
        ):
            assert await client.download_fit_file("tok", "1") == b"FIT-BYTES"

    async def test_a_throttled_cdn_leaves_the_throttle_standing(self):
        """Both routes refused, so there is still no answer about this workout."""
        from backend.app.services.providers.throttling import ProviderThrottled

        throttled = httpx.Response(
            429, request=httpx.Request("GET", "https://api.wahooligan.test/fit")
        )
        cdn_throttled = httpx.Response(
            503, request=httpx.Request("GET", "https://cdn.wahooligan.test/x.fit")
        )
        client = WahooClient()
        client._fit_urls = {"1": "https://cdn.wahooligan.test/x.fit"}

        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            side_effect=[
                _mock_httpx_context(throttled),
                _mock_httpx_context(cdn_throttled),
            ],
        ):
            with pytest.raises(ProviderThrottled) as exc:
                await client.download_fit_file("tok", "1")
        # The API endpoint's refusal is the one reported: it is the request that
        # was actually rate-limited, the CDN being only a fallback.
        assert exc.value.status_code == 429

    async def test_a_404_with_a_dead_cdn_is_still_just_absence(self):
        """A missing file plus an unhelpful CDN must not become a throttle."""
        missing = httpx.Response(
            404, request=httpx.Request("GET", "https://api.wahooligan.test/fit")
        )
        cdn_missing = httpx.Response(
            404, request=httpx.Request("GET", "https://cdn.wahooligan.test/x.fit")
        )
        client = WahooClient()
        client._fit_urls = {"1": "https://cdn.wahooligan.test/x.fit"}

        with patch(
            "backend.app.services.providers.wahoo.httpx.AsyncClient",
            side_effect=[
                _mock_httpx_context(missing),
                _mock_httpx_context(cdn_missing),
            ],
        ):
            assert await client.download_fit_file("tok", "1") is None
