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
