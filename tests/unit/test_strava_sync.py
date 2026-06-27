"""Tests for strava_sync.py webhook event processing."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.strava_sync import process_webhook_event, _process_event_for_team


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_event(event_type="create", object_type="activity", owner_id="12345", object_id=99):
    return {
        "strava_event_type": event_type,
        "strava_owner_id": str(owner_id),
        "payload": {
            "object_type": object_type,
            "object_id": object_id,
            "aspect_type": event_type,
            "updates": {},
        },
    }


def _make_conn(user_id="user-1"):
    conn = MagicMock()
    conn.user_id = user_id
    return conn


def _make_athlete(athlete_id="ath-1", user_id="user-1"):
    ath = MagicMock()
    ath.id = athlete_id
    ath.global_user_id = user_id
    ath.app_settings = {}
    return ath


def _make_activity(start_time=None):
    act = MagicMock()
    act.id = "act-1"
    act.start_time = start_time or datetime(2025, 6, 1, 8, 0, tzinfo=timezone.utc)
    act.sources = []
    return act


def _make_strava_raw(start_date="2025-06-01T08:00:00Z"):
    return {
        "name": "Morning Ride",
        "sport_type": "Ride",
        "start_date": start_date,
        "moving_time": 3600,
        "elapsed_time": 3700,
        "distance": 50000.0,
        "total_elevation_gain": 500.0,
        "average_watts": 220.0,
        "average_heartrate": 155.0,
        "max_heartrate": 178.0,
        "average_speed": 13.9,
        "average_cadence": 88.0,
    }


# ── process_webhook_event: early-exit guards ──────────────────────────────────

class TestProcessWebhookEventGuards:
    async def test_unknown_event_type_returns_early(self):
        event = _make_event(event_type="like")  # not create/update/delete
        with patch("backend.app.db.registry._RegistrySessionLocal") as mock_reg:
            await process_webhook_event(event)
            mock_reg.assert_not_called()

    async def test_non_activity_object_returns_early(self):
        event = _make_event(object_type="athlete")
        with patch("backend.app.db.registry._RegistrySessionLocal") as mock_reg:
            await process_webhook_event(event)
            mock_reg.assert_not_called()

    async def test_unknown_strava_owner_returns_early(self):
        event = _make_event()
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None  # no ProviderConnection found
        session.execute = AsyncMock(return_value=result)

        @asynccontextmanager
        async def _reg_ctx():
            yield session

        with (
            patch("backend.app.db.registry._RegistrySessionLocal", return_value=_reg_ctx()),
            patch("backend.app.db.team_session.get_team_session_factory") as mock_factory,
        ):
            await process_webhook_event(event)
            mock_factory.assert_not_called()


# ── _process_event_for_team: create ──────────────────────────────────────────

class TestProcessEventCreate:
    async def _run_create(self, existing_activity=None, already_imported=False):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()

        dupe_check = MagicMock()
        dupe_check.scalar_one_or_none.return_value = MagicMock() if already_imported else None

        existing_check = MagicMock()
        existing_check.scalar_one_or_none.return_value = existing_activity

        session.execute = AsyncMock(side_effect=[dupe_check, existing_check])

        strava_raw = _make_strava_raw()
        mock_resp = MagicMock()
        mock_resp.json.return_value = strava_raw
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("backend.app.services.metrics_engine.recalculate_from", new_callable=AsyncMock),
            patch("backend.app.services.strava_sync._populate_activity", new_callable=AsyncMock),
            patch("backend.app.services.strava_sync._repopulate_activity", new_callable=AsyncMock),
            patch("httpx.AsyncClient", return_value=mock_http),
        ):
            await _process_event_for_team(
                "create", "99", {"object_id": 99, "object_type": "activity"},
                athlete, conn, "access-token-xxx", "team-1", session,
            )

        return session

    async def test_create_adds_activity_and_source(self):
        session = await self._run_create()
        assert session.add.called

    async def test_create_idempotent_when_already_imported(self):
        session = await self._run_create(already_imported=True)
        # No activity should be added when already imported
        assert not session.flush.called


# ── _process_event_for_team: delete ──────────────────────────────────────────

class TestProcessEventDelete:
    async def test_delete_removes_source_and_activity(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.delete = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()

        act = _make_activity()
        src = MagicMock()
        src.activity = act

        src_result = MagicMock()
        src_result.scalar_one_or_none.return_value = src

        remaining_result = MagicMock()
        remaining_result.scalar_one_or_none.return_value = None  # no other sources

        session.execute = AsyncMock(side_effect=[src_result, remaining_result])

        with patch("backend.app.services.metrics_engine.recalculate_from", new_callable=AsyncMock):
            await _process_event_for_team(
                "delete", "99", {"object_id": 99, "object_type": "activity", "updates": {}},
                athlete, conn, "token", "team-1", session,
            )

        assert session.delete.call_count == 2  # src and act deleted

    async def test_delete_keeps_activity_when_other_sources_remain(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.delete = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()

        act = _make_activity()
        src = MagicMock()
        src.activity = act

        src_result = MagicMock()
        src_result.scalar_one_or_none.return_value = src

        remaining_result = MagicMock()
        remaining_result.scalar_one_or_none.return_value = MagicMock()  # another source remains

        session.execute = AsyncMock(side_effect=[src_result, remaining_result])

        with patch("backend.app.services.metrics_engine.recalculate_from", new_callable=AsyncMock):
            await _process_event_for_team(
                "delete", "99", {"object_id": 99, "object_type": "activity", "updates": {}},
                athlete, conn, "token", "team-1", session,
            )

        # Only the source is deleted, not the activity
        assert session.delete.call_count == 1

    async def test_delete_no_op_when_source_not_found(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.delete = AsyncMock()

        src_result = MagicMock()
        src_result.scalar_one_or_none.return_value = None  # source not found

        session.execute = AsyncMock(return_value=src_result)

        await _process_event_for_team(
            "delete", "99", {"object_id": 99, "object_type": "activity", "updates": {}},
            athlete, conn, "token", "team-1", session,
        )

        session.delete.assert_not_called()


# ── _process_event_for_team: update ──────────────────────────────────────────

class TestProcessEventUpdate:
    async def test_update_sets_name_and_sport_type(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.commit = AsyncMock()

        act = _make_activity()
        src = MagicMock()
        src.activity = act

        src_result = MagicMock()
        src_result.scalar_one_or_none.return_value = src
        session.execute = AsyncMock(return_value=src_result)

        payload = {"object_id": 99, "object_type": "activity",
                   "updates": {"title": "Evening Ride", "sport_type": "VirtualRide"}}

        await _process_event_for_team(
            "update", "99", payload, athlete, conn, "token", "team-1", session,
        )

        assert act.name == "Evening Ride"
        assert act.sport_type == "VirtualRide"
        session.commit.assert_called_once()

    async def test_update_no_op_when_empty_updates(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.commit = AsyncMock()

        payload = {"object_id": 99, "object_type": "activity", "updates": {}}
        await _process_event_for_team(
            "update", "99", payload, athlete, conn, "token", "team-1", session,
        )

        session.commit.assert_not_called()

    async def test_update_no_op_when_source_not_found(self):
        athlete = _make_athlete()
        conn = _make_conn()
        session = AsyncMock()
        session.commit = AsyncMock()

        src_result = MagicMock()
        src_result.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=src_result)

        payload = {"object_id": 99, "object_type": "activity",
                   "updates": {"title": "New name"}}

        await _process_event_for_team(
            "update", "99", payload, athlete, conn, "token", "team-1", session,
        )

        session.commit.assert_not_called()
