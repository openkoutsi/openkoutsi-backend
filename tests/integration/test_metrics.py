"""
Integration tests for /api/metrics endpoints.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from unittest.mock import patch

from sqlalchemy import select

from backend.app.models.team_orm import Activity, ActivityStream, DailyMetric, Athlete


class TestGetFitness:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/metrics/fitness", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_inserted_metrics(self, client, auth_headers, session):
        # Get the athlete ID
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        metric = DailyMetric(
            athlete_id=athlete_id,
            date=today,
            ctl=30.0,
            atl=40.0,
            tsb=-10.0,
            tss_day=80.0,
        )
        session.add(metric)
        await session.commit()

        resp = await client.get("/api/metrics/fitness", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ctl"] == 30.0
        assert data[0]["atl"] == 40.0

    async def test_days_filter_limits_results(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        for offset in [60, 10, 1]:
            session.add(DailyMetric(
                athlete_id=athlete_id,
                date=today - timedelta(days=offset),
                ctl=10.0, atl=10.0, tsb=0.0, tss_day=50.0,
            ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness?days=30", headers=auth_headers)
        data = resp.json()
        # Only the metrics from last 30 days should be returned
        assert len(data) == 2  # 10 and 1 days ago

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/fitness")
        assert resp.status_code == 401


class TestGetFitnessCurrent:
    async def test_returns_zeros_when_no_metrics(self, client, auth_headers):
        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctl"] == 0.0
        assert data["atl"] == 0.0
        assert data["tsb"] == 0.0
        assert "form" in data

    async def test_form_label_computed_from_tsb(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        # TSB > 25 → "peak"
        session.add(DailyMetric(
            athlete_id=athlete_id, date=today,
            ctl=50.0, atl=20.0, tsb=30.0, tss_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.json()["form"] == "peak"

    async def test_tired_form_label(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        session.add(DailyMetric(
            athlete_id=athlete_id, date=today,
            ctl=40.0, atl=60.0, tsb=-20.0, tss_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.json()["form"] == "tired"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/fitness/current")
        assert resp.status_code == 401


class TestCatchUp:
    async def test_creates_today_metric_when_missing(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        # Seed yesterday so CTL/ATL can be inherited
        session.add(DailyMetric(
            athlete_id=athlete_id,
            date=today - timedelta(days=1),
            ctl=40.0, atl=50.0, tsb=-10.0, tss_day=80.0,
        ))
        await session.commit()

        resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

        # Today's DailyMetric should now exist
        resp2 = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp2.json()["date"] == str(today)

    async def test_returns_not_updated_when_already_current(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        session.add(DailyMetric(
            athlete_id=athlete_id,
            date=today,
            ctl=30.0, atl=35.0, tsb=-5.0, tss_day=0.0,
        ))
        await session.commit()

        resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["updated"] is False

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/metrics/catch-up")
        assert resp.status_code == 401


class TestRecalculate:
    async def test_returns_202_immediately(self, client, auth_headers):
        resp = await client.post("/api/metrics/recalculate", headers=auth_headers)
        assert resp.status_code == 202
        assert resp.json()["status"] == "recalculation started"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/metrics/recalculate")
        assert resp.status_code == 401

    async def test_bg_full_recalculate_updates_tss(self, client, auth_headers, session):
        """Call _bg_full_recalculate directly with the test session to verify it updates TSS."""
        from backend.app.api.metrics import _bg_full_recalculate
        from tests.conftest import _TEST_TEAM_ID

        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]

        # Set FTP on athlete
        await client.put("/api/athlete/", json={"ftp": 250}, headers=auth_headers)

        # Create a processed activity with a recent date (within 180-day lookback window)
        from datetime import date, timedelta
        recent_date = (date.today() - timedelta(days=7)).isoformat() + "T10:00:00Z"
        act_resp = await client.post(
            "/api/activities/",
            json={"sport_type": "Ride", "start_time": recent_date, "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = act_resp.json()["id"]

        # Add a short power stream (< 30 points so NP falls back to avg_power)
        act_result = await session.execute(select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        activity.avg_power = 200.0
        activity.tss = None  # clear so we can verify it gets set
        session.add(ActivityStream(
            activity_id=activity_id,
            stream_type="power",
            data=[200] * 20,  # too short for NP calculation
        ))
        await session.commit()

        # Route _bg_full_recalculate to use the test session
        @asynccontextmanager
        async def _test_factory():
            yield session

        with patch("backend.app.api.metrics.get_team_session_factory", return_value=lambda: _test_factory()):
            await _bg_full_recalculate(_TEST_TEAM_ID, athlete_id)

        await session.refresh(activity)
        assert activity.tss is not None


# ── Activity summary ───────────────────────────────────────────────────────────

class TestActivitySummary:
    async def _add_activity(self, client, auth_headers, sport_type, start_time, duration_s, distance_m):
        resp = await client.post(
            "/api/activities/",
            json={
                "sport_type": sport_type,
                "start_time": start_time,
                "duration_s": duration_s,
                "distance_m": distance_m,
            },
            headers=auth_headers,
        )
        assert resp.status_code == 201
        return resp.json()["id"]

    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/metrics/activity-summary", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"num_activities": 0, "total_duration_s": 0, "total_distance_m": 0.0}

    async def test_totals_only_cycling(self, client, auth_headers):
        today = date.today()
        recent = (today - timedelta(days=5)).isoformat() + "T10:00:00Z"
        # Two cycling rides (counted)
        await self._add_activity(client, auth_headers, "Ride", recent, 3600, 30000.0)
        await self._add_activity(client, auth_headers, "VirtualRide", recent, 1800, 15000.0)
        # Non-cycling activities (excluded)
        await self._add_activity(client, auth_headers, "Run", recent, 1200, 5000.0)
        await self._add_activity(client, auth_headers, "Yoga", recent, 2400, 0.0)

        resp = await client.get("/api/metrics/activity-summary?days=30", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_activities"] == 2
        assert data["total_duration_s"] == 5400
        assert data["total_distance_m"] == 45000.0

    async def test_days_filter_excludes_older_activities(self, client, auth_headers):
        today = date.today()
        recent = (today - timedelta(days=5)).isoformat() + "T10:00:00Z"
        old = (today - timedelta(days=120)).isoformat() + "T10:00:00Z"
        await self._add_activity(client, auth_headers, "Ride", recent, 3600, 30000.0)
        await self._add_activity(client, auth_headers, "Ride", old, 7200, 60000.0)

        resp = await client.get("/api/metrics/activity-summary?days=30", headers=auth_headers)
        data = resp.json()
        assert data["num_activities"] == 1
        assert data["total_duration_s"] == 3600
        assert data["total_distance_m"] == 30000.0

    async def test_start_and_end_range(self, client, auth_headers):
        today = date.today()
        in_range = (today - timedelta(days=40)).isoformat() + "T10:00:00Z"
        out_range = (today - timedelta(days=5)).isoformat() + "T10:00:00Z"
        await self._add_activity(client, auth_headers, "Ride", in_range, 3600, 30000.0)
        await self._add_activity(client, auth_headers, "Ride", out_range, 1800, 15000.0)

        start = str(today - timedelta(days=60))
        end = str(today - timedelta(days=20))
        resp = await client.get(
            f"/api/metrics/activity-summary?start={start}&end={end}", headers=auth_headers
        )
        data = resp.json()
        assert data["num_activities"] == 1
        assert data["total_duration_s"] == 3600

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/activity-summary")
        assert resp.status_code == 401


# ── Zones ──────────────────────────────────────────────────────────────────────

class TestZonesEndpoint:
    async def _create_activity_with_streams(self, client, auth_headers, session, hr_data=None, power_data=None):
        resp = await client.post(
            "/api/activities/",
            json={"sport_type": "Ride", "start_time": "2025-04-01T08:00:00Z", "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = resp.json()["id"]
        if hr_data:
            session.add(ActivityStream(activity_id=activity_id, stream_type="heartrate", data=hr_data))
        if power_data:
            session.add(ActivityStream(activity_id=activity_id, stream_type="power", data=power_data))
        await session.commit()
        return activity_id

    async def test_zones_with_power_stream(self, client, auth_headers, session):
        # Configure power zones on athlete
        await client.put(
            "/api/athlete/",
            json={"ftp": 250, "power_zones": [
                {"name": "Z1", "low": 0, "high": 150},
                {"name": "Z2", "low": 151, "high": 210},
                {"name": "Z3", "low": 211, "high": 300},
            ]},
            headers=auth_headers,
        )
        activity_id = await self._create_activity_with_streams(
            client, auth_headers, session,
            power_data=[100] * 60 + [180] * 60 + [250] * 60,
        )
        resp = await client.get(f"/api/metrics/zones/{activity_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "power" in data
        assert sum(data["power"].values()) == 180  # 3 * 60 power values

    async def test_zones_with_hr_stream(self, client, auth_headers, session):
        await client.put(
            "/api/athlete/",
            json={"hr_zones": [
                {"name": "Z1", "low": 0, "high": 120},
                {"name": "Z2", "low": 121, "high": 150},
                {"name": "Z3", "low": 151, "high": 185},
            ]},
            headers=auth_headers,
        )
        activity_id = await self._create_activity_with_streams(
            client, auth_headers, session,
            hr_data=[110] * 30 + [135] * 30 + [160] * 40,
        )
        resp = await client.get(f"/api/metrics/zones/{activity_id}", headers=auth_headers)
        assert resp.status_code == 200
        assert "hr" in resp.json()

    async def test_no_zones_configured_returns_400(self, client, auth_headers, session):
        # Athlete has no zones at all
        activity_id = await self._create_activity_with_streams(
            client, auth_headers, session, hr_data=[150] * 100
        )
        resp = await client.get(f"/api/metrics/zones/{activity_id}", headers=auth_headers)
        assert resp.status_code == 400

    async def test_nonexistent_activity_returns_404(self, client, auth_headers):
        await client.put(
            "/api/athlete/",
            json={"power_zones": [{"name": "Z1", "low": 0, "high": 300}]},
            headers=auth_headers,
        )
        resp = await client.get("/api/metrics/zones/no-such-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/zones/some-id")
        assert resp.status_code == 401


# ── FTP history ────────────────────────────────────────────────────────────────

class TestFtpHistory:
    async def test_empty_when_no_ftp_tests(self, client, auth_headers):
        resp = await client.get("/api/metrics/ftp-history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_ftp_tests(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]

        act_result = await session.execute(select(Athlete).where(Athlete.id == athlete_id))
        athlete = act_result.scalar_one()
        athlete.ftp_tests = [
            {"date": "2025-01-01", "ftp": 240},
            {"date": "2025-04-01", "ftp": 260},
        ]
        await session.commit()

        resp = await client.get("/api/metrics/ftp-history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/ftp-history")
        assert resp.status_code == 401


# ── Fitness date/range filter edge cases ──────────────────────────────────────

class TestFitnessDateRange:
    async def test_start_and_end_filter(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        for offset in [100, 50, 10]:
            session.add(DailyMetric(
                athlete_id=athlete_id,
                date=today - timedelta(days=offset),
                ctl=float(offset), atl=float(offset), tsb=0.0, tss_day=0.0,
            ))
        await session.commit()

        start = str(today - timedelta(days=60))
        end = str(today - timedelta(days=20))
        resp = await client.get(f"/api/metrics/fitness?start={start}&end={end}", headers=auth_headers)
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ctl"] == 50.0

    async def test_current_falls_back_to_latest_when_no_today_metric(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete/", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        yesterday = date.today() - timedelta(days=1)

        session.add(DailyMetric(
            athlete_id=athlete_id, date=yesterday,
            ctl=55.0, atl=60.0, tsb=-5.0, tss_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["ctl"] == 55.0
