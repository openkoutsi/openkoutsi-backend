"""
Integration tests for /api/metrics endpoints.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import date, timedelta
from unittest.mock import patch

from sqlalchemy import select

from backend.app.models.user_orm import (
    Activity, ActivityStream, DailyMetric, Athlete, PlannedWorkout, TrainingPlan,
)


class TestGetFitness:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/metrics/fitness", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_inserted_metrics(self, client, auth_headers, session):
        # Get the athlete ID
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        metric = DailyMetric(
            athlete_id=athlete_id,
            date=today,
            fitness=30.0,
            fatigue=40.0,
            form=-10.0,
            load_day=80.0,
        )
        session.add(metric)
        await session.commit()

        resp = await client.get("/api/metrics/fitness", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["fitness"] == 30.0
        assert data[0]["fatigue"] == 40.0

    async def test_days_filter_limits_results(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        for offset in [60, 10, 1]:
            session.add(DailyMetric(
                athlete_id=athlete_id,
                date=today - timedelta(days=offset),
                fitness=10.0, fatigue=10.0, form=0.0, load_day=50.0,
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
        assert data["fitness"] == 0.0
        assert data["fatigue"] == 0.0
        assert data["form"] == 0.0
        assert "form" in data

    async def test_form_label_computed_from_form(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        # Form > 25 → "peak"
        session.add(DailyMetric(
            athlete_id=athlete_id, date=today,
            fitness=50.0, fatigue=20.0, form=30.0, load_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.json()["form_label"] == "peak"

    async def test_tired_form_label(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        session.add(DailyMetric(
            athlete_id=athlete_id, date=today,
            fitness=40.0, fatigue=60.0, form=-20.0, load_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.json()["form_label"] == "tired"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/fitness/current")
        assert resp.status_code == 401


class TestCatchUp:
    async def test_creates_today_metric_when_missing(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        # Seed yesterday so Fitness/Fatigue can be inherited
        session.add(DailyMetric(
            athlete_id=athlete_id,
            date=today - timedelta(days=1),
            fitness=40.0, fatigue=50.0, form=-10.0, load_day=80.0,
        ))
        await session.commit()

        resp = await client.post("/api/metrics/catch-up", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

        # Today's DailyMetric should now exist
        resp2 = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp2.json()["date"] == str(today)

    async def test_returns_not_updated_when_already_current(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        session.add(DailyMetric(
            athlete_id=athlete_id,
            date=today,
            fitness=30.0, fatigue=35.0, form=-5.0, load_day=0.0,
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
        """Call _bg_full_recalculate directly with the test session to verify it updates Load."""
        from backend.app.api.metrics import _bg_full_recalculate
        from tests.conftest import _TEST_USER_ID

        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]

        # Set FTP on athlete
        await client.patch("/api/athlete", json={"ftp": 250}, headers=auth_headers)

        # Create a processed activity with a recent date (within 180-day lookback window)
        from datetime import date, timedelta
        recent_date = (date.today() - timedelta(days=7)).isoformat() + "T10:00:00Z"
        act_resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": recent_date, "duration_s": 3600},
            headers=auth_headers,
        )
        activity_id = act_resp.json()["id"]

        # Add a short power stream (< 30 points so Weighted Power falls back to avg_power)
        act_result = await session.execute(select(Activity).where(Activity.id == activity_id))
        activity = act_result.scalar_one()
        activity.avg_power = 200.0
        activity.load = None  # clear so we can verify it gets set
        session.add(ActivityStream(
            activity_id=activity_id,
            stream_type="power",
            data=[200] * 20,  # too short for Weighted Power calculation
        ))
        await session.commit()

        # Route _bg_full_recalculate to use the test session
        @asynccontextmanager
        async def _test_factory():
            yield session

        with patch("backend.app.api.metrics.get_user_session_factory", return_value=lambda: _test_factory()):
            await _bg_full_recalculate(_TEST_USER_ID, athlete_id)

        await session.refresh(activity)
        assert activity.load is not None


# ── Activity summary ───────────────────────────────────────────────────────────

class TestActivitySummary:
    async def _add_activity(self, client, auth_headers, sport_type, start_time, duration_s, distance_m):
        resp = await client.post(
            "/api/activities",
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
            "/api/activities",
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
        await client.patch(
            "/api/athlete",
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
        await client.patch(
            "/api/athlete",
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
        await client.patch(
            "/api/athlete",
            json={"power_zones": [{"name": "Z1", "low": 0, "high": 300}]},
            headers=auth_headers,
        )
        resp = await client.get("/api/metrics/zones/no-such-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/zones/some-id")
        assert resp.status_code == 401


# ── Weekly accumulated time in zones (issue #27) ────────────────────────────────

class TestWeeklyZonesEndpoint:
    _POWER_ZONES = [
        {"name": "Z1", "low": 0, "high": 150},
        {"name": "Z2", "low": 151, "high": 210},
        {"name": "Z3", "low": 211, "high": 300},
    ]

    async def _set_power_zones(self, client, auth_headers, zones=None):
        await client.patch(
            "/api/athlete",
            json={"ftp": 250, "power_zones": zones or self._POWER_ZONES},
            headers=auth_headers,
        )

    async def _make_activity(self, client, auth_headers, session, start_time, power_data):
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": start_time, "duration_s": len(power_data)},
            headers=auth_headers,
        )
        activity_id = resp.json()["id"]
        session.add(ActivityStream(activity_id=activity_id, stream_type="power", data=power_data))
        await session.commit()
        return activity_id

    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/metrics/zones/weekly", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_weekly_not_shadowed_by_activity_id_route(self, client, auth_headers):
        # "weekly" must hit the weekly endpoint, not /zones/{activity_id} (404).
        resp = await client.get("/api/metrics/zones/weekly", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_aggregates_within_a_week(self, client, auth_headers, session):
        await self._set_power_zones(client, auth_headers)
        # 2025-04-01 (Tue) and 2025-04-02 (Wed) → same ISO week (Mon 2025-03-31).
        await self._make_activity(
            client, auth_headers, session, "2025-04-01T08:00:00Z",
            [100] * 60 + [250] * 60,  # Z1:60, Z3:60
        )
        await self._make_activity(
            client, auth_headers, session, "2025-04-02T08:00:00Z",
            [180] * 30,  # Z2:30
        )
        resp = await client.get("/api/metrics/zones/weekly", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["week_start"] == "2025-03-31"
        assert data[0]["power"] == {"Z1": 60, "Z2": 30, "Z3": 60}

    async def test_separate_weeks_sorted(self, client, auth_headers, session):
        await self._set_power_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, "2025-04-08T08:00:00Z", [250] * 45,  # Z3:45
        )
        await self._make_activity(
            client, auth_headers, session, "2025-04-01T08:00:00Z", [100] * 20,  # Z1:20
        )
        resp = await client.get("/api/metrics/zones/weekly", headers=auth_headers)
        data = resp.json()
        assert [b["week_start"] for b in data] == ["2025-03-31", "2025-04-07"]
        assert data[0]["power"] == {"Z1": 20}
        assert data[1]["power"] == {"Z3": 45}

    async def test_snapshot_frozen_against_zone_changes(self, client, auth_headers, session):
        await self._set_power_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, "2025-04-01T08:00:00Z",
            [100] * 60 + [250] * 60,  # Z1:60, Z3:60 under the original zones
        )
        first = (await client.get("/api/metrics/zones/weekly", headers=auth_headers)).json()
        assert first[0]["power"] == {"Z1": 60, "Z3": 60}

        # Collapse everything into a single wide zone; past activity must not move.
        await self._set_power_zones(
            client, auth_headers, zones=[{"name": "Z1", "low": 0, "high": 500}],
        )
        second = (await client.get("/api/metrics/zones/weekly", headers=auth_headers)).json()
        assert second[0]["power"] == {"Z1": 60, "Z3": 60}

    async def test_date_range_filter(self, client, auth_headers, session):
        await self._set_power_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, "2025-04-01T08:00:00Z", [100] * 10,
        )
        await self._make_activity(
            client, auth_headers, session, "2025-05-01T08:00:00Z", [250] * 10,
        )
        resp = await client.get(
            "/api/metrics/zones/weekly?start=2025-04-20&end=2025-05-10",
            headers=auth_headers,
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["power"] == {"Z3": 10}

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/zones/weekly")
        assert resp.status_code == 401


# ── FTP history ────────────────────────────────────────────────────────────────

class TestFtpHistory:
    async def test_empty_when_no_ftp_tests(self, client, auth_headers):
        resp = await client.get("/api/metrics/ftp/history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_ftp_tests(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]

        act_result = await session.execute(select(Athlete).where(Athlete.id == athlete_id))
        athlete = act_result.scalar_one()
        athlete.ftp_tests = [
            {"date": "2025-01-01", "ftp": 240},
            {"date": "2025-04-01", "ftp": 260},
        ]
        await session.commit()

        resp = await client.get("/api/metrics/ftp/history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/ftp/history")
        assert resp.status_code == 401


# ── Fitness date/range filter edge cases ──────────────────────────────────────

class TestFitnessDateRange:
    async def test_start_and_end_filter(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        for offset in [100, 50, 10]:
            session.add(DailyMetric(
                athlete_id=athlete_id,
                date=today - timedelta(days=offset),
                fitness=float(offset), fatigue=float(offset), form=0.0, load_day=0.0,
            ))
        await session.commit()

        start = str(today - timedelta(days=60))
        end = str(today - timedelta(days=20))
        resp = await client.get(f"/api/metrics/fitness?start={start}&end={end}", headers=auth_headers)
        data = resp.json()
        assert len(data) == 1
        assert data[0]["fitness"] == 50.0

    async def test_current_falls_back_to_latest_when_no_today_metric(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        yesterday = date.today() - timedelta(days=1)

        session.add(DailyMetric(
            athlete_id=athlete_id, date=yesterday,
            fitness=55.0, fatigue=60.0, form=-5.0, load_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/current", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["fitness"] == 55.0


class TestFitnessForecast:
    """GET /api/metrics/fitness/forecast — the forward projection (issue #34)."""

    async def _seed_plan(self, session, athlete_id, *, loads, status="active",
                         plan_id="p-forecast", start=None):
        """Active plan starting tomorrow, with ``loads`` as week 1 days 1..n."""
        start = start or (date.today() + timedelta(days=1))
        session.add(TrainingPlan(
            id=plan_id, athlete_id=athlete_id, name="Forecast plan",
            start_date=start, end_date=start + timedelta(days=6),
            weeks=1, status=status,
        ))
        for day, load in enumerate(loads, start=1):
            session.add(PlannedWorkout(
                id=f"{plan_id}-d{day}", plan_id=plan_id,
                week_number=1, day_of_week=day,
                workout_type="endurance", target_load=load,
            ))
        await session.commit()
        return start

    async def test_returns_only_dates_after_today(self, client, auth_headers):
        resp = await client.get("/api/metrics/fitness/forecast?days=7", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        today = date.today()
        assert len(data) == 7
        assert data[0]["date"] == str(today + timedelta(days=1))
        assert data[-1]["date"] == str(today + timedelta(days=7))
        assert all(date.fromisoformat(d["date"]) > today for d in data)

    async def test_defaults_to_90_days(self, client, auth_headers):
        resp = await client.get("/api/metrics/fitness/forecast", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 90

    async def test_rows_are_marked_projected_and_labelled(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        session.add(DailyMetric(
            athlete_id=ath_resp.json()["id"], date=date.today(),
            fitness=60.0, fatigue=20.0, form=40.0, load_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/forecast?days=3", headers=auth_headers)
        data = resp.json()

        assert all(d["projected"] is True for d in data)
        # Form well above the "peak" threshold, and rising as fatigue decays.
        assert data[0]["form_label"] == "peak"

    async def test_planned_load_appears_in_the_projection(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        start = await self._seed_plan(
            session, ath_resp.json()["id"], loads=[60, None, 80, None, None, 120, None],
        )

        resp = await client.get("/api/metrics/fitness/forecast?days=10", headers=auth_headers)
        by_date = {d["date"]: d for d in resp.json()}

        assert by_date[str(start)]["daily_load"] == 60.0
        assert by_date[str(start + timedelta(days=2))]["daily_load"] == 80.0
        # A day with no prescribed load is a rest day, not a gap.
        assert by_date[str(start + timedelta(days=1))]["daily_load"] == 0.0
        assert by_date[str(start + timedelta(days=6))]["fitness"] > 0.0

    async def test_no_active_plan_is_pure_decay(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        session.add(DailyMetric(
            athlete_id=ath_resp.json()["id"], date=date.today(),
            fitness=70.0, fatigue=50.0, form=20.0, load_day=0.0,
        ))
        await session.commit()

        resp = await client.get("/api/metrics/fitness/forecast?days=21", headers=auth_headers)
        data = resp.json()

        assert all(d["daily_load"] == 0.0 for d in data)
        fitness = [d["fitness"] for d in data]
        assert fitness == sorted(fitness, reverse=True)
        assert fitness[0] < 70.0

    async def test_archived_plan_does_not_contribute(self, client, auth_headers, session):
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        await self._seed_plan(
            session, ath_resp.json()["id"], loads=[100] * 7, status="archived",
        )

        resp = await client.get("/api/metrics/fitness/forecast?days=10", headers=auth_headers)
        assert all(d["daily_load"] == 0.0 for d in resp.json())

    async def test_other_athletes_plans_are_not_projected(self, client, auth_headers, session):
        other = Athlete(id=f"other-{uuid.uuid4().hex[:8]}", global_user_id="other-user", ftp_tests=[])
        session.add(other)
        await session.commit()
        await self._seed_plan(session, other.id, loads=[150] * 7, plan_id="p-other")

        resp = await client.get("/api/metrics/fitness/forecast?days=10", headers=auth_headers)
        assert all(d["daily_load"] == 0.0 for d in resp.json())

    async def test_days_bounds_enforced(self, client, auth_headers):
        assert (await client.get("/api/metrics/fitness/forecast?days=0", headers=auth_headers)).status_code == 422
        assert (await client.get("/api/metrics/fitness/forecast?days=366", headers=auth_headers)).status_code == 422
        assert (await client.get("/api/metrics/fitness/forecast?days=365", headers=auth_headers)).status_code == 200

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/fitness/forecast")
        assert resp.status_code == 401

    async def test_the_projection_itself_is_never_persisted(self, client, auth_headers, session):
        # The endpoint catches metrics up first, so it may write a row for today
        # — but no projected day may ever reach the DB.
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        await self._seed_plan(session, ath_resp.json()["id"], loads=[100] * 7)

        await client.get("/api/metrics/fitness/forecast?days=30", headers=auth_headers)

        stored = (await session.execute(select(DailyMetric))).scalars().all()
        assert all(m.date <= date.today() for m in stored)

    async def test_result_does_not_depend_on_catch_up_having_run_first(
        self, client, auth_headers, session
    ):
        """The forecast catches up itself, so client call order can't change it.

        Without that, an athlete whose metrics were a few days stale got a
        different (and always fresher-looking) series depending on whether the
        dashboard had been visited first.
        """
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        session.add(DailyMetric(
            athlete_id=athlete_id, date=date.today() - timedelta(days=5),
            fitness=50.0, fatigue=40.0, form=10.0, load_day=0.0,
        ))
        await self._seed_plan(session, athlete_id, loads=[100] * 7)

        cold = await client.get("/api/metrics/fitness/forecast?days=14", headers=auth_headers)
        await client.post("/api/metrics/catch-up", headers=auth_headers)
        warm = await client.get("/api/metrics/fitness/forecast?days=14", headers=auth_headers)

        assert cold.status_code == 200
        assert cold.json() == warm.json()

    async def test_historical_endpoint_is_unchanged(self, client, auth_headers, session):
        """Regression guard: adding the forecast must not alter GET /metrics/fitness."""
        ath_resp = await client.get("/api/athlete", headers=auth_headers)
        athlete_id = ath_resp.json()["id"]
        today = date.today()

        session.add(DailyMetric(
            athlete_id=athlete_id, date=today,
            fitness=30.0, fatigue=40.0, form=-10.0, load_day=80.0,
        ))
        await self._seed_plan(session, athlete_id, loads=[100] * 7)

        resp = await client.get("/api/metrics/fitness", headers=auth_headers)
        data = resp.json()

        assert data == [{
            "date": str(today),
            "fitness": 30.0,
            "fatigue": 40.0,
            "form": -10.0,
            "load_day": 80.0,
            "daily_load": 80.0,
        }]
