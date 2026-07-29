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

# Zone lists are fixed at seven power / five HR zones (issue #38), so every
# test that configures zones has to send a full set.
_CANONICAL_POWER_ZONES = [
    {"name": "Z1 Recovery", "low": 0, "high": 137},
    {"name": "Z2 Endurance", "low": 137, "high": 187},
    {"name": "Z3 Tempo", "low": 187, "high": 217},
    {"name": "Z4 Threshold", "low": 217, "high": 237},
    {"name": "Z5 VO2max", "low": 237, "high": 265},
    {"name": "Z6 Anaerobic", "low": 265, "high": 300},
    {"name": "Z7 Neuromuscular", "low": 300, "high": 9999},
]

_CANONICAL_HR_ZONES = [
    {"name": "Z1 Recovery", "low": 0, "high": 120},
    {"name": "Z2 Endurance", "low": 120, "high": 140},
    {"name": "Z3 Tempo", "low": 140, "high": 160},
    {"name": "Z4 Threshold", "low": 160, "high": 172},
    {"name": "Z5 VO2max", "low": 172, "high": 200},
]


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


# ── Aerobic efficiency trend ───────────────────────────────────────────────────

class TestEfficiencyTrend:
    """Issue #37 — GET /api/metrics/efficiency."""

    async def _add_ride(
        self, client, auth_headers, session, *,
        days_ago=5, sport_type="Ride", duration_s=3600,
        avg_power=200.0, weighted_power=206.0, avg_hr=140.0,
        workout_category="endurance", decoupling_pct=None,
    ):
        start = (date.today() - timedelta(days=days_ago)).isoformat() + "T10:00:00Z"
        resp = await client.post(
            "/api/activities",
            json={"sport_type": sport_type, "start_time": start, "duration_s": duration_s},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        activity_id = resp.json()["id"]

        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()
        activity.avg_power = avg_power
        activity.weighted_power = weighted_power
        activity.avg_hr = avg_hr
        activity.workout_category = workout_category
        activity.decoupling_pct = decoupling_pct
        await session.commit()
        return activity_id

    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/metrics/efficiency", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_returns_point_for_steady_ride(self, client, auth_headers, session):
        activity_id = await self._add_ride(
            client, auth_headers, session, decoupling_pct=3.4
        )
        resp = await client.get("/api/metrics/efficiency", headers=auth_headers)
        assert resp.status_code == 200
        points = resp.json()
        assert len(points) == 1
        assert points[0]["activity_id"] == activity_id
        assert points[0]["efficiency_factor"] == round(206.0 / 140.0, 3)
        assert points[0]["decoupling_pct"] == 3.4
        assert points[0]["duration_s"] == 3600

    async def test_ordered_oldest_first(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, days_ago=2)
        await self._add_ride(client, auth_headers, session, days_ago=30)
        points = (await client.get("/api/metrics/efficiency", headers=auth_headers)).json()
        assert [p["date"] for p in points] == sorted(p["date"] for p in points)

    async def test_excludes_interval_rides(self, client, auth_headers, session):
        # High variability index — the ride was too surgy for the number to mean
        # anything, the same test the decoupling gate applies.
        await self._add_ride(
            client, auth_headers, session, avg_power=180.0, weighted_power=260.0
        )
        assert (await client.get("/api/metrics/efficiency", headers=auth_headers)).json() == []

    async def test_excludes_interval_categories(self, client, auth_headers, session):
        for category in ("vo2max", "anaerobic", "sprint"):
            await self._add_ride(
                client, auth_headers, session, workout_category=category
            )
        assert (await client.get("/api/metrics/efficiency", headers=auth_headers)).json() == []

    async def test_excludes_non_cycling(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, sport_type="Run")
        assert (await client.get("/api/metrics/efficiency", headers=auth_headers)).json() == []

    async def test_excludes_rides_without_heart_rate(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, avg_hr=None)
        assert (await client.get("/api/metrics/efficiency", headers=auth_headers)).json() == []

    async def test_excludes_very_short_rides(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, duration_s=600)
        assert (await client.get("/api/metrics/efficiency", headers=auth_headers)).json() == []

    async def test_includes_uncategorized_steady_rides(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, workout_category=None)
        assert len(
            (await client.get("/api/metrics/efficiency", headers=auth_headers)).json()
        ) == 1

    async def test_days_filter(self, client, auth_headers, session):
        await self._add_ride(client, auth_headers, session, days_ago=5)
        await self._add_ride(client, auth_headers, session, days_ago=200)
        points = (
            await client.get("/api/metrics/efficiency?days=30", headers=auth_headers)
        ).json()
        assert len(points) == 1

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/efficiency")
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
            json={"ftp": 250, "power_zones": _CANONICAL_POWER_ZONES},
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
            json={"hr_zones": _CANONICAL_HR_ZONES},
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
            json={"power_zones": _CANONICAL_POWER_ZONES},
            headers=auth_headers,
        )
        resp = await client.get("/api/metrics/zones/no-such-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/metrics/zones/some-id")
        assert resp.status_code == 401


# ── Weekly accumulated time in zones (issue #27) ────────────────────────────────

class TestWeeklyZonesEndpoint:
    # The original three zones, extended to the canonical seven that the API
    # now requires (issue #38). The lower boundaries are unchanged, so every
    # expectation below still describes the same behaviour.
    _POWER_ZONES = [
        {"name": "Z1", "low": 0, "high": 150},
        {"name": "Z2", "low": 151, "high": 210},
        {"name": "Z3", "low": 211, "high": 300},
        {"name": "Z4", "low": 301, "high": 330},
        {"name": "Z5", "low": 331, "high": 360},
        {"name": "Z6", "low": 361, "high": 400},
        {"name": "Z7", "low": 401, "high": 9999},
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

        # Shift every boundary so the same samples would now land in different
        # zones (100 → Z2, 250 → Z5); the past activity must not move.
        await self._set_power_zones(
            client, auth_headers, zones=[
                {"name": "Z1", "low": 0, "high": 90},
                {"name": "Z2", "low": 90, "high": 140},
                {"name": "Z3", "low": 140, "high": 190},
                {"name": "Z4", "low": 190, "high": 230},
                {"name": "Z5", "low": 230, "high": 260},
                {"name": "Z6", "low": 260, "high": 300},
                {"name": "Z7", "low": 300, "high": 9999},
            ],
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


# ── Intensity distribution over a block (issue #38) ────────────────────────────

class TestIntensityDistribution:
    """Three-band distribution and its shape over a training block.

    Power samples are chosen against ``_CANONICAL_POWER_ZONES``:
    100 W → Z1 and 160 W → Z2 (band 1), 200 W → Z3 (band 2), 250 W → Z5
    (band 3).
    """

    _URL = "/api/metrics/intensity-distribution"

    async def _set_zones(self, client, auth_headers, power=True, hr=False):
        body: dict = {"ftp": 250}
        if power:
            body["power_zones"] = _CANONICAL_POWER_ZONES
        if hr:
            body["hr_zones"] = _CANONICAL_HR_ZONES
        resp = await client.patch("/api/athlete", json=body, headers=auth_headers)
        assert resp.status_code == 200

    async def _make_activity(
        self, client, auth_headers, session, days_ago, power_data=None, hr_data=None,
    ):
        start = (date.today() - timedelta(days=days_ago)).isoformat() + "T08:00:00Z"
        length = len(power_data or hr_data or [])
        resp = await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": start, "duration_s": length},
            headers=auth_headers,
        )
        activity_id = resp.json()["id"]
        if power_data:
            session.add(ActivityStream(
                activity_id=activity_id, stream_type="power", data=power_data,
            ))
        if hr_data:
            session.add(ActivityStream(
                activity_id=activity_id, stream_type="heartrate", data=hr_data,
            ))
        await session.commit()
        return activity_id

    @staticmethod
    def _bands(body):
        return {b["band"]: b for b in body["bands"]}

    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get(self._URL, headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is None
        assert body["coverage"]["activities_total"] == 0
        assert body["coverage"]["activities_used"] == 0
        assert [b["pct"] for b in body["bands"]] == [0.0, 0.0, 0.0]

    async def test_time_method_splits_into_three_bands(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, 10,
            power_data=[100] * 600 + [200] * 100 + [250] * 50,
        )
        body = (await client.get(self._URL, headers=auth_headers)).json()

        bands = self._bands(body)
        assert bands[1]["seconds"] == 600
        assert bands[2]["seconds"] == 100
        assert bands[3]["seconds"] == 50
        assert bands[1]["pct"] == 80.0
        assert body["method"] == "time"
        assert body["basis"] == "power"
        assert body["classification"] == "pyramidal"
        # The time method counts seconds, so there is no session count.
        assert all(b["sessions"] is None for b in body["bands"])

    async def test_polarized_block(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, 10,
            power_data=[100] * 800 + [200] * 50 + [250] * 150,
        )
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["classification"] == "polarized"

    async def test_predominantly_low_block(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, 10,
            power_data=[100] * 970 + [200] * 20 + [250] * 10,
        )
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["classification"] == "predominantly_low"

    async def test_aggregates_across_activities(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 20, power_data=[100] * 300)
        await self._make_activity(client, auth_headers, session, 10, power_data=[200] * 100)
        body = (await client.get(self._URL, headers=auth_headers)).json()

        bands = self._bands(body)
        assert bands[1]["seconds"] == 300
        assert bands[2]["seconds"] == 100
        assert body["coverage"]["activities_used"] == 2
        assert body["coverage"]["seconds_total"] == 400

    async def test_hr_basis(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers, power=False, hr=True)
        # 110 bpm → Z1 (band 1), 150 → Z3 (band 2), 180 → Z5 (band 3).
        await self._make_activity(
            client, auth_headers, session, 10,
            hr_data=[110] * 300 + [150] * 60 + [180] * 40,
        )
        body = (await client.get(f"{self._URL}?basis=hr", headers=auth_headers)).json()

        bands = self._bands(body)
        assert body["basis"] == "hr"
        assert bands[1]["seconds"] == 300
        assert bands[2]["seconds"] == 60
        assert bands[3]["seconds"] == 40

    async def test_basis_defaults_to_hr_without_power_zones(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers, power=False, hr=True)
        await self._make_activity(client, auth_headers, session, 10, hr_data=[110] * 300)
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["basis"] == "hr"
        assert self._bands(body)[1]["seconds"] == 300

    async def test_basis_prefers_power_when_both_exist(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers, power=True, hr=True)
        await self._make_activity(
            client, auth_headers, session, 10,
            power_data=[100] * 300, hr_data=[110] * 300,
        )
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["basis"] == "power"

    async def test_requested_basis_without_data_is_empty_not_an_error(
        self, client, auth_headers, session
    ):
        await self._set_zones(client, auth_headers, power=True, hr=False)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)
        resp = await client.get(f"{self._URL}?basis=hr", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["classification"] is None
        assert body["coverage"]["activities_used"] == 0
        assert body["coverage"]["activities_total"] == 1

    async def test_session_method_counts_whole_sessions(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        # Three easy rides and one VO2max session. By time the hard session is
        # a rounding error; by session goal it is a quarter of the block.
        for days_ago, category in (
            (30, "endurance"), (20, "endurance"), (10, "recovery"), (5, "vo2max"),
        ):
            activity_id = await self._make_activity(
                client, auth_headers, session, days_ago, power_data=[100] * 3600,
            )
            resp = await client.patch(
                f"/api/activities/{activity_id}",
                json={"workout_category": category},
                headers=auth_headers,
            )
            assert resp.status_code == 200

        body = (await client.get(f"{self._URL}?method=session", headers=auth_headers)).json()
        bands = self._bands(body)
        assert body["method"] == "session"
        # Session counting works off the workout category, so power vs HR
        # does not apply.
        assert body["basis"] is None
        assert bands[1]["sessions"] == 3
        assert bands[2]["sessions"] == 0
        assert bands[3]["sessions"] == 1
        assert bands[1]["pct"] == 75.0
        assert bands[3]["pct"] == 25.0

    async def test_session_method_excludes_non_cycling_and_unset(
        self, client, auth_headers, session
    ):
        await self._set_zones(client, auth_headers)
        endurance = await self._make_activity(
            client, auth_headers, session, 20, power_data=[100] * 3600,
        )
        await client.patch(
            f"/api/activities/{endurance}",
            json={"workout_category": "endurance"},
            headers=auth_headers,
        )
        strength = await self._make_activity(
            client, auth_headers, session, 15, power_data=[100] * 3600,
        )
        await client.patch(
            f"/api/activities/{strength}",
            json={"workout_category": "strength"},
            headers=auth_headers,
        )
        # A third ride keeps whatever category processing gave it — with no
        # power-derived intensity that is nothing at all.
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 3600)

        body = (await client.get(f"{self._URL}?method=session", headers=auth_headers)).json()
        assert body["coverage"]["activities_total"] == 3
        assert body["coverage"]["activities_used"] == 1
        assert self._bands(body)[1]["sessions"] == 1

    async def test_coverage_counts_activities_without_snapshots(
        self, client, auth_headers, session
    ):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)
        # No stream at all → nothing to freeze, so it can't reach the bands.
        start = (date.today() - timedelta(days=8)).isoformat() + "T08:00:00Z"
        await client.post(
            "/api/activities",
            json={"sport_type": "Ride", "start_time": start, "duration_s": 3600},
            headers=auth_headers,
        )

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["coverage"]["activities_total"] == 2
        assert body["coverage"]["activities_used"] == 1

    async def test_mixed_zone_flag_is_off_for_a_stable_window(
        self, client, auth_headers, session
    ):
        await self._set_zones(client, auth_headers)
        # Two rides that touched different zones under identical definitions —
        # a partial snapshot must not read as a zone change.
        await self._make_activity(client, auth_headers, session, 20, power_data=[100] * 300)
        await self._make_activity(
            client, auth_headers, session, 10, power_data=[100] * 100 + [250] * 200,
        )
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is False

    async def test_mixed_zone_flag_on_ftp_change_inside_window(
        self, client, auth_headers, session
    ):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)

        athlete = (await session.execute(select(Athlete))).scalars().first()
        athlete.ftp_tests = [
            {"date": (date.today() - timedelta(days=200)).isoformat(), "ftp": 250, "method": "test"},
            {"date": (date.today() - timedelta(days=20)).isoformat(), "ftp": 265, "method": "test"},
        ]
        await session.commit()

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is True

    async def test_restating_the_same_ftp_does_not_flag(self, client, auth_headers, session):
        # Every profile save that includes an FTP appends a test entry, so the
        # presence of one inside the window means nothing on its own.
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)

        athlete = (await session.execute(select(Athlete))).scalars().first()
        athlete.ftp_tests = [
            {"date": (date.today() - timedelta(days=200)).isoformat(), "ftp": 250, "method": "manual"},
            {"date": (date.today() - timedelta(days=20)).isoformat(), "ftp": 250, "method": "manual"},
        ]
        await session.commit()

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is False

    async def test_first_ever_ftp_entry_does_not_flag(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)
        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is False

    async def test_mixed_zone_flag_on_renamed_zones(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 20, power_data=[100] * 300)
        # Read once so the first snapshot is frozen under the original names —
        # otherwise the lazy backfill would compute both with the new zones.
        await client.get(self._URL, headers=auth_headers)

        renamed = [dict(z, name=f"Zone {i + 1}") for i, z in enumerate(_CANONICAL_POWER_ZONES)]
        await client.patch(
            "/api/athlete", json={"power_zones": renamed}, headers=auth_headers,
        )
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is True

    async def test_ftp_change_outside_window_does_not_flag(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)

        athlete = (await session.execute(select(Athlete))).scalars().first()
        athlete.ftp_tests = [
            {"date": (date.today() - timedelta(days=500)).isoformat(), "ftp": 230, "method": "test"},
            {"date": (date.today() - timedelta(days=400)).isoformat(), "ftp": 265, "method": "test"},
        ]
        await session.commit()

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["zone_definitions_changed"] is False

    async def test_default_window_is_twelve_weeks(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 10, power_data=[100] * 300)
        await self._make_activity(client, auth_headers, session, 200, power_data=[250] * 999)

        body = (await client.get(self._URL, headers=auth_headers)).json()
        assert body["start"] == (date.today() - timedelta(days=84)).isoformat()
        assert body["coverage"]["activities_total"] == 1
        assert self._bands(body)[3]["seconds"] == 0

    async def test_days_widens_the_window(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 200, power_data=[100] * 300)

        body = (await client.get(f"{self._URL}?days=365", headers=auth_headers)).json()
        assert body["coverage"]["activities_used"] == 1

    async def test_explicit_start_and_end(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 40, power_data=[100] * 300)
        await self._make_activity(client, auth_headers, session, 5, power_data=[250] * 300)

        start = (date.today() - timedelta(days=50)).isoformat()
        end = (date.today() - timedelta(days=30)).isoformat()
        body = (await client.get(
            f"{self._URL}?start={start}&end={end}", headers=auth_headers,
        )).json()
        assert body["start"] == start
        assert body["end"] == end
        assert body["coverage"]["activities_total"] == 1
        assert self._bands(body)[1]["seconds"] == 300

    async def test_explicit_start_beats_days(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(client, auth_headers, session, 200, power_data=[100] * 300)

        start = (date.today() - timedelta(days=365)).isoformat()
        body = (await client.get(
            f"{self._URL}?start={start}&days=7", headers=auth_headers,
        )).json()
        assert body["start"] == start
        assert body["coverage"]["activities_used"] == 1

    async def test_weekly_zones_output_is_unchanged(self, client, auth_headers, session):
        await self._set_zones(client, auth_headers)
        await self._make_activity(
            client, auth_headers, session, 10, power_data=[100] * 60 + [200] * 30,
        )
        weekly = (await client.get(
            "/api/metrics/zones/weekly?days=90", headers=auth_headers,
        )).json()
        assert len(weekly) == 1
        # Still zone names and seconds, untouched by the band mapping.
        assert weekly[0]["power"] == {"Z1 Recovery": 60, "Z3 Tempo": 30}

    async def test_invalid_method_returns_422(self, client, auth_headers):
        resp = await client.get(f"{self._URL}?method=vibes", headers=auth_headers)
        assert resp.status_code == 422

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get(self._URL)
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
