"""
Integration tests for /api/power/bests endpoint.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.models.team_orm import Activity, ActivityPowerBest, ActivitySource, ActivityStream, Athlete

TESTDATA = Path(__file__).parent.parent.parent / "testdata"
SAMPLE_FIT = TESTDATA / "Zwift_Aerobic_Foundation_Forge.fit"


async def _get_athlete(client, auth_headers, session) -> Athlete:
    resp = await client.get("/api/athlete/", headers=auth_headers)
    athlete_id = resp.json()["id"]
    result = await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    return result.scalar_one()


async def _insert_activity_with_power(
    session, athlete: Athlete, power_stream: list[float], start_time: str
) -> Activity:
    activity = Activity(
        athlete_id=athlete.id,
        name="Test Power Activity",
        sport_type="Ride",
        start_time=datetime.fromisoformat(start_time),
        duration_s=len(power_stream),
        status="processed",
    )
    session.add(activity)
    await session.flush()

    session.add(
        ActivityStream(
            activity_id=activity.id,
            stream_type="power",
            data=power_stream,
        )
    )

    from openkoutsi.training_math import compute_power_bests

    bests = compute_power_bests(power_stream)
    for duration_s, power_w in bests.items():
        session.add(
            ActivityPowerBest(
                activity_id=activity.id,
                athlete_id=athlete.id,
                duration_s=duration_s,
                power_w=power_w,
                activity_start_time=activity.start_time,
            )
        )

    await session.commit()
    await session.refresh(activity)
    return activity


class TestGetPowerBestsEmpty:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/power/bests", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"bests": []}

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/power/bests")
        assert resp.status_code == 401

    async def test_no_power_stream_activity_produces_no_bests(
        self, client, auth_headers, session
    ):
        """Manual activity without a power stream should not produce any bests."""
        await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-01T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        resp = await client.get("/api/power/bests", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["bests"] == []


class TestGetPowerBestsSingleActivity:
    async def test_returns_bests_after_activity_with_power(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 60-second stream at constant 250 W
        stream = [250.0] * 60
        await _insert_activity_with_power(session, athlete, stream, "2025-06-01T10:00:00+00:00")

        resp = await client.get("/api/power/bests", headers=auth_headers)
        assert resp.status_code == 200
        bests = resp.json()["bests"]
        assert len(bests) > 0

        # All returned entries should have rank=1 (only one activity)
        for entry in bests:
            assert entry["rank"] == 1

        # 1s best from a constant 250 W stream must be 250
        ones = [e for e in bests if e["duration_s"] == 1]
        assert len(ones) == 1
        assert ones[0]["power_w"] == pytest.approx(250.0, abs=0.1)

    async def test_durations_longer_than_stream_absent(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # Only 30-second stream — durations > 30s must be absent
        stream = [200.0] * 30
        await _insert_activity_with_power(session, athlete, stream, "2025-06-02T10:00:00+00:00")

        resp = await client.get("/api/power/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        for entry in bests:
            assert entry["duration_s"] <= 30, (
                f"duration {entry['duration_s']}s should not appear for a 30s stream"
            )

    async def test_activity_id_and_name_present(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        activity = await _insert_activity_with_power(
            session, athlete, [300.0] * 60, "2025-06-03T10:00:00+00:00"
        )

        resp = await client.get("/api/power/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        assert len(bests) > 0
        for entry in bests:
            assert entry["activity_id"] == activity.id
            assert entry["activity_name"] == "Test Power Activity"

    async def test_entries_ordered_by_duration_then_rank(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        await _insert_activity_with_power(
            session, athlete, [250.0] * 120, "2025-06-04T10:00:00+00:00"
        )

        resp = await client.get("/api/power/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        durations = [e["duration_s"] for e in bests]
        assert durations == sorted(durations), "bests must be sorted by duration_s"


class TestGetPowerBestsMultipleActivities:
    async def test_top_3_ranking(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)

        # Insert 4 activities with different 60s average powers
        for i, power in enumerate([200.0, 350.0, 300.0, 280.0]):
            await _insert_activity_with_power(
                session,
                athlete,
                [power] * 60,
                f"2025-06-0{i + 1}T10:00:00+00:00",
            )

        resp = await client.get("/api/power/bests", headers=auth_headers)
        bests = resp.json()["bests"]

        sixty_s = [e for e in bests if e["duration_s"] == 60]
        # Must have exactly 3 entries (top-3 cap)
        assert len(sixty_s) == 3
        assert sixty_s[0]["rank"] == 1
        assert sixty_s[0]["power_w"] == pytest.approx(350.0, abs=0.1)
        assert sixty_s[1]["rank"] == 2
        assert sixty_s[1]["power_w"] == pytest.approx(300.0, abs=0.1)
        assert sixty_s[2]["rank"] == 3
        assert sixty_s[2]["power_w"] == pytest.approx(280.0, abs=0.1)

    async def test_rank_capped_at_3(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)

        for i, power in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
            await _insert_activity_with_power(
                session,
                athlete,
                [power] * 60,
                f"2025-07-0{i + 1}T10:00:00+00:00",
            )

        resp = await client.get("/api/power/bests", headers=auth_headers)
        sixty_s = [e for e in resp.json()["bests"] if e["duration_s"] == 60]
        assert len(sixty_s) == 3
        assert max(e["rank"] for e in sixty_s) == 3


class TestPowerBestsDaysFilter:
    """The ?days=N parameter restricts bests to a rolling time window."""

    async def test_activity_within_window_is_included(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        # Activity from 10 days ago — inside a 30-day window.
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        await _insert_activity_with_power(
            session, athlete, [300.0] * 60, recent.isoformat()
        )

        resp = await client.get("/api/power/bests?days=30", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["bests"]) > 0

    async def test_activity_outside_window_is_excluded(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        # Activity from 60 days ago — outside a 30-day window.
        old = datetime.now(timezone.utc) - timedelta(days=60)
        await _insert_activity_with_power(
            session, athlete, [300.0] * 60, old.isoformat()
        )

        resp = await client.get("/api/power/bests?days=30", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["bests"] == []

    async def test_no_days_param_returns_all_activities(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        old = datetime.now(timezone.utc) - timedelta(days=365)
        await _insert_activity_with_power(
            session, athlete, [300.0] * 60, old.isoformat()
        )

        all_resp = await client.get("/api/power/bests", headers=auth_headers)
        filtered_resp = await client.get("/api/power/bests?days=30", headers=auth_headers)

        assert len(all_resp.json()["bests"]) > 0
        assert filtered_resp.json()["bests"] == []

    async def test_days_filter_only_returns_bests_within_window(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        old = datetime.now(timezone.utc) - timedelta(days=200)
        # Old activity has higher power — should NOT appear in the 90-day window.
        await _insert_activity_with_power(session, athlete, [400.0] * 60, old.isoformat())
        await _insert_activity_with_power(session, athlete, [250.0] * 60, recent.isoformat())

        resp = await client.get("/api/power/bests?days=90", headers=auth_headers)
        bests = resp.json()["bests"]
        sixty_s = [e for e in bests if e["duration_s"] == 60]
        assert len(sixty_s) == 1
        assert sixty_s[0]["power_w"] == pytest.approx(250.0, abs=0.1)

    async def test_days_must_be_at_least_1(self, client, auth_headers):
        resp = await client.get("/api/power/bests?days=0", headers=auth_headers)
        assert resp.status_code == 422


@pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
class TestPowerBestsFromFitFile:
    async def test_power_bests_created_after_fit_processing(
        self, client, auth_headers, session
    ):
        """End-to-end: upload + process a real FIT file; bests must appear."""
        with open(SAMPLE_FIT, "rb") as f:
            resp = await client.post(
                "/api/activities/upload",
                files={"file": ("test.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code == 201
        activity_id = resp.json()["id"]

        act_result = await session.execute(
            select(Activity).where(Activity.id == activity_id)
        )
        activity = act_result.scalar_one()
        src_result = await session.execute(
            select(ActivitySource).where(
                ActivitySource.activity_id == activity_id,
                ActivitySource.provider == "upload",
            )
        )
        upload_src = src_result.scalar_one()
        ath_result = await session.execute(
            select(Athlete).where(Athlete.id == activity.athlete_id)
        )
        athlete = ath_result.scalar_one()

        from backend.app.services.fit_processor import process_fit_file

        await process_fit_file(upload_src.fit_file_path, athlete, activity, session)

        resp = await client.get("/api/power/bests", headers=auth_headers)
        assert resp.status_code == 200
        bests = resp.json()["bests"]

        # The sample FIT is a Zwift ride, so it should have power and cover at least
        # durations up to 1 min.
        durations_returned = {e["duration_s"] for e in bests}
        assert 1 in durations_returned
        assert 60 in durations_returned
        # All entries must link back to the uploaded activity
        for entry in bests:
            assert entry["activity_id"] == activity_id


class TestGetFtpEstimate:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/power/ftp-estimate", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["ftp_simple"] is None
        assert body["ftp_cp"] is None
        assert body["simple_available"] is False
        assert body["cp_available"] is False

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/power/ftp-estimate")
        assert resp.status_code == 401

    async def test_both_methods_available_with_long_stream(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 1300s of constant 250 W covers all CP durations (120–1200).
        await _insert_activity_with_power(
            session, athlete, [250.0] * 1300, "2025-06-01T10:00:00+00:00"
        )

        resp = await client.get("/api/power/ftp-estimate", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["simple_available"] is True
        assert body["twenty_min_power"] == pytest.approx(250.0, abs=0.1)
        assert body["ftp_simple"] == round(0.95 * 250.0)
        assert body["cp_available"] is True
        # Constant power → CP ≈ 250 W, W' ≈ 0 J.
        assert body["cp"] == pytest.approx(250.0, abs=0.5)
        assert body["w_prime"] == pytest.approx(0.0, abs=1.0)
        assert body["ftp_cp"] == 250

    async def test_cp_available_without_20min(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        # 500s stream: covers 120/180/300/480 (>=2 CP points) but not 1200.
        await _insert_activity_with_power(
            session, athlete, [250.0] * 500, "2025-06-02T10:00:00+00:00"
        )

        resp = await client.get("/api/power/ftp-estimate", headers=auth_headers)
        body = resp.json()
        assert body["simple_available"] is False
        assert body["ftp_simple"] is None
        assert body["cp_available"] is True
        assert body["ftp_cp"] is not None

    async def test_neither_available_with_short_stream(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 130s stream covers only the 120s CP duration → fewer than 2 points.
        await _insert_activity_with_power(
            session, athlete, [250.0] * 130, "2025-06-03T10:00:00+00:00"
        )

        resp = await client.get("/api/power/ftp-estimate", headers=auth_headers)
        body = resp.json()
        assert body["simple_available"] is False
        assert body["cp_available"] is False

    async def test_days_filter_excludes_old_activity(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        old = datetime.now(timezone.utc) - timedelta(days=200)
        await _insert_activity_with_power(
            session, athlete, [250.0] * 1300, old.isoformat()
        )

        resp = await client.get("/api/power/ftp-estimate?days=90", headers=auth_headers)
        body = resp.json()
        assert body["simple_available"] is False
        assert body["cp_available"] is False
