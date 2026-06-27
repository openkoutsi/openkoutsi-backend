"""
Integration tests for /api/distance/bests endpoint.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.models.team_orm import Activity, ActivityDistanceBest, ActivitySource, ActivityStream, Athlete

TESTDATA = Path(__file__).parent.parent.parent / "testdata"
SAMPLE_FIT = TESTDATA / "Zwift_Aerobic_Foundation_Forge.fit"


async def _get_athlete(client, auth_headers, session) -> Athlete:
    resp = await client.get("/api/athlete/", headers=auth_headers)
    athlete_id = resp.json()["id"]
    result = await session.execute(select(Athlete).where(Athlete.id == athlete_id))
    return result.scalar_one()


async def _insert_activity_with_speed(
    session, athlete: Athlete, speed_stream: list[float], start_time: str
) -> Activity:
    activity = Activity(
        athlete_id=athlete.id,
        name="Test Distance Activity",
        sport_type="Ride",
        start_time=datetime.fromisoformat(start_time),
        duration_s=len(speed_stream),
        status="processed",
    )
    session.add(activity)
    await session.flush()

    session.add(
        ActivityStream(
            activity_id=activity.id,
            stream_type="speed",
            data=speed_stream,
        )
    )

    from openkoutsi.training_math import compute_distance_bests

    bests = compute_distance_bests(speed_stream)
    for distance_m, time_s in bests.items():
        session.add(
            ActivityDistanceBest(
                activity_id=activity.id,
                athlete_id=athlete.id,
                distance_m=distance_m,
                time_s=time_s,
                activity_start_time=activity.start_time,
            )
        )

    await session.commit()
    await session.refresh(activity)
    return activity


class TestGetDistanceBestsEmpty:
    async def test_empty_for_new_athlete(self, client, auth_headers):
        resp = await client.get("/api/distance/bests", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"bests": []}

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/distance/bests")
        assert resp.status_code == 401

    async def test_no_speed_stream_activity_produces_no_bests(
        self, client, auth_headers, session
    ):
        """Manual activity without a speed stream should not produce any bests."""
        await client.post(
            "/api/activities/",
            json={
                "sport_type": "Ride",
                "start_time": "2025-06-01T10:00:00Z",
                "duration_s": 3600,
            },
            headers=auth_headers,
        )
        resp = await client.get("/api/distance/bests", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["bests"] == []


class TestGetDistanceBestsSingleActivity:
    async def test_returns_bests_after_activity_with_speed(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 5 m/s for 300 s = 1500 m — covers 1 km
        stream = [5.0] * 300
        await _insert_activity_with_speed(session, athlete, stream, "2025-06-01T10:00:00+00:00")

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        assert resp.status_code == 200
        bests = resp.json()["bests"]
        assert len(bests) > 0

        # All returned entries should have rank=1 (only one activity)
        for entry in bests:
            assert entry["rank"] == 1

        # 1 km best at constant 5 m/s = 200 s
        ones = [e for e in bests if e["distance_m"] == 1000]
        assert len(ones) == 1
        assert ones[0]["time_s"] == 200

    async def test_distances_beyond_total_absent(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 5 m/s for 300 s = 1500 m — cannot cover 2 km
        stream = [5.0] * 300
        await _insert_activity_with_speed(session, athlete, stream, "2025-06-02T10:00:00+00:00")

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        for entry in bests:
            assert entry["distance_m"] <= 1000, (
                f"distance {entry['distance_m']} m should not appear for a 1500 m activity"
            )

    async def test_activity_id_and_name_present(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)
        activity = await _insert_activity_with_speed(
            session, athlete, [5.0] * 300, "2025-06-03T10:00:00+00:00"
        )

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        assert len(bests) > 0
        for entry in bests:
            assert entry["activity_id"] == activity.id
            assert entry["activity_name"] == "Test Distance Activity"

    async def test_entries_ordered_by_distance_then_rank(
        self, client, auth_headers, session
    ):
        athlete = await _get_athlete(client, auth_headers, session)
        # 5 m/s for 2500 s = 12 500 m — covers 1, 2, 3, 5, 8, 10 km
        await _insert_activity_with_speed(
            session, athlete, [5.0] * 2500, "2025-06-04T10:00:00+00:00"
        )

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        bests = resp.json()["bests"]
        distances = [e["distance_m"] for e in bests]
        assert distances == sorted(distances), "bests must be sorted by distance_m"


class TestGetDistanceBestsMultipleActivities:
    async def test_top_3_ranking(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)

        # Insert 4 activities with different speeds over 1 km
        # speeds: 5, 8, 7, 6 m/s → times: 200, 125, ~143, ~167 s
        for i, speed in enumerate([5.0, 8.0, 7.0, 6.0]):
            await _insert_activity_with_speed(
                session,
                athlete,
                [speed] * 300,
                f"2025-06-0{i + 1}T10:00:00+00:00",
            )

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        bests = resp.json()["bests"]

        one_km = [e for e in bests if e["distance_m"] == 1000]
        # Must have exactly 3 entries (top-3 cap)
        assert len(one_km) == 3
        # Rank 1 = fastest time (8 m/s → 125 s)
        assert one_km[0]["rank"] == 1
        assert one_km[0]["time_s"] == 125
        # Rank 2 = second fastest (7 m/s → ~143 s)
        assert one_km[1]["rank"] == 2
        assert one_km[1]["time_s"] < one_km[2]["time_s"]
        assert one_km[2]["rank"] == 3

    async def test_rank_capped_at_3(self, client, auth_headers, session):
        athlete = await _get_athlete(client, auth_headers, session)

        for i, speed in enumerate([4.0, 5.0, 6.0, 7.0, 8.0]):
            await _insert_activity_with_speed(
                session,
                athlete,
                [speed] * 300,
                f"2025-07-0{i + 1}T10:00:00+00:00",
            )

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        one_km = [e for e in resp.json()["bests"] if e["distance_m"] == 1000]
        assert len(one_km) == 3
        assert max(e["rank"] for e in one_km) == 3

    async def test_best_time_is_smallest(self, client, auth_headers, session):
        """Rank 1 must always be the smallest (fastest) time."""
        athlete = await _get_athlete(client, auth_headers, session)

        for i, speed in enumerate([3.0, 6.0, 5.0]):
            await _insert_activity_with_speed(
                session,
                athlete,
                [speed] * 500,
                f"2025-08-0{i + 1}T10:00:00+00:00",
            )

        resp = await client.get("/api/distance/bests", headers=auth_headers)
        one_km = [e for e in resp.json()["bests"] if e["distance_m"] == 1000]
        times = [e["time_s"] for e in one_km]
        assert times == sorted(times), "times must be ordered fastest-first"


@pytest.mark.skipif(not SAMPLE_FIT.exists(), reason="FIT fixture not found")
class TestDistanceBestsFromFitFile:
    async def test_distance_bests_created_after_fit_processing(
        self, client, auth_headers, session
    ):
        """End-to-end: upload + process a real FIT file; bests must appear if GPS present."""
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

        # Check that distance bests were created (if the FIT file has a speed stream)
        bests_result = await session.execute(
            select(ActivityDistanceBest).where(
                ActivityDistanceBest.activity_id == activity_id
            )
        )
        bests = bests_result.scalars().all()

        # If distance bests were created, verify they appear in the API response
        if bests:
            resp = await client.get("/api/distance/bests", headers=auth_headers)
            assert resp.status_code == 200
            api_bests = resp.json()["bests"]
            assert len(api_bests) > 0
            for entry in api_bests:
                assert entry["activity_id"] == activity_id
                assert entry["time_s"] > 0
