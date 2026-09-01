"""Integration tests for the garage (issue #64).

The invariant every test here circles: **``bike_source`` protects a choice the
athlete made by hand.** Automapping is applied rather than suggested — a bike
assignment mints no badge and hides no ride from a prompt, and a per-bike total
that counted only individually-confirmed rides would be a chore rather than a
garage — so the only thing standing between an athlete's correction and the
next reprocess is that column. It is tested from every direction that could
overwrite it: reprocess, a fresh provider import, the history scan, and an edit
to what a bike claims.

The second theme is that numbers must not move on their own. Distance is
derived on read, an odometer reading is absolute, and retiring a bike is not
deleting it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from backend.app.models.user_orm import (
    Activity,
    Bike,
    BikeAccessory,
    BikeMaintenance,
    Course,
)


async def _create_bike(client, auth_headers, **overrides):
    payload = {"name": "Road bike", "tyre_width_mm": 28, "riding_position": "hoods"}
    payload.update(overrides)
    resp = await client.post("/api/bikes", json=payload, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _ride(client, auth_headers, *, sport_type="Ride", distance_m=20_000.0, **extra):
    body = {
        "sport_type": sport_type,
        "start_time": "2026-08-26T07:30:00Z",
        "duration_s": 3600,
        "distance_m": distance_m,
    }
    body.update(extra)
    resp = await client.post("/api/activities", json=body, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _get_bike(client, auth_headers, bike_id):
    resp = await client.get(f"/api/bikes/{bike_id}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _get_activity(client, auth_headers, activity_id):
    resp = await client.get(f"/api/activities/{activity_id}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── The bike row, promoted ─────────────────────────────────────────────────


class TestBikeFields:
    async def test_the_new_columns_round_trip(self, client, auth_headers):
        bike = await _create_bike(
            client,
            auth_headers,
            odometer_base_km=4200.0,
            default_sports=["GravelRide"],
        )
        assert bike["odometer_base_km"] == 4200.0
        assert bike["default_sports"] == ["GravelRide"]
        assert bike["retired_at"] is None
        # No rides yet: the baseline is the whole of the lifetime figure, and
        # `tracked_km` says openkoutsi has seen none of it.
        assert bike["tracked_km"] == 0.0
        assert bike["lifetime_km"] == 4200.0

    async def test_claims_are_normalised_to_canonical_spellings(self, client, auth_headers):
        """`gravel_ride` and `GravelRide` are one claim, not two that never both fire."""
        bike = await _create_bike(
            client, auth_headers, default_sports=["gravel_ride", "GravelRide", "ride"]
        )
        assert bike["default_sports"] == ["GravelRide", "Ride"]

    async def test_a_non_cycling_sport_is_refused(self, client, auth_headers):
        """A bike claiming `Run` would silently never match — worse than a 422."""
        resp = await client.post(
            "/api/bikes",
            json={"name": "Confused", "default_sports": ["Run"]},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_an_unknown_sport_is_refused(self, client, auth_headers):
        resp = await client.post(
            "/api/bikes",
            json={"name": "Confused", "default_sports": ["Unicycling"]},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_two_bikes_cannot_claim_the_same_sport(self, client, auth_headers):
        """No correct resolution exists, so the second claim is refused by name."""
        first = await _create_bike(
            client, auth_headers, name="Gravel", default_sports=["GravelRide"]
        )
        resp = await client.post(
            "/api/bikes",
            json={"name": "Other gravel", "default_sports": ["GravelRide"]},
            headers=auth_headers,
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["sport"] == "GravelRide"
        assert detail["bike_id"] == first["id"]
        assert detail["bike_name"] == "Gravel"

    async def test_the_collision_check_ignores_the_bike_being_edited(
        self, client, auth_headers
    ):
        """Re-sending a bike's own claims is not a collision with itself."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        resp = await client.patch(
            f"/api/bikes/{bike['id']}",
            json={"default_sports": ["Ride", "VirtualRide"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["default_sports"] == ["Ride", "VirtualRide"]

    async def test_a_retired_bike_does_not_hold_its_claim_hostage(
        self, client, auth_headers
    ):
        """Retiring the gravel bike and buying another is the ordinary case."""
        old = await _create_bike(
            client, auth_headers, name="Old gravel", default_sports=["GravelRide"]
        )
        await client.patch(
            f"/api/bikes/{old['id']}",
            json={"retired_at": "2026-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        resp = await client.post(
            "/api/bikes",
            json={"name": "New gravel", "default_sports": ["GravelRide"]},
            headers=auth_headers,
        )
        assert resp.status_code == 201

    async def test_retiring_is_reversible(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        await client.patch(
            f"/api/bikes/{bike['id']}",
            json={"retired_at": "2026-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        resp = await client.patch(
            f"/api/bikes/{bike['id']}", json={"retired_at": None}, headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["retired_at"] is None

    async def test_the_garage_does_not_need_course_recon(self, client, auth_headers):
        """Issue #55 gated bikes behind course recon because a bike was only a
        pacing input. It is now where the athlete's kilometres and maintenance
        history live, and this test fixes that: no `course_recon_on` fixture."""
        bike = await _create_bike(client, auth_headers)
        assert (await client.get("/api/bikes", headers=auth_headers)).status_code == 200
        assert (
            await client.get(f"/api/bikes/{bike['id']}", headers=auth_headers)
        ).status_code == 200


# ── Distance ───────────────────────────────────────────────────────────────


class TestDistance:
    async def test_tracked_and_lifetime_are_reported_apart(self, client, auth_headers):
        """One is what openkoutsi observed; the other leans on a typed number."""
        bike = await _create_bike(
            client, auth_headers, odometer_base_km=1000.0, default_sports=["Ride"]
        )
        await _ride(client, auth_headers, distance_m=20_000.0)
        await _ride(client, auth_headers, distance_m=30_000.0)

        data = await _get_bike(client, auth_headers, bike["id"])
        assert data["tracked_km"] == pytest.approx(50.0)
        assert data["lifetime_km"] == pytest.approx(1050.0)

    async def test_without_a_baseline_lifetime_is_just_the_tracked_rides(
        self, client, auth_headers
    ):
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        await _ride(client, auth_headers, distance_m=25_000.0)
        data = await _get_bike(client, auth_headers, bike["id"])
        assert data["tracked_km"] == pytest.approx(25.0)
        assert data["lifetime_km"] == pytest.approx(25.0)

    async def test_a_ride_with_no_distance_contributes_nothing_and_nulls_nothing(
        self, client, auth_headers
    ):
        """SUM over a set containing NULLs must not become NULL — a trainer ride
        logged without a distance would otherwise erase the whole total."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        await _ride(client, auth_headers, distance_m=40_000.0)
        await _ride(client, auth_headers, distance_m=None, duration_s=1800)

        data = await _get_bike(client, auth_headers, bike["id"])
        assert data["tracked_km"] == pytest.approx(40.0)

    async def test_every_ride_can_be_distanceless(self, client, auth_headers):
        """The all-NULL group is the case a bare SUM returns NULL for."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        await _ride(client, auth_headers, distance_m=None, duration_s=1800)
        data = await _get_bike(client, auth_headers, bike["id"])
        assert data["tracked_km"] == 0.0
        assert data["lifetime_km"] == 0.0

    async def test_the_list_carries_the_same_totals_as_the_detail(
        self, client, auth_headers
    ):
        """The list builds them in one grouped query; the detail in one filtered
        sum. Two code paths for one number is exactly how they drift."""
        bike = await _create_bike(
            client, auth_headers, odometer_base_km=500.0, default_sports=["Ride"]
        )
        await _ride(client, auth_headers, distance_m=10_000.0)

        listed = (await client.get("/api/bikes", headers=auth_headers)).json()[0]
        detail = await _get_bike(client, auth_headers, bike["id"])
        assert listed["tracked_km"] == detail["tracked_km"] == pytest.approx(10.0)
        assert listed["lifetime_km"] == detail["lifetime_km"] == pytest.approx(510.0)

    async def test_reassigning_a_ride_moves_the_distance_immediately(
        self, client, auth_headers
    ):
        """Derived on read is the whole point: there is no counter to rebuild."""
        road = await _create_bike(client, auth_headers, name="Road", default_sports=["Ride"])
        gravel = await _create_bike(client, auth_headers, name="Gravel")
        ride = await _ride(client, auth_headers, distance_m=60_000.0)

        assert (await _get_bike(client, auth_headers, road["id"]))["tracked_km"] == 60.0

        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"bike_id": gravel["id"]},
            headers=auth_headers,
        )
        assert (await _get_bike(client, auth_headers, road["id"]))["tracked_km"] == 0.0
        assert (await _get_bike(client, auth_headers, gravel["id"]))["tracked_km"] == 60.0


# ── Automapping ────────────────────────────────────────────────────────────


class TestAutomapping:
    async def test_a_claimed_sport_is_assigned_on_ingest(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] == bike["id"]
        assert ride["bike_source"] == "auto"

    async def test_a_sport_claimed_by_nobody_is_left_alone(self, client, auth_headers):
        """Guessing is worse than a blank: the athlete can see and fix a blank."""
        await _create_bike(client, auth_headers, default_sports=["GravelRide"])
        ride = await _ride(client, auth_headers, sport_type="Ride")
        assert ride["bike_id"] is None
        assert ride["bike_source"] is None

    async def test_an_athlete_with_no_claims_gets_nothing(self, client, auth_headers):
        await _create_bike(client, auth_headers)
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] is None

    @pytest.mark.parametrize("sport", ["Run", "Swim", "WeightTraining", "Hike"])
    async def test_a_non_cycling_activity_is_never_assigned(
        self, client, auth_headers, sport
    ):
        await _create_bike(client, auth_headers, default_sports=["Ride"])
        ride = await _ride(client, auth_headers, sport_type=sport)
        assert ride["bike_id"] is None
        assert ride["bike_source"] is None

    async def test_each_bike_takes_the_sports_it_claims(self, client, auth_headers):
        road = await _create_bike(client, auth_headers, name="Road", default_sports=["Ride"])
        gravel = await _create_bike(
            client, auth_headers, name="Gravel", default_sports=["GravelRide"]
        )
        ebike = await _create_bike(
            client, auth_headers, name="E-bike", default_sports=["EBikeRide"]
        )

        assert (await _ride(client, auth_headers, sport_type="Ride"))["bike_id"] == road["id"]
        assert (
            await _ride(client, auth_headers, sport_type="GravelRide")
        )["bike_id"] == gravel["id"]
        assert (
            await _ride(client, auth_headers, sport_type="EBikeRide")
        )["bike_id"] == ebike["id"]

    async def test_a_retired_bike_collects_no_new_rides(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        await client.patch(
            f"/api/bikes/{bike['id']}",
            json={"retired_at": "2026-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] is None


class TestManualAssignmentIsDurable:
    """The invariant this whole feature turns on."""

    async def _corrected(self, client, auth_headers):
        """A ride automapped to the road bike, corrected by hand onto the gravel bike."""
        road = await _create_bike(client, auth_headers, name="Road", default_sports=["Ride"])
        gravel = await _create_bike(client, auth_headers, name="Gravel")
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] == road["id"]

        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"bike_id": gravel["id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["bike_id"] == gravel["id"]
        assert resp.json()["bike_source"] == "manual"
        return road, gravel, ride

    async def test_setting_a_bike_by_hand_stamps_manual(self, client, auth_headers):
        await self._corrected(client, auth_headers)

    async def test_it_survives_a_reprocess(self, client, auth_headers):
        _road, gravel, ride = await self._corrected(client, auth_headers)
        resp = await client.post(
            f"/api/activities/{ride['id']}/reprocess", headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["bike_id"] == gravel["id"]
        assert resp.json()["bike_source"] == "manual"

    async def test_it_survives_the_history_scan(self, client, auth_headers):
        _road, gravel, ride = await self._corrected(client, auth_headers)
        resp = await client.post("/api/bikes/assign-history", headers=auth_headers)
        assert resp.status_code == 200
        # Skipped before it is even examined — the scan reads only unassigned rows.
        assert resp.json()["assigned"] == 0
        after = await _get_activity(client, auth_headers, ride["id"])
        assert after["bike_id"] == gravel["id"]
        assert after["bike_source"] == "manual"

    async def test_it_survives_an_edit_to_what_a_bike_claims(self, client, auth_headers):
        road, gravel, ride = await self._corrected(client, auth_headers)
        # Widen the road bike's claim to cover everything.
        resp = await client.patch(
            f"/api/bikes/{road['id']}",
            json={"default_sports": ["Ride", "VirtualRide", "GravelRide"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        after = await _get_activity(client, auth_headers, ride["id"])
        assert after["bike_id"] == gravel["id"]

    async def test_it_survives_a_fresh_provider_import(
        self, client, auth_headers, session, seeded_athlete
    ):
        """A re-sync re-runs the whole processing pass over the same row."""
        from backend.app.services.provider_sync import _apply_import

        _road, gravel, ride = await self._corrected(client, auth_headers)
        activity = (
            await session.execute(select(Activity).where(Activity.id == ride["id"]))
        ).scalar_one()
        await _apply_import(
            activity,
            seeded_athlete,
            session,
            fields={"sport_type": "Ride", "distance_m": 20_000.0},
            streams={},
            weight_log=[],
            load_duration_s=3600,
        )
        await session.commit()
        await session.refresh(activity)
        assert activity.bike_id == gravel["id"]
        assert activity.bike_source == "manual"

    async def test_clearing_it_returns_the_ride_to_unassigned(self, client, auth_headers):
        """An explicit null means "no bike", which is not the same as "never asked"
        — both columns go, so automapping is free to fill the gap again."""
        _road, _gravel, ride = await self._corrected(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}", json={"bike_id": None}, headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.json()["bike_id"] is None
        assert resp.json()["bike_source"] is None

    async def test_a_bike_that_is_not_yours_is_not_assignable(
        self, client, auth_headers, session
    ):
        session.add(Bike(id="foreign", athlete_id="other-athlete", name="Not yours"))
        await session.commit()
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"bike_id": "foreign"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_a_retired_bike_is_still_correctable_onto(self, client, auth_headers):
        """Correcting an old ride onto the bike it was actually done on is the
        point of the override, and that bike is often the one since sold."""
        sold = await _create_bike(client, auth_headers, name="Sold")
        await client.patch(
            f"/api/bikes/{sold['id']}",
            json={"retired_at": "2026-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        ride = await _ride(client, auth_headers)
        resp = await client.patch(
            f"/api/activities/{ride['id']}",
            json={"bike_id": sold["id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["bike_id"] == sold["id"]


class TestAutoAssignmentIsRecomputed:
    async def test_narrowing_a_claim_withdraws_an_automatic_assignment(
        self, client, auth_headers
    ):
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] == bike["id"]

        await client.patch(
            f"/api/bikes/{bike['id']}", json={"default_sports": []}, headers=auth_headers
        )
        resp = await client.post(
            f"/api/activities/{ride['id']}/reprocess", headers=auth_headers
        )
        assert resp.json()["bike_id"] is None
        assert resp.json()["bike_source"] is None

    async def test_retiring_a_bike_does_not_strip_the_rides_it_already_has(
        self, client, auth_headers
    ):
        """The entire difference between retiring and deleting."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        ride = await _ride(client, auth_headers)
        await client.patch(
            f"/api/bikes/{bike['id']}",
            json={"retired_at": "2026-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        resp = await client.post(
            f"/api/activities/{ride['id']}/reprocess", headers=auth_headers
        )
        assert resp.json()["bike_id"] == bike["id"]
        assert (await _get_bike(client, auth_headers, bike["id"]))["tracked_km"] == 20.0


class TestIngestPaths:
    """Assignment must happen on every path, or a garage is right for the rides
    that arrived one way and quietly short for the rest."""

    async def test_the_manual_activity_endpoint(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        assert (await _ride(client, auth_headers))["bike_id"] == bike["id"]

    async def test_reprocess(self, client, auth_headers):
        """A ride logged before the athlete described their bikes picks one up."""
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] is None
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        resp = await client.post(
            f"/api/activities/{ride['id']}/reprocess", headers=auth_headers
        )
        assert resp.json()["bike_id"] == bike["id"]
        assert resp.json()["bike_source"] == "auto"

    async def test_provider_sync(self, client, auth_headers, session, seeded_athlete):
        from backend.app.services.provider_sync import _apply_import

        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        activity = Activity(
            id="synced-1", athlete_id=seeded_athlete.id, status="pending"
        )
        session.add(activity)
        await session.commit()

        await _apply_import(
            activity,
            seeded_athlete,
            session,
            fields={"sport_type": "Ride", "distance_m": 20_000.0, "duration_s": 3600},
            streams={},
            weight_log=[],
            load_duration_s=3600,
        )
        await session.commit()
        await session.refresh(activity)
        assert activity.bike_id == bike["id"]
        assert activity.bike_source == "auto"

    async def test_file_processing(self, client, auth_headers, session, seeded_athlete):
        """A real uploaded file, through the real processing pass.

        The bike claims every cycling sport rather than one, so the assertion is
        "the ride the file describes landed on the bike", not "this fixture
        happens to say GravelRide".
        """
        from pathlib import Path

        from openkoutsi.sport_matching import CYCLING_SPORT_TYPES

        from backend.app.models.user_orm import ActivitySource
        from backend.app.services.fit_processor import process_fit_file

        fixture = (
            Path(__file__).parent.parent.parent
            / "testdata"
            / "fixtures"
            / "synthetic_bike_power_gps.fit"
        )
        if not fixture.exists():
            pytest.skip("no FIT fixture available")

        bike = await _create_bike(
            client, auth_headers, default_sports=sorted(CYCLING_SPORT_TYPES)
        )
        with open(fixture, "rb") as f:
            resp = await client.post(
                "/api/activities/upload",
                files={"file": ("ride.fit", f, "application/octet-stream")},
                headers=auth_headers,
            )
        assert resp.status_code == 201, resp.text
        activity_id = resp.json()["id"]

        activity = (
            await session.execute(select(Activity).where(Activity.id == activity_id))
        ).scalar_one()
        source = (
            await session.execute(
                select(ActivitySource).where(ActivitySource.activity_id == activity_id)
            )
        ).scalars().first()
        # The background task is suppressed under test, so drive the processing
        # pass the same way the upload tests do.
        await process_fit_file(source.fit_file_path, seeded_athlete, activity, session)
        await session.commit()
        await session.refresh(activity)
        assert activity.sport_type in CYCLING_SPORT_TYPES
        assert activity.bike_id == bike["id"]
        assert activity.bike_source == "auto"


class TestAssignHistory:
    async def test_it_picks_up_the_back_catalogue(self, client, auth_headers):
        """A garage that only counts rides from today is empty when it matters."""
        old = await _ride(client, auth_headers, distance_m=30_000.0)
        assert old["bike_id"] is None

        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        resp = await client.post("/api/bikes/assign-history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {"scanned": 1, "assigned": 1}

        after = await _get_activity(client, auth_headers, old["id"])
        assert after["bike_id"] == bike["id"]
        assert after["bike_source"] == "auto"
        assert (await _get_bike(client, auth_headers, bike["id"]))["tracked_km"] == 30.0

    async def test_it_leaves_non_cycling_history_alone(self, client, auth_headers):
        run = await _ride(client, auth_headers, sport_type="Run")
        await _create_bike(client, auth_headers, default_sports=["Ride"])
        resp = await client.post("/api/bikes/assign-history", headers=auth_headers)
        assert resp.json()["assigned"] == 0
        assert (await _get_activity(client, auth_headers, run["id"]))["bike_id"] is None

    async def test_with_no_claims_it_does_nothing(self, client, auth_headers):
        await _ride(client, auth_headers)
        await _create_bike(client, auth_headers)
        resp = await client.post("/api/bikes/assign-history", headers=auth_headers)
        assert resp.json() == {"scanned": 0, "assigned": 0}

    async def test_it_never_re_homes_an_automatic_assignment(self, client, auth_headers):
        """Rides already assigned, by hand *or* by an earlier rule, are left
        exactly as they are — this pass only fills blanks."""
        road = await _create_bike(client, auth_headers, name="Road", default_sports=["Ride"])
        ride = await _ride(client, auth_headers)
        assert ride["bike_id"] == road["id"]

        await client.patch(
            f"/api/bikes/{road['id']}", json={"default_sports": []}, headers=auth_headers
        )
        gravel = await _create_bike(
            client, auth_headers, name="Gravel", default_sports=["Ride"]
        )
        resp = await client.post("/api/bikes/assign-history", headers=auth_headers)
        assert resp.json()["assigned"] == 0
        after = await _get_activity(client, auth_headers, ride["id"])
        assert after["bike_id"] == road["id"] != gravel["id"]


# ── Maintenance ────────────────────────────────────────────────────────────


class TestMaintenance:
    async def _entry(self, client, auth_headers, bike_id, **overrides):
        payload = {
            "performed_on": "2026-01-01",
            "component": "tyres",
            "odometer_km": 1000.0,
        }
        payload.update(overrides)
        resp = await client.post(
            f"/api/bikes/{bike_id}/maintenance", json=payload, headers=auth_headers
        )
        assert resp.status_code == 201, resp.text
        return resp.json()

    async def test_create_and_list(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        entry = await self._entry(
            client, auth_headers, bike["id"], note="New Gravelkings"
        )
        assert entry["component"] == "tyres"
        assert entry["note"] == "New Gravelkings"

        listed = await client.get(
            f"/api/bikes/{bike['id']}/maintenance", headers=auth_headers
        )
        assert [e["id"] for e in listed.json()] == [entry["id"]]

    async def test_component_life_is_the_gap_between_consecutive_entries(
        self, client, auth_headers
    ):
        """The original question — "how many kilometres did these tyres last?" —
        is about two events of the same kind, which is why `component` exists."""
        bike = await _create_bike(client, auth_headers)
        await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-01-01", odometer_km=1000.0
        )
        second = await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-06-01", odometer_km=4200.0
        )
        assert second["previous_component_km"] == pytest.approx(3200.0)

    async def test_a_different_component_does_not_close_the_span(
        self, client, auth_headers
    ):
        bike = await _create_bike(client, auth_headers)
        await self._entry(
            client, auth_headers, bike["id"], component="tyres", odometer_km=1000.0
        )
        chain = await self._entry(
            client,
            auth_headers,
            bike["id"],
            component="chain",
            performed_on="2026-03-01",
            odometer_km=2000.0,
        )
        assert chain["previous_component_km"] is None

    async def test_the_open_ended_current_entry_counts_from_the_lifetime(
        self, client, auth_headers
    ):
        """Tyres fitted at 4 200 km on a bike now at 6 000 have done 1 800 — the
        number the athlete actually wants to see."""
        bike = await _create_bike(
            client, auth_headers, odometer_base_km=4000.0, default_sports=["Ride"]
        )
        await _ride(client, auth_headers, distance_m=2_000_000.0)  # 2 000 km

        await self._entry(client, auth_headers, bike["id"], odometer_km=4200.0)
        detail = await _get_bike(client, auth_headers, bike["id"])
        assert detail["lifetime_km"] == pytest.approx(6000.0)
        current = detail["maintenance"][0]
        assert current["is_current"] is True
        assert current["km_since"] == pytest.approx(1800.0)

    async def test_a_missing_earlier_reading_leaves_the_span_unknown(
        self, client, auth_headers
    ):
        """Unknown is not zero, and reporting zero would be a lie about wear."""
        bike = await _create_bike(client, auth_headers)
        await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-01-01", odometer_km=None
        )
        second = await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-06-01", odometer_km=4200.0
        )
        assert second["previous_component_km"] is None
        assert second["is_current"] is True

    async def test_only_the_newest_entry_of_a_component_is_current(
        self, client, auth_headers
    ):
        bike = await _create_bike(client, auth_headers)
        first = await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-01-01", odometer_km=1000.0
        )
        second = await self._entry(
            client, auth_headers, bike["id"], performed_on="2026-06-01", odometer_km=4200.0
        )
        detail = await _get_bike(client, auth_headers, bike["id"])
        current = {e["id"]: e["is_current"] for e in detail["maintenance"]}
        assert current == {first["id"]: False, second["id"]: True}

    async def test_the_log_is_newest_first(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        first = await self._entry(client, auth_headers, bike["id"], performed_on="2026-01-01")
        second = await self._entry(client, auth_headers, bike["id"], performed_on="2026-06-01")
        detail = await _get_bike(client, auth_headers, bike["id"])
        assert [e["id"] for e in detail["maintenance"]] == [second["id"], first["id"]]

    async def test_an_odometer_reading_does_not_move_when_a_ride_is_reassigned(
        self, client, auth_headers
    ):
        """A maintenance log that rewrites itself is worse than none."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        other = await _create_bike(client, auth_headers, name="Other")
        ride = await _ride(client, auth_headers, distance_m=50_000.0)
        entry = await self._entry(client, auth_headers, bike["id"], odometer_km=1234.0)

        await client.patch(
            f"/api/activities/{ride['id']}",
            json={"bike_id": other["id"]},
            headers=auth_headers,
        )
        detail = await _get_bike(client, auth_headers, bike["id"])
        assert detail["maintenance"][0]["odometer_km"] == 1234.0
        assert detail["maintenance"][0]["id"] == entry["id"]
        # The *derived* figure moves, because the bike really has ridden less.
        assert detail["tracked_km"] == 0.0

    async def test_update_and_delete(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        entry = await self._entry(client, auth_headers, bike["id"])
        resp = await client.patch(
            f"/api/bikes/{bike['id']}/maintenance/{entry['id']}",
            json={"note": "Corrected"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["note"] == "Corrected"
        assert resp.json()["component"] == "tyres"  # untouched

        resp = await client.delete(
            f"/api/bikes/{bike['id']}/maintenance/{entry['id']}", headers=auth_headers
        )
        assert resp.status_code == 204
        listed = await client.get(
            f"/api/bikes/{bike['id']}/maintenance", headers=auth_headers
        )
        assert listed.json() == []

    async def test_a_component_outside_the_suggested_list_is_accepted(
        self, client, auth_headers
    ):
        """Advisory, not validated — the list cannot anticipate everything."""
        bike = await _create_bike(client, auth_headers)
        entry = await self._entry(client, auth_headers, bike["id"], component="dropper_post")
        assert entry["component"] == "dropper_post"

    async def test_another_athletes_bike_has_no_log_to_read(
        self, client, auth_headers, session
    ):
        session.add(Bike(id="foreign", athlete_id="other-athlete", name="Not yours"))
        await session.commit()
        for resp in (
            await client.get("/api/bikes/foreign/maintenance", headers=auth_headers),
            await client.post(
                "/api/bikes/foreign/maintenance",
                json={"performed_on": "2026-01-01", "component": "tyres"},
                headers=auth_headers,
            ),
        ):
            assert resp.status_code == 404

    async def test_an_entry_from_another_bike_is_not_reachable_through_this_one(
        self, client, auth_headers
    ):
        mine = await _create_bike(client, auth_headers, name="Mine")
        other = await _create_bike(client, auth_headers, name="Other")
        entry = await self._entry(client, auth_headers, other["id"])
        resp = await client.patch(
            f"/api/bikes/{mine['id']}/maintenance/{entry['id']}",
            json={"note": "x"},
            headers=auth_headers,
        )
        assert resp.status_code == 404


# ── Accessories ────────────────────────────────────────────────────────────


class TestAccessories:
    async def test_crud(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        resp = await client.post(
            f"/api/bikes/{bike['id']}/accessories",
            json={"name": "Child trailer", "note": "Croozer"},
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.text
        accessory = resp.json()
        assert accessory["name"] == "Child trailer"

        resp = await client.patch(
            f"/api/bikes/{bike['id']}/accessories/{accessory['id']}",
            json={"note": "Croozer Kid Plus"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["note"] == "Croozer Kid Plus"
        assert resp.json()["name"] == "Child trailer"

        detail = await _get_bike(client, auth_headers, bike["id"])
        assert [a["id"] for a in detail["accessories"]] == [accessory["id"]]

        resp = await client.delete(
            f"/api/bikes/{bike['id']}/accessories/{accessory['id']}", headers=auth_headers
        )
        assert resp.status_code == 204
        listed = await client.get(
            f"/api/bikes/{bike['id']}/accessories", headers=auth_headers
        )
        assert listed.json() == []

    async def test_an_accessory_does_not_touch_the_pacing_inputs(
        self, client, auth_headers
    ):
        """Deliberately a plain record: mass and drag are a separate issue."""
        bike = await _create_bike(client, auth_headers, tyre_width_mm=32)
        await client.post(
            f"/api/bikes/{bike['id']}/accessories",
            json={"name": "Child trailer"},
            headers=auth_headers,
        )
        detail = await _get_bike(client, auth_headers, bike["id"])
        assert detail["tyre_width_mm"] == 32
        assert detail["riding_position"] == "hoods"

    async def test_another_athletes_bike_has_no_accessories_to_read(
        self, client, auth_headers, session
    ):
        session.add(Bike(id="foreign", athlete_id="other-athlete", name="Not yours"))
        await session.commit()
        resp = await client.get("/api/bikes/foreign/accessories", headers=auth_headers)
        assert resp.status_code == 404


# ── Deletion ───────────────────────────────────────────────────────────────


class TestDeletion:
    async def test_deleting_a_bike_keeps_its_rides_and_unassigns_them(
        self, client, auth_headers, session
    ):
        """The rides survive; they are simply no longer on a bike. `bike_source`
        goes too — "manual" against no bike would claim a choice that cannot be
        true, and would then block automapping from ever filling the gap."""
        bike = await _create_bike(client, auth_headers, default_sports=["Ride"])
        auto = await _ride(client, auth_headers, distance_m=10_000.0)
        manual = await _ride(client, auth_headers, distance_m=20_000.0)
        await client.patch(
            f"/api/activities/{manual['id']}",
            json={"bike_id": bike["id"]},
            headers=auth_headers,
        )

        resp = await client.delete(f"/api/bikes/{bike['id']}", headers=auth_headers)
        assert resp.status_code == 204

        for activity_id in (auto["id"], manual["id"]):
            row = (
                await session.execute(select(Activity).where(Activity.id == activity_id))
            ).scalar_one()
            await session.refresh(row)
            assert row.bike_id is None
            assert row.bike_source is None

    async def test_deleting_a_bike_takes_its_log_and_accessories(
        self, client, auth_headers, session
    ):
        bike = await _create_bike(client, auth_headers)
        await client.post(
            f"/api/bikes/{bike['id']}/maintenance",
            json={"performed_on": "2026-01-01", "component": "chain"},
            headers=auth_headers,
        )
        await client.post(
            f"/api/bikes/{bike['id']}/accessories",
            json={"name": "Rack"},
            headers=auth_headers,
        )

        assert (
            await client.delete(f"/api/bikes/{bike['id']}", headers=auth_headers)
        ).status_code == 204
        assert (await session.execute(select(BikeMaintenance))).scalars().all() == []
        assert (await session.execute(select(BikeAccessory))).scalars().all() == []

    async def test_deleting_a_bike_still_leaves_courses_intact(
        self, client, auth_headers, session, seeded_athlete
    ):
        """Existing behaviour from issue #55 — guarded against regression."""
        bike = await _create_bike(client, auth_headers)
        session.add(
            Course(
                id="course-1",
                athlete_id=seeded_athlete.id,
                name="Loop",
                bike_id=bike["id"],
                gpx_file_key="course-course-1.gpx",
                distance_m=10_000.0,
            )
        )
        await session.commit()

        assert (
            await client.delete(f"/api/bikes/{bike['id']}", headers=auth_headers)
        ).status_code == 204
        survivor = (
            await session.execute(select(Course).where(Course.id == "course-1"))
        ).scalar_one()
        await session.refresh(survivor)
        assert survivor.bike_id is None

    async def test_deleting_an_activity_leaves_the_bike_and_its_history_alone(
        self, client, auth_headers, session
    ):
        bike = await _create_bike(
            client, auth_headers, odometer_base_km=100.0, default_sports=["Ride"]
        )
        ride = await _ride(client, auth_headers, distance_m=10_000.0)
        await client.post(
            f"/api/bikes/{bike['id']}/maintenance",
            json={"performed_on": "2026-01-01", "component": "chain", "odometer_km": 50.0},
            headers=auth_headers,
        )

        assert (
            await client.delete(f"/api/activities/{ride['id']}", headers=auth_headers)
        ).status_code == 204

        detail = await _get_bike(client, auth_headers, bike["id"])
        assert detail["odometer_base_km"] == 100.0
        assert len(detail["maintenance"]) == 1
        # Only the derived figure moves, and it should: the ride is gone.
        assert detail["tracked_km"] == 0.0
        assert detail["lifetime_km"] == 100.0


class TestExport:
    async def test_the_garage_is_in_the_data_export(self, client, auth_headers):
        import io
        import json
        import zipfile

        bike = await _create_bike(
            client, auth_headers, odometer_base_km=4200.0, default_sports=["Ride"]
        )
        await client.post(
            f"/api/bikes/{bike['id']}/maintenance",
            json={"performed_on": "2026-01-01", "component": "tyres", "odometer_km": 1000.0},
            headers=auth_headers,
        )
        await client.post(
            f"/api/bikes/{bike['id']}/accessories",
            json={"name": "Child trailer"},
            headers=auth_headers,
        )

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            bikes = json.loads(zf.read("bikes.json"))
        assert len(bikes) == 1
        assert bikes[0]["odometer_base_km"] == 4200.0
        assert bikes[0]["default_sports"] == ["Ride"]
        assert [m["component"] for m in bikes[0]["maintenance"]] == ["tyres"]
        assert [a["name"] for a in bikes[0]["accessories"]] == ["Child trailer"]
