"""Integration tests for the bikes CRUD (issue #55).

A bike is the small equipment concept the course physics reads: tyre width →
rolling resistance, riding position → drag area. The interesting behaviour is
around deletion — a deleted bike must never take a course with it, only leave
the course needing a bike picked again.

Issue #64 promoted the same row into the garage and, with it, lifted the
course-recon gate from this router: a bike is now where an athlete's own
kilometres and maintenance history live, which has nothing to do with whether
the self-hoster switched on GPX course analysis. These tests therefore run with
the switch at its default (off) — see ``test_garage.py`` for the one that
states that directly.
"""
from __future__ import annotations

from sqlalchemy import select

from backend.app.models.user_orm import Bike, Course


async def _create_bike(client, auth_headers, **overrides):
    payload = {"name": "Road bike", "tyre_width_mm": 28, "riding_position": "hoods"}
    payload.update(overrides)
    resp = await client.post("/api/bikes", json=payload, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestBikesCrud:
    async def test_create_and_list(self, client, auth_headers):
        created = await _create_bike(client, auth_headers)
        assert created["riding_position"] == "hoods"
        assert created["tyre_width_mm"] == 28

        listed = await client.get("/api/bikes", headers=auth_headers)
        assert listed.status_code == 200
        assert [b["id"] for b in listed.json()] == [created["id"]]

    async def test_position_must_be_a_known_value(self, client, auth_headers):
        resp = await client.post(
            "/api/bikes",
            json={"name": "Bad", "riding_position": "recumbent"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_tyre_width_is_bounded(self, client, auth_headers):
        resp = await client.post(
            "/api/bikes",
            json={"name": "Bad", "tyre_width_mm": 300},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    async def test_update_is_partial(self, client, auth_headers):
        bike = await _create_bike(client, auth_headers)
        resp = await client.patch(
            f"/api/bikes/{bike['id']}",
            json={"riding_position": "aero"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["riding_position"] == "aero"
        assert data["tyre_width_mm"] == 28  # untouched

    async def test_another_athletes_bike_is_not_found(self, client, auth_headers, session):
        session.add(Bike(id="foreign", athlete_id="other-athlete", name="Not yours"))
        await session.commit()
        for resp in (
            await client.patch("/api/bikes/foreign", json={"name": "x"}, headers=auth_headers),
            await client.delete("/api/bikes/foreign", headers=auth_headers),
        ):
            assert resp.status_code == 404
        listed = await client.get("/api/bikes", headers=auth_headers)
        assert listed.json() == []

    async def test_delete_clears_course_references_without_deleting_them(
        self, client, auth_headers, session, seeded_athlete
    ):
        bike = await _create_bike(client, auth_headers)
        course = Course(
            id="course-1",
            athlete_id=seeded_athlete.id,
            name="Loop",
            bike_id=bike["id"],
            gpx_file_key="course-course-1.gpx",
            distance_m=10_000.0,
        )
        session.add(course)
        await session.commit()

        resp = await client.delete(f"/api/bikes/{bike['id']}", headers=auth_headers)
        assert resp.status_code == 204

        survivor = (
            await session.execute(select(Course).where(Course.id == "course-1"))
        ).scalar_one()
        await session.refresh(survivor)
        assert survivor.bike_id is None
        assert (await session.execute(select(Bike))).scalars().all() == []
