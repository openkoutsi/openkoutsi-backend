"""Integration tests for course upload, analysis, storage and deletion (issue #55).

The properties that matter, in the order the issue states them:

* an uploaded course produces a segment table and a chart profile;
* the raw GPX is **encrypted at rest** and round-trips through decryption —
  the same property the FIT path asserts;
* **no coordinates appear in any API response** — the track lives in
  ``course_tracks`` and the encrypted blob, nowhere the API serves;
* re-analysis works from the stored track, without re-upload;
* delete removes the rows **and the blob**;
* consent and ownership are enforced exactly as upload has them.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.file_encryption import decrypt_file
from backend.app.models.user_orm import Course, CourseSegment, CourseTrack

_TEST_USER_ID = "test-user-00000000"

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
COURSE_GPX = FIXTURES / "synthetic_course.gpx"
COURSE_NO_ELE_GPX = FIXTURES / "synthetic_course_no_ele.gpx"

# The fixture's geography, restated from the generator: due north from open
# water. These exact strings must never appear in an API response.
FIXTURE_LAT_PREFIX = "61.5"
FIXTURE_LON = "20.5"


async def _seed_rider(session, seeded_athlete):
    seeded_athlete.ftp = 250
    seeded_athlete.weight_kg = 75.0
    await session.commit()
    return seeded_athlete


async def _create_bike(client, auth_headers, **overrides):
    payload = {"name": "Road bike", "tyre_width_mm": 28, "riding_position": "hoods"}
    payload.update(overrides)
    resp = await client.post("/api/bikes", json=payload, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _upload_course(client, auth_headers, bike_id: str, *, path=COURSE_GPX, **form):
    data = {"bike_id": bike_id, **{k: str(v) for k, v in form.items() if v is not None}}
    resp = await client.post(
        "/api/courses",
        headers=auth_headers,
        data=data,
        files={"file": (path.name, path.read_bytes(), "application/gpx+xml")},
    )
    return resp


class TestCourseUpload:
    async def test_upload_produces_a_segment_table_and_profile(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(client, auth_headers, bike["id"])
        assert resp.status_code == 201, resp.text
        data = resp.json()

        assert data["name"] == "Synthetic Course"
        assert data["status"] == "ready"
        assert data["distance_m"] == pytest.approx(15_000, rel=0.01)
        # 140 m climb + ~37 m of false flat; the descent must not count.
        assert 150 <= data["elevation_gain_m"] <= 200
        assert 80 <= data["elevation_loss_m"] <= 100

        segments = data["segments"]
        assert len(segments) >= 3
        types = [s["segment_type"] for s in segments]
        assert "climb" in types and "descent" in types
        # Segments tile the course and the splits accumulate monotonically.
        for a, b in zip(segments, segments[1:]):
            assert abs(a["end_distance_m"] - b["start_distance_m"]) < 1e-6
        offsets = [s["start_offset_s"] for s in segments]
        assert offsets == sorted(offsets)
        assert all(s["power_w"] is not None for s in segments)

        profile = data["profile"]
        assert profile is not None and len(profile) <= 400
        assert data["feasible"] is True
        assert data["predicted_time_s"] > 0

    async def test_no_coordinates_in_any_response(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        created = await _upload_course(client, auth_headers, bike["id"])
        course_id = created.json()["id"]

        for resp in (
            created,
            await client.get("/api/courses", headers=auth_headers),
            await client.get(f"/api/courses/{course_id}", headers=auth_headers),
            await client.get(f"/api/courses/{course_id}/plan", headers=auth_headers),
        ):
            text = resp.text.lower()
            for forbidden in ("latitude", "longitude", '"lat"', '"lon"', "track", FIXTURE_LON):
                assert forbidden not in text, f"{forbidden!r} leaked into {resp.request.url}"

    async def test_the_gpx_is_encrypted_at_rest_and_round_trips(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(client, auth_headers, bike["id"])
        course_id = resp.json()["id"]

        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        assert course.gpx_file_encrypted is True
        # An opaque key, not a path.
        assert "/" not in course.gpx_file_key and "\\" not in course.gpx_file_key

        blob = settings.user_fit_dir(_TEST_USER_ID) / course.gpx_file_key
        assert blob.exists()
        # Genuinely encrypted on disk — not the plaintext file with a flag.
        raw = blob.read_bytes()
        assert b"<trkpt" not in raw and b"<gpx" not in raw
        assert decrypt_file(blob, _TEST_USER_ID) == COURSE_GPX.read_bytes()

    async def test_the_thinned_track_is_stored_for_reanalysis(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(client, auth_headers, bike["id"])
        course_id = resp.json()["id"]

        track = (
            await session.execute(
                select(CourseTrack).where(CourseTrack.course_id == course_id)
            )
        ).scalar_one()
        assert len(track.points) > 500
        lat, lon, ele, dist = track.points[0]
        assert str(lat).startswith(FIXTURE_LAT_PREFIX)
        assert str(lon).startswith(FIXTURE_LON)

    async def test_an_infeasible_target_is_a_result_not_an_error(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        # 15 km in 5 minutes is faster than physics at any human power.
        resp = await _upload_course(
            client, auth_headers, bike["id"], target_time_s=300
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["feasible"] is False
        assert data["refusal_reason"] == "target_faster_than_physics"
        # The athlete still gets the course: a segment table with no splits.
        assert len(data["segments"]) >= 3
        assert all(s["power_w"] is None for s in data["segments"])

    async def test_a_file_without_elevation_is_rejected_with_a_reason(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(
            client, auth_headers, bike["id"], path=COURSE_NO_ELE_GPX
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "no_elevation_data"

    async def test_garbage_is_a_400(self, client, auth_headers, session, seeded_athlete):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await client.post(
            "/api/courses",
            headers=auth_headers,
            data={"bike_id": bike["id"]},
            files={"file": ("nonsense.gpx", b"this is not xml", "application/gpx+xml")},
        )
        assert resp.status_code == 400

    async def test_missing_rider_data_is_a_422(self, client, auth_headers, session, seeded_athlete):
        # No FTP, no weight on the seeded athlete.
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(client, auth_headers, bike["id"])
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "missing_rider_data"

    async def test_an_unowned_bike_is_a_404(self, client, auth_headers, session, seeded_athlete):
        await _seed_rider(session, seeded_athlete)
        resp = await _upload_course(client, auth_headers, "no-such-bike")
        assert resp.status_code == 404

    async def test_requires_consent(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        from backend.app.models.registry_orm import User

        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)

        user = (
            await registry_session.execute(select(User).where(User.id == _TEST_USER_ID))
        ).scalar_one()
        user.consented_at = None
        user.consent_version = None
        await registry_session.commit()

        resp = await _upload_course(client, auth_headers, bike["id"])
        assert resp.status_code == 403
        assert "Consent" in resp.json()["detail"]


class TestCourseLifecycle:
    async def test_another_athletes_course_is_not_found(
        self, client, auth_headers, session, seeded_athlete
    ):
        session.add(
            Course(
                id="foreign",
                athlete_id="other-athlete",
                name="Not yours",
                gpx_file_key="course-foreign.gpx",
                distance_m=1000.0,
            )
        )
        await session.commit()
        for resp in (
            await client.get("/api/courses/foreign", headers=auth_headers),
            await client.delete("/api/courses/foreign", headers=auth_headers),
            await client.post(
                "/api/courses/foreign/reanalyze", json={}, headers=auth_headers
            ),
        ):
            assert resp.status_code == 404
        listed = await client.get("/api/courses", headers=auth_headers)
        assert listed.json()["total"] == 0

    async def test_delete_removes_rows_and_the_blob(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        course_id = (await _upload_course(client, auth_headers, bike["id"])).json()["id"]

        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        blob = settings.user_fit_dir(_TEST_USER_ID) / course.gpx_file_key
        assert blob.exists()

        resp = await client.delete(f"/api/courses/{course_id}", headers=auth_headers)
        assert resp.status_code == 204

        assert not blob.exists()
        for model in (Course, CourseSegment, CourseTrack):
            rows = (await session.execute(select(model))).scalars().all()
            assert rows == [], f"{model.__tablename__} not emptied"

    async def test_reanalysis_uses_the_stored_track_and_clears_the_plan(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        created = (await _upload_course(client, auth_headers, bike["id"])).json()
        course_id = created["id"]

        # A written plan exists; re-analysis must invalidate it.
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        course.plan = "Old prose about old segments."
        course.plan_status = "done"
        course.plan_mood = "cheer"
        await session.commit()

        aero = await _create_bike(
            client, auth_headers, name="TT bike", riding_position="aero", tyre_width_mm=23
        )
        resp = await client.post(
            f"/api/courses/{course_id}/reanalyze",
            json={"bike_id": aero["id"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        # Aero position is faster; the prediction must move.
        assert data["predicted_time_s"] < created["predicted_time_s"]
        assert data["plan_status"] is None

        await session.refresh(course)
        assert course.plan is None and course.plan_mood is None

    async def test_reanalysis_without_a_bike_is_a_409(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        course_id = (await _upload_course(client, auth_headers, bike["id"])).json()["id"]

        await client.delete(f"/api/bikes/{bike['id']}", headers=auth_headers)
        resp = await client.post(
            f"/api/courses/{course_id}/reanalyze", json={}, headers=auth_headers
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "no_bike"

    async def test_reanalysis_can_set_and_clear_a_target(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        course_id = (await _upload_course(client, auth_headers, bike["id"])).json()["id"]

        with_target = await client.post(
            f"/api/courses/{course_id}/reanalyze",
            json={"target_time_s": 3600},
            headers=auth_headers,
        )
        assert with_target.status_code == 200
        assert with_target.json()["target_time_s"] == 3600

        cleared = await client.post(
            f"/api/courses/{course_id}/reanalyze",
            json={"target_time_s": None},
            headers=auth_headers,
        )
        assert cleared.status_code == 200
        assert cleared.json()["target_time_s"] is None
        assert cleared.json()["feasible"] is True


class TestBlobLifecycle:
    """A stored route that no row points at is unreachable by design: both
    per-course delete and the GDPR export iterate `Course` rows, and nothing
    sweeps the uploads directory. So a failure part-way through the upload
    must not leave one behind — least of all an unencrypted one."""

    async def test_a_failed_encryption_leaves_no_plaintext_behind(
        self, client, auth_headers, session, seeded_athlete
    ):
        from backend.app.services import course_analysis

        uploads = settings.user_fit_dir(_TEST_USER_ID)
        uploads.mkdir(parents=True, exist_ok=True)
        before = set(uploads.iterdir())

        # encrypt_file is *designed* to raise hard on a missing or rotated key.
        with patch.object(
            course_analysis, "encrypt_file", side_effect=RuntimeError("no key")
        ):
            with pytest.raises(RuntimeError):
                course_analysis.store_course_blob(
                    COURSE_GPX.read_bytes(), _TEST_USER_ID, "boom"
                )

        assert set(uploads.iterdir()) == before

    async def test_a_failed_commit_leaves_no_orphan_blob(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)

        uploads = settings.user_fit_dir(_TEST_USER_ID)
        uploads.mkdir(parents=True, exist_ok=True)
        before = set(uploads.iterdir())

        from backend.app.services import course_analysis

        with patch.object(
            course_analysis, "store_track", side_effect=RuntimeError("db went away")
        ):
            with pytest.raises(RuntimeError):
                await _upload_course(client, auth_headers, bike["id"])

        assert set(uploads.iterdir()) == before
        assert (await session.execute(select(Course))).scalars().all() == []

    async def test_an_over_long_name_is_refused(
        self, client, auth_headers, session, seeded_athlete
    ):
        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        resp = await _upload_course(
            client, auth_headers, bike["id"], name="x" * 201
        )
        assert resp.status_code == 422


class TestCourseExport:
    async def test_courses_and_their_originals_are_in_the_gdpr_export(
        self, client, auth_headers, session, seeded_athlete
    ):
        import io
        import json
        import zipfile

        await _seed_rider(session, seeded_athlete)
        bike = await _create_bike(client, auth_headers)
        course_id = (await _upload_course(client, auth_headers, bike["id"])).json()["id"]

        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
            assert "bikes.json" in names
            assert "courses.json" in names
            # The original file, decrypted, byte-identical to the upload.
            assert f"courses/{course_id}.gpx" in names
            assert zf.read(f"courses/{course_id}.gpx") == COURSE_GPX.read_bytes()

            courses = json.loads(zf.read("courses.json"))
            assert [c["id"] for c in courses] == [course_id]
            assert courses[0]["segments"]
            # Rows are the coordinate-free derived data; the blob carries the
            # track, and server-side plumbing stays out.
            dumped = json.dumps(courses).lower()
            assert "gpx_file_key" not in dumped
            assert "latitude" not in dumped and "longitude" not in dumped

            bikes = json.loads(zf.read("bikes.json"))
            assert [b["id"] for b in bikes] == [bike["id"]]
