"""Bulk activity import: archives, mixed formats and per-file outcomes (issue #36).

The background task is suppressed in tests (see ``conftest``), so these drive
``execute_import_job`` directly against the in-memory database the client is
wired to — the same code path the endpoint schedules, not a re-implementation
of it.

The privacy assertion in ``TestNoLocationIsStored`` is the one to read first: a
GPX file is nothing but coordinates, and the whole point of parsing one is to
arrive at an activity that contains none.
"""
from __future__ import annotations

import gzip
import io
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.models.user_orm import (
    Activity,
    ActivityInterval,
    ActivitySource,
    ActivityStream,
    ImportJob,
)
from backend.app.services.activity_import import execute_import_job

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
RIDE_GPX = FIXTURES / "synthetic_ride.gpx"
HR_ONLY_GPX = FIXTURES / "synthetic_hr_only.gpx"
RIDE_TCX = FIXTURES / "synthetic_ride.tcx"

_TEST_USER_ID = "test-user-00000000"

pytestmark = pytest.mark.skipif(
    not RIDE_GPX.exists(),
    reason="synthetic GPX/TCX fixtures missing — run scripts/generate_synthetic_activity_fixtures.py",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def shifted_gpx(minutes: int) -> bytes:
    """The ride fixture moved along the calendar, for a second distinct activity.

    The hour digits are rewritten rather than the file re-synthesised, which
    keeps every other property of the fixture identical — two activities that
    differ only in when they happened is exactly what the duplicate window is
    about.
    """
    text = RIDE_GPX.read_text()
    hour = 9 + minutes // 60
    return text.replace("T09:", f"T{hour:02d}:").encode()


async def post_import(client, auth_headers, parts: list[tuple[str, bytes]]):
    files = [("files", (name, payload, "application/octet-stream")) for name, payload in parts]
    return await client.post("/api/activities/import", files=files, headers=auth_headers)


async def run_job(session, job_id: str, names: list[str]) -> ImportJob:
    """Run the staged job the endpoint created, then return the finished row.

    Reconstructs the runner's arguments from where the endpoint staged them,
    because the background task that would normally carry them is suppressed.
    """
    work_dir = settings.user_fit_dir(_TEST_USER_ID) / "imports" / job_id
    uploads = [
        (path, names[index])
        for index, path in enumerate(sorted(work_dir.glob("part-*")))
    ]
    job = (
        await session.execute(select(ImportJob).where(ImportJob.id == job_id))
    ).scalar_one()
    await execute_import_job(job_id, job.athlete_id, _TEST_USER_ID, uploads, work_dir, session)
    await session.refresh(job)
    return job


async def import_files(client, auth_headers, session, parts: list[tuple[str, bytes]]) -> ImportJob:
    response = await post_import(client, auth_headers, parts)
    assert response.status_code == 202, response.text
    return await run_job(session, response.json()["id"], [name for name, _ in parts])


def outcomes(job: ImportJob) -> dict[str, str]:
    return {row["filename"]: row["outcome"] for row in (job.results or [])}


@pytest.fixture
async def athlete_with_thresholds(session, seeded_athlete):
    """The seeded athlete with an FTP and a max HR.

    ``calculate_load`` needs one or the other to produce a Load at all, and the
    bare fixture has neither — so without this an assertion about Load would be
    testing the fixture rather than the import.
    """
    seeded_athlete.ftp = 250
    seeded_athlete.max_hr = 190
    await session.commit()
    return seeded_athlete


# ── the endpoint ─────────────────────────────────────────────────────────────


class TestImportEndpoint:
    async def test_returns_a_job_to_poll(self, client, auth_headers):
        response = await post_import(client, auth_headers, [("ride.gpx", RIDE_GPX.read_bytes())])

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "pending"
        assert body["source_name"] == "ride.gpx"
        assert body["processed"] == 0
        assert body["results"] == []

    async def test_names_a_multi_file_import_by_its_count(self, client, auth_headers):
        response = await post_import(
            client,
            auth_headers,
            [("a.gpx", RIDE_GPX.read_bytes()), ("b.tcx", RIDE_TCX.read_bytes())],
        )
        assert response.json()["source_name"] == "2 files"

    async def test_only_one_import_may_run_at_a_time(self, client, auth_headers):
        first = await post_import(client, auth_headers, [("ride.gpx", RIDE_GPX.read_bytes())])
        assert first.status_code == 202

        second = await post_import(client, auth_headers, [("ride.gpx", RIDE_GPX.read_bytes())])
        assert second.status_code == 409
        assert "already running" in second.json()["detail"]

    async def test_job_is_listed_and_fetchable(self, client, auth_headers):
        created = await post_import(client, auth_headers, [("ride.gpx", RIDE_GPX.read_bytes())])
        job_id = created.json()["id"]

        listing = await client.get("/api/activities/imports", headers=auth_headers)
        assert listing.status_code == 200
        assert [item["id"] for item in listing.json()["items"]] == [job_id]
        assert listing.json()["total"] == 1
        # The list view omits the per-file detail, which can be thousands of rows.
        assert "results" not in listing.json()["items"][0]

        detail = await client.get(f"/api/activities/imports/{job_id}", headers=auth_headers)
        assert detail.status_code == 200
        assert detail.json()["id"] == job_id

    async def test_another_athletes_job_is_not_visible(self, client, auth_headers, session):
        job = ImportJob(id="someone-elses", athlete_id="other-athlete", status="completed")
        session.add(job)
        await session.commit()

        response = await client.get("/api/activities/imports/someone-elses", headers=auth_headers)
        assert response.status_code == 404

    async def test_requires_authentication(self, client):
        response = await post_import(client, {}, [("ride.gpx", RIDE_GPX.read_bytes())])
        assert response.status_code == 401

    async def test_requires_consent(self, client, auth_headers, registry_session):
        from backend.app.models.registry_orm import User

        user = (
            await registry_session.execute(select(User).where(User.id == _TEST_USER_ID))
        ).scalar_one()
        user.consented_at = None
        user.consent_version = None
        await registry_session.commit()

        response = await post_import(client, auth_headers, [("ride.gpx", RIDE_GPX.read_bytes())])
        assert response.status_code == 403
        assert "Consent" in response.json()["detail"]


# ── running the job ──────────────────────────────────────────────────────────


class TestImportRun:
    async def test_imports_a_single_gpx(
        self, client, auth_headers, session, athlete_with_thresholds
    ):
        job = await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        assert job.status == "completed"
        assert (job.total_files, job.imported, job.skipped_duplicate, job.failed) == (1, 1, 0, 0)

        activity = (
            await session.execute(
                select(Activity).where(Activity.athlete_id == athlete_with_thresholds.id)
            )
        ).scalar_one()
        assert activity.status == "processed"
        assert activity.name == "Synthetic Ride"
        assert activity.sport_type == "Ride"
        assert activity.distance_m == pytest.approx(4800, rel=0.01)
        assert activity.load is not None

    async def test_walks_a_mixed_archive(self, client, auth_headers, session, seeded_athlete):
        fit = next(FIXTURES.glob("synthetic_bike_power_gps.fit"), None)
        assert fit is not None, "FIT fixture is expected alongside the XML ones"

        archive = make_zip(
            {
                "activities/one.gpx": shifted_gpx(60),
                "activities/two.tcx": RIDE_TCX.read_bytes(),
                "activities/three.fit.gz": gzip.compress(fit.read_bytes()),
                "activities/notes.txt": b"a stray file the export put there",
            }
        )
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.status == "completed"
        assert job.total_files == 4
        assert job.imported == 3
        assert job.failed == 1

        result = outcomes(job)
        assert result["activities/notes.txt"] == "failed"
        # The `.gz` is reported under the name of what came out of it.
        assert result["activities/three.fit"] == "imported"

        formats = {
            source.format
            for source in (await session.execute(select(ActivitySource))).scalars()
        }
        assert formats == {"fit", "gpx", "tcx"}

    async def test_gzipped_single_file(self, client, auth_headers, session):
        job = await import_files(
            client,
            auth_headers,
            session,
            [("ride.gpx.gz", gzip.compress(RIDE_GPX.read_bytes()))],
        )
        assert job.imported == 1
        assert outcomes(job) == {"ride.gpx": "imported"}

    async def test_one_corrupt_file_does_not_abort_the_job(self, client, auth_headers, session):
        archive = make_zip(
            {
                "good.gpx": RIDE_GPX.read_bytes(),
                "truncated.gpx": RIDE_GPX.read_bytes()[:4000],
                "later.tcx": RIDE_TCX.read_bytes().replace(b"T09:", b"T14:"),
            }
        )
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.status == "completed"
        assert job.imported == 2
        assert job.failed == 1

        failure = next(r for r in job.results if r["outcome"] == "failed")
        assert failure["filename"] == "truncated.gpx"
        assert failure["reason"], "a failed file must say why"

    async def test_a_file_that_is_not_an_activity_is_reported(self, client, auth_headers, session):
        job = await import_files(client, auth_headers, session, [("photo.jpg", b"\xff\xd8\xff\xe0nope")])

        assert job.imported == 0
        assert job.failed == 1
        assert "Not a FIT, GPX or TCX" in job.results[0]["reason"]

    async def test_progress_counts_add_up(self, client, auth_headers, session):
        archive = make_zip(
            {
                "one.gpx": RIDE_GPX.read_bytes(),
                "same-ride.tcx": RIDE_TCX.read_bytes(),
                "broken.gpx": b"<gpx>",
            }
        )
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.imported + job.skipped_duplicate + job.failed == job.total_files
        assert len(job.results) == job.total_files


# ── deduplication ────────────────────────────────────────────────────────────


class TestDeduplication:
    async def test_the_same_ride_twice_in_one_archive_keeps_the_richer_format(
        self, client, auth_headers, session, seeded_athlete
    ):
        archive = make_zip(
            {
                "activities/ride.gpx": RIDE_GPX.read_bytes(),
                "activities/ride.tcx": RIDE_TCX.read_bytes(),
            }
        )
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.imported == 1
        assert job.skipped_duplicate == 1

        result = outcomes(job)
        assert result["activities/ride.tcx"] == "imported"
        assert result["activities/ride.gpx"] == "skipped_duplicate"

        activities = (
            await session.execute(select(Activity).where(Activity.athlete_id == seeded_athlete.id))
        ).scalars().all()
        assert len(activities) == 1

        source = (await session.execute(select(ActivitySource))).scalar_one()
        assert source.format == "tcx"

        skipped = next(r for r in job.results if r["outcome"] == "skipped_duplicate")
        assert "activities/ride.tcx" in skipped["reason"]

    async def test_a_fit_beats_both_xml_formats(self, client, auth_headers, session):
        # The FIT fixture is a different ride, so its timestamps are rewritten
        # onto the GPX's start time to make the three collide.
        fit = FIXTURES / "synthetic_bike_power_gps.fit"
        from openkoutsi.fit import getStartTime

        fit_start = getStartTime(str(fit))
        gpx_text = RIDE_GPX.read_text().replace(
            "2024-03-02T09:", fit_start.strftime("%Y-%m-%dT%H:")
        ).replace("2024-03-02", fit_start.strftime("%Y-%m-%d"))

        archive = make_zip({"ride.gpx": gpx_text.encode(), "ride.fit": fit.read_bytes()})
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.imported == 1
        assert outcomes(job)["ride.fit"] == "imported"

        source = (await session.execute(select(ActivitySource))).scalar_one()
        assert source.format == "fit"

    async def test_re_importing_the_same_archive_skips_everything(
        self, client, auth_headers, session
    ):
        archive = make_zip({"ride.gpx": RIDE_GPX.read_bytes()})

        first = await import_files(client, auth_headers, session, [("export.zip", archive)])
        assert first.imported == 1

        second = await import_files(client, auth_headers, session, [("export.zip", archive)])
        assert second.status == "completed"
        assert second.imported == 0
        assert second.skipped_duplicate == 1
        assert "already exists" in second.results[0]["reason"]

        activities = (await session.execute(select(Activity))).scalars().all()
        assert len(activities) == 1

    async def test_rides_outside_the_window_are_separate_activities(
        self, client, auth_headers, session
    ):
        archive = make_zip(
            {"morning.gpx": RIDE_GPX.read_bytes(), "afternoon.gpx": shifted_gpx(300)}
        )
        job = await import_files(client, auth_headers, session, [("export.zip", archive)])

        assert job.imported == 2
        assert job.skipped_duplicate == 0


# ── the privacy promise ──────────────────────────────────────────────────────


class TestNoLocationIsStored:
    """A GPX is coordinates; an imported activity must not be.

    openkoutsi stores no location data — ``scripts/strip_fit_location.py``
    exists to hold FIT uploads to that, and importing a format made entirely of
    coordinates must not be the way around it.
    """

    #: Every channel an activity is allowed to store. A whitelist rather than a
    #: blocklist on purpose: the failure this guards against is a *new* channel
    #: appearing, and a blocklist only catches the ones someone thought to name.
    ALLOWED_STREAMS = {
        "power",
        "heartrate",
        "cadence",
        "speed",
        "altitude",
        "torque",
        "w_bal",
    }

    async def test_only_known_channels_are_stored(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        streams = (await session.execute(select(ActivityStream))).scalars().all()
        assert streams, "the import should have produced streams"
        assert {s.stream_type for s in streams} <= self.ALLOWED_STREAMS

    async def test_no_stream_is_the_coordinate_series(self, client, auth_headers, session):
        from openkoutsi import gpx

        route = gpx.extract_route(str(RIDE_GPX))
        assert route.points, "the fixture must actually contain coordinates"

        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])
        streams = (await session.execute(select(ActivityStream))).scalars().all()

        # Compared as *sequences*, not as sets of values: a torque reading and a
        # longitude can coincide at three decimal places by arithmetic accident,
        # but a stored coordinate track would reproduce the route's own series
        # position by position.
        latitudes = [round(p.latitude, 4) for p in route.points]
        longitudes = [round(p.longitude, 4) for p in route.points]
        for stream in streams:
            values = [round(v, 4) for v in (stream.data or []) if v is not None]
            if len(values) < 10:
                continue
            overlap = min(len(values), len(longitudes))
            assert values[:overlap] != longitudes[:overlap], (
                f"{stream.stream_type} is the longitude track"
            )
            assert values[:overlap] != latitudes[:overlap], (
                f"{stream.stream_type} is the latitude track"
            )

    async def test_the_stored_original_is_the_only_place_coordinates_remain(
        self, client, auth_headers, session
    ):
        """The uploaded file itself is kept as it arrived — including its track.

        That is the deliberate trade: an athlete can download exactly what they
        imported, and a reprocess re-reads exactly what was parsed. What matters
        is that nothing extracts the track *out* of the file into the database,
        where it would be queryable, exportable and analysable.
        """
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        source = (await session.execute(select(ActivitySource))).scalar_one()
        assert source.fit_file_path, "the original is retained"
        assert source.format == "gpx"
        # ...and the path is all the database knows about it.
        assert "lat" not in json.dumps(
            {c.name: str(getattr(source, c.name)) for c in source.__table__.columns}
        ).lower()

    async def test_the_activity_row_holds_no_coordinates(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        activity = (await session.execute(select(Activity))).scalar_one()
        for column in activity.__table__.columns:
            assert "lat" not in column.name.lower()
            assert "lon" not in column.name.lower()

        # Nothing sneaks into a JSON column either.
        payload = json.dumps(
            {
                "zone_times": activity.zone_times,
                "labels": activity.labels,
                "notes": activity.notes,
            }
        ).lower()
        for token in ("lat", "lon", "coord"):
            assert token not in payload

    async def test_distance_and_elevation_still_survive(self, client, auth_headers, session):
        """The two numbers the coordinates existed to produce."""
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        activity = (await session.execute(select(Activity))).scalar_one()
        assert activity.distance_m == pytest.approx(4800, rel=0.01)
        assert activity.elevation_m == pytest.approx(60, abs=3)


# ── what each format yields ──────────────────────────────────────────────────


class TestFormatDifferences:
    async def test_a_gpx_without_power_is_a_complete_import(
        self, client, auth_headers, session, athlete_with_thresholds
    ):
        job = await import_files(
            client, auth_headers, session, [("hr_only.gpx", HR_ONLY_GPX.read_bytes())]
        )
        assert job.imported == 1

        activity = (await session.execute(select(Activity))).scalar_one()
        assert activity.status == "processed"
        # No power meter means no power-derived metric — and a Load anyway,
        # because `calculate_load` falls back to heart rate.
        assert activity.avg_power is None
        assert activity.weighted_power is None
        assert activity.avg_hr is not None
        assert activity.load is not None

        types = {
            s.stream_type
            for s in (await session.execute(select(ActivityStream))).scalars()
        }
        assert "heartrate" in types
        assert "power" not in types

    async def test_tcx_laps_become_intervals(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.tcx", RIDE_TCX.read_bytes())])

        intervals = (
            await session.execute(
                select(ActivityInterval).order_by(ActivityInterval.interval_number)
            )
        ).scalars().all()
        assert len(intervals) == 2
        assert all(not iv.is_auto_split for iv in intervals)

    async def test_gpx_falls_back_to_auto_split_intervals(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])

        intervals = (await session.execute(select(ActivityInterval))).scalars().all()
        assert intervals, "every processed activity gets a breakdown"
        assert all(iv.is_auto_split for iv in intervals)


# ── living with an imported activity ─────────────────────────────────────────


class TestImportedActivityLifecycle:
    async def test_the_activity_reports_its_source_format(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])
        activity = (await session.execute(select(Activity))).scalar_one()

        response = await client.get(f"/api/activities/{activity.id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["has_fit_file"] is True
        assert response.json()["original_format"] == "gpx"

    async def test_the_original_downloads_as_a_gpx(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])
        activity = (await session.execute(select(Activity))).scalar_one()

        response = await client.get(f"/api/activities/{activity.id}/fit", headers=auth_headers)
        assert response.status_code == 200
        # Starlette encodes the filename per RFC 5987, so match the extension
        # rather than the exact header spelling.
        assert response.headers["content-disposition"].endswith(".gpx")
        # Byte-for-byte what was imported, not a conversion of it.
        assert b"<trkpt" in response.content

    async def test_reprocess_works_on_a_gpx_sourced_activity(
        self, client, auth_headers, session, athlete_with_thresholds
    ):
        await import_files(client, auth_headers, session, [("ride.gpx", RIDE_GPX.read_bytes())])
        activity = (await session.execute(select(Activity))).scalar_one()

        response = await client.post(
            f"/api/activities/{activity.id}/reprocess", headers=auth_headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == activity.id
        assert body["load"] is not None
        assert body["intervals"], "reprocess rebuilds the interval breakdown"

    async def test_reprocess_keeps_tcx_laps(self, client, auth_headers, session):
        await import_files(client, auth_headers, session, [("ride.tcx", RIDE_TCX.read_bytes())])
        activity = (await session.execute(select(Activity))).scalar_one()

        response = await client.post(
            f"/api/activities/{activity.id}/reprocess", headers=auth_headers
        )
        assert response.status_code == 200
        # The laps are re-read from the stored original, so they survive a
        # reprocess rather than degrading into an auto-split.
        assert len(response.json()["intervals"]) == 2
        assert not response.json()["intervals"][0]["is_auto_split"]
