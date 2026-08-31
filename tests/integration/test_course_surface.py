"""Surface classification end to end (issue #56, Stage 2).

No live engine: the matcher is a double, so CI never needs 15 GB of routing
tiles. What these tests hold onto are the properties that make the feature
honest rather than merely present.

* **Confidence survives the whole way out.** A tag-confirmed segment and a
  road-class-inferred one are distinguishable in the API response, not
  flattened into one comfortable-looking answer.
* **Degrade, don't fail.** Unconfigured, unreachable, timing out, or matching
  only part of a course — every one of them leaves a complete Stage 1 result
  and no error in front of the athlete.
* **The toggle refuses the capability**, not merely the entry point — and
  never the data export, because a right to your own data is not a feature an
  instance switches off.
* **A short severe sector survives.** 130 m of mud inside 40 km of asphalt is
  the most important thing on that course, and it has to reach the segment
  table, the ribbon, the sector list and the prose.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from backend.app.models.registry_orm import InstanceSettings
from backend.app.models.user_orm import Course, CourseSegment, CourseTrack
from backend.app.services import course_surface
from openkoutsi import surface as surface_math

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
COURSE_GPX = FIXTURES / "synthetic_course.gpx"

FIXTURE_LAT_PREFIX = "61.5"
FIXTURE_LON = "20.5"
_LON_TOKEN = re.compile(rf"(?<![\d.]){re.escape(FIXTURE_LON)}0*(?![\d])")


# ── doubles ───────────────────────────────────────────────────────────────────


class _FakeMatcher:
    """A matcher whose answers a test dictates. Duck-typed, like the email double.

    ``answers`` is either a callable taking the point count and returning one
    raw engine value per point, or ``None`` to mean "could not answer" — which
    is how an unreachable or empty-tiled sidecar looks from here.
    """

    def __init__(self, answers=None, configured: bool = True):
        self._answers = answers
        self._configured = configured
        self.calls = 0

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def match(self, points):
        self.calls += 1
        if self._answers is None:
            return None
        return self._answers(len(points))


def _all(value: str):
    return lambda n: [value] * n


def _mud_sector(fraction_start=0.4, fraction_end=0.45):
    """Asphalt with a short stretch of dirt in the middle of it."""

    def _answer(n):
        lo, hi = int(n * fraction_start), int(n * fraction_end)
        return ["dirt" if lo <= i < hi else "paved_smooth" for i in range(n)]

    return _answer


def _using(matcher):
    """Patch the matcher accessor everywhere it is looked up.

    It is fetched by name in two modules rather than resolved through a FastAPI
    dependency, because the background job is not a request and has no
    dependency graph to override.
    """
    return (
        patch("backend.app.api.courses.get_surface_matcher", return_value=matcher),
        patch(
            "backend.app.services.course_surface.get_surface_matcher",
            return_value=matcher,
        ),
    )


# ── helpers ───────────────────────────────────────────────────────────────────


async def _seed_rider(session, seeded_athlete):
    seeded_athlete.ftp = 250
    seeded_athlete.weight_kg = 75.0
    await session.commit()


async def _create_bike(client, auth_headers):
    resp = await client.post(
        "/api/bikes",
        json={"name": "Gravel", "tyre_width_mm": 40, "riding_position": "hoods"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _upload(client, auth_headers, bike_id):
    """Upload the fixture course with the background match suppressed.

    `asyncio.create_task` is not patched by the test client, so an unsuppressed
    match would run against a session this test is still using. Every test here
    drives the job by hand instead, which is also how `test_course_plan` drives
    the plan.
    """
    with patch.object(course_surface, "match_course_surface"):
        resp = await client.post(
            "/api/courses",
            headers=auth_headers,
            data={"bike_id": bike_id},
            files={
                "file": (
                    COURSE_GPX.name,
                    COURSE_GPX.read_bytes(),
                    "application/gpx+xml",
                )
            },
        )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture
def run_match(session, registry_session):
    """Drive the real background job against the test databases.

    The job runs outside a request, so it opens both of its own sessions rather
    than receiving them: the user database through `get_user_session_factory`
    and the registry — for the capability check — through `registry_session`.
    Neither is the FastAPI dependency the test client overrides, so both are
    redirected here, the same way `test_course_plan` redirects the streaming
    machinery's registry session.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _user_session():
        yield session

    @asynccontextmanager
    async def _registry():
        yield registry_session

    async def _run(course_id, athlete_id, matcher):
        with (
            patch.object(
                course_surface,
                "get_user_session_factory",
                return_value=lambda: _user_session(),
            ),
            # `failure_recovery` deliberately opens a *fresh* session — the
            # original may be exactly what failed — so it resolves the factory
            # through its own module and needs redirecting separately.
            patch(
                "backend.app.services.llm_streaming.get_user_session_factory",
                return_value=lambda: _user_session(),
            ),
            patch.object(course_surface, "registry_session", _registry),
        ):
            await course_surface.match_course_surface(
                athlete_id, course_id, "test-user-00000000", None, matcher
            )

    return _run



async def _always_enabled(_session) -> bool:
    """Stand in for the capability check when the test drives the job directly."""
    return True


@pytest.fixture(autouse=True)
def _course_recon_enabled(course_recon_on):
    """These tests are about an instance whose admin switched course recon on.

    `TestTheInstanceSwitch` below is the one that covers it being off.
    """


@pytest.fixture
async def course(client, auth_headers, session, seeded_athlete, course_recon_on):
    # `course_recon_on` is requested explicitly rather than relied on through
    # the autouse fixture above: pytest is free to build this one first, and a
    # course uploaded against a switched-off instance is a 404, not a course.
    await _seed_rider(session, seeded_athlete)
    bike_id = await _create_bike(client, auth_headers)
    return await _upload(client, auth_headers, bike_id)


# ── the tests ─────────────────────────────────────────────────────────────────


class TestConfidenceEndToEnd:
    async def test_a_confirmed_and_an_inferred_segment_are_distinguishable(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """The property the whole stage exists for.

        Flattening these two into one field would be exactly the quiet
        dishonesty the project refuses elsewhere — reason codes instead of
        misleading figures, badges hidden rather than shown permanently locked.
        """
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_mud_sector())
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        segments = detail.json()["segments"]

        confidences = {s["surface_confidence"] for s in segments}
        assert surface_math.CONFIRMED in confidences
        assert surface_math.INFERRED in confidences

        dirt = [s for s in segments if s["surface"] == surface_math.DIRT]
        asphalt = [s for s in segments if s["surface"] == surface_math.ASPHALT]
        assert dirt and asphalt
        assert all(s["surface_confidence"] == surface_math.CONFIRMED for s in dirt)
        # Untagged and explicitly-paved roads are indistinguishable, so the
        # weaker claim is the only honest one.
        assert all(s["surface_confidence"] == surface_math.INFERRED for s in asphalt)

    async def test_the_raw_matcher_value_is_preserved_not_discarded(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("gravel"))
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert {s["surface_raw"] for s in detail.json()["segments"]} == {"gravel"}

    async def test_the_rolling_resistance_used_is_reported(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """A number the athlete is asked to trust should be inspectable."""
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("gravel"))
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        crr = {s["crr_used"] for s in detail.json()["segments"] if s["crr_used"]}
        assert crr == {surface_math.crr_for(surface_math.GRAVEL, 40)}

    async def test_matching_changes_the_predicted_time(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """Surface is not decoration: it re-solves the physics."""
        before = course["predicted_time_s"]
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("dirt"))
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["predicted_time_s"] > before


class TestShortSevereSectors:
    """130 m of mud inside 40 km of asphalt must not be smoothed away."""

    async def test_the_sector_gets_its_own_segment(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_mud_sector())
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        dirt = [
            s for s in detail.json()["segments"] if s["surface"] == surface_math.DIRT
        ]
        assert dirt, "a severe short sector must keep its own pacing row"

    async def test_it_is_named_in_the_rough_sector_list(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_mud_sector())
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        sectors = detail.json()["rough_sectors"]
        assert sectors and any(s[2] == surface_math.DIRT for s in sectors)

    async def test_the_ribbon_carries_it_at_full_resolution(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """The ribbon has no minimum length, unlike the pacing table."""
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_mud_sector())
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        ribbon = detail.json()["surface_ribbon"]
        assert ribbon and any(entry[2] == surface_math.DIRT for entry in ribbon)

    async def test_a_smooth_course_has_no_rough_sectors(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("paved_smooth"))
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["rough_sectors"] == []


class TestDegradation:
    """Absent rather than broken, at every step."""

    async def test_an_unconfigured_instance_leaves_a_complete_stage_1_course(
        self, client, auth_headers, course
    ):
        assert course["status"] == "ready"
        assert course["surface_status"] is None
        assert course["segments"]
        assert all(s["surface"] is None for s in course["segments"])
        assert course["predicted_time_s"] is not None

    async def test_an_unreachable_matcher_surfaces_no_error(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(None))
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        data = detail.json()
        assert detail.status_code == 200
        assert data["status"] == "ready", "the course itself is not broken"
        assert data["error"] is None
        assert data["surface_status"] == course_surface.UNAVAILABLE
        assert data["segments"] and data["predicted_time_s"] is not None

    async def test_an_unavailable_match_leaves_the_stage_1_numbers_alone(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        before = course["predicted_time_s"]
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(None))
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["predicted_time_s"] == before

    async def test_a_partial_match_keeps_what_was_matched(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        def _half(n):
            return ["gravel" if i < n // 2 else None for i in range(n)]

        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_half))
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        classes = {s["surface"] for s in detail.json()["segments"]}
        assert surface_math.GRAVEL in classes
        assert surface_math.UNKNOWN in classes, "unmatched reads as unknown"

    async def test_an_unmatched_stretch_never_reads_as_confirmed(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        def _half(n):
            return ["gravel" if i < n // 2 else None for i in range(n)]

        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_half))
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        unknown = [
            s
            for s in detail.json()["segments"]
            if s["surface"] == surface_math.UNKNOWN
        ]
        assert unknown
        assert all(
            s["surface_confidence"] == surface_math.INFERRED for s in unknown
        )

    async def test_a_matcher_that_snaps_nothing_reads_as_unavailable(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """A course outside this instance's tile region, most likely.

        Every point comes back unmatched. Drawing a full-length grey "unknown"
        band would claim we had looked and found something when we had only
        looked, so this reads as no surface data at all.
        """
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(lambda n: [None] * n)
        )
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["surface_status"] == course_surface.UNAVAILABLE
        assert detail.json()["surface_ribbon"] is None


class TestReMatchingAStoredCourse:
    """Most of the value of turning the sidecar on: no re-upload."""

    async def test_an_instance_that_enables_matching_later_can_enrich_a_course(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        assert course["surface_status"] is None
        matcher = _FakeMatcher(_all("gravel"))
        with _using(matcher)[0], _using(matcher)[1]:
            resp = await client.post(
                f"/api/courses/{course['id']}/surface", headers=auth_headers
            )
        assert resp.status_code == 202
        assert resp.json()["status"] == course_surface.PENDING

        await run_match(course["id"], seeded_athlete.id, matcher)
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["surface_status"] == course_surface.DONE
        assert {s["surface"] for s in detail.json()["segments"]} == {
            surface_math.GRAVEL
        }

    async def test_it_is_idempotent(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        matcher = _FakeMatcher(_all("gravel"))
        await run_match(course["id"], seeded_athlete.id, matcher)
        first = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)

        await run_match(course["id"], seeded_athlete.id, matcher)
        second = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)

        assert first.json()["segments"] == second.json()["segments"]
        assert first.json()["surface_ribbon"] == second.json()["surface_ribbon"]

    async def test_a_match_already_running_is_not_started_twice(
        self, client, auth_headers, session, course
    ):
        matcher = _FakeMatcher(_all("gravel"))
        with _using(matcher)[0], _using(matcher)[1]:
            with patch.object(course_surface, "match_course_surface"):
                await client.post(
                    f"/api/courses/{course['id']}/surface", headers=auth_headers
                )
                second = await client.post(
                    f"/api/courses/{course['id']}/surface", headers=auth_headers
                )
        assert second.status_code == 202
        assert second.json()["status"] == course_surface.PENDING

    async def test_asking_without_a_matcher_says_so_plainly(
        self, client, auth_headers, course
    ):
        resp = await client.post(
            f"/api/courses/{course['id']}/surface", headers=auth_headers
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "no_surface_matcher"

    async def test_re_solving_reuses_the_stored_match_without_asking_again(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """Changing bike or target costs no sidecar round trip.

        The track has not moved, so what the matcher said about it still holds.
        """
        matcher = _FakeMatcher(_all("gravel"))
        await run_match(course["id"], seeded_athlete.id, matcher)
        calls_before = matcher.calls

        with _using(matcher)[0], _using(matcher)[1]:
            resp = await client.post(
                f"/api/courses/{course['id']}/reanalyze",
                json={"target_time_s": 3600},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        assert matcher.calls == calls_before, "re-solving must not re-match"
        assert {s["surface"] for s in resp.json()["segments"]} == {surface_math.GRAVEL}


class TestStrandedMatches:
    async def test_a_match_a_restart_interrupted_is_settled(self, session, course):
        """Not left pending for ever, and not called an error either.

        Nothing about the course is wrong — the match simply did not finish.
        """
        from backend.app.services.stranded_runs import settle_stranded_user_runs

        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        row.surface_status = course_surface.PENDING
        row.surface_updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
        await session.commit()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _factory():
            yield session

        # `settle_stranded_user_runs` imports the factory inside the
        # function, so it has to be patched where it is defined.
        with patch(
            "backend.app.db.user_session.get_user_session_factory",
            return_value=lambda: _factory(),
        ):
            await settle_stranded_user_runs("test-user-00000000")

        await session.refresh(row)
        assert row.surface_status == course_surface.UNAVAILABLE
        assert row.surface_run_id is None

    async def test_a_fresh_pending_match_is_left_alone(self, session, course):
        """"There is a pending row and we just booted" is not evidence of death."""
        from backend.app.services.stranded_runs import settle_stranded_user_runs

        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        row.surface_status = course_surface.PENDING
        row.surface_updated_at = datetime.now(timezone.utc)
        await session.commit()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _factory():
            yield session

        # `settle_stranded_user_runs` imports the factory inside the
        # function, so it has to be patched where it is defined.
        with patch(
            "backend.app.db.user_session.get_user_session_factory",
            return_value=lambda: _factory(),
        ):
            await settle_stranded_user_runs("test-user-00000000")

        await session.refresh(row)
        assert row.surface_status == course_surface.PENDING


class TestTheInstanceSwitch:
    """Off refuses the capability, not merely the entry point."""

    @pytest.fixture(autouse=True)
    async def _switch_off(self, registry_session, course):
        # Depends on `course` so the upload happens while the instance still
        # offers the feature. Switching off is what these tests do to an
        # instance that already has courses in it — which is exactly the
        # situation an existing deployment upgrading into #56 lands in.
        instance = (
            await registry_session.execute(select(InstanceSettings).limit(1))
        ).scalar_one_or_none()
        instance.allow_course_recon = False
        await registry_session.commit()

    async def test_courses_are_not_found(self, client, auth_headers, course):
        for path in ("/api/courses", f"/api/courses/{course['id']}"):
            assert (await client.get(path, headers=auth_headers)).status_code == 404

    async def test_bikes_go_with_them(self, client, auth_headers, course):
        """A bike exists for nothing but course pacing."""
        assert (await client.get("/api/bikes", headers=auth_headers)).status_code == 404

    async def test_re_analysis_and_matching_are_refused_too(
        self, client, auth_headers, course
    ):
        for path in ("reanalyze", "surface", "plan"):
            resp = await client.post(
                f"/api/courses/{course['id']}/{path}", json={}, headers=auth_headers
            )
            assert resp.status_code == 404, path

    async def test_the_background_matcher_declines(
        self, session, seeded_athlete, course, run_match
    ):
        """The check lives in the job, not only on the route that schedules it.

        Honouring a switch at the entry point alone is the mistake the personal
        access token gate made by checking issuance and leaving /mcp open.
        """
        matcher = _FakeMatcher(_all("gravel"))
        await run_match(course["id"], seeded_athlete.id, matcher)
        assert matcher.calls == 0

        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        await session.refresh(row)
        assert row.surface_status is None

    async def test_nothing_is_deleted(self, session, course):
        """Turning it back on returns every stored course intact."""
        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        assert row is not None
        segments = (
            await session.execute(
                select(CourseSegment).where(CourseSegment.course_id == course["id"])
            )
        ).scalars().all()
        assert segments
        assert await session.get(CourseTrack, course["id"]) is not None

    async def test_the_data_export_still_contains_the_course(
        self, client, auth_headers, course
    ):
        """A right to your own data is not a feature an instance toggles.

        The export is deliberately outside the gate — deleting nothing but
        withholding it would be the same harm by a slower route.
        """
        resp = await client.get("/api/athlete/export", headers=auth_headers)
        assert resp.status_code == 200
        assert course["id"] in resp.text


class TestTheRouteLlmWall:
    """Storing route data is expected. Route data *leaving* is not."""

    async def test_no_coordinate_reaches_a_surface_response(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_mud_sector())
        )
        for path in ("", f"/{course['id']}"):
            body = (
                await client.get(f"/api/courses{path}", headers=auth_headers)
            ).text
            assert FIXTURE_LAT_PREFIX not in body
            assert not _LON_TOKEN.search(body)

    async def test_the_stored_per_point_series_carries_no_coordinate(
        self, session, seeded_athlete, course, run_match
    ):
        """The matched series is one entry per track point but holds none of it."""
        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("gravel"))
        )
        track = await session.get(CourseTrack, course["id"])
        assert track.surfaces
        assert all(
            entry == ["gravel", surface_math.CONFIRMED] for entry in track.surfaces
        )


class TestTheRunToken:
    """The guards that stop two runs writing over each other.

    A match holds its columns by token, exactly as a plan run does. These are
    the branches that would silently corrupt a segment table if they were
    wrong — a stale run writing surface classes onto segments a later
    re-analysis has already replaced.
    """

    async def test_a_run_that_lost_its_token_before_starting_does_nothing(
        self, session, seeded_athlete, course, run_match
    ):
        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        row.surface_run_id = "somebody-elses-run"
        await session.commit()

        matcher = _FakeMatcher(_all("gravel"))
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _user_session():
            yield session

        @asynccontextmanager
        async def _registry():
            yield session  # unused on this path; the run bails before matching

        with (
            patch.object(
                course_surface,
                "get_user_session_factory",
                return_value=lambda: _user_session(),
            ),
            patch.object(course_surface, "registry_session", _registry),
            patch.object(course_surface, "course_recon_enabled", _always_enabled),
        ):
            await course_surface.match_course_surface(
                seeded_athlete.id, course["id"], "test-user-00000000", "my-run", matcher
            )
        assert matcher.calls == 0, "a superseded run must not even ask"

    async def test_a_run_superseded_while_it_was_matching_discards_its_writes(
        self, session, seeded_athlete, course
    ):
        """The expensive case: the sidecar answered, but a re-analysis landed.

        Its segment table is the current one, so this run's classes describe
        distances that no longer exist and must be thrown away.
        """
        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        run_id = course_surface.claim_run(row)
        await session.commit()

        class _StealsTheToken(_FakeMatcher):
            async def match(self, points):
                # A re-analysis lands mid-flight and clears the token.
                row.surface_run_id = None
                await session.commit()
                return await super().match(points)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _user_session():
            yield session

        @asynccontextmanager
        async def _registry():
            yield session

        with (
            patch.object(
                course_surface,
                "get_user_session_factory",
                return_value=lambda: _user_session(),
            ),
            patch.object(course_surface, "registry_session", _registry),
            patch.object(course_surface, "course_recon_enabled", _always_enabled),
        ):
            await course_surface.match_course_surface(
                seeded_athlete.id,
                course["id"],
                "test-user-00000000",
                run_id,
                _StealsTheToken(_all("gravel")),
            )

        await session.refresh(row)
        segments = (
            await session.execute(
                select(CourseSegment).where(CourseSegment.course_id == course["id"])
            )
        ).scalars().all()
        assert all(s.surface is None for s in segments), "stale run wrote anyway"

    async def test_a_course_that_vanished_is_not_an_error(
        self, session, seeded_athlete, run_match
    ):
        matcher = _FakeMatcher(_all("gravel"))
        await run_match("no-such-course", seeded_athlete.id, matcher)
        assert matcher.calls == 0

    async def test_settling_a_match_that_is_not_pending_is_a_no_op(self):
        class _Row:
            surface_status = course_surface.DONE
            surface_run_id = None
            surface_updated_at = None

        assert course_surface.settle_course_surface(_Row()) is False


class TestMoreDegradation:
    async def test_a_course_with_no_stored_track_cannot_be_matched(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """Nothing to send, so nothing is claimed — and no error is raised."""
        track = await session.get(CourseTrack, course["id"])
        await session.delete(track)
        await session.commit()

        matcher = _FakeMatcher(_all("gravel"))
        await run_match(course["id"], seeded_athlete.id, matcher)
        assert matcher.calls == 0

        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["surface_status"] == course_surface.UNAVAILABLE
        assert detail.json()["status"] == "ready"

    async def test_an_athlete_without_ftp_keeps_the_match_but_not_a_resolve(
        self, session, seeded_athlete, course, run_match
    ):
        """The physics cannot run, but what the matcher said is still true.

        Keeping it means the surface is there the moment the athlete fills in
        their profile, without another trip to the sidecar.
        """
        seeded_athlete.ftp = None
        await session.commit()

        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("gravel")))

        track = await session.get(CourseTrack, course["id"])
        await session.refresh(track)
        assert track.surfaces, "the match itself is worth keeping"
        segments = (
            await session.execute(
                select(CourseSegment).where(CourseSegment.course_id == course["id"])
            )
        ).scalars().all()
        assert all(
            s.surface is None for s in segments
        ), "segments must not be half-updated"

    async def test_a_crash_mid_match_settles_the_course_rather_than_stranding_it(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """An unexpected failure must not leave a spinner running for ever.

        `stream_into_db`-style recovery only settles what it is running; a
        failure *outside* the inner handler needs the fresh-session fallback,
        which is the branch this exercises.
        """
        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        course_surface.claim_run(row)
        await session.commit()

        class _Explodes(_FakeMatcher):
            async def match(self, points):
                raise RuntimeError("the sidecar returned something impossible")

        await run_match(course["id"], seeded_athlete.id, _Explodes())

        await session.refresh(row)
        assert row.surface_status == course_surface.UNAVAILABLE
        assert row.surface_run_id is None

        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.status_code == 200
        assert detail.json()["status"] == "ready", "the course itself is untouched"

    async def test_a_course_with_no_bike_still_matches_on_defaults(
        self, client, auth_headers, session, seeded_athlete, course, run_match
    ):
        """Deleting a bike nulls `course.bike_id`; that must not break a match."""
        row = (
            await session.execute(select(Course).where(Course.id == course["id"]))
        ).scalar_one()
        row.bike_id = None
        await session.commit()

        await run_match(course["id"], seeded_athlete.id, _FakeMatcher(_all("gravel")))
        detail = await client.get(f"/api/courses/{course['id']}", headers=auth_headers)
        assert detail.json()["surface_status"] == course_surface.DONE
