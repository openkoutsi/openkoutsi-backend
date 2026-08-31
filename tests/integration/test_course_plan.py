"""Integration tests for the LLM-written course pacing plan (issue #55).

Same shape as ``test_goal_guidance``: the trigger/pending contract, a full run
against a mocked SSE stream with the MOOD tag parsed off, the subscription
gate, and stranded-run recovery. Plus the invariant this feature exists to
uphold: **no coordinate reaches the LLM context** — the mocked transport
captures the exact request payload and the test greps it.
"""
from __future__ import annotations

import pytest

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from backend.app.models.user_orm import Course
from backend.app.services.stranded_runs import settle_stranded_user_runs

_TEST_USER_ID = "test-user-00000000"

FIXTURES = Path(__file__).parent.parent.parent / "testdata" / "fixtures"
COURSE_GPX = FIXTURES / "synthetic_course.gpx"

_PLAN_SSE = [
    "MOOD: knowing",
    "\n\n",
    "Ride the opening kilometres well inside yourself — the day is decided ",
    "on the climb starting at km 4.0, not before it.\n\n",
    "Eat every 40 minutes from the start. All predictions assume still air.",
]


def _make_streaming_lines(chunks):
    async def _gen():
        for chunk in chunks:
            escaped = chunk.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"choices":[{{"delta":{{"content":"{escaped}"}}}}]}}'
        yield "data: [DONE]"
    return _gen()


def _mock_httpx_stream(chunks, requests_out: list | None = None):
    """Patch object for httpx.AsyncClient that streams *chunks* as SSE and
    records every request payload into *requests_out*."""
    mock_resp = AsyncMock()
    mock_resp.aiter_lines = MagicMock(return_value=_make_streaming_lines(chunks))
    mock_resp.is_error = False

    @asynccontextmanager
    async def _mock_stream(*args, **kwargs):
        if requests_out is not None:
            requests_out.append(kwargs.get("json") or {})
        yield mock_resp

    mock_client = AsyncMock()
    mock_client.stream = _mock_stream

    @asynccontextmanager
    async def _mock_httpx(*args, **kwargs):
        yield mock_client

    return _mock_httpx()


@asynccontextmanager
async def _mock_registry_session(instance):
    reg = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = instance
    reg.execute = AsyncMock(return_value=result)
    yield reg


async def _seed_course(client, auth_headers, session, seeded_athlete, **form) -> str:
    """Upload the fixture course for a BYOK-configured athlete; return its id."""
    seeded_athlete.ftp = 250
    seeded_athlete.weight_kg = 75.0
    # BYOK config so config resolution needs no instance and usage isn't recorded.
    seeded_athlete.app_settings = {
        "llm_base_url": "http://localhost:11434/v1", "llm_model": "x",
    }
    await session.commit()

    bike = await client.post(
        "/api/bikes",
        json={"name": "Road", "tyre_width_mm": 28, "riding_position": "hoods"},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/courses",
        headers=auth_headers,
        data={"bike_id": bike.json()["id"], **{k: str(v) for k, v in form.items()}},
        files={"file": (COURSE_GPX.name, COURSE_GPX.read_bytes(), "application/gpx+xml")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _run_plan_bg(session, course_id: str, athlete_id: str, requests_out=None, run_id=None):
    """Drive the real background task with the LLM transport mocked."""
    from backend.app.services import llm_course_plan as svc

    @asynccontextmanager
    async def _factory():
        yield session

    with (
        patch.object(svc, "get_user_session_factory", return_value=lambda: _factory()),
        patch("backend.app.services.llm_streaming._RegistrySessionLocal",
              return_value=_mock_registry_session(None)),
        patch("httpx.AsyncClient",
              return_value=_mock_httpx_stream(_PLAN_SSE, requests_out)),
    ):
        await svc.generate_course_plan_bg(
            athlete_id, course_id, _TEST_USER_ID, None, run_id
        )


async def _match_course(session, course_id: str, rough: bool = False) -> None:
    """Give a stored course a surface, by running the real analysis path.

    No matcher and no HTTP: the per-point series is written straight onto the
    track row, which is exactly what a completed match leaves behind, and the
    course is then re-solved from it.
    """
    from backend.app.models.user_orm import Bike, CourseTrack
    from backend.app.services import course_analysis
    from openkoutsi import course as course_math

    track = await session.get(CourseTrack, course_id)
    n = len(track.points)
    lo, hi = int(n * 0.40), int(n * 0.45)
    values = [
        "dirt" if (rough and lo <= i < hi) else "paved_smooth" for i in range(n)
    ]
    track.surfaces = [[v, "confirmed" if v != "paved_smooth" else "inferred"] for v in values]

    course = (
        await session.execute(select(Course).where(Course.id == course_id))
    ).scalar_one()
    bike = await session.get(Bike, course.bike_id)
    analysis, _ = course_analysis.analyze_stored_track(
        track.points,
        course_math.RiderParams(ftp_w=250.0, weight_kg=75.0),
        course_math.BikeParams(
            tyre_width_mm=bike.tyre_width_mm, riding_position=bike.riding_position
        ),
        course.target_time_s,
        course.target_power_w,
        track.surfaces,
    )
    await course_analysis.persist_analysis(
        course,
        analysis,
        session,
        rider=course_math.RiderParams(ftp_w=250.0, weight_kg=75.0),
    )
    await session.commit()


@pytest.fixture(autouse=True)
def _course_recon_enabled(course_recon_on):
    """Course recon defaults off (issue #56); these tests are about an instance
    whose admin switched it on. `TestInstanceSwitch` covers the other case."""


class TestCoursePlan:
    async def test_no_plan_yet(self, client, auth_headers, session, seeded_athlete):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        resp = await client.get(f"/api/courses/{course_id}/plan", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] is None
        assert data["plan"] is None
        assert data["plan_ai_generated"] is False

    async def test_trigger_sets_pending_and_double_trigger_is_stable(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            first = await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
            second = await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        assert first.status_code == 202 and second.status_code == 202
        assert second.json()["status"] == "pending"

        status = await client.get(f"/api/courses/{course_id}/plan", headers=auth_headers)
        assert status.json()["status"] == "pending"

    async def test_full_run_parses_mood_and_prose(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            resp = await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        assert resp.status_code == 202

        await _run_plan_bg(session, course_id, seeded_athlete.id)

        resp = await client.get(f"/api/courses/{course_id}/plan", headers=auth_headers)
        data = resp.json()
        assert data["status"] == "done"
        assert data["mood"] == "knowing"
        assert data["plan"].startswith("Ride the opening kilometres")
        assert "MOOD" not in data["plan"]
        assert data["plan_ai_generated"] is True

    async def test_no_coordinates_reach_the_llm_context(
        self, client, auth_headers, session, seeded_athlete
    ):
        """The invariant Stage 0 left standing: route data being stored is
        expected; route data *leaving* is not. The captured request payload —
        system prompt, user prompt, everything — must be coordinate-free."""
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )

        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)

        assert requests, "the mocked transport saw no request"
        import json as jsonlib

        payload = jsonlib.dumps(requests).lower()
        # The fixture rides due north from (61.5, 20.5); its latitudes span
        # 61.50–61.63 and its longitude is constant. None of that — nor any
        # coordinate-shaped vocabulary — may appear in the context.
        for forbidden in ("latitude", "longitude", "61.50", "61.6", "20.50", "trkpt", "coordinate"):
            assert forbidden not in payload, f"{forbidden!r} leaked into the LLM context"
        # And the table itself did make it — this is a real prompt, not an empty one.
        assert "segments" in payload
        assert "km" in payload

    async def test_the_prompt_states_the_still_air_assumption(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )

        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)
        payload = " ".join(
            m.get("content", "") for req in requests for m in req.get("messages", [])
        )
        assert "still air" in payload
        # With no surface match, the physics really did solve this as pavement
        # and the plan has to admit it rather than quietly assume it.
        assert "dry pavement" in payload

    async def test_an_unmatched_course_is_not_told_about_surfaces(
        self, client, auth_headers, session, seeded_athlete
    ):
        """A course with no match must not get the confidence paragraph.

        Handing the model rules about inferred surfaces on a course that has
        none invites it to write about a distinction that is not in the table.
        """
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)
        payload = " ".join(
            m.get("content", "") for req in requests for m in req.get("messages", [])
        )
        assert "inferred" not in payload
        assert "Rough sectors" not in payload

    async def test_a_matched_course_names_inferred_surfaces_as_guesses(
        self, client, auth_headers, session, seeded_athlete
    ):
        """Confidence has to reach the *prose*, not stop at the API.

        OSM coverage is uneven, and a plan that talked about gravel at km 47
        with the same certainty either way would be the quiet dishonesty this
        project refuses elsewhere.
        """
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        await _match_course(session, course_id)

        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)
        payload = " ".join(
            m.get("content", "") for req in requests for m in req.get("messages", [])
        )
        assert "inferred" in payload, "the weaker claim must be labelled"
        assert "still air" in payload, "the wind caveat still applies"
        assert "dry pavement" not in payload, "the surface is known now"

    async def test_a_matched_course_names_its_rough_sectors(
        self, client, auth_headers, session, seeded_athlete
    ):
        """Including the short ones, which is the whole point of the list.

        A rider expecting smooth asphalt needs to be told about a short mud
        sector before they hit it, and a sector too short to have earned its
        own row in the segment table is exactly the one they would otherwise
        meet by surprise.
        """
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        await _match_course(session, course_id, rough=True)

        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)
        payload = " ".join(
            m.get("content", "") for req in requests for m in req.get("messages", [])
        )
        assert "Rough sectors" in payload
        assert "dirt" in payload

    async def test_the_prompt_names_a_power_target_and_its_splits(
        self, client, auth_headers, session, seeded_athlete
    ):
        # Issue #61: a target power the athlete cannot sustain for the whole
        # ride is flagged in the prompt *and* keeps its splits, so the coach
        # writes about a real plan rather than about a refusal.
        course_id = await _seed_course(
            client, auth_headers, session, seeded_athlete, target_power_w=300
        )
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )

        requests: list = []
        await _run_plan_bg(session, course_id, seeded_athlete.id, requests)
        payload = " ".join(
            m.get("content", "") for req in requests for m in req.get("messages", [])
        )
        assert "average of 300 W" in payload
        assert "above what anyone sustains" in payload
        assert "Predicted total time" in payload

    async def test_gated_instance_denies_without_entitlement(
        self, client, auth_headers, session, seeded_athlete, registry_session
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        # Gate on, and strip the athlete's BYOK config so the gate binds.
        seeded_athlete.app_settings = {}
        await session.commit()
        gate = await client.patch(
            "/api/admin/settings",
            json={"llm_requires_subscription": True},
            headers=auth_headers,
        )
        assert gate.status_code == 200

        resp = await client.post(
            f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "llm_subscription_required"

    async def test_a_stuck_pending_run_is_settled_on_read(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        course.plan_status = "pending"
        course.plan_updated_at = datetime.now(timezone.utc) - timedelta(minutes=16)
        await session.commit()

        resp = await client.get(f"/api/courses/{course_id}/plan", headers=auth_headers)
        assert resp.json()["status"] == "error"

    async def test_the_boot_sweep_settles_a_stranded_plan(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        course.plan_status = "pending"
        await session.commit()

        @asynccontextmanager
        async def _factory():
            yield session

        with patch(
            "backend.app.db.user_session.get_user_session_factory",
            return_value=lambda: _factory(),
        ):
            settled = await settle_stranded_user_runs(_TEST_USER_ID)

        assert settled == 1
        await session.refresh(course)
        assert course.plan_status == "error"

    async def test_reanalysis_discards_a_plan_that_is_still_streaming(
        self, client, auth_headers, session, seeded_athlete
    ):
        """The plan task runs on its own session and commits *after* the
        request that cleared the plan, so clearing the columns alone does not
        stop it — it would put prose describing the old segment table straight
        back, ending on `done` with splits keyed to distances that no longer
        exist. The run token is what makes the clear stick."""
        from backend.app.services import course_analysis
        from openkoutsi.course import RiderParams

        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)

        # A plan run is in flight, holding its token.
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        run_id = course.plan_run_id
        assert run_id, "the trigger must stamp a run token"

        # Re-analysis lands while that run is still going.
        resp = await client.post(
            f"/api/courses/{course_id}/reanalyze", json={}, headers=auth_headers
        )
        assert resp.status_code == 200
        await session.refresh(course)
        assert course.plan_run_id is None

        # Now the in-flight run finishes and writes its prose.
        await _run_plan_bg(session, course_id, seeded_athlete.id, run_id=run_id)

        await session.refresh(course)
        assert course.plan_status is None, "a superseded run resurrected its plan"
        assert course.plan is None
        assert course.plan_mood is None

    async def test_a_current_run_still_writes_its_plan(
        self, client, auth_headers, session, seeded_athlete
    ):
        """The guard must not cost the ordinary case its result."""
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        with patch(
            "backend.app.services.llm_course_plan.generate_course_plan_bg",
            new_callable=AsyncMock,
        ):
            await client.post(
                f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
            )
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()

        await _run_plan_bg(session, course_id, seeded_athlete.id, run_id=course.plan_run_id)

        await session.refresh(course)
        assert course.plan_status == "done"
        assert course.plan.startswith("Ride the opening kilometres")

    async def test_a_plan_for_a_broken_course_is_a_409(
        self, client, auth_headers, session, seeded_athlete
    ):
        course_id = await _seed_course(client, auth_headers, session, seeded_athlete)
        course = (
            await session.execute(select(Course).where(Course.id == course_id))
        ).scalar_one()
        course.status = "error"
        await session.commit()

        resp = await client.post(
            f"/api/courses/{course_id}/plan", json={}, headers=auth_headers
        )
        assert resp.status_code == 409
