"""
Integration tests for POST /api/plans/{plan_id}/generate-upcoming/workouts.

Synthesizes structured workouts (LLM mocked) for a plan's upcoming days and
caches them on the planned workouts, without uploading anywhere. Asserts
generation + caching, the date / window filter, reuse of already-generated
definitions, refresh, and error handling.
"""
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from backend.app.models.team_orm import (
    Athlete, PlannedWorkout, TrainingPlan, WahooWorkoutUpload, WorkoutDefinition,
)

_TEST_USER_ID = "test-user-00000000"

_DEF_STEPS = [
    {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 1200},
     "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90.0}}},
]


def _workout_json() -> str:
    return json.dumps({"steps": [
        {"kind": "step", "step_type": "warmup", "duration": {"type": "time", "seconds": 600},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 50}}},
        {"kind": "step", "step_type": "active", "duration": {"type": "time", "seconds": 1200},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 90}}},
        {"kind": "step", "step_type": "cooldown", "duration": {"type": "time", "seconds": 600},
         "target": {"metric": "power", "spec": {"type": "pct_ftp", "pct": 45}}},
    ]})


def _mock_llm(raw_json: str):
    """Patch httpx.AsyncClient so the LLM call returns *raw_json*."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": raw_json}}]}
    http = AsyncMock()
    http.post = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)
    return http


async def _configure_athlete(session, *, llm: bool = True):
    athlete = (await session.execute(select(Athlete))).scalar_one()
    athlete.ftp = 250
    if llm:
        athlete.app_settings = {
            "llm_base_url": "http://localhost:11434/v1",
            "llm_model": "llama3.2",
        }
    await session.commit()
    return athlete


def _utc_today() -> date:
    # Match the endpoint, which selects against datetime.now(timezone.utc).date().
    return datetime.now(timezone.utc).date()


def _this_monday() -> date:
    today = _utc_today()
    return today - timedelta(days=today.isoweekday() - 1)


async def _seed_plan(session, workouts: list[dict], start_date: date | None = None):
    """Create an active plan with the given planned-workout rows. Returns (plan, [pw…]).

    Defaults the plan start to this week's Monday so that a week-1 row on today's
    weekday maps (via week/day → date) onto today's calendar date.
    """
    athlete = (await session.execute(select(Athlete))).scalar_one()
    plan = TrainingPlan(
        athlete_id=athlete.id, name="Test Plan",
        start_date=start_date or _this_monday(), weeks=4, status="active",
    )
    session.add(plan)
    await session.flush()
    pws = []
    for w in workouts:
        pw = PlannedWorkout(plan_id=plan.id, **w)
        session.add(pw)
        pws.append(pw)
    await session.commit()
    return plan, pws


def _today_dow() -> int:
    return _utc_today().isoweekday()


async def _link_definition(session, athlete, pw) -> WorkoutDefinition:
    wd = WorkoutDefinition(
        id=str(uuid.uuid4()), athlete_id=athlete.id, name="Cached",
        sport_type="Ride", steps=_DEF_STEPS, estimated_duration_s=1200, estimated_tss=30.0,
    )
    session.add(wd)
    await session.flush()
    pw.workout_definition_id = wd.id
    await session.commit()
    return wd


class TestGenerateUpcomingWorkouts:
    async def test_generates_and_caches(self, client, auth_headers, session):
        await _configure_athlete(session, llm=True)
        # A single structured workout scheduled today (week 1, day = today's weekday).
        plan, (pw,) = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "threshold", "description": "2x20", "duration_min": 60, "target_tss": 80},
        ])

        with patch("httpx.AsyncClient", return_value=_mock_llm(_workout_json())):
            resp = await client.post(
                f"/api/plans/{plan.id}/generate-upcoming/workouts", json={}, headers=auth_headers
            )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["status"] == "generated"
        assert results[0]["workout_definition_id"] is not None

        # A definition was generated and cached on the planned workout.
        pw_row = (await session.execute(
            select(PlannedWorkout).where(PlannedWorkout.id == pw.id)
        )).scalar_one()
        assert pw_row.workout_definition_id is not None

        # The definition exists in the workout library; nothing was uploaded.
        defs = (await session.execute(select(WorkoutDefinition))).scalars().all()
        assert len(defs) == 1
        uploads = (await session.execute(select(WahooWorkoutUpload))).scalars().all()
        assert len(uploads) == 0

    async def test_rest_and_out_of_window_excluded(self, client, auth_headers, session):
        await _configure_athlete(session, llm=False)  # no LLM — nothing should need it
        plan, (rest_pw, future_pw) = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "rest", "duration_min": None, "target_tss": None},
            # Week 4 → ~3 weeks out, well beyond the today→+6 window.
            {"week_number": 4, "day_of_week": _today_dow(),
             "workout_type": "threshold", "duration_min": 60, "target_tss": 80},
        ])

        resp = await client.post(
            f"/api/plans/{plan.id}/generate-upcoming/workouts", json={}, headers=auth_headers
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        # Only the in-window rest day appears, marked skipped; the future workout is excluded.
        assert len(results) == 1
        assert results[0]["planned_workout_id"] == rest_pw.id
        assert results[0]["status"] == "skipped"
        assert results[0]["reason"] == "rest_or_unstructured"

    async def test_already_generated_is_skipped(self, client, auth_headers, session):
        # No LLM configured: a call to generate would 400. Reuse proves caching.
        athlete = await _configure_athlete(session, llm=False)
        plan, (pw,) = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "threshold", "duration_min": 60, "target_tss": 80},
        ])
        wd = await _link_definition(session, athlete, pw)

        resp = await client.post(
            f"/api/plans/{plan.id}/generate-upcoming/workouts", json={}, headers=auth_headers
        )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["status"] == "skipped"
        assert results[0]["reason"] == "already_generated"
        assert results[0]["workout_definition_id"] == wd.id

        # No new definition was created.
        defs = (await session.execute(select(WorkoutDefinition))).scalars().all()
        assert len(defs) == 1

    async def test_invalid_llm_output_fails(self, client, auth_headers, session):
        await _configure_athlete(session, llm=True)
        plan, (pw,) = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "threshold", "duration_min": 60, "target_tss": 80},
        ])

        with patch("httpx.AsyncClient", return_value=_mock_llm("not valid json")):
            resp = await client.post(
                f"/api/plans/{plan.id}/generate-upcoming/workouts", json={}, headers=auth_headers
            )

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert "generation_failed" in results[0]["reason"]
        # Nothing generated.
        defs = (await session.execute(select(WorkoutDefinition))).scalars().all()
        assert len(defs) == 0

    async def test_llm_not_configured_returns_400(self, client, auth_headers, session):
        await _configure_athlete(session, llm=False)
        plan, _ = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "threshold", "duration_min": 60, "target_tss": 80},
        ])
        resp = await client.post(
            f"/api/plans/{plan.id}/generate-upcoming/workouts", json={}, headers=auth_headers
        )
        assert resp.status_code == 400

    async def test_unknown_plan_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/api/plans/does-not-exist/generate-upcoming/workouts", json={}, headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_refresh_regenerates_definition(self, client, auth_headers, session):
        athlete = await _configure_athlete(session, llm=True)
        plan, (pw,) = await _seed_plan(session, [
            {"week_number": 1, "day_of_week": _today_dow(),
             "workout_type": "threshold", "duration_min": 60, "target_tss": 80},
        ])
        old_def = await _link_definition(session, athlete, pw)

        with patch("httpx.AsyncClient", return_value=_mock_llm(_workout_json())):
            resp = await client.post(
                f"/api/plans/{plan.id}/generate-upcoming/workouts",
                json={"refresh": True}, headers=auth_headers,
            )

        assert resp.status_code == 200
        assert resp.json()["results"][0]["status"] == "generated"
        pw_row = (await session.execute(
            select(PlannedWorkout).where(PlannedWorkout.id == pw.id)
        )).scalar_one()
        # A fresh definition replaced the cached one.
        assert pw_row.workout_definition_id != old_def.id
