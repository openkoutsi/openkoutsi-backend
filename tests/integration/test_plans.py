"""
Integration tests for /api/plans endpoints.
"""
import json
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


_START = date(2025, 6, 2)  # A Monday


class TestCreatePlan:
    async def test_creates_rule_based_plan_with_correct_structure(self, client, auth_headers):
        resp = await client.post(
            "/api/plans",
            json={"name": "Base Build", "start_date": str(_START), "weeks": 8},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Base Build"
        assert data["generation_method"] == "rule_based"
        assert data["status"] == "active"
        assert len(data["workouts"]) == 56  # 8 weeks × 7 days

    def _expected_end_date(self, start: date, weeks: int) -> str:
        return str(start + timedelta(weeks=weeks) - timedelta(days=1))

    async def test_end_date_calculated_correctly(self, client, auth_headers):
        resp = await client.post(
            "/api/plans",
            json={"name": "Plan", "start_date": str(_START), "weeks": 8},
            headers=auth_headers,
        )
        assert resp.json()["end_date"] == self._expected_end_date(_START, 8)

    async def test_creating_second_plan_archives_first(self, client, auth_headers):
        resp1 = await client.post(
            "/api/plans",
            json={"name": "Plan 1", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan1_id = resp1.json()["id"]

        await client.post(
            "/api/plans",
            json={"name": "Plan 2", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )

        # First plan should now be archived
        resp = await client.get(f"/api/plans/{plan1_id}", headers=auth_headers)
        assert resp.json()["status"] == "archived"

    async def test_non_overlapping_plan_leaves_first_active(self, client, auth_headers):
        resp1 = await client.post(
            "/api/plans",
            json={"name": "Plan 1", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan1_id = resp1.json()["id"]

        # Start the second plan the day after the first one ends → no overlap.
        second_start = _START + timedelta(weeks=4)
        await client.post(
            "/api/plans",
            json={"name": "Plan 2", "start_date": str(second_start), "weeks": 4},
            headers=auth_headers,
        )

        # First plan should still be active because the ranges don't overlap.
        resp = await client.get(f"/api/plans/{plan1_id}", headers=auth_headers)
        assert resp.json()["status"] == "active"

    async def test_overlapping_plan_archives_first(self, client, auth_headers):
        resp1 = await client.post(
            "/api/plans",
            json={"name": "Plan 1", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan1_id = resp1.json()["id"]

        # Overlap by starting within the first plan's window.
        second_start = _START + timedelta(weeks=2)
        await client.post(
            "/api/plans",
            json={"name": "Plan 2", "start_date": str(second_start), "weeks": 4},
            headers=auth_headers,
        )

        resp = await client.get(f"/api/plans/{plan1_id}", headers=auth_headers)
        assert resp.json()["status"] == "archived"

    async def test_llm_without_configured_url_returns_400(self, client, auth_headers):
        resp = await client.post(
            "/api/plans",
            json={
                "name": "LLM Plan",
                "start_date": str(_START),
                "weeks": 4,
                "use_llm": True,
                "config": {
                    "days_per_week": 3,
                    "day_configs": [{"day_of_week": 2, "workout_type": "threshold"}],
                    "periodization": "base_building",
                    "intensity_preference": "moderate",
                },
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post(
            "/api/plans",
            json={"name": "X", "start_date": str(_START)},
        )
        assert resp.status_code == 401


class TestGetPlan:
    async def test_returns_plan_with_workouts(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "My Plan", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]

        resp = await client.get(f"/api/plans/{plan_id}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == plan_id
        assert len(data["workouts"]) == 28

    async def test_nonexistent_plan_returns_404(self, client, auth_headers):
        resp = await client.get("/api/plans/no-such-id", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.get("/api/plans/some-id")
        assert resp.status_code == 401


class TestUpdatePlan:
    async def test_update_plan_name(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "Old Name", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]

        resp = await client.put(
            f"/api/plans/{plan_id}",
            json={"name": "New Name"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"

    async def test_archive_plan(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "Active Plan", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]

        resp = await client.put(
            f"/api/plans/{plan_id}",
            json={"status": "archived"},
            headers=auth_headers,
        )
        assert resp.json()["status"] == "archived"

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.put("/api/plans/some-id", json={"name": "X"})
        assert resp.status_code == 401


class TestUnarchivePlan:
    async def test_unarchive_reactivates_plan(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "Plan", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]
        await client.put(
            f"/api/plans/{plan_id}",
            json={"status": "archived"},
            headers=auth_headers,
        )

        resp = await client.post(f"/api/plans/{plan_id}/unarchive", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    async def test_unarchive_archives_overlapping_active_plan(self, client, auth_headers):
        # Plan 1 is created, then archived by an overlapping Plan 2.
        resp1 = await client.post(
            "/api/plans",
            json={"name": "Plan 1", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan1_id = resp1.json()["id"]
        resp2 = await client.post(
            "/api/plans",
            json={"name": "Plan 2", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan2_id = resp2.json()["id"]

        # Unarchiving Plan 1 should archive the overlapping active Plan 2.
        resp = await client.post(f"/api/plans/{plan1_id}/unarchive", headers=auth_headers)
        assert resp.json()["status"] == "active"

        resp = await client.get(f"/api/plans/{plan2_id}", headers=auth_headers)
        assert resp.json()["status"] == "archived"

    async def test_unarchive_keeps_non_overlapping_active_plan(self, client, auth_headers):
        resp1 = await client.post(
            "/api/plans",
            json={"name": "Plan 1", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan1_id = resp1.json()["id"]
        await client.put(
            f"/api/plans/{plan1_id}",
            json={"status": "archived"},
            headers=auth_headers,
        )

        # A non-overlapping active plan later in the calendar.
        second_start = _START + timedelta(weeks=4)
        resp2 = await client.post(
            "/api/plans",
            json={"name": "Plan 2", "start_date": str(second_start), "weeks": 4},
            headers=auth_headers,
        )
        plan2_id = resp2.json()["id"]

        resp = await client.post(f"/api/plans/{plan1_id}/unarchive", headers=auth_headers)
        assert resp.json()["status"] == "active"

        # Plan 2 does not overlap Plan 1, so it stays active.
        resp = await client.get(f"/api/plans/{plan2_id}", headers=auth_headers)
        assert resp.json()["status"] == "active"

    async def test_unarchive_nonexistent_returns_404(self, client, auth_headers):
        resp = await client.post("/api/plans/nope/unarchive", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/plans/some-id/unarchive")
        assert resp.status_code == 401


class TestDeletePlan:
    async def test_delete_plan_returns_204(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "Deletable", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]
        resp = await client.delete(f"/api/plans/{plan_id}", headers=auth_headers)
        assert resp.status_code == 204

    async def test_deleted_plan_returns_404(self, client, auth_headers):
        create_resp = await client.post(
            "/api/plans",
            json={"name": "Gone", "start_date": str(_START), "weeks": 4},
            headers=auth_headers,
        )
        plan_id = create_resp.json()["id"]
        await client.delete(f"/api/plans/{plan_id}", headers=auth_headers)
        resp = await client.get(f"/api/plans/{plan_id}", headers=auth_headers)
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.delete("/api/plans/some-id")
        assert resp.status_code == 401


# ── LLM plan generation ────────────────────────────────────────────────────────

def _make_llm_plan_json(num_weeks=4) -> str:
    """Build a minimal valid LLM response for num_weeks weeks."""
    weeks = []
    for w in range(1, num_weeks + 1):
        workouts = []
        for day in range(1, 8):
            if day in (2, 4, 6):
                workouts.append({"day_of_week": day, "workout_type": "endurance",
                                  "description": "Easy ride", "duration_min": 60, "target_load": 50})
            else:
                workouts.append({"day_of_week": day, "workout_type": "rest",
                                  "description": None, "duration_min": None, "target_load": None})
        weeks.append({"week_number": w, "workouts": workouts})
    return json.dumps({"weeks": weeks})


_LLM_REQUEST_BODY = {
    "name": "LLM Plan",
    "start_date": str(_START),
    "weeks": 4,
    "use_llm": True,
    "config": {
        "days_per_week": 3,
        "day_configs": [
            {"day_of_week": 2, "workout_type": "endurance"},
            {"day_of_week": 4, "workout_type": "threshold"},
            {"day_of_week": 6, "workout_type": "long"},
        ],
        "periodization": "base_building",
        "intensity_preference": "moderate",
    },
}


class TestLlmPlanGeneration:
    async def _mock_llm_call(self, raw_json: str):
        mock_resp = MagicMock()
        mock_resp.is_error = False
        mock_resp.json.return_value = {"choices": [{"message": {"content": raw_json}}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        return mock_http

    async def test_llm_plan_created_when_url_configured(self, client, auth_headers, session):
        from sqlalchemy import select as sa_select
        from backend.app.models.user_orm import Athlete

        # Set LLM URL on athlete
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )

        mock_http = await self._mock_llm_call(_make_llm_plan_json(4))

        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        data = resp.json()
        assert data["generation_method"] == "llm"
        assert len(data["workouts"]) == 28  # 4 weeks × 7 days

    async def test_llm_plan_retries_on_parse_failure(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )

        # First call returns garbage, second returns valid JSON
        bad_resp = MagicMock()
        bad_resp.is_error = False
        bad_resp.json.return_value = {"choices": [{"message": {"content": "not json at all"}}]}

        good_resp = MagicMock()
        good_resp.is_error = False
        good_resp.json.return_value = {"choices": [{"message": {"content": _make_llm_plan_json(4)}}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[bad_resp, good_resp])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        assert resp.json()["generation_method"] == "llm"

    async def test_llm_plan_fails_gracefully_on_double_parse_error(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )

        bad_resp = MagicMock()
        bad_resp.is_error = False
        bad_resp.json.return_value = {"choices": [{"message": {"content": "still not json"}}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=bad_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        # API returns 4xx/5xx when both LLM attempts produce unparseable JSON
        assert resp.status_code >= 400

    async def test_llm_plan_sends_schema_by_default(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )
        mock_http = await self._mock_llm_call(_make_llm_plan_json(4))
        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        rf = mock_http.post.call_args.kwargs["json"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "training_plan"

    async def test_llm_plan_opts_out_via_preset(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {
                "llm_model": "local",
                "llm_models": [{"name": "local", "base_url": "http://localhost:11434/v1",
                                "model": "llama3.2", "structured_outputs": False}],
            }},
            headers=auth_headers,
        )
        mock_http = await self._mock_llm_call(_make_llm_plan_json(4))
        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        assert "response_format" not in mock_http.post.call_args.kwargs["json"]

    async def test_llm_plan_falls_back_when_schema_unsupported(self, client, auth_headers):
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )
        err_body = '{"error": {"message": "response_format is not supported"}}'
        err_resp = MagicMock()
        err_resp.is_error = True
        err_resp.status_code = 400
        err_resp.request = MagicMock()
        err_resp.text = err_body
        err_resp.aread = AsyncMock(return_value=err_body.encode())
        good_resp = MagicMock()
        good_resp.is_error = False
        good_resp.json.return_value = {"choices": [{"message": {"content": _make_llm_plan_json(4)}}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[err_resp, good_resp])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        assert mock_http.post.await_count == 2
        first, second = mock_http.post.await_args_list
        assert "response_format" in first.kwargs["json"]
        assert "response_format" not in second.kwargs["json"]

    async def test_correction_retry_skips_already_rejected_schema(self, client, auth_headers):
        # Provider rejects the schema (call 1) → fallback (call 2) yields unparseable
        # output → correction retry (call 3) must NOT re-send the rejected schema.
        await client.patch(
            "/api/athlete",
            json={"app_settings": {"llm_base_url": "http://localhost:11434/v1",
                                   "llm_model": "llama3.2"}},
            headers=auth_headers,
        )
        err_body = '{"error": {"message": "response_format is not supported"}}'
        err_resp = MagicMock()
        err_resp.is_error = True
        err_resp.status_code = 400
        err_resp.request = MagicMock()
        err_resp.text = err_body
        err_resp.aread = AsyncMock(return_value=err_body.encode())

        bad_resp = MagicMock()
        bad_resp.is_error = False
        bad_resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
        good_resp = MagicMock()
        good_resp.is_error = False
        good_resp.json.return_value = {"choices": [{"message": {"content": _make_llm_plan_json(4)}}]}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=[err_resp, bad_resp, good_resp])
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http):
            resp = await client.post("/api/plans", json=_LLM_REQUEST_BODY, headers=auth_headers)

        assert resp.status_code == 201
        assert mock_http.post.await_count == 3
        c1, c2, c3 = mock_http.post.await_args_list
        assert "response_format" in c1.kwargs["json"]      # first tries the schema
        assert "response_format" not in c2.kwargs["json"]  # fallback drops it
        assert "response_format" not in c3.kwargs["json"]  # retry doesn't re-send it

    async def test_build_user_prompt_includes_ftp(self, client, auth_headers):
        from backend.app.services.llm_plan_generator import _build_user_prompt
        from backend.app.schemas.plans import PlanConfig, DayConfig

        config = PlanConfig(
            days_per_week=3,
            day_configs=[DayConfig(day_of_week=2, workout_type="endurance")],
            periodization="base_building",
            intensity_preference="moderate",
        )
        prompt = _build_user_prompt(config, "Gran Fondo 2025", 8, 280, 45.0)
        assert "280" in prompt  # FTP
        assert "45.0" in prompt  # Fitness
        assert "Gran Fondo" in prompt

    async def test_extract_json_strips_markdown_fences(self):
        from backend.app.services.llm_plan_generator import _extract_json
        raw = '```json\n{"foo": "bar"}\n```'
        assert _extract_json(raw) == '{"foo": "bar"}'

    async def test_parse_response_validates_week_count(self):
        from backend.app.services.llm_plan_generator import _parse_response
        import pytest
        valid = _make_llm_plan_json(4)
        with pytest.raises(ValueError, match="Expected 6 weeks"):
            _parse_response(valid, 6)


class TestSkipWorkout:
    async def _create_plan_and_get_workout(self, client, auth_headers):
        resp = await client.post(
            "/api/plans",
            json={"name": "Skip Test Plan", "start_date": str(_START), "weeks": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        plan = resp.json()
        workout = next(w for w in plan["workouts"] if w["workout_type"] != "rest")
        return plan["id"], workout["id"]

    async def test_skip_sets_reason(self, client, auth_headers):
        plan_id, workout_id = await self._create_plan_and_get_workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            json={"reason": "illness"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skip_reason"] == "illness"

    async def test_skip_reason_persists_in_plan_response(self, client, auth_headers):
        plan_id, workout_id = await self._create_plan_and_get_workout(client, auth_headers)
        await client.put(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            json={"reason": "Travel"},
            headers=auth_headers,
        )
        resp = await client.get(f"/api/plans/{plan_id}", headers=auth_headers)
        assert resp.status_code == 200
        workout = next(w for w in resp.json()["workouts"] if w["id"] == workout_id)
        assert workout["skip_reason"] == "Travel"

    async def test_clear_skip_removes_reason(self, client, auth_headers):
        plan_id, workout_id = await self._create_plan_and_get_workout(client, auth_headers)
        await client.put(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            json={"reason": "busy"},
            headers=auth_headers,
        )
        resp = await client.delete(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            headers=auth_headers,
        )
        assert resp.status_code == 204

        plan_resp = await client.get(f"/api/plans/{plan_id}", headers=auth_headers)
        workout = next(w for w in plan_resp.json()["workouts"] if w["id"] == workout_id)
        assert workout["skip_reason"] is None

    async def test_skip_unknown_plan_returns_404(self, client, auth_headers):
        resp = await client.put(
            "/api/plans/nonexistent/workouts/nonexistent/skip",
            json={"reason": "illness"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_skip_unknown_workout_returns_404(self, client, auth_headers):
        plan_id, _ = await self._create_plan_and_get_workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/nonexistent/skip",
            json={"reason": "illness"},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_skip_already_completed_workout_returns_409(self, client, auth_headers, session):
        from sqlalchemy import select
        from backend.app.models.user_orm import PlannedWorkout, Activity, Athlete

        plan_id, workout_id = await self._create_plan_and_get_workout(client, auth_headers)

        # Insert a dummy activity and mark the workout as completed directly in the DB
        athlete_result = await session.execute(select(Athlete))
        athlete = athlete_result.scalar_one()
        import uuid
        activity = Activity(
            id=str(uuid.uuid4()),
            athlete_id=athlete.id,
            name="dummy",
            sport_type="Ride",
            status="ok",
        )
        session.add(activity)
        await session.flush()

        workout_result = await session.execute(select(PlannedWorkout).where(PlannedWorkout.id == workout_id))
        workout = workout_result.scalar_one()
        workout.completed_activity_id = activity.id
        await session.commit()

        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            json={"reason": "illness"},
            headers=auth_headers,
        )
        assert resp.status_code == 409

    async def test_unauthenticated_returns_401(self, client, auth_headers):
        plan_id, workout_id = await self._create_plan_and_get_workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout_id}/skip",
            json={"reason": "illness"},
        )
        assert resp.status_code == 401


async def _create_plan(client, auth_headers, *, name="Edit Plan", weeks=4):
    resp = await client.post(
        "/api/plans",
        json={"name": name, "start_date": str(_START), "weeks": weeks, "goal": "Original goal"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.json()


async def _mark_completed(session, workout_id: str):
    """Attach a dummy activity to a planned workout to mark it completed."""
    import uuid
    from sqlalchemy import select
    from backend.app.models.user_orm import PlannedWorkout, Activity, Athlete

    athlete = (await session.execute(select(Athlete))).scalar_one()
    activity = Activity(
        id=str(uuid.uuid4()), athlete_id=athlete.id, name="dummy",
        sport_type="Ride", status="ok",
    )
    session.add(activity)
    await session.flush()
    workout = (await session.execute(
        select(PlannedWorkout).where(PlannedWorkout.id == workout_id)
    )).scalar_one()
    workout.completed_activity_id = activity.id
    await session.commit()
    return activity.id


class TestUpdatePlanMetadata:
    async def test_update_goal(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan['id']}",
            json={"goal": "New goal"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["goal"] == "New goal"

    async def test_update_weeks_recomputes_end_date(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=4)
        resp = await client.put(
            f"/api/plans/{plan['id']}",
            json={"weeks": 6},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["weeks"] == 6
        assert data["end_date"] == str(_START + timedelta(weeks=6) - timedelta(days=1))

    async def test_update_start_date_recomputes_end_date(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=4)
        new_start = _START + timedelta(days=7)
        resp = await client.put(
            f"/api/plans/{plan['id']}",
            json={"start_date": str(new_start)},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_date"] == str(new_start)
        assert data["end_date"] == str(new_start + timedelta(weeks=4) - timedelta(days=1))


class TestEditWorkout:
    async def _workout(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=1)
        workout = next(w for w in plan["workouts"] if w["workout_type"] != "rest")
        return plan["id"], workout

    async def test_edit_fields(self, client, auth_headers):
        plan_id, workout = await self._workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            json={"workout_type": "vo2max", "description": "5x3min", "duration_min": 75, "target_load": 95},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["workout_type"] == "vo2max"
        assert data["description"] == "5x3min"
        assert data["duration_min"] == 75
        assert data["target_load"] == 95

    async def test_partial_update_leaves_other_fields(self, client, auth_headers):
        plan_id, workout = await self._workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            json={"description": "just a note"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "just a note"
        assert data["workout_type"] == workout["workout_type"]

    async def test_edit_skipped_workout_succeeds(self, client, auth_headers):
        plan_id, workout = await self._workout(client, auth_headers)
        await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}/skip",
            json={"reason": "busy"},
            headers=auth_headers,
        )
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            json={"duration_min": 30},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["duration_min"] == 30

    async def test_edit_completed_workout_returns_409(self, client, auth_headers, session):
        plan_id, workout = await self._workout(client, auth_headers)
        await _mark_completed(session, workout["id"])
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            json={"duration_min": 30},
            headers=auth_headers,
        )
        assert resp.status_code == 409

    async def test_edit_unknown_workout_returns_404(self, client, auth_headers):
        plan_id, _ = await self._workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/nonexistent",
            json={"duration_min": 30},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client, auth_headers):
        plan_id, workout = await self._workout(client, auth_headers)
        resp = await client.put(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            json={"duration_min": 30},
        )
        assert resp.status_code == 401


class TestAddWorkout:
    async def test_add_workout(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=1)
        resp = await client.post(
            f"/api/plans/{plan['id']}/workouts",
            json={"week_number": 1, "day_of_week": 3, "workout_type": "tempo",
                  "description": "Tempo intervals", "duration_min": 50, "target_load": 60},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["week_number"] == 1
        assert data["day_of_week"] == 3
        assert data["workout_type"] == "tempo"

        plan_resp = await client.get(f"/api/plans/{plan['id']}", headers=auth_headers)
        assert any(w["id"] == data["id"] for w in plan_resp.json()["workouts"])

    async def test_add_to_unknown_plan_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/api/plans/nonexistent/workouts",
            json={"week_number": 1, "day_of_week": 3, "workout_type": "tempo"},
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestDeleteWorkout:
    async def _workout(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=1)
        workout = next(w for w in plan["workouts"] if w["workout_type"] != "rest")
        return plan["id"], workout

    async def test_delete_workout(self, client, auth_headers):
        plan_id, workout = await self._workout(client, auth_headers)
        resp = await client.delete(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 204
        plan_resp = await client.get(f"/api/plans/{plan_id}", headers=auth_headers)
        assert not any(w["id"] == workout["id"] for w in plan_resp.json()["workouts"])

    async def test_delete_completed_workout_returns_409(self, client, auth_headers, session):
        plan_id, workout = await self._workout(client, auth_headers)
        await _mark_completed(session, workout["id"])
        resp = await client.delete(
            f"/api/plans/{plan_id}/workouts/{workout['id']}",
            headers=auth_headers,
        )
        assert resp.status_code == 409

    async def test_delete_unknown_workout_returns_404(self, client, auth_headers):
        plan_id, _ = await self._workout(client, auth_headers)
        resp = await client.delete(
            f"/api/plans/{plan_id}/workouts/nonexistent",
            headers=auth_headers,
        )
        assert resp.status_code == 404


class TestRegeneratePlan:
    async def test_regenerate_rule_based_replaces_workouts(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=4)
        original_ids = {w["id"] for w in plan["workouts"]}
        resp = await client.post(
            f"/api/plans/{plan['id']}/regenerate",
            json={"config": {
                "days_per_week": 3,
                "day_configs": [
                    {"day_of_week": 1, "workout_type": "endurance"},
                    {"day_of_week": 3, "workout_type": "threshold"},
                    {"day_of_week": 5, "workout_type": "long"},
                ],
                "periodization": "base_building",
                "intensity_preference": "moderate",
            }},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        new_ids = {w["id"] for w in data["workouts"]}
        assert new_ids.isdisjoint(original_ids)  # all replaced
        assert data["generation_method"] == "rule_based"

    async def test_regenerate_preserves_completed_workout(self, client, auth_headers, session):
        plan = await _create_plan(client, auth_headers, weeks=2)
        workout = next(w for w in plan["workouts"] if w["workout_type"] != "rest")
        await _mark_completed(session, workout["id"])

        resp = await client.post(
            f"/api/plans/{plan['id']}/regenerate",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        ids = {w["id"] for w in resp.json()["workouts"]}
        assert workout["id"] in ids  # completed workout preserved

        # And no duplicate row for the preserved slot.
        slot = [w for w in resp.json()["workouts"]
                if w["week_number"] == workout["week_number"]
                and w["day_of_week"] == workout["day_of_week"]]
        assert len(slot) == 1

    async def test_regenerate_changes_weeks(self, client, auth_headers):
        plan = await _create_plan(client, auth_headers, weeks=4)
        resp = await client.post(
            f"/api/plans/{plan['id']}/regenerate",
            json={"weeks": 2},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["weeks"] == 2
        assert max(w["week_number"] for w in data["workouts"]) == 2

    async def test_regenerate_unknown_plan_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/api/plans/nonexistent/regenerate",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_unauthenticated_returns_401(self, client):
        resp = await client.post("/api/plans/some-id/regenerate", json={})
        assert resp.status_code == 401
