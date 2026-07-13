"""Integration tests for per-goal AI guidance (issue #17).

POST /api/goals/{id}/guidance triggers a background task that streams coaching
prose from an OpenAI-compatible endpoint, parses the leading REALISM verdict,
and persists both on the goal; GET returns the current state. These tests mock
the streamed SSE response (no network) and cover the happy path, the gated-denied
path, and the trigger/pending contract.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from backend.app.models.user_orm import Athlete, Goal

_TEST_USER_ID = "test-user-00000000"

_GUIDANCE_SSE = [
    "REALISM: ambitious",
    "\n\n",
    "Your FTP goal is a real stretch on this timeline, ",
    "but the trend is going the right way.\n\n",
    "Keep two threshold sessions a week and one long ride.",
]


def _make_streaming_lines(chunks):
    """Return an async generator of OpenAI-style SSE lines from text chunks."""
    async def _gen():
        for chunk in chunks:
            escaped = chunk.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            yield f'data: {{"choices":[{{"delta":{{"content":"{escaped}"}}}}]}}'
        yield "data: [DONE]"
    return _gen()


def _mock_httpx_stream(chunks):
    """Patch object for httpx.AsyncClient that streams *chunks* as SSE."""
    mock_resp = AsyncMock()
    mock_resp.aiter_lines = MagicMock(return_value=_make_streaming_lines(chunks))
    mock_resp.is_error = False

    @asynccontextmanager
    async def _mock_stream(*args, **kwargs):
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


async def _seed_goal(session, **kwargs) -> Goal:
    athlete = (await session.execute(select(Athlete))).scalar_one()
    # BYOK config so config resolution needs no instance and usage isn't recorded.
    athlete.app_settings = {"llm_base_url": "http://localhost:11434/v1", "llm_model": "x"}
    athlete.ftp = 250
    goal = Goal(
        id=str(uuid.uuid4()),
        athlete_id=athlete.id,
        title=kwargs.get("title", "Reach FTP 300 W by December"),
        target_value=kwargs.get("target_value", 300.0),
        current_value=kwargs.get("current_value", 250.0),
        status="active",
    )
    session.add(goal)
    await session.commit()
    return goal


async def _set_gate(client, auth_headers, on: bool):
    resp = await client.patch(
        "/api/admin/settings",
        json={"llm_requires_subscription": on},
        headers=auth_headers,
    )
    assert resp.status_code == 200


class TestGoalGuidance:
    async def test_no_guidance_yet(self, client, auth_headers, session):
        goal = await _seed_goal(session)
        resp = await client.get(f"/api/goals/{goal.id}/guidance", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] is None
        assert data["verdict"] is None
        assert data["guidance"] is None

    async def test_trigger_sets_pending(self, client, auth_headers, session):
        goal = await _seed_goal(session)
        with patch(
            "backend.app.services.llm_goal_guidance.generate_goal_guidance_bg",
            new_callable=AsyncMock,
        ):
            resp = await client.post(
                f"/api/goals/{goal.id}/guidance", json={}, headers=auth_headers
            )
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"

        status = await client.get(f"/api/goals/{goal.id}/guidance", headers=auth_headers)
        assert status.json()["status"] == "pending"

    async def test_trigger_while_pending_returns_pending(self, client, auth_headers, session):
        goal = await _seed_goal(session)
        with patch(
            "backend.app.services.llm_goal_guidance.generate_goal_guidance_bg",
            new_callable=AsyncMock,
        ):
            await client.post(f"/api/goals/{goal.id}/guidance", json={}, headers=auth_headers)
            resp = await client.post(
                f"/api/goals/{goal.id}/guidance", json={}, headers=auth_headers
            )
        assert resp.status_code == 202
        assert resp.json()["status"] == "pending"

    async def test_full_run_parses_verdict_and_prose(self, client, auth_headers, session):
        goal = await _seed_goal(session)

        # POST → pending (background task suppressed so we can drive it explicitly).
        with patch(
            "backend.app.services.llm_goal_guidance.generate_goal_guidance_bg",
            new_callable=AsyncMock,
        ):
            resp = await client.post(
                f"/api/goals/{goal.id}/guidance", json={}, headers=auth_headers
            )
        assert resp.status_code == 202

        # Drive the real background task against the same in-memory session,
        # with the LLM streamed response mocked.
        from backend.app.services import llm_goal_guidance as svc

        @asynccontextmanager
        async def _factory():
            yield session

        with (
            patch.object(svc, "get_user_session_factory", return_value=lambda: _factory()),
            patch.object(svc, "_RegistrySessionLocal", return_value=_mock_registry_session(None)),
            patch("httpx.AsyncClient", return_value=_mock_httpx_stream(_GUIDANCE_SSE)),
        ):
            await svc.generate_goal_guidance_bg(goal.athlete_id, goal.id, _TEST_USER_ID)

        resp = await client.get(f"/api/goals/{goal.id}/guidance", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["verdict"] == "ambitious"
        assert "REALISM" not in data["guidance"]
        assert "real stretch" in data["guidance"]
        assert data["updated_at"] is not None

    async def test_denied_on_gated_instance(self, client, auth_headers, session):
        goal = await _seed_goal(session)
        # BYOK would bypass the gate, so clear the athlete's own LLM config first.
        athlete = (await session.execute(select(Athlete))).scalar_one()
        athlete.app_settings = {}
        await session.commit()
        await _set_gate(client, auth_headers, True)

        resp = await client.post(
            f"/api/goals/{goal.id}/guidance", json={}, headers=auth_headers
        )
        assert resp.status_code == 403
        assert resp.json()["detail"]["code"] == "llm_subscription_required"

    async def test_trigger_unknown_goal_404(self, client, auth_headers, session):
        await _seed_goal(session)
        resp = await client.post(
            "/api/goals/does-not-exist/guidance", json={}, headers=auth_headers
        )
        assert resp.status_code == 404

    async def test_pending_timeout_recovers_to_error(self, client, auth_headers, session):
        goal = await _seed_goal(session)
        goal.guidance_status = "pending"
        goal.guidance_updated_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        await session.commit()

        resp = await client.get(f"/api/goals/{goal.id}/guidance", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"
