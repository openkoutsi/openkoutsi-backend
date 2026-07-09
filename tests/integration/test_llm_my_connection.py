"""Integration tests for POST /api/llm/test-my-connection (BYOK, any user).

Body values override the athlete's saved config so Test works before saving;
an omitted ``api_key`` falls back to the saved encrypted key. The upstream HTTP
call is mocked so no real LLM is contacted.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.core import config
from backend.app.models.user_orm import Athlete
from sqlalchemy import select

_TEST_USER_ID = "test-user-00000000"


def _mock_httpx_client(*, status_code=200, json_body=None, text="", raises=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_body or {})

    client = AsyncMock()
    if raises is not None:
        client.post = AsyncMock(side_effect=raises)
    else:
        client.post = AsyncMock(return_value=resp)

    @asynccontextmanager
    async def _cm(*args, **kwargs):
        yield client

    _cm.client = client
    return _cm


_OK_COMPLETION = {"choices": [{"message": {"role": "assistant", "content": "Hi there."}}]}


async def _set_app_settings(session, **settings):
    result = await session.execute(select(Athlete).where(Athlete.global_user_id == _TEST_USER_ID))
    athlete = result.scalar_one()
    athlete.app_settings = settings
    await session.commit()


class TestTestMyConnection:
    async def test_requires_auth(self, client):
        resp = await client.post("/api/llm/test-my-connection")
        assert resp.status_code == 401

    async def test_no_base_url_anywhere_is_400(self, client, auth_headers):
        resp = await client.post("/api/llm/test-my-connection", json={}, headers=auth_headers)
        assert resp.status_code == 400
        assert "base URL" in resp.json()["detail"]

    async def test_success_with_body_values(self, client, auth_headers):
        factory = _mock_httpx_client(json_body=_OK_COMPLETION)
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post(
                "/api/llm/test-my-connection",
                json={"base_url": "http://127.0.0.1:11434", "model": "my-model"},
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["model_configured"] == "my-model"
        assert data["response_text"] == "Hi there."
        # Body api_key omitted and none saved → no Authorization header sent.
        sent = factory.client.post.call_args
        assert "Authorization" not in sent.kwargs["headers"]

    async def test_uses_saved_config_when_body_omits_values(self, client, auth_headers, session):
        await _set_app_settings(
            session, llm_base_url="http://127.0.0.1:11434", llm_model="saved-model"
        )
        factory = _mock_httpx_client(json_body=_OK_COMPLETION)
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post("/api/llm/test-my-connection", json={}, headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["base_url"] == "http://127.0.0.1:11434"
        assert data["model_configured"] == "saved-model"

    async def test_uses_saved_key_when_api_key_omitted(self, client, auth_headers, session):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        with patch.object(config.settings, "encryption_key", key):
            from backend.app.core.file_encryption import encrypt_secret
            enc = encrypt_secret("sk-saved", _TEST_USER_ID)
            await _set_app_settings(
                session, llm_base_url="http://127.0.0.1:11434",
                llm_model="m", llm_api_key_enc=enc,
            )
            factory = _mock_httpx_client(json_body=_OK_COMPLETION)
            with patch("httpx.AsyncClient", return_value=factory()):
                resp = await client.post(
                    "/api/llm/test-my-connection", json={}, headers=auth_headers
                )
        assert resp.json()["ok"] is True
        sent = factory.client.post.call_args
        assert sent.kwargs["headers"]["Authorization"] == "Bearer sk-saved"

    async def test_body_api_key_overrides_saved(self, client, auth_headers, session):
        factory = _mock_httpx_client(json_body=_OK_COMPLETION)
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post(
                "/api/llm/test-my-connection",
                json={"base_url": "http://127.0.0.1:11434", "model": "m", "api_key": "sk-typed"},
                headers=auth_headers,
            )
        assert resp.json()["ok"] is True
        sent = factory.client.post.call_args
        assert sent.kwargs["headers"]["Authorization"] == "Bearer sk-typed"

    async def test_auth_failure_surfaced(self, client, auth_headers):
        factory = _mock_httpx_client(status_code=401, text="nope")
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post(
                "/api/llm/test-my-connection",
                json={"base_url": "http://127.0.0.1:11434", "model": "m", "api_key": "bad"},
                headers=auth_headers,
            )
        data = resp.json()
        assert data["ok"] is False
        assert data["http_status"] == 401
        assert "API key" in data["error"]

    async def test_timeout_surfaced(self, client, auth_headers):
        factory = _mock_httpx_client(raises=httpx.ConnectTimeout("slow"))
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post(
                "/api/llm/test-my-connection",
                json={"base_url": "http://127.0.0.1:11434", "model": "m"},
                headers=auth_headers,
            )
        data = resp.json()
        assert data["ok"] is False
        assert "timed out" in data["error"].lower()

    async def test_allowlist_rejection_is_403(self, client, auth_headers):
        with patch.object(config.settings, "llm_allowed_servers", "http://allowed/v1"):
            resp = await client.post(
                "/api/llm/test-my-connection",
                json={"base_url": "http://blocked/v1", "model": "m"},
                headers=auth_headers,
            )
        assert resp.status_code == 403
        assert "allowed list" in resp.json()["detail"]

    async def test_allowlisted_url_is_accepted(self, client, auth_headers):
        factory = _mock_httpx_client(json_body=_OK_COMPLETION)
        with patch.object(config.settings, "llm_allowed_servers", "http://127.0.0.1:11434"):
            with patch("httpx.AsyncClient", return_value=factory()):
                resp = await client.post(
                    "/api/llm/test-my-connection",
                    json={"base_url": "http://127.0.0.1:11434", "model": "m"},
                    headers=auth_headers,
                )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
