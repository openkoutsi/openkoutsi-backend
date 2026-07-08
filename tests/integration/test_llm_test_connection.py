"""Integration tests for POST /api/llm/test-connection.

The endpoint sends a minimal "hello world" chat completion to the instance's
configured LLM and reports whether a usable response came back. The upstream
HTTP call is mocked so no real LLM is contacted.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


def _mock_httpx_client(*, status_code=200, json_body=None, text="", raises=None):
    """Return a patchable stand-in for ``httpx.AsyncClient`` used by the endpoint.

    The endpoint does ``async with httpx.AsyncClient(...) as client: await
    client.post(...)``, so we yield a client whose ``post`` returns a fake
    response (or raises the given exception).
    """
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

    # Expose the client so tests can inspect the outbound request.
    _cm.client = client
    return _cm


async def _configure_llm(client, auth_headers, *, base_url="http://127.0.0.1:11434", model="test-model"):
    resp = await client.patch(
        "/api/admin/settings",
        json={"llm_base_url": base_url, "llm_model": model},
        headers=auth_headers,
    )
    assert resp.status_code == 200


class TestLlmTestConnection:
    async def test_requires_auth(self, client):
        resp = await client.post("/api/llm/test-connection")
        assert resp.status_code == 401

    async def test_no_base_url_configured(self, client, auth_headers):
        resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        assert resp.status_code == 400
        assert "base URL" in resp.json()["detail"]

    async def test_no_model_configured(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={"llm_base_url": "https://llm.example.com", "llm_model": ""},
            headers=auth_headers,
        )
        resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "model" in data["error"]

    async def test_successful_hello_world_roundtrip(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        completion = {"choices": [{"message": {"role": "assistant", "content": "Hello! I'm here."}}]}
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(json_body=completion)()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["model_configured"] == "test-model"
        # The prompt actually sent is echoed back so the UI can show it.
        assert data["prompt_sent"] and "greeting" in data["prompt_sent"].lower()
        assert data["response_text"] == "Hello! I'm here."
        assert data["http_status"] == 200
        # The old model-listing fields are gone.
        assert "models_available" not in data
        assert "model_found" not in data

    async def test_empty_reply_is_not_ok(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        completion = {"choices": [{"message": {"role": "assistant", "content": "   "}}]}
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(json_body=completion)()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert "empty" in data["error"].lower()

    async def test_unexpected_response_shape(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(json_body={"unexpected": True})()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert "format" in data["error"].lower()

    async def test_auth_failure_surfaced(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(status_code=401, text="nope")()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert data["http_status"] == 401
        assert "API key" in data["error"]

    async def test_http_error_surfaced(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(status_code=500, text="boom")()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert data["http_status"] == 500
        assert "500" in data["error"]

    async def test_connection_refused_surfaced(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        refused = httpx.ConnectError("no route to host")
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(raises=refused)()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert "refused" in data["error"].lower()

    async def test_timeout_surfaced(self, client, auth_headers):
        await _configure_llm(client, auth_headers)
        timeout = httpx.ConnectTimeout("slow")
        with patch("httpx.AsyncClient", return_value=_mock_httpx_client(raises=timeout)()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        data = resp.json()
        assert data["ok"] is False
        assert "timed out" in data["error"].lower()

    async def test_applies_extra_headers_and_per_model_body(self, client, auth_headers):
        # Configure a model with its own body extras and a global ZDR header.
        resp = await client.patch(
            "/api/admin/settings",
            json={
                "llm_base_url": "http://127.0.0.1:11434",
                "llm_model": "thinker",
                "llm_models": [{"name": "thinker", "body": {"max_tokens": 32, "reasoning_effort": "high"}}],
                "llm_extra_headers": {"X-Wafer-ZDR": "true"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

        completion = {"choices": [{"message": {"content": "hi"}}]}
        factory = _mock_httpx_client(json_body=completion)
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post("/api/llm/test-connection", headers=auth_headers)
        assert resp.json()["ok"] is True

        sent = factory.client.post.call_args
        assert sent.kwargs["headers"]["X-Wafer-ZDR"] == "true"
        body = sent.kwargs["json"]
        assert body["model"] == "thinker"
        assert body["max_tokens"] == 32
        assert body["reasoning_effort"] == "high"

    async def test_model_override_picks_that_models_body(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={
                "llm_base_url": "http://127.0.0.1:11434",
                "llm_model": "plain",
                "llm_models": [
                    {"name": "plain", "body": {}},
                    {"name": "thinker", "body": {"reasoning_effort": "high"}},
                ],
            },
            headers=auth_headers,
        )
        completion = {"choices": [{"message": {"content": "hi"}}]}
        factory = _mock_httpx_client(json_body=completion)
        with patch("httpx.AsyncClient", return_value=factory()):
            resp = await client.post(
                "/api/llm/test-connection?model=thinker", headers=auth_headers
            )
        data = resp.json()
        assert data["ok"] is True
        assert data["model_configured"] == "thinker"
        body = factory.client.post.call_args.kwargs["json"]
        assert body["model"] == "thinker"
        assert body["reasoning_effort"] == "high"


class TestLlmModelsEndpoint:
    async def test_lists_configured_models_and_selection(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={
                "llm_base_url": "http://127.0.0.1:11434",
                "llm_model": "b",
                "llm_models": [
                    {"name": "a", "label": "Model A"},
                    {"name": "b"},
                ],
            },
            headers=auth_headers,
        )
        resp = await client.get("/api/llm/models", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        # Options carry a display label (falling back to the name).
        assert data["models"] == [
            {"name": "a", "label": "Model A"},
            {"name": "b", "label": "b"},
        ]
        assert data["selected"] == "b"

    async def test_single_legacy_model_is_offered(self, client, auth_headers):
        await client.patch(
            "/api/admin/settings",
            json={"llm_base_url": "http://127.0.0.1:11434", "llm_model": "solo"},
            headers=auth_headers,
        )
        resp = await client.get("/api/llm/models", headers=auth_headers)
        data = resp.json()
        assert data["models"] == [{"name": "solo", "label": "solo"}]
        assert data["selected"] == "solo"

    async def test_requires_auth(self, client):
        resp = await client.get("/api/llm/models")
        assert resp.status_code == 401


class TestInstanceSettingsPersistModelsHeaders:
    async def test_round_trips_models_and_headers(self, client, auth_headers):
        resp = await client.patch(
            "/api/admin/settings",
            json={
                "llm_models": [{"name": "gpt", "body": {"max_tokens": 10}}],
                "llm_extra_headers": {"X-ZDR": "true"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        got = await client.get("/api/admin/settings", headers=auth_headers)
        data = got.json()
        assert data["llm_models"] == [{
            "name": "gpt",
            "label": None,
            "base_url": None,
            "model": None,
            "api_key_set": False,
            "headers": {},
            "body": {"max_tokens": 10},
        }]
        assert data["llm_extra_headers"] == {"X-ZDR": "true"}

    async def test_full_preset_and_key_lifecycle(self, client, auth_headers):
        from cryptography.fernet import Fernet

        from backend.app.core import config

        with patch.object(config.settings, "encryption_key", Fernet.generate_key().decode()):
            # A preset carries its own base URL, model id, headers, body and key.
            resp = await client.patch(
                "/api/admin/settings",
                json={
                    "llm_models": [{
                        "name": "Anthropic (US)",
                        "base_url": "https://api.anthropic.com/v1",
                        "model": "claude-x",
                        "api_key": "sk-secret",
                        "headers": {"anthropic-version": "2023-06-01"},
                        "body": {"max_tokens": 1024},
                    }],
                },
                headers=auth_headers,
            )
            assert resp.status_code == 200
            preset = resp.json()["llm_models"][0]
            assert preset["base_url"] == "https://api.anthropic.com/v1"
            assert preset["model"] == "claude-x"
            assert preset["api_key_set"] is True
            # The encrypted key is never returned.
            assert "api_key" not in preset and "api_key_enc" not in preset

            # Editing another field without resending the key preserves it.
            resp = await client.patch(
                "/api/admin/settings",
                json={"llm_models": [{
                    "name": "Anthropic (US)",
                    "base_url": "https://api.anthropic.com/v1",
                    "model": "claude-y",
                }]},
                headers=auth_headers,
            )
            assert resp.json()["llm_models"][0]["api_key_set"] is True
            assert resp.json()["llm_models"][0]["model"] == "claude-y"

            # Clearing removes the key.
            resp = await client.patch(
                "/api/admin/settings",
                json={"llm_models": [{"name": "Anthropic (US)", "api_key_clear": True}]},
                headers=auth_headers,
            )
            assert resp.json()["llm_models"][0]["api_key_set"] is False
