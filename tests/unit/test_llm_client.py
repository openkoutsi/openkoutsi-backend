"""Unit tests for the shared LLM client helpers.

Covers the two behaviours added to make third-party LLM failures diagnosable:

* ``temperature_param`` omits the ``temperature`` field unless a caller passes
  an explicit value, so thinking-enabled models (which reject any temperature
  other than 1) work by default.
* ``raise_for_llm_status`` surfaces the upstream response body in the raised
  error instead of discarding it like ``httpx.Response.raise_for_status``.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.app.services.llm_client import (
    call_llm,
    is_response_format_unsupported_error,
    raise_for_llm_status,
    temperature_param,
)


class TestTemperatureParam:
    def test_omitted_by_default(self):
        assert temperature_param() == {}

    def test_explicit_none_omitted(self):
        assert temperature_param(None) == {}

    def test_explicit_value_included(self):
        assert temperature_param(0.2) == {"temperature": 0.2}


def _response(status_code: int, body: str) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        text=body,
    )


class TestRaiseForLlmStatus:
    async def test_noop_on_success(self):
        # Should not raise for a 2xx response.
        await raise_for_llm_status(_response(200, "ok"), "https://api.example.com/v1/chat/completions")

    async def test_includes_body_on_error(self):
        body = '{"error": {"message": "temperature must be 1 for thinking models"}}'
        resp = _response(400, body)
        url = "https://api.example.com/v1/chat/completions"
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await raise_for_llm_status(resp, url)
        message = str(exc_info.value)
        assert "400" in message
        assert "temperature must be 1 for thinking models" in message
        assert url in message

    async def test_truncates_long_body(self):
        resp = _response(400, "Z" * 5000)
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await raise_for_llm_status(resp, "https://api.example.com/v1/chat/completions")
        message = str(exc_info.value)
        # Body is truncated to 1000 chars: a run of 1000 is present, 1001 is not.
        assert "Z" * 1000 in message
        assert "Z" * 1001 not in message


def _mock_httpx_client(resp: httpx.Response | MagicMock) -> AsyncMock:
    """An httpx.AsyncClient stand-in whose ``post`` returns ``resp``."""
    http = AsyncMock()
    http.post = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)
    return http


def _ok_response() -> MagicMock:
    resp = MagicMock()
    resp.is_error = False
    resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    return resp


class TestCallLlmResponseFormat:
    _RF = {"type": "json_schema", "json_schema": {"name": "x", "strict": True, "schema": {}}}

    async def _posted_payload(self, **call_kwargs) -> dict:
        http = _mock_httpx_client(_ok_response())
        with patch("httpx.AsyncClient", return_value=http):
            await call_llm(
                "hi", "http://localhost:11434/v1", "m", None,
                system_prompt="sys", **call_kwargs,
            )
        return http.post.call_args.kwargs["json"]

    async def test_response_format_included_when_passed(self):
        payload = await self._posted_payload(response_format=self._RF)
        assert payload["response_format"] == self._RF

    async def test_response_format_omitted_by_default(self):
        payload = await self._posted_payload()
        assert "response_format" not in payload

    async def test_response_format_wins_over_extra_body(self):
        # A core field must not be overridden by a preset's extra_body.
        payload = await self._posted_payload(
            response_format=self._RF,
            extra_body={"response_format": {"type": "text"}, "max_tokens": 8},
        )
        assert payload["response_format"] == self._RF
        assert payload["max_tokens"] == 8


class TestIsResponseFormatUnsupportedError:
    def _error(self, status: int, body: str) -> httpx.HTTPStatusError:
        resp = _response(status, body)
        return httpx.HTTPStatusError(
            f"LLM request failed with status {status}: {body}",
            request=resp.request, response=resp,
        )

    def test_matches_representative_400_body(self):
        body = '{"error": {"message": "response_format is not supported by this model"}}'
        assert is_response_format_unsupported_error(self._error(400, body)) is True

    def test_matches_json_schema_422(self):
        body = '{"error": "Unknown parameter: json_schema"}'
        assert is_response_format_unsupported_error(self._error(422, body)) is True

    def test_ignores_unrelated_400(self):
        body = '{"error": {"message": "temperature must be 1 for thinking models"}}'
        assert is_response_format_unsupported_error(self._error(400, body)) is False

    def test_ignores_non_client_error_status(self):
        body = '{"error": "response_format server exploded"}'
        assert is_response_format_unsupported_error(self._error(500, body)) is False

    def test_invalid_schema_body_not_swallowed(self):
        # OpenAI-style "our schema is broken" — must NOT be treated as an
        # unsupported param, so the error surfaces instead of silently degrading
        # structured outputs to prompt-only everywhere.
        body = '{"error": {"message": "Invalid schema for response_format \'training_plan\': bad"}}'
        assert is_response_format_unsupported_error(self._error(400, body)) is False

    def test_matches_body_not_exception_message(self):
        # A marker only in the URL/exception message (not the response body) must
        # not trigger a match — we classify on the body, not str(exc).
        resp = _response(400, "temperature must be 1")
        exc = httpx.HTTPStatusError(
            "LLM request to https://host/response_format/v1 failed: temperature must be 1",
            request=resp.request, response=resp,
        )
        assert is_response_format_unsupported_error(exc) is False

    def test_drops_spaced_marker_variants(self):
        # Only the API-style underscore forms count; a spaced "json schema" in an
        # unrelated 400 should not false-match.
        body = '{"error": "invalid json schema in tool definition"}'
        assert is_response_format_unsupported_error(self._error(400, body)) is False
