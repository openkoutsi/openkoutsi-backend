"""Unit tests for the shared LLM client helpers.

Covers the two behaviours added to make third-party LLM failures diagnosable:

* ``temperature_param`` omits the ``temperature`` field unless a caller passes
  an explicit value, so thinking-enabled models (which reject any temperature
  other than 1) work by default.
* ``raise_for_llm_status`` surfaces the upstream response body in the raised
  error instead of discarding it like ``httpx.Response.raise_for_status``.
"""
import httpx
import pytest

from backend.app.services.llm_client import raise_for_llm_status, temperature_param


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
