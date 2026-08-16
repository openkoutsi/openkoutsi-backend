"""Unit tests for the SSRF guard's DNS pinning (guarded_async_client).

F-02: `check_url_safe` resolved the hostname and returned the vetted IP so the
caller could connect to it directly — its own docstring said so — but every
caller discarded the return value and handed httpx the hostname, which resolved
it a second time. A short-TTL record can answer differently on that second
lookup: the check sees a public address, the connection goes to loopback.

These tests drive the transport with a stub underneath, so they assert what the
socket layer would have been asked for without opening one.
"""
import socket
from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from backend.app.core import config
from backend.app.core.ssrf import guarded_async_client


def _resolving_to(*ips: str):
    infos = [
        (socket.AF_INET6, socket.SOCK_STREAM, 0, '', (ip, 80, 0, 0))
        if ":" in ip
        else (socket.AF_INET, socket.SOCK_STREAM, 0, '', (ip, 80))
        for ip in ips
    ]
    return patch("socket.getaddrinfo", return_value=infos)


class _Recorder:
    """Stands in for the real network: records the request, answers 200.

    Patched over ``httpx.AsyncHTTPTransport.handle_async_request``, which is
    the ``super()`` hop at the end of the pinning transport — so the pinning
    logic under test runs for real and only the socket is stubbed. The pinning
    transport defines its own ``handle_async_request``, so patching the parent
    does not shadow the code being tested.
    """

    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)


def _client_with_recorder():
    """A guarded client that records outbound requests instead of sending them."""
    recorder = _Recorder()
    patcher = patch.object(
        httpx.AsyncHTTPTransport, "handle_async_request", new=recorder
    )
    return guarded_async_client(timeout=5.0), recorder, patcher


class TestConnectionPinning:
    async def test_connects_to_vetted_ip_not_hostname(self):
        client, recorder, patcher = _client_with_recorder()
        with patcher, _resolving_to("1.2.3.4"):
            async with client:
                await client.post("https://llm.example.com/v1/chat/completions", json={})

        sent = recorder.requests[0]
        assert sent.url.host == "1.2.3.4", "connection must target the vetted IP"

    async def test_host_header_keeps_the_original_name(self):
        client, recorder, patcher = _client_with_recorder()
        with patcher, _resolving_to("1.2.3.4"):
            async with client:
                await client.post("https://llm.example.com/v1/chat/completions", json={})

        sent = recorder.requests[0]
        assert sent.headers["Host"] == "llm.example.com"

    async def test_tls_verifies_against_the_original_name(self):
        """Pinning must not silently downgrade certificate verification."""
        client, recorder, patcher = _client_with_recorder()
        with patcher, _resolving_to("1.2.3.4"):
            async with client:
                await client.post("https://llm.example.com/v1/chat/completions", json={})

        sent = recorder.requests[0]
        assert sent.extensions["sni_hostname"] == "llm.example.com"

    async def test_ip_literal_passes_through_unrewritten(self):
        client, recorder, patcher = _client_with_recorder()
        with patcher, patch.object(config.settings, "llm_allow_private_networks", True):
            async with client:
                await client.post("http://127.0.0.1:11434/v1/chat/completions", json={})

        sent = recorder.requests[0]
        assert sent.url.host == "127.0.0.1"
        assert "sni_hostname" not in sent.extensions


class TestRebindingRefused:
    async def test_rebound_second_lookup_never_connects(self):
        """
        The guard's own resolution is the one that gets connected to, so a
        record that flips to loopback between check and connect cannot land the
        request on loopback — the flip is simply what the guard now sees, and
        it refuses.
        """
        client, recorder, patcher = _client_with_recorder()
        deny = patch.object(config.settings, "llm_allow_private_networks", False)
        with patcher, deny, _resolving_to("127.0.0.1"):
            async with client:
                with pytest.raises(HTTPException) as exc_info:
                    await client.post("https://rebind.example.com/v1/chat/completions", json={})

        assert exc_info.value.status_code == 403
        assert recorder.requests == [], "no connection may be attempted"

    async def test_guard_runs_on_every_request_not_just_the_first(self):
        client, recorder, patcher = _client_with_recorder()
        with patcher:
            async with client:
                with _resolving_to("1.2.3.4"):
                    await client.post("https://llm.example.com/v1/chat/completions", json={})
                # DNS flips under us; the next request must be re-checked.
                with _resolving_to("169.254.169.254"):
                    with pytest.raises(HTTPException):
                        await client.post("https://llm.example.com/v1/chat/completions", json={})

        assert len(recorder.requests) == 1


class TestRedirectsDisabled:
    def test_client_does_not_follow_redirects_by_default(self):
        """A 302 into a blocked range must not be chased automatically."""
        client = guarded_async_client(timeout=5.0)
        assert client.follow_redirects is False
