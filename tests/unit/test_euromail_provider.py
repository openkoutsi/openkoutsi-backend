"""Unit tests for the EuroMail email provider.

Mirrors the Lettermint provider tests (test_email_provider.py): factory
selection, the outbound send (mocking the ``euromail`` SDK), and the inbound
webhook verify/parse operations (HMAC-SHA256 over ``{timestamp}.{body}`` carried
in the ``X-Euromail-Signature`` header).
"""
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.config import Settings
from backend.app.services.email import (
    EmailConfigurationError,
    EmailError,
    EuromailProvider,
    InboundEmail,
    InboundParseError,
    OutboundMessage,
    build_email_provider,
)

# Enough to satisfy the SECRET_KEY validator when building Settings in-test.
_SECRET = "0" * 64


def _settings(**overrides) -> Settings:
    return Settings(secret_key=_SECRET, **overrides)


# ── Factory / provider selection ─────────────────────────────────────────────


def test_factory_returns_euromail_when_selected():
    provider = build_email_provider(_settings(email_provider="euromail"))
    assert isinstance(provider, EuromailProvider)
    assert provider.PROVIDER_NAME == "euromail"


# ── Outbound send ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_calls_sdk_with_html_and_text():
    provider = EuromailProvider.from_settings(
        _settings(
            email_provider="euromail",
            euromail_api_key="key",
            email_from="ops@koutsi.dev",
        )
    )
    msg = OutboundMessage(
        to="user@example.com", subject="Hi", html="<p>hi</p>", text="hi"
    )

    response = MagicMock()
    response.message_id = "msg-123"
    client = MagicMock()
    client.send_email = AsyncMock(return_value=response)
    client.close = AsyncMock()

    with patch(
        "backend.app.services.email.euromail.AsyncEuroMail", return_value=client
    ) as ctor:
        message_id = await provider.send(msg)

    ctor.assert_called_once_with(api_key="key")
    client.send_email.assert_awaited_once_with(
        from_address="ops@koutsi.dev",
        to="user@example.com",
        subject="Hi",
        html_body="<p>hi</p>",
        text_body="hi",
    )
    client.close.assert_awaited_once()
    assert message_id == "msg-123"


@pytest.mark.asyncio
async def test_send_wraps_transport_error_and_still_closes():
    """A pre-response transport failure (ConnectError) is a raw httpx.HTTPError,
    not a EuroMailError — send() must still surface it as EmailError."""
    import httpx

    provider = EuromailProvider.from_settings(
        _settings(euromail_api_key="key", email_from="ops@koutsi.dev")
    )
    client = MagicMock()
    client.send_email = AsyncMock(side_effect=httpx.ConnectError("no route to host"))
    client.close = AsyncMock()

    with patch(
        "backend.app.services.email.euromail.AsyncEuroMail", return_value=client
    ):
        with pytest.raises(EmailError, match="EuroMail send failed"):
            await provider.send(
                OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
            )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_wraps_sdk_error_and_still_closes():
    from euromail.errors import EuroMailError as SdkError

    provider = EuromailProvider.from_settings(
        _settings(euromail_api_key="key", email_from="ops@koutsi.dev")
    )
    client = MagicMock()
    client.send_email = AsyncMock(
        side_effect=SdkError(500, "server_error", "boom")
    )
    client.close = AsyncMock()

    with patch(
        "backend.app.services.email.euromail.AsyncEuroMail", return_value=client
    ):
        with pytest.raises(EmailError, match="EuroMail send failed"):
            await provider.send(
                OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
            )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_unconfigured_raises_configuration_error():
    provider = EuromailProvider.from_settings(_settings())  # no api key / from
    with pytest.raises(EmailConfigurationError):
        await provider.send(
            OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
        )


# ── Inbound verify / parse ───────────────────────────────────────────────────

_WEBHOOK_SECRET = "whsec_test_secret"


def _sign(body: bytes, secret: str = _WEBHOOK_SECRET, ts: int | None = None):
    """Build the EuroMail X-Euromail-Signature header for ``body``."""
    ts = ts if ts is not None else int(time.time())
    signed = f"{ts}.{body.decode()}".encode()
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return {"X-Euromail-Signature": f"t={ts},v1={digest}"}


def _inbound_body() -> bytes:
    return json.dumps(
        {
            "event": "email.inbound",
            "data": {
                "id": "in-42",
                "from_address": "sender@example.com",
                "to_addresses": ["lassi@koutsi.dev"],
                "subject": "Hello there",
                "text_body": "the body",
                "created_at": "2026-07-15T08:00:00Z",
            },
        }
    ).encode()


def test_verify_inbound_signature_accepts_valid():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body)) is True


def test_verify_inbound_signature_is_case_insensitive_on_header():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    headers = {k.lower(): v for k, v in _sign(body).items()}
    assert provider.verify_inbound_signature(body, headers) is True


def test_verify_inbound_signature_rejects_tampered_body():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    headers = _sign(body)
    assert provider.verify_inbound_signature(body + b" ", headers) is False


def test_verify_inbound_signature_rejects_wrong_secret():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body, secret="other")) is False


def test_verify_inbound_signature_false_when_unconfigured():
    provider = EuromailProvider.from_settings(_settings())
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body)) is False


def test_verify_inbound_signature_false_without_header():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    assert provider.verify_inbound_signature(_inbound_body(), {}) is False


def test_parse_inbound_returns_inbound_email():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    parsed = provider.parse_inbound(body, _sign(body))
    assert isinstance(parsed, InboundEmail)
    assert parsed.from_addr == "sender@example.com"
    assert parsed.to_addr == "lassi@koutsi.dev"
    assert parsed.subject == "Hello there"
    assert parsed.text == "the body"
    assert parsed.message_id == "in-42"
    assert parsed.received_at == datetime(2026, 7, 15, 8, 0, tzinfo=timezone.utc)


def test_parse_inbound_rejects_bad_signature():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    with pytest.raises(InboundParseError):
        provider.parse_inbound(body, _sign(body, secret="wrong"))


def test_parse_inbound_rejects_stale_timestamp():
    provider = EuromailProvider.from_settings(
        _settings(euromail_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    stale = _sign(body, ts=int(time.time()) - 3600)  # outside 5-min tolerance
    assert provider.verify_inbound_signature(body, stale) is False
    with pytest.raises(InboundParseError):
        provider.parse_inbound(body, stale)


def test_parse_inbound_unconfigured_raises():
    provider = EuromailProvider.from_settings(_settings())
    body = _inbound_body()
    with pytest.raises(EmailConfigurationError):
        provider.parse_inbound(body, _sign(body))
