"""Unit tests for the generic email module and the Lettermint provider.

Covers the swappable interface (dataclasses, factory selection), the outbound
body rendering, and the Lettermint-specific outbound send and inbound
verify/parse operations — the shared foundation both #15 (outbound) and #38
(inbound) build on.
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
    InboundEmail,
    InboundParseError,
    LettermintProvider,
    OutboundMessage,
    build_email_provider,
    render_transactional_email,
)

# Enough to satisfy the SECRET_KEY validator when building Settings in-test.
_SECRET = "0" * 64


def _settings(**overrides) -> Settings:
    return Settings(secret_key=_SECRET, **overrides)


# ── Rendering ────────────────────────────────────────────────────────────────


def test_render_produces_inline_styled_html_and_text():
    html, text = render_transactional_email(
        title="Verify your email",
        intro="Tap the button to finish.",
        body_paragraphs=["This link expires in 1 hour."],
        action_label="Verify email",
        action_url="https://example.com/verify?token=abc",
    )
    # HTML: inline CSS (no <style> block), the copy, and a working CTA link.
    assert "<style" not in html
    assert "style=" in html
    assert "Verify your email" in html
    assert "This link expires in 1 hour." in html
    assert "https://example.com/verify?token=abc" in html
    # Text alternative carries the same content and the labelled link.
    assert "Verify your email" in text
    assert "Verify email: https://example.com/verify?token=abc" in text
    assert "<" not in text  # genuinely plain text


def test_render_escapes_html_in_content():
    html, _text = render_transactional_email(
        title="Hi <script>alert(1)</script>",
        intro="ok",
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_without_action_omits_button():
    html, text = render_transactional_email(title="No CTA", intro="body only")
    assert "href=" not in html
    assert "body only" in text


# ── Factory / provider selection ─────────────────────────────────────────────


def test_factory_returns_lettermint_by_default():
    provider = build_email_provider(_settings())
    assert isinstance(provider, LettermintProvider)
    assert provider.PROVIDER_NAME == "lettermint"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown email_provider"):
        build_email_provider(_settings(email_provider="smtp"))


def test_dataclasses_are_frozen():
    msg = OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
    with pytest.raises(Exception):
        msg.subject = "x"  # type: ignore[misc]


# ── Outbound send ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_calls_sdk_with_html_and_text():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_api_key="key", email_from="ops@koutsi.dev")
    )
    msg = OutboundMessage(
        to="user@example.com", subject="Hi", html="<p>hi</p>", text="hi"
    )

    # Chainable builder mock: every setter returns the builder, send() is async.
    builder = MagicMock()
    for setter in ("from_", "to", "subject", "html", "text"):
        getattr(builder, setter).return_value = builder
    builder.send = AsyncMock(return_value={"message_id": "msg-123", "status": "queued"})
    client = MagicMock()
    client.email = builder
    client.close = AsyncMock()

    with patch(
        "backend.app.services.email.lettermint.AsyncLettermint", return_value=client
    ) as ctor:
        message_id = await provider.send(msg)

    ctor.assert_called_once_with(api_token="key")
    builder.from_.assert_called_once_with("ops@koutsi.dev")
    builder.to.assert_called_once_with("user@example.com")
    builder.subject.assert_called_once_with("Hi")
    builder.html.assert_called_once_with("<p>hi</p>")
    builder.text.assert_called_once_with("hi")
    builder.send.assert_awaited_once()
    client.close.assert_awaited_once()
    assert message_id == "msg-123"


@pytest.mark.asyncio
async def test_send_unconfigured_raises_configuration_error():
    provider = LettermintProvider.from_settings(_settings())  # no api key / from
    with pytest.raises(EmailConfigurationError):
        await provider.send(
            OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
        )


@pytest.mark.asyncio
async def test_send_wraps_sdk_error_and_still_closes():
    from lettermint.exceptions import LettermintError as SdkError

    provider = LettermintProvider.from_settings(
        _settings(lettermint_api_key="key", email_from="ops@koutsi.dev")
    )
    builder = MagicMock()
    for setter in ("from_", "to", "subject", "html", "text"):
        getattr(builder, setter).return_value = builder
    builder.send = AsyncMock(side_effect=SdkError("boom"))
    client = MagicMock()
    client.email = builder
    client.close = AsyncMock()

    with patch(
        "backend.app.services.email.lettermint.AsyncLettermint", return_value=client
    ):
        with pytest.raises(EmailError, match="Lettermint send failed"):
            await provider.send(
                OutboundMessage(to="a@b.co", subject="s", html="<p>h</p>", text="h")
            )
    client.close.assert_awaited_once()


# ── Inbound verify / parse ───────────────────────────────────────────────────

_WEBHOOK_SECRET = "whsec_test_secret"


def _sign(body: bytes, secret: str = _WEBHOOK_SECRET, ts: int | None = None):
    """Build the Lettermint signature + delivery headers for ``body``."""
    ts = ts if ts is not None else int(time.time())
    signed = f"{ts}.{body.decode()}".encode()
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return {
        "X-Lettermint-Signature": f"t={ts},v1={digest}",
        "X-Lettermint-Delivery": str(ts),
    }


def _inbound_body() -> bytes:
    return json.dumps(
        {
            "event": "message.inbound",
            "message": {
                "id": "in-42",
                "from_email": "sender@example.com",
                "to": [{"email": "lassi@koutsi.dev", "name": "Ops"}],
                "subject": "Hello there",
                "text": "the body",
                "created_at": "2026-07-15T08:00:00Z",
            },
        }
    ).encode()


def test_verify_inbound_signature_accepts_valid():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body)) is True


def test_verify_inbound_signature_rejects_tampered_body():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    headers = _sign(body)
    assert provider.verify_inbound_signature(body + b" ", headers) is False


def test_verify_inbound_signature_rejects_wrong_secret():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body, secret="other")) is False


def test_verify_inbound_signature_false_when_unconfigured():
    provider = LettermintProvider.from_settings(_settings())
    body = _inbound_body()
    assert provider.verify_inbound_signature(body, _sign(body)) is False


def test_parse_inbound_returns_inbound_email():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
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
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    with pytest.raises(InboundParseError):
        provider.parse_inbound(body, _sign(body, secret="wrong"))


def test_parse_inbound_rejects_stale_timestamp():
    provider = LettermintProvider.from_settings(
        _settings(lettermint_webhook_secret=_WEBHOOK_SECRET)
    )
    body = _inbound_body()
    stale = _sign(body, ts=int(time.time()) - 3600)  # outside 5-min tolerance
    assert provider.verify_inbound_signature(body, stale) is False
    with pytest.raises(InboundParseError):
        provider.parse_inbound(body, stale)


def test_parse_inbound_unconfigured_raises():
    provider = LettermintProvider.from_settings(_settings())
    body = _inbound_body()
    with pytest.raises(EmailConfigurationError):
        provider.parse_inbound(body, _sign(body))
