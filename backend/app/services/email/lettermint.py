"""Lettermint email provider.

Isolates every Lettermint-specific detail behind the generic
:class:`~backend.app.services.email.base.EmailProvider` interface:

  * **Outbound** — sending via the official ``lettermint-python`` SDK, whose send
    surface exposes only ``.html()`` / ``.text()`` (bodies are rendered by
    :mod:`.rendering`, not templated server-side).
  * **Inbound** — the webhook signature scheme (HMAC-SHA256 over
    ``{timestamp}.{body}`` carried in the ``X-Lettermint-Signature`` /
    ``X-Lettermint-Delivery`` headers) and the ``message.inbound`` payload shape.

Lettermint (https://lettermint.co) is EU-based, which is why it is the default
provider; nothing outside this module depends on any of the above.
"""

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from lettermint import AsyncLettermint, Webhook
from lettermint.exceptions import LettermintError, WebhookVerificationError

from backend.app.core.config import Settings
from backend.app.services.email.base import (
    EmailConfigurationError,
    EmailError,
    EmailProvider,
    InboundEmail,
    InboundParseError,
    OutboundMessage,
)

log = logging.getLogger(__name__)


class LettermintProvider(EmailProvider):
    PROVIDER_NAME = "lettermint"

    def __init__(
        self,
        *,
        api_key: str = "",
        from_addr: str = "",
        webhook_secret: str = "",
    ) -> None:
        self._api_key = api_key
        self._from_addr = from_addr
        self._webhook_secret = webhook_secret

    @classmethod
    def from_settings(cls, settings: Settings) -> "LettermintProvider":
        return cls(
            api_key=settings.lettermint_api_key,
            from_addr=settings.email_from,
            webhook_secret=settings.lettermint_webhook_secret,
        )

    # ── Outbound ───────────────────────────────────────────────────────────

    async def send(self, message: OutboundMessage) -> str:
        if not self._api_key or not self._from_addr:
            raise EmailConfigurationError(
                "Lettermint outbound sending is not configured "
                "(set LETTERMINT_API_KEY and EMAIL_FROM)."
            )
        client = AsyncLettermint(api_token=self._api_key)
        try:
            response = await (
                client.email.from_(self._from_addr)
                .to(message.to)
                .subject(message.subject)
                .html(message.html)
                .text(message.text)
                .send()
            )
        except LettermintError as exc:
            raise EmailError(f"Lettermint send failed: {exc}") from exc
        finally:
            await client.close()
        return response["message_id"]

    # ── Inbound ────────────────────────────────────────────────────────────

    def verify_inbound_signature(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> bool:
        if not self._webhook_secret:
            return False
        try:
            Webhook(self._webhook_secret).verify_headers(
                dict(headers), raw_body.decode("utf-8")
            )
        except (WebhookVerificationError, ValueError, UnicodeDecodeError):
            return False
        return True

    def parse_inbound(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> InboundEmail:
        if not self._webhook_secret:
            raise EmailConfigurationError(
                "Lettermint inbound is not configured (set LETTERMINT_WEBHOOK_SECRET)."
            )
        try:
            payload = Webhook(self._webhook_secret).verify_headers(
                dict(headers), raw_body.decode("utf-8")
            )
        except (WebhookVerificationError, ValueError, UnicodeDecodeError) as exc:
            raise InboundParseError(f"Inbound signature invalid: {exc}") from exc
        return _parse_inbound_payload(payload)


def _parse_inbound_payload(payload: dict[str, Any]) -> InboundEmail:
    """Map a verified ``message.inbound`` webhook payload to :class:`InboundEmail`.

    This is the one place that knows Lettermint's inbound JSON shape. Lettermint
    wraps the message under a ``message``/``data`` key (older/flat payloads put
    the fields at the top level), so we look inside first and fall back to the
    root, and accept the small field-name variants Lettermint has used.
    """
    message = payload.get("message") or payload.get("data") or payload
    if not isinstance(message, dict):
        raise InboundParseError("Inbound payload has no message object")

    from_addr = _as_address(message.get("from") or message.get("from_email"))
    to_addr = _as_address(message.get("to") or message.get("to_email"))
    subject = message.get("subject") or ""
    text = (
        message.get("text")
        or message.get("text_body")
        or message.get("plain")
        or ""
    )
    message_id = message.get("message_id") or message.get("id") or ""
    received_at = _parse_timestamp(
        message.get("received_at")
        or message.get("created_at")
        or payload.get("timestamp")
    )

    if not from_addr or not message_id:
        raise InboundParseError("Inbound payload missing sender or message id")

    return InboundEmail(
        from_addr=from_addr,
        to_addr=to_addr,
        subject=subject,
        text=text,
        message_id=str(message_id),
        received_at=received_at,
    )


def _as_address(value: Any) -> str:
    """Normalise a recipient field to a bare address.

    Lettermint may render an address as a plain string, a ``{"email", "name"}``
    object, or a list of either (for ``to``); we take the first address.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return _as_address(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("email") or "")
    return str(value)


def _parse_timestamp(value: Any) -> datetime:
    """Best-effort parse of a webhook timestamp, defaulting to 'now' (UTC)."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)
