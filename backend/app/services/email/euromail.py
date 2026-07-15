"""EuroMail email provider.

Isolates every EuroMail-specific detail behind the generic
:class:`~backend.app.services.email.base.EmailProvider` interface:

  * **Outbound** — sending via the official ``euromail`` SDK
    (:class:`euromail.AsyncEuroMail`), whose ``send_email`` takes ``html_body`` /
    ``text_body`` (bodies are rendered by :mod:`.rendering`, not templated
    server-side).
  * **Inbound** — the webhook signature scheme (HMAC-SHA256 over
    ``{timestamp}.{body}`` carried in the ``X-Euromail-Signature`` header as
    ``t={timestamp},v1={hex}``) and the ``email.inbound`` payload shape.

EuroMail (https://euromail.dev) is EU-based (Finland) with inbound email on its
free tier (issue #41), which is why it is offered as a cheaper alternative to
Lettermint; nothing outside this module depends on any of the above.
"""

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import httpx
from euromail import AsyncEuroMail
from euromail.errors import EuroMailError

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

# EuroMail signs webhook bodies as HMAC-SHA256 over ``{timestamp}.{body}`` and
# carries the result in the ``X-Euromail-Signature`` header. We reject deliveries
# whose timestamp is outside this tolerance to blunt replay attacks (the same
# 5-minute window Lettermint uses).
_SIGNATURE_HEADER = "x-euromail-signature"
_SIGNATURE_TOLERANCE_SECONDS = 300


class EuromailProvider(EmailProvider):
    PROVIDER_NAME = "euromail"

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
    def from_settings(cls, settings: Settings) -> "EuromailProvider":
        return cls(
            api_key=settings.euromail_api_key,
            from_addr=settings.email_from,
            webhook_secret=settings.euromail_webhook_secret,
        )

    # ── Outbound ───────────────────────────────────────────────────────────

    async def send(self, message: OutboundMessage) -> str:
        if not self._api_key or not self._from_addr:
            raise EmailConfigurationError(
                "EuroMail outbound sending is not configured "
                "(set EUROMAIL_API_KEY and EMAIL_FROM)."
            )
        client = AsyncEuroMail(api_key=self._api_key)
        try:
            response = await client.send_email(
                from_address=self._from_addr,
                to=message.to,
                subject=message.subject,
                html_body=message.html,
                text_body=message.text,
            )
        except (EuroMailError, httpx.HTTPError) as exc:
            # The SDK wraps API status errors as EuroMailError, but pre-response
            # transport failures (ConnectError, DNS/TLS) escape as raw
            # httpx.HTTPError. Catch both so send() always honours the documented
            # "raises EmailError on delivery failure" contract.
            raise EmailError(f"EuroMail send failed: {exc}") from exc
        finally:
            await client.close()
        return response.message_id

    # ── Inbound ────────────────────────────────────────────────────────────

    def verify_inbound_signature(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> bool:
        if not self._webhook_secret:
            return False
        try:
            return _verify_signature(self._webhook_secret, raw_body, headers)
        except Exception:  # never raises, per the interface contract
            return False

    def parse_inbound(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> InboundEmail:
        if not self._webhook_secret:
            raise EmailConfigurationError(
                "EuroMail inbound is not configured (set EUROMAIL_WEBHOOK_SECRET)."
            )
        if not _verify_signature(self._webhook_secret, raw_body, headers):
            raise InboundParseError("Inbound signature invalid")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise InboundParseError(f"Inbound body is not valid JSON: {exc}") from exc
        return _parse_inbound_payload(payload)


# ── Signature verification ─────────────────────────────────────────────────


def _header_value(headers: Mapping[str, str], name: str) -> str:
    """Case-insensitively look up ``name`` in ``headers`` (HTTP headers are
    case-insensitive but a plain mapping preserves whatever case it was built
    with)."""
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def _parse_signature_header(header: str) -> tuple[str, list[str]]:
    """Split a ``t={ts},v1={hex}`` header into its timestamp and v1 signatures.

    Multiple ``v1`` entries are tolerated (the scheme allows several signatures,
    e.g. during secret rotation); any matching one accepts the delivery.
    """
    timestamp = ""
    signatures: list[str] = []
    for part in header.split(","):
        key, _, value = part.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1":
            signatures.append(value)
    return timestamp, signatures


def _verify_signature(
    secret: str, raw_body: bytes, headers: Mapping[str, str]
) -> bool:
    header = _header_value(headers, _SIGNATURE_HEADER)
    if not header:
        return False
    timestamp, signatures = _parse_signature_header(header)
    if not timestamp or not signatures:
        return False

    # Reject stale (replayed) deliveries outside the tolerance window.
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > _SIGNATURE_TOLERANCE_SECONDS:
        return False

    signed = f"{timestamp}.{raw_body.decode('utf-8')}".encode()
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, sig) for sig in signatures)


# ── Inbound payload parsing ─────────────────────────────────────────────────


def _parse_inbound_payload(payload: dict[str, Any]) -> InboundEmail:
    """Map a verified ``email.inbound`` webhook payload to :class:`InboundEmail`.

    This is the one place that knows EuroMail's inbound JSON shape. EuroMail
    wraps the inbound email object under a ``data``/``email`` key (the REST API
    uses the same ``data`` envelope); we look inside first and fall back to the
    root, and accept the small field-name variants EuroMail uses across its REST
    and webhook payloads.
    """
    if not isinstance(payload, dict):
        raise InboundParseError("Inbound payload is not an object")
    message = payload.get("data") or payload.get("email") or payload
    if not isinstance(message, dict):
        raise InboundParseError("Inbound payload has no email object")

    from_addr = _as_address(message.get("from_address") or message.get("from"))
    to_addr = _as_address(
        message.get("to_addresses")
        or message.get("to")
        or message.get("to_address")
    )
    subject = message.get("subject") or ""
    text = message.get("text_body") or message.get("text") or ""
    message_id = message.get("id") or message.get("message_id") or ""
    received_at = _parse_timestamp(
        message.get("created_at")
        or message.get("received_at")
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

    EuroMail may render an address as a plain string, a ``{"email", "name"}``
    object, or a list of either (``to_addresses`` is a list); we take the first
    address.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return _as_address(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("email") or value.get("address") or "")
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
