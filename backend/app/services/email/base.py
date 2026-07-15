"""Generic, swappable email-provider interface.

This is the single seam through which the rest of the backend talks to an email
provider. Both consumers build on it:

  * **Outbound** transactional mail — verification / password-reset messages
    (openkoutsi/openkoutsi#15) — calls :meth:`EmailProvider.send`.
  * **Inbound** operator mail — surfaced in-app via the optional webhook bridge
    (#38) — calls :meth:`EmailProvider.verify_inbound_signature` and
    :meth:`EmailProvider.parse_inbound`.

Everything provider-specific (auth, endpoints, signature scheme, payload shapes)
lives in a concrete implementation such as :class:`LettermintProvider`; selecting
a different provider should touch only this package. Use
:func:`backend.app.services.email.get_email_provider` to obtain the configured
implementation.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from backend.app.core.config import Settings


class EmailError(Exception):
    """Base class for all email-provider errors."""


class EmailConfigurationError(EmailError):
    """Raised when an operation is attempted without the required configuration.

    Callers that offer email-dependent features (self-serve signup, password
    reset by email, the inbound bridge) should degrade gracefully — check that
    the provider is configured before offering the feature — rather than letting
    this surface as a 500.
    """


class InboundParseError(EmailError):
    """Raised when an inbound webhook payload cannot be verified or parsed."""


@dataclass(frozen=True)
class OutboundMessage:
    """A transactional message to send.

    Lettermint's send API (and providers like it) accept only ``html`` and
    ``text`` — no markdown field and no server-side templating — so the backend
    renders both parts itself (see :mod:`.rendering`) and always sends both.
    """

    to: str
    subject: str
    html: str
    text: str


@dataclass(frozen=True)
class InboundEmail:
    """A provider-agnostic representation of a received email."""

    from_addr: str
    to_addr: str
    subject: str
    text: str
    message_id: str
    received_at: datetime


class EmailProvider(ABC):
    """Abstract email provider covering the operations both consumers need."""

    #: Stable identifier used for provider selection in config.
    PROVIDER_NAME: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: "Settings") -> "EmailProvider":
        """Build the provider from application config."""

    # ── Outbound ───────────────────────────────────────────────────────────

    @abstractmethod
    async def send(self, message: OutboundMessage) -> str:
        """Send a transactional message.

        Returns the provider's message id. Raises
        :class:`EmailConfigurationError` when outbound sending is not configured
        and :class:`EmailError` (or a subclass) on delivery failure.
        """

    # ── Inbound ────────────────────────────────────────────────────────────

    @abstractmethod
    def verify_inbound_signature(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> bool:
        """Return ``True`` iff the inbound webhook signature over ``raw_body`` is
        valid for this provider's signing scheme. Never raises."""

    @abstractmethod
    def parse_inbound(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> InboundEmail:
        """Verify and parse an inbound webhook body into an :class:`InboundEmail`.

        Raises :class:`InboundParseError` if the signature is invalid or the
        payload cannot be understood — a parsed message is always an
        authenticated one. Because this re-verifies the signature internally,
        callers should use *either* :meth:`verify_inbound_signature` (a
        never-raises gate) *or* this method, not both.
        """
