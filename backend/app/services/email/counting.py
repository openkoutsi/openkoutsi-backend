"""Counting outbound email (issue #66).

Email is the one third-party surface the counting HTTP transport cannot see:
both providers talk to their vendor through an SDK (``AsyncLettermint``,
``AsyncEuroMail``), not through an ``httpx`` client we construct. Its seam is
the one :mod:`.base` already documents as *the* single seam —
:meth:`EmailProvider.send` — so the counting goes there, wrapped around the
constructed provider in :func:`~.factory.build_email_provider`.

Per **message** is also the right unit here, unlike the per-request unit used
for the providers: a message is how email is billed. Email has no rate-limit
quota in the headroom sense, so these rows serve the cost goal only and carry no
``ratelimit_*`` reading.

The *kind* of message travels in a context variable rather than on
:class:`~.base.OutboundMessage`. ``send()`` is the generic interface every
caller depends on, and "this is a password reset" is accounting metadata, not
part of what it means to send a message — so the five helpers in :mod:`.messages`
label their call and the interface is untouched.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

from backend.app.services.api_usage import schedule_api_usage
from backend.app.services.email.base import (
    EmailConfigurationError,
    EmailProvider,
    InboundEmail,
    OutboundMessage,
)

if TYPE_CHECKING:
    from backend.app.core.config import Settings

#: Message kinds, matching the five ``provider.send()`` call sites in
#: :mod:`.messages`. Recorded in the ``endpoint`` column as ``send/<kind>``,
#: which is what makes ``group_by=endpoint`` a per-kind email breakdown.
KIND_VERIFICATION = "verification"
KIND_EMAIL_CHANGE = "email_change"
KIND_EMAIL_CHANGE_AUTHORISATION = "email_change_authorisation"
KIND_PASSWORD_RESET = "password_reset"
KIND_TOKEN_EXPIRY = "token_expiry"

_kind: ContextVar[str | None] = ContextVar("email_kind", default=None)


@contextlib.contextmanager
def email_kind(kind: str) -> Iterator[None]:
    """Label every message sent inside this block as *kind*."""
    token = _kind.set(kind)
    try:
        yield
    finally:
        _kind.reset(token)


def current_endpoint() -> str:
    kind = _kind.get()
    return f"send/{kind}" if kind else "send"


class CountingEmailProvider(EmailProvider):
    """Delegates to a real provider, recording one row per message sent.

    A send that never left the building is not recorded:
    :class:`EmailConfigurationError` is raised before the vendor is contacted, so
    nothing was spent and nothing was billed. A delivery failure *is* recorded —
    with a null status, because the generic :class:`~.base.EmailProvider`
    contract surfaces failures as :class:`~.base.EmailError` and deliberately
    does not carry the vendor's HTTP status.
    """

    def __init__(self, inner: EmailProvider) -> None:
        self._inner = inner
        self.PROVIDER_NAME = inner.PROVIDER_NAME

    @property
    def wrapped(self) -> EmailProvider:
        """The concrete provider underneath.

        Exposed because the wrapper is transparent by intent: callers that need
        the provider's own type (a test asserting which one config selected)
        should not have to reach for a private attribute to get it.
        """
        return self._inner

    @classmethod
    def from_settings(cls, settings: "Settings") -> "EmailProvider":  # pragma: no cover
        raise NotImplementedError(
            "CountingEmailProvider wraps an already-constructed provider; "
            "build one with build_email_provider()."
        )

    @property
    def is_configured(self) -> bool:
        return self._inner.is_configured

    async def send(self, message: OutboundMessage) -> str:
        endpoint = current_endpoint()
        started = time.perf_counter()
        try:
            message_id = await self._inner.send(message)
        except EmailConfigurationError:
            raise
        except Exception:
            schedule_api_usage(
                service=self._inner.PROVIDER_NAME,
                endpoint=endpoint,
                method="POST",
                status_code=None,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        schedule_api_usage(
            service=self._inner.PROVIDER_NAME,
            endpoint=endpoint,
            method="POST",
            status_code=200,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        return message_id

    # ── Inbound: pure delegation. Nothing inbound is a third-party call of
    # ours, so nothing here is counted. ────────────────────────────────────

    def verify_inbound_signature(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> bool:
        return self._inner.verify_inbound_signature(raw_body, headers)

    def parse_inbound(
        self, raw_body: bytes, headers: Mapping[str, str]
    ) -> InboundEmail:
        return self._inner.parse_inbound(raw_body, headers)
