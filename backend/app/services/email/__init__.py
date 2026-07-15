"""Generic, swappable email module.

A single seam for all provider-specific email behaviour — outbound transactional
mail and inbound webhook handling — so call sites depend on the generic
:class:`EmailProvider` interface and the provider can be swapped via config
without touching them.

Typical use::

    from backend.app.services.email import (
        get_email_provider,
        OutboundMessage,
        render_transactional_email,
    )

    html, text = render_transactional_email(
        title="Verify your email",
        intro="Tap the button to finish creating your account.",
        action_label="Verify email",
        action_url=verify_url,
    )
    await get_email_provider().send(
        OutboundMessage(to=addr, subject="Verify your email", html=html, text=text)
    )
"""

from backend.app.services.email.base import (
    EmailConfigurationError,
    EmailError,
    EmailProvider,
    InboundEmail,
    InboundParseError,
    OutboundMessage,
)
from backend.app.services.email.factory import (
    build_email_provider,
    get_email_provider,
)
from backend.app.services.email.lettermint import LettermintProvider
from backend.app.services.email.rendering import render_transactional_email

__all__ = [
    "EmailConfigurationError",
    "EmailError",
    "EmailProvider",
    "InboundEmail",
    "InboundParseError",
    "OutboundMessage",
    "LettermintProvider",
    "build_email_provider",
    "get_email_provider",
    "render_transactional_email",
]
