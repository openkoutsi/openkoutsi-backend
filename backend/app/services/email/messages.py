"""Transactional messages sent by the auth flows (issue #15).

Thin helpers that render a body with :func:`render_transactional_email` and hand
it to a provider's :meth:`~backend.app.services.email.base.EmailProvider.send`.
The two messages this feature introduces — *verify email* on signup and
*password reset* on forgot-password — live here so the endpoints stay focused on
token/account handling.

These are addressed to a bare email address (before an account is even
activated), so they call the provider directly rather than routing through
:mod:`backend.app.services.notifications` (whose ``_dispatch_external`` seam is
for per-existing-user in-app notifications keyed by a per-user DB).
"""

from backend.app.services.email.base import EmailProvider, OutboundMessage
from backend.app.services.email.rendering import render_transactional_email

_VERIFY_SUBJECT = "Confirm your email"
_RESET_SUBJECT = "Reset your password"
_TOKEN_EXPIRY_SUBJECT = "Your openkoutsi access token"


async def send_verification_email(
    provider: EmailProvider, *, to: str, action_url: str
) -> str:
    """Render and send the signup email-verification message."""
    html, text = render_transactional_email(
        title="Confirm your email",
        intro=(
            "Welcome to openkoutsi. Confirm this email address to activate your "
            "account."
        ),
        action_label="Verify email",
        action_url=action_url,
        outro="This link expires in 1 hour.",
        footer=(
            "If you didn't create an account, you can safely ignore this email."
        ),
    )
    return await provider.send(
        OutboundMessage(to=to, subject=_VERIFY_SUBJECT, html=html, text=text)
    )


async def send_password_reset_email(
    provider: EmailProvider, *, to: str, action_url: str
) -> str:
    """Render and send the self-serve password-reset message."""
    html, text = render_transactional_email(
        title="Reset your password",
        intro=(
            "We received a request to reset the password for your openkoutsi "
            "account. Choose a new password with the link below."
        ),
        action_label="Reset password",
        action_url=action_url,
        outro="This link expires in 1 hour and can only be used once.",
        footer=(
            "If you didn't request a password reset, you can safely ignore this "
            "email — your password won't change."
        ),
    )
    return await provider.send(
        OutboundMessage(to=to, subject=_RESET_SUBJECT, html=html, text=text)
    )


async def send_token_expiry_email(
    provider: EmailProvider,
    *,
    to: str,
    token_name: str,
    days_left: int | None,
    manage_url: str,
) -> str:
    """Render and send the personal-access-token expiry warning (issue #46).

    ``days_left`` is ``None`` once the token has already expired. Email is
    best-effort on top of the inbox message, which is unconditional — this is the
    channel that reaches somebody whose integration is about to break while they
    are not looking at the app, which is the whole point of warning at all.
    """
    label = f'"{token_name}"' if token_name else "A personal access token"
    if days_left is None:
        title = "Your access token has expired"
        intro = (
            f"{label} has expired and no longer works. Anything still using it "
            "is now being refused."
        )
        outro = "Create a replacement if you still need it."
    else:
        when = "tomorrow" if days_left <= 1 else f"in {days_left} days"
        title = f"Your access token expires {when}"
        intro = (
            f"{label} expires {when}. Anything using it will stop working once "
            "it does."
        )
        outro = (
            "Tokens cannot be extended — create a replacement and update "
            "whatever holds the old one."
        )

    html, text = render_transactional_email(
        title=title,
        intro=intro,
        action_label="Manage tokens",
        action_url=manage_url,
        outro=outro,
        footer=(
            "You can turn these emails off in your profile settings. The "
            "in-app notification will still be sent."
        ),
    )
    return await provider.send(
        OutboundMessage(to=to, subject=_TOKEN_EXPIRY_SUBJECT, html=html, text=text)
    )
