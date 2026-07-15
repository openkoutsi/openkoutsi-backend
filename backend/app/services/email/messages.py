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
