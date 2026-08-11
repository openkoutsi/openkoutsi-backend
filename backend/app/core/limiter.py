"""The shared limiter.

Keyed by *principal* rather than purely by address (issue #46). A personal access
token makes a per-principal key both necessary — one script hammering from one
address is not one anonymous visitor — and finally possible, because an
authenticated request carries a stable identity in a way an IP never did.

The principal is the **user**, not the token. Keying on the token id would be
better for observability, but a user may mint tokens freely and each new one
would be a fresh bucket, so the effective limit would be multiplicative in a
number nothing caps — inverting the intent above the moment a script holds two
credentials. The token id is still recorded on ``request.state`` (and in the
audit log), which is where per-token attribution belongs.

Unauthenticated traffic is unaffected: with no token in play the key falls back
to the remote address, so the limits protecting login, signup and password reset
behave exactly as before.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def principal_key(request: Request) -> str:
    """Rate-limit key: the authenticated user when there is one, else the IP.

    ``request.state.principal_user_id`` is set by both resolvers in ``core.auth``
    — the personal-access-token path and the session-JWT path — which run as
    dependencies and therefore before the endpoint the limiter wraps.

    Session requests were originally left on the address key, which was the
    right call when the only per-principal limits protected token traffic. Chat
    (issue #44) is what makes it wrong: it is athlete-triggered, expensive, and
    closed to tokens, so an address key would rate-limit a household behind one
    NAT as a single user while leaving an actual user free to open two browsers.

    ``pat_user_id`` stays readable as a fallback: it is the attribute the audit
    log and per-token attribution use, and reading it here keeps the key stable
    for anything that sets it directly.
    """
    user_id = getattr(request.state, "principal_user_id", None) or getattr(
        request.state, "pat_user_id", None
    )
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=principal_key)
