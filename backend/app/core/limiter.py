"""The shared rate limiter.

Keyed by *principal* rather than purely by address (issue #46). A personal access
token makes a per-principal key both necessary — one script hammering from one
address is not one anonymous visitor — and finally possible, because a token id
is a stable principal in a way an IP never was.

Unauthenticated traffic is unaffected: with no token in play the key falls back
to the remote address, so the limits protecting login, signup and password reset
behave exactly as before.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def principal_key(request: Request) -> str:
    """Rate-limit key: the personal access token when there is one, else the IP.

    ``request.state.pat_token_id`` is set by the resolver in ``core.auth``, which
    runs as a dependency and therefore before the endpoint the limiter wraps.
    """
    token_id = getattr(request.state, "pat_token_id", None)
    if token_id:
        return f"pat:{token_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=principal_key)
