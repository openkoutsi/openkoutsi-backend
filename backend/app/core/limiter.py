"""The shared limiter.

Keyed by *principal* rather than purely by address (issue #46): one script
hammering from one address is not one anonymous visitor, and an authenticated
request carries a stable identity in a way an IP never did.

The principal is the **user**, not the token — a user may mint tokens freely, and
each new one would be a fresh bucket, making the limit multiplicative in a number
nothing caps. The token id is still on ``request.state`` and in the audit log,
which is where per-token attribution belongs.

Unauthenticated traffic falls back to the remote address, so the limits
protecting login, signup and password reset behave as before.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def principal_key(request: Request) -> str:
    """Rate-limit key: the authenticated user when there is one, else the IP.

    ``request.state.principal_user_id`` is set by both resolvers in ``core.auth``
    — the personal-access-token path and the session-JWT path — which run as
    dependencies and therefore before the endpoint the limiter wraps.

    Session requests were originally left on the address key, which chat
    (issue #44) makes wrong: it is athlete-triggered, expensive and closed to
    tokens, so an address key would limit a household behind one NAT as a single
    user while leaving an actual user free to open two browsers.

    ``pat_user_id`` stays readable as a fallback, since it is the attribute the
    audit log and per-token attribution use.
    """
    user_id = getattr(request.state, "principal_user_id", None) or getattr(
        request.state, "pat_user_id", None
    )
    if user_id:
        return f"user:{user_id}"
    return get_remote_address(request)


# `key_style="endpoint"` scopes each limit to the *route*, not the request path.
# slowapi defaults to "url" — the substituted path — so on any route with a path
# parameter every distinct value got its own bucket and the limit never fired.
# That silently applied to the admin password-reset mint, both chat write routes,
# and the public avatar route added for issue #102's F-14, whose whole point is
# bounding requests for arbitrary user ids.
#
# Per-route is what those limits read as meaning: a cap on how often *this
# caller* may call *this endpoint*, not with one particular id.
limiter = Limiter(key_func=principal_key, key_style="endpoint")
