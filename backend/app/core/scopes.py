"""Scope vocabulary and per-route access policy for personal access tokens.

A personal access token (PAT, issue #46) is the first credential this API issues
that is *less* than a session: it carries a fixed set of scopes and can only
reach routes that have explicitly declared themselves reachable.

The declaration is a no-op dependency — :class:`PatAccess` — attached to a router
or a single route::

    router = APIRouter(prefix="/activities", tags=["activities"], dependencies=[
        pat_scopes(read="activities:read", write="activities:write"),
    ])

    @router.get("/export", dependencies=[pat_scope("athlete:export")])
    async def export(...): ...

    router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[pat_forbidden()])

Nothing is enforced *by* the dependency, and nothing is read from it at request
time either. :func:`build_access_map` resolves every route's declaration **once,
at app construction**, into an endpoint → :class:`PatAccess` map, and
:func:`backend.app.core.auth.get_current_user` looks the current endpoint up in
it.

That indirection is not incidental. Publishing the declaration on
``request.state`` for the resolver to read back is correct only while every
declaration runs before ``get_current_user`` — which it silently does not when a
route-level dependency (``require_consent``, say) resolves ``get_current_user``
as a sub-dependency. FastAPI caches that, so the *first* dependency to ask for an
identity fixes the answer and a later ``pat_forbidden()`` never speaks. Resolving
statically removes the ordering question entirely.

Deriving the policy from the route rather than from what ran is what makes this
default-deny: a route with **no** declaration is absent from the map and
unreachable by a PAT, so a router added later is closed by default.

``tests/integration/test_pat_scopes.py`` walks ``app.routes`` and fails when an
authenticated route carries no declaration, turning the convention into a control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from fastapi import Depends, Request

# ── Vocabulary ───────────────────────────────────────────────────────────────
#
# Per-resource read/write, named after the router tags `openapi.json` already
# groups operations by, so "what can this token do?" and "what does the API
# reference call this?" have the same answer.
#
# Two deviations from the sketch in issue #46, both found by walking the routes:
# `/api/metrics` and `/api/achievements` are not purely read-only (the `catch-up`
# / `recalculate` endpoints and the `seen` marker), so both get a write scope
# rather than folding those endpoints into the read one.

SCOPES: dict[str, str] = {
    "activities:read": "Read activities, their streams, laps and intervals.",
    "activities:write": "Upload, edit, reprocess and delete activities.",
    "athlete:read": "Read the athlete profile, zones and settings.",
    "athlete:write": "Edit the athlete profile, zones and settings.",
    "athlete:export": "Download the complete data export (the entire record, in one call).",
    "metrics:read": "Read daily metrics, fitness/fatigue/form, power and distance bests.",
    "metrics:write": "Recompute derived metrics from stored activities.",
    "goals:read": "Read goals and their progress.",
    "goals:write": "Create, edit and delete goals.",
    "plans:read": "Read training plans and their planned workouts.",
    "plans:write": "Create, edit and delete training plans and planned workouts.",
    "workouts:read": "Read workout definitions.",
    "workouts:write": "Create, edit and delete workout definitions.",
    "achievements:read": "Read earned achievements.",
    "achievements:write": "Mark achievements as seen.",
    "integrations:read": "Read connected provider integrations.",
    "integrations:write": "Connect, sync and disconnect provider integrations.",
    "bikes:read": "Read bikes.",
    "bikes:write": "Create, edit and delete bikes.",
    "courses:read": "Read courses, their segment tables and pacing plans.",
    "courses:write": "Upload, re-analyse and delete courses.",
}

#: Scopes that grant more than their name suggests and are presented apart from
#: the ordinary read scopes in the UI. ``athlete:export`` returns the entire
#: record — including the inbox — in a single call, so it is never folded into
#: ``athlete:read``; it is a box the user deliberately ticks.
SENSITIVE_SCOPES: frozenset[str] = frozenset({"athlete:export"})

_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def validate_scopes(scopes: Iterable[str]) -> list[str]:
    """Normalise a requested scope list, rejecting anything unknown.

    Returns the scopes de-duplicated and in the canonical :data:`SCOPES` order so
    a token's stored scope list does not depend on how the client happened to
    order the request.
    """
    requested = set(scopes)
    unknown = sorted(requested - SCOPES.keys())
    if unknown:
        raise ValueError(f"Unknown scopes: {', '.join(unknown)}")
    return [scope for scope in SCOPES if scope in requested]


# ── Route declarations ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatAccess:
    """What a personal access token may do on the route(s) this is attached to.

    ``read`` applies to GET/HEAD/OPTIONS, ``write`` to everything else. Either
    may be ``None``, which means that half of the router is not PAT-reachable
    even though the other half is — distinct from ``allowed=False``, which
    closes the route to tokens entirely.

    As a FastAPI dependency it does nothing at all — it exists to be *found* on
    ``route.dependant`` by :func:`build_access_map`; see the module docstring.
    """

    read: Optional[str] = None
    write: Optional[str] = None
    allowed: bool = True

    def __post_init__(self) -> None:
        for scope in (self.read, self.write):
            if scope is not None and scope not in SCOPES:
                raise ValueError(f"Unknown scope: {scope}")

    def scope_for(self, method: str) -> Optional[str]:
        """The scope a token must hold to call this route with ``method``."""
        if not self.allowed:
            return None
        return self.read if method.upper() in _READ_METHODS else self.write

    async def __call__(self) -> None:  # pragma: no cover - a declaration, not logic
        return None


def pat_scopes(*, read: Optional[str] = None, write: Optional[str] = None):
    """Declare per-method scopes for a router (or a single route)."""
    return Depends(PatAccess(read=read, write=write))


def pat_scope(scope: str):
    """Declare one scope covering every method on a route."""
    return Depends(PatAccess(read=scope, write=scope))


def pat_forbidden():
    """Declare that no personal access token may reach this router/route.

    Used for the surfaces a PAT must never touch: the admin API, the auth and
    setup routers, the inbox, the LLM endpoints and the triggers that spend
    money, and the token endpoints themselves — a token can never mint another.
    """
    return Depends(PatAccess(allowed=False))


# ── Reading declarations back off the app ────────────────────────────────────


def _flatten(dependant) -> list:
    """Every sub-dependency of ``dependant``, depth-first."""
    found = []
    for sub in dependant.dependencies:
        found.append(sub)
        found.extend(_flatten(sub))
    return found


def route_pat_access(route) -> Optional[PatAccess]:
    """The :class:`PatAccess` declared on ``route``, or ``None`` if there is none.

    Router-level dependencies are registered before route-level ones, so the
    last match wins and a route can narrow (or close) what its router opened.
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return None
    declared = [
        sub.call for sub in _flatten(dependant) if isinstance(sub.call, PatAccess)
    ]
    return declared[-1] if declared else None


def build_access_map(app) -> dict:
    """Resolve every route's declaration once, keyed by endpoint function.

    Called from ``create_app()``; the result lives on ``app.state`` and is what
    ``get_current_user`` consults. Routes with no declaration are simply absent,
    which is the default-deny case.

    Keying on the endpoint is safe because Starlette puts ``endpoint`` (not
    ``route``) in the request scope. Registering one function on two routes is
    fine as long as they agree — ``PUT``/``POST /athlete/avatar`` share a handler
    and a declaration — and a **disagreement raises at startup** rather than
    resolving to whichever route was registered last.
    """
    from fastapi.routing import APIRoute

    resolved: dict = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        access = route_pat_access(route)
        if access is None:
            continue
        existing = resolved.get(route.endpoint)
        if existing is not None and existing != access:
            raise RuntimeError(
                f"{route.endpoint.__qualname__} is registered on routes with "
                f"conflicting personal-access-token declarations: "
                f"{existing!r} vs {access!r}. Split the handler, or make them agree."
            )
        resolved[route.endpoint] = access
    return resolved


def access_for_request(request: Request) -> Optional[PatAccess]:
    """The declaration covering the route this request matched, if any."""
    resolved = getattr(request.app.state, "pat_access_by_endpoint", None)
    if not resolved:
        return None
    return resolved.get(request.scope.get("endpoint"))


def route_requires_auth(route) -> bool:
    """Whether ``route`` resolves an identity at all.

    True for anything that reaches ``get_current_user``, directly or through
    ``get_ctx_and_session`` / ``get_ctx_session_athlete`` / ``require_admin`` /
    the inbox's own session dependency — every authenticated route funnels
    through that one resolver, which is the point of it.
    """
    from backend.app.core.auth import get_current_user

    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return False
    return any(sub.call is get_current_user for sub in _flatten(dependant))
