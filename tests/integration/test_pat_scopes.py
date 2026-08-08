"""The default-deny guarantee for personal access tokens (issue #46).

Scope declarations on routes are a *convention* until something fails when one
is missing. This module is that something: it walks ``app.routes`` and refuses
any authenticated route that has not said what a token may do with it.

Without this test the hole arrives silently six months from now, in a router
somebody adds while not thinking about tokens at all.
"""
import pytest
from fastapi.routing import APIRoute

from backend.app.core.scopes import (
    SCOPES,
    PatAccess,
    route_pat_access,
    route_requires_auth,
)

#: Surfaces a personal access token must never reach, whatever scopes it holds.
#: Enforced by allowlist — these prefixes are asserted *closed*, and everything
#: else must carry a positive declaration.
CLOSED_PREFIXES = (
    "/api/admin",     # admin status must not widen the athlete-data surface
    "/api/auth",      # a token must never mint or refresh a credential
    "/api/setup",
    "/api/tokens",    # a token cannot create, list or revoke a token
    "/api/messages",  # the inbox is platform correspondence, not training data
    "/api/llm",       # LLM endpoints spend money
    "/api/consent",   # consent is the account holder's act, not a credential's
)


def _api_routes(app):
    return [r for r in app.routes if isinstance(r, APIRoute)]


def _authenticated_routes(app):
    return [r for r in _api_routes(app) if route_requires_auth(r)]


def test_there_are_authenticated_routes_to_check(app):
    """Guards the rest of this module against passing vacuously."""
    assert len(_authenticated_routes(app)) > 50


def test_every_authenticated_route_declares_what_a_token_may_do(app):
    """**The default-deny control.**

    A route with no declaration is closed at runtime, so a missing one is not a
    security hole — but it is an accident, and an unreachable endpoint nobody
    meant to close is its own kind of bug. Declare it, either with a scope or
    with ``pat_forbidden()``.
    """
    undeclared = [
        f"{sorted(r.methods - {'HEAD', 'OPTIONS'})} {r.path}"
        for r in _authenticated_routes(app)
        if route_pat_access(r) is None
    ]
    assert undeclared == [], (
        "These authenticated routes declare no personal-access-token policy. "
        "Add `pat_scopes(...)` / `pat_scope(...)` to the router or route, or "
        "`pat_forbidden()` if tokens must not reach it:\n  "
        + "\n  ".join(undeclared)
    )


def test_an_undeclared_route_would_fail_this_check(app):
    """The check above only means something if it can fail."""

    class _Fake:
        dependant = type("D", (), {"dependencies": []})()

    assert route_pat_access(_Fake()) is None


def test_every_declared_scope_is_in_the_vocabulary(app):
    """No route may demand a scope the token endpoints cannot grant."""
    for route in _authenticated_routes(app):
        access = route_pat_access(route)
        for scope in (access.read, access.write):
            assert scope is None or scope in SCOPES, f"{route.path}: {scope}"


def test_every_scope_in_the_vocabulary_is_reachable(app):
    """No dead scopes — a grant the user ticks must open something."""
    declared = set()
    for route in _authenticated_routes(app):
        access = route_pat_access(route)
        for method in route.methods - {"HEAD", "OPTIONS"}:
            scope = access.scope_for(method)
            if scope:
                declared.add(scope)
    assert set(SCOPES) - declared == set()


@pytest.mark.parametrize("prefix", CLOSED_PREFIXES)
def test_the_closed_surfaces_stay_closed(app, prefix):
    """Excluded by allowlist, not denylist: each of these must declare
    ``pat_forbidden()`` rather than merely happen to lack a scope.

    Applied to every route under the prefix, authenticated or not. ``/api/setup``
    is the one that resolves no identity today, so nothing reads its declaration
    at request time — a token presented there is ignored rather than refused.
    The declaration is still the right thing to carry: it is what makes setup
    closed on the day it gains an authenticated route, instead of open.
    """
    routes = [r for r in _api_routes(app) if r.path.startswith(prefix)]
    assert routes, f"expected routes under {prefix}"
    for route in routes:
        access = route_pat_access(route)
        assert access is not None, route.path
        assert access.allowed is False, f"{route.path} is reachable by a token"


def test_the_llm_triggers_are_closed_individually(app):
    """The endpoints that spend money but live on otherwise-open routers."""
    closed = {
        ("POST", "/api/athlete/training-status"),
        ("POST", "/api/activities/{activity_id}/analyze"),
        ("POST", "/api/goals/{goal_id}/guidance"),
        ("POST", "/api/plans/{plan_id}/regenerate"),
        ("POST", "/api/plans/{plan_id}/generate-upcoming/workouts"),
    }
    seen = set()
    for route in _authenticated_routes(app):
        for method in route.methods:
            if (method, route.path) in closed:
                seen.add((method, route.path))
                assert route_pat_access(route).allowed is False, route.path
    assert seen == closed, f"missing routes: {closed - seen}"


def test_export_carries_its_own_scope(app):
    """One call that returns the entire record is never folded into a general read."""
    export = next(
        r for r in _authenticated_routes(app) if r.path == "/api/athlete/export"
    )
    access = route_pat_access(export)
    assert access.scope_for("GET") == "athlete:export"

    # …and no other route grants that scope, so ticking it grants exactly this.
    granting = [
        r.path
        for r in _authenticated_routes(app)
        if (a := route_pat_access(r))
        and "athlete:export" in {a.read, a.write}
    ]
    assert granting == ["/api/athlete/export"]


def test_a_route_level_declaration_overrides_its_routers(app):
    """`/api/athlete` is open for reads; `/api/athlete/export` narrows it."""
    profile = next(r for r in _authenticated_routes(app) if r.path == "/api/athlete")
    assert route_pat_access(profile).scope_for("GET") == "athlete:read"

    export = next(
        r for r in _authenticated_routes(app) if r.path == "/api/athlete/export"
    )
    assert route_pat_access(export).scope_for("GET") == "athlete:export"


def test_unauthenticated_routes_need_no_declaration(app):
    """The check is about *credentials*, so it has nothing to say about
    endpoints that never resolve one."""
    public = [
        r for r in _api_routes(app)
        if r.path.startswith("/api/public") or r.path == "/api/health"
    ]
    assert public
    for route in public:
        assert not route_requires_auth(route)


def test_read_methods_never_resolve_to_a_write_scope(app):
    """A GET must not be able to satisfy itself with a write grant."""
    for route in _authenticated_routes(app):
        access = route_pat_access(route)
        if not access.allowed:
            continue
        scope = access.scope_for("GET")
        assert scope is None or scope.endswith((":read", ":export")), (
            f"{route.path}: GET resolves to {scope}"
        )


def test_the_policy_is_resolved_from_the_route_not_from_what_ran(app):
    """Regression: a route-level declaration must bind regardless of ordering.

    An earlier version had the declaration publish itself on `request.state` for
    the resolver to read back, which works only while every declaration runs
    before `get_current_user` does. It silently does not when a route-level
    dependency resolves `get_current_user` as a sub-dependency of its own —
    FastAPI caches that resolution, so the first dependency to ask for an
    identity fixes the answer and a later `pat_forbidden()` never speaks.

    `GET /api/integrations/{provider}/connect` is exactly that shape: its
    `require_consent` dependency precedes its `pat_forbidden()`. Resolving the
    policy statically from `route.dependant` is what makes the order irrelevant.
    """
    connect = next(
        r for r in _authenticated_routes(app)
        if r.path == "/api/integrations/{provider}/connect"
    )
    # The declaration is found, and it is the route's own, not its router's.
    assert route_pat_access(connect).allowed is False
    # And the map the resolver actually consults agrees.
    assert app.state.pat_access_by_endpoint[connect.endpoint].allowed is False


def test_every_declared_route_is_in_the_resolver_map(app):
    """The static walk and the runtime lookup must not be able to disagree."""
    for route in _api_routes(app):
        access = route_pat_access(route)
        if access is None:
            continue
        assert app.state.pat_access_by_endpoint.get(route.endpoint) == access, route.path


def test_conflicting_declarations_on_one_handler_fail_loudly():
    """Keying the map on the endpoint is only safe if disagreement is caught."""
    from fastapi import FastAPI
    from backend.app.core.scopes import build_access_map, pat_forbidden, pat_scopes

    conflicted = FastAPI()

    async def handler():
        return None

    conflicted.get("/a", dependencies=[pat_scopes(read="activities:read")])(handler)
    conflicted.get("/b", dependencies=[pat_forbidden()])(handler)

    with pytest.raises(RuntimeError, match="conflicting"):
        build_access_map(conflicted)


def test_declaring_a_scope_outside_the_vocabulary_is_impossible():
    with pytest.raises(ValueError):
        PatAccess(read="everything:always")
