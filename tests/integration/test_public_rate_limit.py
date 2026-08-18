"""The unauthenticated router declares limits like every other one (F-14).

`/api/public` was the only router with no `@limiter.limit` anywhere on it —
every other credential-accepting router declares one. `instance-info` hits the
registry on each call, and `avatar` was F-03.

The limits are deliberately generous. F-03 (#105) already removed the expensive
part of the avatar route, so what is left to bound is ordinary request volume
rather than a resource leak. They are also *keyed on the remote address* for
unauthenticated callers, and behind a reverse proxy that address is the proxy's
unless `FORWARDED_ALLOW_IPS` is set — so on a default container deployment these
are one instance-wide bucket, not a per-client one. That is why they sit far
above plausible real usage, and why the nginx `limit_req` in DEPLOY.md is the
per-IP half of this.

The shared `client` fixture disables the limiter (rate limits would make the
rest of the suite flaky), so these tests build their own client and turn it
back on.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api import public
from backend.app.core.limiter import limiter
from backend.app.db.registry import get_registry_session


@pytest.fixture
async def limited_client(app, registry_session):
    """A client with the limiter *enabled* and a clean counter store."""

    async def _override_registry():
        yield registry_session

    app.dependency_overrides[get_registry_session] = _override_registry
    limiter.enabled = True
    limiter.reset()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
    finally:
        limiter.reset()
        limiter.enabled = False
        app.dependency_overrides.clear()


class TestLimitsAreDeclared:
    """Structural: the router must not go back to declaring nothing."""

    def test_instance_info_declares_a_limit(self):
        assert public._INSTANCE_INFO_LIMIT

    def test_avatar_declares_a_limit(self):
        assert public._AVATAR_LIMIT

    def test_every_public_route_is_limited(self, app):
        """A route added to this router later must not quietly arrive unlimited.

        Checked against slowapi's own registry rather than by inspecting the
        endpoint, so it cannot pass because the marker moved.
        """
        from fastapi.routing import APIRoute

        registered = set(getattr(limiter, "_route_limits", {}))
        public_routes = [
            r for r in app.routes
            if isinstance(r, APIRoute) and r.path.startswith("/api/public")
        ]
        assert public_routes, "no public routes found — has the prefix changed?"

        unlimited = [
            r.path for r in public_routes
            if f"{r.endpoint.__module__}.{r.endpoint.__name__}" not in registered
        ]
        assert unlimited == [], f"public routes with no rate limit: {unlimited}"

    def test_limits_are_scoped_per_route_not_per_path(self, app):
        """The avatar path carries a user id, so a url-scoped key is per id.

        slowapi's default `key_style="url"` buckets on the substituted path, so
        every distinct user id got a fresh allowance and the limit never fired —
        on the one route whose whole purpose is bounding requests for arbitrary
        ids. Measured: 500 requests, zero refusals.
        """
        assert limiter._key_style == "endpoint"


class TestInstanceInfoLimit:
    async def test_normal_use_is_not_limited(self, limited_client):
        """A handful of page loads must not trip it."""
        for _ in range(10):
            resp = await limited_client.get("/api/public/instance-info")
            assert resp.status_code == 200

    async def test_a_flood_is_refused(self, limited_client, monkeypatch):
        monkeypatch.setattr(public, "_INSTANCE_INFO_LIMIT", "3/minute")
        limiter.reset()

        statuses = [
            (await limited_client.get("/api/public/instance-info")).status_code
            for _ in range(200)
        ]
        assert 429 in statuses, "the endpoint never refused"


class TestAvatarLimit:
    async def test_normal_use_is_not_limited(self, limited_client):
        """Unknown ids still 404 rather than 429 at ordinary volume."""
        import uuid

        for _ in range(10):
            resp = await limited_client.get(
                f"/api/public/users/{uuid.uuid4()}/avatar"
            )
            assert resp.status_code == 404

    async def test_a_flood_is_refused(self, limited_client):
        """The F-03 reproduction rate, now bounded."""
        import uuid

        statuses = {
            (await limited_client.get(
                f"/api/public/users/{uuid.uuid4()}/avatar"
            )).status_code
            for _ in range(400)
        }
        assert 429 in statuses, "the avatar route never refused"
        assert statuses <= {404, 429}


class TestTheLimiterStaysOffForEveryoneElse:
    async def test_the_shared_client_is_unlimited(self, client):
        """The rest of the suite must not start tripping limits."""
        for _ in range(50):
            assert (await client.get("/api/public/instance-info")).status_code == 200
