import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from backend.app.core.config import settings
from backend.app.core.limiter import limiter
from backend.app.db.registry import init_registry_db
from backend.app.db.usage import init_usage_db

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.app.api.strava import strava_bridge_poller
    from backend.app.api.wahoo import wahoo_bridge_poller
    from backend.app.services.pat_expiry import pat_expiry_sweeper

    await init_registry_db()
    await init_usage_db()

    # Background work here is periodic asyncio tasks rather than a scheduler
    # dependency; the token-expiry sweep (issue #46) joins the bridge pollers on
    # that pattern, and inherits their single-process assumption.
    background = [
        asyncio.create_task(strava_bridge_poller()),
        asyncio.create_task(wahoo_bridge_poller()),
        asyncio.create_task(pat_expiry_sweeper()),
    ]

    yield

    for task in background:
        task.cancel()
    for task in background:
        try:
            await task
        except asyncio.CancelledError:
            pass


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def _annotate_pat_scopes(app: FastAPI) -> None:
    """Record each operation's personal-access-token policy in the schema (#46).

    A PAT is presented in the same ``Authorization: Bearer …`` header as a
    session token, so it needs no new security scheme, no new ``allow_headers``
    entry and no CORS change. What it does need is for the generated reference to
    say which scope each operation wants — otherwise a token holder has to
    discover that by being refused.

    Written as an ``x-`` extension so tooling can read it, and repeated in the
    description so a human reading the rendered docs sees it too.
    """
    from fastapi.routing import APIRoute

    from backend.app.core.scopes import route_pat_access, route_requires_auth

    for route in app.routes:
        if not isinstance(route, APIRoute) or not route_requires_auth(route):
            continue
        access = route_pat_access(route)
        methods = sorted(route.methods - {"HEAD", "OPTIONS"})
        scopes = {m: (access.scope_for(m) if access else None) for m in methods}
        # One route, one policy: every method here shares a description, so only
        # annotate when they agree (they do for every router in this app).
        distinct = set(scopes.values())
        if len(distinct) != 1:
            continue
        scope = distinct.pop()
        # A nested object rather than a bare `scope: null`, because the schema is
        # encoded with `exclude_none=True` — a null would simply vanish and a
        # closed operation would be indistinguishable from an un-annotated one.
        route.openapi_extra = {
            **(route.openapi_extra or {}),
            "x-personal-access-token": (
                {"allowed": True, "scope": scope} if scope else {"allowed": False}
            ),
        }
        note = (
            f"\n\n**Personal access token scope:** `{scope}`"
            if scope
            else "\n\n**Not available to personal access tokens.**"
        )
        route.description = (route.description or "") + note


def create_app() -> FastAPI:
    from backend.app.api.auth import router as auth_router
    from backend.app.api.setup import router as setup_router
    from backend.app.api.admin import router as admin_router
    from backend.app.api.athlete import router as athlete_router
    from backend.app.api.activities import router as activities_router
    from backend.app.api.integrations import router as integrations_router
    from backend.app.api.metrics import router as metrics_router
    from backend.app.api.distance import router as distance_router
    from backend.app.api.power import router as power_router
    from backend.app.api.goals import router as goals_router
    from backend.app.api.strava import router as strava_router
    from backend.app.api.wahoo import router as wahoo_router
    from backend.app.api.plans import router as plans_router
    from backend.app.api.llm import router as llm_router
    from backend.app.api.public import router as public_router
    from backend.app.api.workouts import router as workouts_router
    from backend.app.api.consent import router as consent_router
    from backend.app.api.health import router as health_router
    from backend.app.api.messages import router as messages_router
    from backend.app.api.achievements import router as achievements_router
    from backend.app.api.tokens import router as tokens_router

    app = FastAPI(title="openkoutsi API", version="2.0.0", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(_SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(auth_router, prefix="/api")
    app.include_router(setup_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")
    app.include_router(athlete_router, prefix="/api")
    app.include_router(activities_router, prefix="/api")
    app.include_router(integrations_router, prefix="/api")
    app.include_router(metrics_router, prefix="/api")
    app.include_router(distance_router, prefix="/api")
    app.include_router(power_router, prefix="/api")
    app.include_router(goals_router, prefix="/api")
    app.include_router(strava_router, prefix="/api")
    app.include_router(wahoo_router, prefix="/api")
    app.include_router(plans_router, prefix="/api")
    app.include_router(llm_router, prefix="/api")
    app.include_router(public_router, prefix="/api")
    app.include_router(workouts_router, prefix="/api")
    app.include_router(consent_router, prefix="/api")
    app.include_router(health_router, prefix="/api")
    app.include_router(messages_router, prefix="/api")
    app.include_router(achievements_router, prefix="/api")
    app.include_router(tokens_router, prefix="/api")

    _annotate_pat_scopes(app)

    @app.get("/api/version")
    async def get_version():
        try:
            from importlib.metadata import version
            v = version("openkoutsi")
        except Exception:
            v = "dev"
        return {"version": v}

    return app


app = create_app()
