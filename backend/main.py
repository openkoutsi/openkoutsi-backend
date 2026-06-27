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

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.app.api.strava import strava_bridge_poller
    from backend.app.api.wahoo import wahoo_bridge_poller

    await init_registry_db()

    strava_poller = asyncio.create_task(strava_bridge_poller())
    wahoo_poller = asyncio.create_task(wahoo_bridge_poller())

    yield

    strava_poller.cancel()
    wahoo_poller.cancel()
    try:
        await strava_poller
    except asyncio.CancelledError:
        pass
    try:
        await wahoo_poller
    except asyncio.CancelledError:
        pass


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def create_app() -> FastAPI:
    from backend.app.api.auth import router as auth_router
    from backend.app.api.setup import router as setup_router
    from backend.app.api.signup import router as signup_router
    from backend.app.api.superadmin import router as superadmin_router
    from backend.app.api.teams import router as teams_router
    from backend.app.api.athlete import router as athlete_router
    from backend.app.api.activities import router as activities_router
    from backend.app.api.integrations import router as integrations_router
    from backend.app.api.metrics import router as metrics_router
    from backend.app.api.goals import router as goals_router
    from backend.app.api.distance import router as distance_router
    from backend.app.api.power import router as power_router
    from backend.app.api.strava import router as strava_router
    from backend.app.api.wahoo import router as wahoo_router
    from backend.app.api.plans import router as plans_router
    from backend.app.api.llm import router as llm_router
    from backend.app.api.members import router as members_router
    from backend.app.api.public import router as public_router
    from backend.app.api.workouts import router as workouts_router
    from backend.app.api.consent import router as consent_router
    from backend.app.api.health import router as health_router
    from backend.app.api.messages import router as messages_router
    from backend.app.api.join_requests import router as join_requests_router

    app = FastAPI(title="openkoutsi API", version="1.0.0", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(_SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-Superadmin-Secret"],
    )

    app.include_router(auth_router, prefix="/api")
    app.include_router(setup_router, prefix="/api")
    app.include_router(signup_router, prefix="/api")
    app.include_router(superadmin_router, prefix="/api")
    app.include_router(teams_router, prefix="/api")
    app.include_router(athlete_router, prefix="/api")
    app.include_router(activities_router, prefix="/api")
    app.include_router(integrations_router, prefix="/api")
    app.include_router(metrics_router, prefix="/api")
    app.include_router(goals_router, prefix="/api")
    app.include_router(distance_router, prefix="/api")
    app.include_router(power_router, prefix="/api")
    app.include_router(strava_router, prefix="/api")
    app.include_router(wahoo_router, prefix="/api")
    app.include_router(plans_router, prefix="/api")
    app.include_router(llm_router, prefix="/api")
    app.include_router(members_router, prefix="/api")
    app.include_router(public_router, prefix="/api")
    app.include_router(workouts_router, prefix="/api")
    app.include_router(consent_router, prefix="/api")
    app.include_router(health_router, prefix="/api")
    app.include_router(messages_router, prefix="/api")
    app.include_router(join_requests_router, prefix="/api")

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
