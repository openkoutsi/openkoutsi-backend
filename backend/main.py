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
from backend.app.core.scopes import build_access_map
from backend.app.db.registry import init_registry_db
from backend.app.db.usage import init_usage_db

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.app.services.stranded_runs import settle_stranded_runs

    if not settings.encryption_key:
        # Reachable only via ALLOW_PLAINTEXT_SECRETS — the settings validator
        # refuses to construct otherwise. Said once per start, at WARNING, so
        # the state is visible in the log of a running instance and not only in
        # whatever the operator remembers configuring (issue #102, F-08).
        log.warning(
            "ALLOW_PLAINTEXT_SECRETS is set and ENCRYPTION_KEY is not: Strava "
            "and Wahoo OAuth tokens are being stored UNENCRYPTED in the "
            "registry database. Anyone who can read that file can use them. "
            "Set ENCRYPTION_KEY to encrypt them."
        )

    await init_registry_db()
    await init_usage_db()

    # Nothing that writes a `pending` LLM status survives this process (issue
    # #91): the auto-analyse paths run under `asyncio.create_task` and the
    # explicit triggers under `BackgroundTasks`, so an ordinary redeploy strands
    # whatever was in flight. Settling that here — before the first request is
    # served — is what stops a redeploy costing an athlete an analysis they can
    # never re-request.
    #
    # It settles only the runs whose heartbeat has run down, not every `pending`
    # row (issue #50). "We just booted, so nothing else is running" is a claim
    # about the whole deployment, not about this process, and it stops being
    # true the moment a rolling redeploy overlaps two of them.
    try:
        settled = await settle_stranded_runs()
        if settled:
            log.info("Settled %d LLM run(s) stranded by the last shutdown", settled)
    except Exception:
        log.exception("Could not settle stranded LLM runs")

    # Periodic asyncio tasks rather than a scheduler dependency. All three run
    # under one claim on the registry rather than in whichever process booted
    # (issue #50); see `services.leadership` for why it is per-cycle.
    supervisor = asyncio.create_task(_background_work())

    yield

    supervisor.cancel()
    try:
        await supervisor
    except asyncio.CancelledError:
        pass
    except Exception:
        # `cancel()` on an already-failed task is a no-op, so this is the
        # original failure resurfacing on the shutdown path. Reported here
        # rather than escaping as an ASGI lifespan error.
        log.exception("Background work supervisor had already failed")


async def _background_work() -> None:
    """Hold the background-work claim, and run everything it covers.

    One claim for all three, re-taken from standby whenever it is lost.
    """
    from backend.app.api.strava import (
        strava_bridge_poller_configured,
        strava_bridge_poller_once,
    )
    from backend.app.api.wahoo import (
        wahoo_bridge_poller_configured,
        wahoo_bridge_poller_once,
    )
    from backend.app.services.leadership import hold_background_work, run_until_lost
    from backend.app.services.pat_expiry import (
        SWEEP_INTERVAL_SECONDS,
        pat_expiry_sweep_once,
    )

    # Asked before contending: an instance with no bridge configured should not
    # take a claim it has nothing to do with.
    jobs: list[tuple[str, float, object]] = [
        ("PAT expiry sweep", float(SWEEP_INTERVAL_SECONDS), pat_expiry_sweep_once),
    ]
    if strava_bridge_poller_configured():
        jobs.insert(0, ("Strava bridge poll", 60.0, strava_bridge_poller_once))
    if wahoo_bridge_poller_configured():
        jobs.insert(0, ("Wahoo bridge poll", 60.0, wahoo_bridge_poller_once))

    from backend.app.services.leadership import STANDBY_POLL_S

    while True:
        # Guarded from taking the claim through to driving the jobs, not just
        # around the work: `leadership._guarded` wraps `work()` only, so a pool
        # `TimeoutError` from `registry_session()` used to kill this task
        # outright and stop all background work until the container restarted —
        # silently, since nothing awaits it until shutdown. This design is also
        # what introduces that registry contention in the first place.
        try:
            async with hold_background_work() as lost:
                running = [
                    asyncio.create_task(
                        run_until_lost(lost, work, interval, label=label)
                    )
                    for label, interval, work in jobs
                ]
                try:
                    await asyncio.gather(*running)
                finally:
                    for task in running:
                        task.cancel()
                    await asyncio.gather(*running, return_exceptions=True)
        except asyncio.CancelledError:
            # Shutdown. Unwinds through the context manager, which releases.
            raise
        except Exception:
            log.exception("Background work supervisor failed — retrying")
            await asyncio.sleep(STANDBY_POLL_S)
            continue
        # The claim was lost rather than the process shutting down: go back to
        # standby and try to take it again.
        log.info("Background work stood down — waiting to reclaim")


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
    from backend.app.api.chat import router as chat_router
    from backend.app.api.tokens import router as tokens_router
    from backend.app.api.bikes import router as bikes_router
    from backend.app.api.courses import router as courses_router
    from backend.app.mcp.server import create_mcp_router

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
    app.include_router(chat_router, prefix="/api")
    app.include_router(tokens_router, prefix="/api")
    app.include_router(bikes_router, prefix="/api")
    app.include_router(courses_router, prefix="/api")

    # Resolve every route's personal-access-token policy once, here, rather
    # than per request — see `core.scopes` for why this is static.
    app.state.pat_access_by_endpoint = build_access_map(app)
    _annotate_pat_scopes(app)

    # The MCP tool server (issue #42). Added *after* the walk above, and
    # deliberately outside it: the scope a call needs is a property of the tool
    # being invoked, not of the URL, so no single declaration on this path could
    # be honest about nine differently-scoped tools. It resolves its own
    # credential and applies default-deny per tool instead — see
    # `backend.app.mcp.server` for why, and `test_pat_scopes.py` for the test
    # that stops a second endpoint doing the same thing unnoticed.
    app.include_router(create_mcp_router())

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
