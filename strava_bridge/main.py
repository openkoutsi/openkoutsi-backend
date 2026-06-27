"""
Strava Webhook Bridge — standalone FastAPI service.

Receives Strava webhook events, stores them in a SQLite queue, and exposes
them to the main openkoutsi app for polling.

Endpoints:
  GET  /webhook          — Strava hub challenge verification
  POST /webhook          — Receive Strava event (optionally HMAC-validated)
  GET  /events/pending   — Return unclaimed events (Bearer auth)
  POST /events/{id}/claim — Mark an event as claimed (Bearer auth)
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import JSON, DateTime, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── Settings ──────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_path: str = "bridge.db"
    strava_client_secret: str = ""
    bridge_secret: str = "changeme"


settings = Settings()


# ── Database ──────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    strava_event_type: Mapped[str] = mapped_column(String, nullable=False)
    strava_owner_id: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


engine = create_async_engine(f"sqlite+aiosqlite:///{settings.database_path}")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# ── Cleanup background task ───────────────────────────────────────────────


async def _cleanup_loop() -> None:
    """Delete events older than 7 days every hour."""
    while True:
        await asyncio.sleep(3600)
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(WebhookEvent).where(WebhookEvent.received_at < cutoff)
            )
            await session.commit()


# ── App factory ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="openkoutsi Strava Bridge", lifespan=lifespan)


# ── Auth helper ───────────────────────────────────────────────────────────


def _require_bearer(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {settings.bridge_secret}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _verify_hmac_256(body: bytes, signature_header: str | None) -> bool:
    """
    Validate X-Hub-Signature-256. Returns True if valid or if validation is
    not applicable (header missing, or client_secret not configured).
    """
    if not signature_header:
        return True  # Strava doesn't always send this header — accept anyway
    if not settings.strava_client_secret:
        return True  # Not configured — skip
    expected = (
        "sha256="
        + hmac.new(
            settings.strava_client_secret.encode(), body, hashlib.sha256
        ).hexdigest()
    )
    return hmac.compare_digest(expected, signature_header)


# ── Webhook endpoints ─────────────────────────────────────────────────────


@app.get("/webhook")
async def hub_challenge(request: Request):
    """
    Strava subscription verification. Strava GETs this with:
      ?hub.mode=subscribe&hub.verify_token=<token>&hub.challenge=<challenge>
    We echo back hub.challenge if the verify_token matches.
    """
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    verify_token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")

    if mode == "subscribe" and verify_token == settings.bridge_secret:
        return {"hub.challenge": challenge}

    raise HTTPException(status_code=403, detail="Invalid verify_token")


@app.post("/webhook", status_code=200)
async def receive_webhook(request: Request):
    """Receive a Strava webhook event and store it in the queue."""
    body = await request.body()

    sig = request.headers.get("x-hub-signature-256")
    if not _verify_hmac_256(body, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Only store activity events
    if payload.get("object_type") != "activity":
        return Response(status_code=200)

    event = WebhookEvent(
        id=str(uuid.uuid4()),
        strava_event_type=payload.get("aspect_type", "unknown"),
        strava_owner_id=str(payload.get("owner_id", "")),
        payload=payload,
        received_at=datetime.now(timezone.utc),
    )
    async with AsyncSessionLocal() as session:
        session.add(event)
        await session.commit()

    return Response(status_code=200)


# ── Polling endpoints (called by main app) ────────────────────────────────


@app.get("/events/pending")
async def get_pending_events(request: Request):
    """Return all unclaimed events. Auth: Bearer bridge_secret."""
    _require_bearer(request)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(WebhookEvent)
            .where(WebhookEvent.claimed_at.is_(None))
            .order_by(WebhookEvent.received_at)
            .limit(100)
        )
        events = result.scalars().all()
        return [
            {
                "id": e.id,
                "strava_event_type": e.strava_event_type,
                "strava_owner_id": e.strava_owner_id,
                "payload": e.payload,
                "received_at": e.received_at.isoformat(),
            }
            for e in events
        ]


@app.post("/events/{event_id}/claim", status_code=200)
async def claim_event(event_id: str, request: Request):
    """Mark an event as claimed. Auth: Bearer bridge_secret."""
    _require_bearer(request)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(WebhookEvent).where(WebhookEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        event.claimed_at = datetime.now(timezone.utc)
        await session.commit()
    return {"status": "claimed"}
