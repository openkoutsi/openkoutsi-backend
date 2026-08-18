"""
Wahoo Webhook Bridge — standalone FastAPI service.

Receives Wahoo webhook events, stores them in a SQLite queue, and exposes
them to the main openkoutsi app for polling.

Wahoo authenticates via a token embedded in the JSON payload (webhook_token
field) rather than an HMAC header. No hub-challenge endpoint is needed.

Endpoints:
  POST /webhook             — Receive Wahoo event (token-validated)
  GET  /events/pending      — Return unclaimed events (Bearer auth)
  POST /events/{id}/claim   — Mark an event as claimed (Bearer auth)
"""

import asyncio
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import JSON, DateTime, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── Settings ──────────────────────────────────────────────────────────────

_INSECURE_DEFAULT = "changeme"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", secrets_dir="/run/secrets", extra="ignore"
    )

    database_path: str = "bridge.db"
    wahoo_webhook_token: str = ""
    wahoo_bridge_secret: str = _INSECURE_DEFAULT

    @model_validator(mode="after")
    def _validate_bridge_secret(self) -> "Settings":
        """Refuse to start on the placeholder, the way the main app does.

        This bridge is on a public HTTPS URL. Left at the default, anyone who
        finds it can drain the event queue with `Bearer changeme` (issue #102,
        F-10).
        """
        if self.wahoo_bridge_secret == _INSECURE_DEFAULT or len(self.wahoo_bridge_secret) < 32:
            raise ValueError(
                "WAHOO_BRIDGE_SECRET is not set or is too weak. It must match "
                "the main app's WAHOO_BRIDGE_SECRET. Generate one with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return self


settings = Settings()


# ── Database ──────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    wahoo_event_type: Mapped[str] = mapped_column(String, nullable=False)
    wahoo_owner_id: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


engine = create_async_engine(f"sqlite+aiosqlite:///{settings.database_path}")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


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


app = FastAPI(title="openkoutsi Wahoo Bridge", lifespan=lifespan)


# ── Auth helper ───────────────────────────────────────────────────────────


def _secret_equals(supplied: str, expected: str) -> bool:
    """Compare two secrets in constant time.

    `!=` returns as soon as two bytes differ, so how long the comparison takes
    is a function of how much of the secret the caller guessed right (issue
    #102, F-09). The webhook token below already used compare_digest; the
    bearer check did not, which is the inconsistency that made it a finding.

    Encoded first: compare_digest raises TypeError on a str holding non-ASCII,
    and `supplied` is whatever the caller sent, so comparing strs would make
    this a 500 generator instead.
    """
    return secrets.compare_digest(
        supplied.encode("utf-8", errors="replace"),
        expected.encode("utf-8", errors="replace"),
    )


def _require_bearer(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if not _secret_equals(auth, f"Bearer {settings.wahoo_bridge_secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Webhook endpoint ──────────────────────────────────────────────────────


@app.post("/webhook", status_code=200)
async def receive_webhook(request: Request):
    """Receive a Wahoo webhook event and store it in the queue."""
    try:
        body = await request.body()
        payload = json.loads(body)
    except (json.JSONDecodeError, Exception):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not settings.wahoo_webhook_token:
        raise HTTPException(status_code=403, detail="Wahoo webhooks not configured")

    token = str(payload.get("webhook_token", ""))
    if not _secret_equals(token, settings.wahoo_webhook_token):
        raise HTTPException(status_code=403, detail="Invalid webhook token")

    if payload.get("event_type") != "workout_summary":
        return Response(status_code=200)

    event = WebhookEvent(
        id=str(uuid.uuid4()),
        wahoo_event_type=str(payload.get("event_type", "")),
        wahoo_owner_id=str(payload.get("user", {}).get("id", "")),
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
    """Return all unclaimed events. Auth: Bearer wahoo_bridge_secret."""
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
                "wahoo_event_type": e.wahoo_event_type,
                "wahoo_owner_id": e.wahoo_owner_id,
                "payload": e.payload,
                "received_at": e.received_at.isoformat(),
            }
            for e in events
        ]


@app.post("/events/{event_id}/claim", status_code=200)
async def claim_event(event_id: str, request: Request):
    """Mark an event as claimed. Auth: Bearer wahoo_bridge_secret."""
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
