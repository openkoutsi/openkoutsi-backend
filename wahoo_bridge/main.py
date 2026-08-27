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
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    JSON, DateTime, Integer, String, delete, func, or_, select, update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── Settings ──────────────────────────────────────────────────────────────

#: Nack backoff, doubling from two seconds, so the attempt budget spans about
#: half an hour rather than the four minutes immediate redelivery gave it.
_NACK_BASE_BACKOFF_S = 2.0
_NACK_MAX_BACKOFF_S = 900.0

_INSECURE_DEFAULT = "changeme"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", secrets_dir="/run/secrets", extra="ignore"
    )

    database_path: str = "bridge.db"
    wahoo_webhook_token: str = ""
    wahoo_bridge_secret: str = _INSECURE_DEFAULT

    # Ceiling on unclaimed events. The queue had none: every accepted event is
    # a row kept for seven days, and the main app drains 100 a minute, so a
    # burst that outpaces the drain grew the database until the disk did
    # (issue #102, F-11). Roughly 100 minutes of backlog at the drain rate —
    # far past any real webhook volume, and still bounded.
    max_queue_events: int = 10000

    # How long a claim survives without an ack; a lapsed one means the consumer
    # died mid-import. Generously longer than a drain of 100 events, because
    # expiring under a merely slow consumer hands the event to two of them.
    claim_visibility_seconds: int = 900

    # After this many attempts an event is retired rather than served again.
    max_delivery_attempts: int = 5

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

log = logging.getLogger(__name__)


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
    # Terminal: acked, or retired after `max_delivery_attempts`. Unchanged in
    # meaning, so the cleanup and the queue-depth guard still read it.
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # In flight: held by token until `claim_expires_at`, after which it is
    # visible again. The token stops a lapsed consumer acking someone else's
    # event (issue #50).
    claim_token: Mapped[str | None] = mapped_column(String, nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0,
                                          server_default="0")


engine = create_async_engine(f"sqlite+aiosqlite:///{settings.database_path}")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# ── Cleanup background task ───────────────────────────────────────────────


#: Columns added after the table's first release. `create_all` does not add
#: columns to an existing table and these bridges have no Alembic, so without
#: this an existing `bridge.db` starts cleanly then fails every query naming
#: them.
_CLAIM_COLUMNS = {
    "claim_token": "VARCHAR",
    "claim_expires_at": "DATETIME",
    "attempts": "INTEGER NOT NULL DEFAULT 0",
}


async def _ensure_claim_columns(conn) -> None:
    """Add the claim columns to a `webhook_events` that predates them."""
    rows = await conn.exec_driver_sql('PRAGMA table_info("webhook_events")')
    existing = {row[1] for row in rows.fetchall()}
    for column, ddl in _CLAIM_COLUMNS.items():
        if column not in existing:
            log.info("Adding webhook_events.%s", column)
            await conn.exec_driver_sql(
                f'ALTER TABLE webhook_events ADD COLUMN {column} {ddl}'
            )


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
        await _ensure_claim_columns(conn)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="openkoutsi Wahoo Bridge", lifespan=lifespan)


async def _pending_count(session) -> int:
    """How many events are waiting to be claimed.

    Counts only *unclaimed* rows: claimed ones are already the main app's
    problem and are swept after seven days, so counting them would let a busy
    week close the queue to a quiet one.
    """
    return await session.scalar(
        select(func.count())
        .select_from(WebhookEvent)
        .where(WebhookEvent.claimed_at.is_(None))
    ) or 0

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
        pending = await _pending_count(session)
        if pending >= settings.max_queue_events:
            # Refuse rather than evict. The events already queued are the ones
            # the main app is about to process; dropping the oldest to make
            # room for the newest would lose exactly those. A queue this deep
            # means the drain has stopped, which is worth an operator's
            # attention rather than a silent ring buffer.
            log.warning(
                "Event queue is full (%d unclaimed, max_queue_events=%d) — "
                "refusing new events until the main app drains it. Check that "
                "the poller is running and can reach this bridge.",
                pending,
                settings.max_queue_events,
            )
            raise HTTPException(
                status_code=503, detail="Event queue is full; try again later"
            )
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


def _required_claim_token(request: Request) -> str:
    """The claim token, or a 400 — never ``None``.

    `Column == None` renders as ``IS NULL``, so an ack with the parameter
    omitted would match the events with *no* live claim and retire them — the
    inverse of what ack guarantees.
    """
    token = request.query_params.get("claim_token")
    if not token:
        raise HTTPException(status_code=400, detail="claim_token is required")
    return token


@app.post("/events/claim")
async def claim_events(request: Request):
    """Claim a batch of deliverable events, with a deadline. Auth: Bearer.

    Replaces fetch-then-process-then-claim (issue #50): that order re-served an
    event whose consumer crashed, and claiming first would have lost it
    instead. The deadline gives at-least-once with a bound.

    The UPDATE is conditional on the event still being free, so two consumers
    racing is safe — each comes away with only the rows carrying its token.
    """
    _require_bearer(request)
    token = uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(seconds=settings.claim_visibility_seconds)

    async with AsyncSessionLocal() as session:
        free = (
            WebhookEvent.claim_expires_at.is_(None),
            WebhookEvent.claim_expires_at <= now,
        )
        candidates = (
            await session.execute(
                select(WebhookEvent.id)
                .where(
                    WebhookEvent.claimed_at.is_(None),
                    or_(*free),
                    WebhookEvent.attempts < settings.max_delivery_attempts,
                )
                .order_by(WebhookEvent.received_at)
                .limit(100)
            )
        ).scalars().all()
        if not candidates:
            return []

        await session.execute(
            update(WebhookEvent)
            .where(
                WebhookEvent.id.in_(candidates),
                WebhookEvent.claimed_at.is_(None),
                or_(*free),
            )
            .values(
                claim_token=token,
                claim_expires_at=deadline,
                attempts=WebhookEvent.attempts + 1,
            )
            .execution_options(synchronize_session=False)
        )
        # Retire anything that has spent its last attempt. Dropping out of the
        # `WHERE` above would retire it silently, leaving `claimed_at` unset —
        # so it would hold a `max_queue_events` slot for seven days and be
        # reported to operators as "unclaimed".
        retired = await session.execute(
            update(WebhookEvent)
            .where(
                WebhookEvent.claimed_at.is_(None),
                WebhookEvent.attempts >= settings.max_delivery_attempts,
            )
            .values(claimed_at=now, claim_token=None, claim_expires_at=None)
            .execution_options(synchronize_session=False)
        )
        if retired.rowcount:
            log.error(
                "Retired %d event(s) after %d failed delivery attempts",
                retired.rowcount,
                settings.max_delivery_attempts,
            )
        await session.commit()

        # Only the rows that actually carry our token — another consumer may
        # have won some of the candidates between the SELECT and the UPDATE.
        events = (
            await session.execute(
                select(WebhookEvent)
                .where(WebhookEvent.claim_token == token)
                .order_by(WebhookEvent.received_at)
            )
        ).scalars().all()
        return [
            {
                "id": e.id,
                "wahoo_event_type": e.wahoo_event_type,
                "wahoo_owner_id": e.wahoo_owner_id,
                "payload": e.payload,
                "received_at": e.received_at.isoformat(),
                "claim_token": token,
                "attempts": e.attempts,
            }
            for e in events
        ]


@app.post("/events/{event_id}/ack", status_code=200)
async def ack_event(event_id: str, request: Request):
    """Terminal success: never serve this event again.

    Conditional on the token, so a lapsed consumer cannot retire an event
    another one now owns.
    """
    _require_bearer(request)
    token = _required_claim_token(request)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            update(WebhookEvent)
            .where(
                WebhookEvent.id == event_id,
                WebhookEvent.claim_token == token,
            )
            .values(claimed_at=datetime.now(timezone.utc), claim_expires_at=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    if result.rowcount != 1:
        # Not an error worth failing the consumer's loop over: it means the
        # claim lapsed and somebody else has the event. Saying so is enough.
        log.warning("Ack for event %s did not match a live claim", event_id)
        return {"status": "stale"}
    return {"status": "acked"}


@app.post("/events/{event_id}/nack", status_code=200)
async def nack_event(event_id: str, request: Request):
    """Known failure: return the event to the queue, after a backoff.

    The backoff is what makes the attempt budget mean anything. Releasing
    immediately put the event back on the next poll, so five attempts spanned
    four minutes — shorter than the provider incidents and expired tokens it
    exists for. Doubling from two seconds spreads them over about half an hour.

    No new column: a future `claim_expires_at` with `claim_token` cleared
    already reads as "not deliverable yet" to the claim query.
    """
    _require_bearer(request)
    token = _required_claim_token(request)
    async with AsyncSessionLocal() as session:
        # The row is needed for `attempts`; a blind UPDATE cannot compute the
        # delay from it.
        event = (
            await session.execute(
                select(WebhookEvent).where(
                    WebhookEvent.id == event_id,
                    WebhookEvent.claim_token == token,
                )
            )
        ).scalar_one_or_none()
        if event is None:
            log.warning("Nack for event %s did not match a live claim", event_id)
            return {"status": "stale"}

        delay = min(_NACK_BASE_BACKOFF_S * (2 ** max(event.attempts - 1, 0)),
                    _NACK_MAX_BACKOFF_S)
        await session.execute(
            update(WebhookEvent)
            .where(
                WebhookEvent.id == event_id,
                WebhookEvent.claim_token == token,
            )
            .values(
                claim_token=None,
                claim_expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=delay),
            )
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return {"status": "released", "retry_in_s": delay}


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
