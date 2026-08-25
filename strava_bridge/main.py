"""
Strava Webhook Bridge — standalone FastAPI service.

Receives Strava webhook events, stores them in a SQLite queue, and exposes
them to the main openkoutsi app for polling.

Endpoints:
  GET  /webhook          — Strava hub challenge verification
  POST /webhook          — Receive Strava event (signature check off by default)
  POST /events/claim     — Claim a batch, with a deadline (Bearer auth)
  POST /events/{id}/ack  — Terminal success (Bearer auth)
  POST /events/{id}/nack — Failed; make deliverable again (Bearer auth)
  GET  /events/pending   — Deprecated: pre-#50 flow, kept for one release
  POST /events/{id}/claim — Deprecated: pre-#50 terminal claim
"""

import asyncio
import hashlib
import hmac
import json
import logging
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

_INSECURE_DEFAULT = "changeme"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", secrets_dir="/run/secrets", extra="ignore"
    )

    database_path: str = "bridge.db"
    strava_client_secret: str = ""
    bridge_secret: str = _INSECURE_DEFAULT

    # Off until Strava supports webhook signatures officially. Strava documents
    # the hub.challenge subscription handshake and nothing else: no signing
    # secret, no header name, and no statement of which bytes are signed. An
    # `X-Hub-Signature-256` verified against the client secret is a guess at an
    # undocumented feature, and a guess that fails closed rejects the real
    # events too — Strava sends no such header today, so every delivery was
    # answered with 401 and webhooks did not work at all.
    #
    # The verification is kept below rather than deleted. Set
    # STRAVA_VERIFY_WEBHOOK_SIGNATURE=true to require the header again once
    # Strava documents the header, the secret, and the exact validation
    # sequence — and check that what they document is what is implemented here
    # before trusting it.
    strava_verify_webhook_signature: bool = False

    # Ceiling on unclaimed events. The queue had none: every accepted event is
    # a row kept for seven days, and the main app drains 100 a minute, so a
    # burst that outpaces the drain grew the database until the disk did
    # (issue #102, F-11). Roughly 100 minutes of backlog at the drain rate —
    # far past any real webhook volume, and still bounded.
    max_queue_events: int = 10000

    # How long a consumer's claim on an event survives without an ack. The
    # consumer processes an event and only then acks it, so a claim that
    # outlives its holder means the process died mid-import — and the event has
    # to become visible again rather than being lost. Generously longer than a
    # drain of 100 events takes, because expiring under a consumer that is
    # merely slow hands the same event to two of them.
    claim_visibility_seconds: int = 900

    # After this many delivery attempts an event is retired rather than served
    # again. Without a bound, an event that always throws is redelivered every
    # visibility window until the seven-day cleanup reaches it.
    max_delivery_attempts: int = 5

    @model_validator(mode="after")
    def _validate_bridge_secret(self) -> "Settings":
        """Refuse to start on the placeholder, the way the main app does.

        This bridge is on a public HTTPS URL. Left at the default, anyone who
        finds it can drain the event queue with `Bearer changeme` — and on this
        bridge the same value is also the `hub.verify_token` that authorises a
        Strava subscription against it (issue #102, F-10).
        """
        if self.bridge_secret == _INSECURE_DEFAULT or len(self.bridge_secret) < 32:
            raise ValueError(
                "BRIDGE_SECRET is not set or is too weak. It must match the "
                "main app's BRIDGE_SECRET. Generate one with: "
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
    strava_event_type: Mapped[str] = mapped_column(String, nullable=False)
    strava_owner_id: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Terminal. Set when the consumer acks a successfully processed event, or
    # when it is retired after `max_delivery_attempts`. Kept as the column it
    # always was so the seven-day cleanup and the queue-depth guard are
    # unchanged by any of this.
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # In flight. A consumer holds an event by token until `claim_expires_at`,
    # and an unacked claim past its deadline makes the event visible again
    # (issue #50). The token is what stops a consumer whose claim already
    # lapsed from acking an event somebody else now owns.
    claim_token: Mapped[str | None] = mapped_column(String, nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0,
                                          server_default="0")


engine = create_async_engine(f"sqlite+aiosqlite:///{settings.database_path}")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


# ── Cleanup background task ───────────────────────────────────────────────


#: Columns added after the table's first release. `create_all` builds a table
#: that does not exist; it does **not** add columns to one that does, and these
#: bridges have no Alembic. Without this, a bridge with an existing `bridge.db`
#: would start cleanly and then fail every query that names them.
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
    if not settings.strava_verify_webhook_signature:
        log.info(
            "Strava webhook signature verification is off — POST /webhook "
            "accepts unsigned events, because Strava does not document "
            "webhook signing and sends no signature. Set "
            "STRAVA_VERIFY_WEBHOOK_SIGNATURE=true once it does."
        )
    elif not settings.strava_client_secret:
        log.warning(
            "STRAVA_VERIFY_WEBHOOK_SIGNATURE is on but STRAVA_CLIENT_SECRET is "
            "not set — webhook events cannot be authenticated, so POST /webhook "
            "will reject every request with 403. Set the secret to the same "
            "value as the main app, or turn verification off."
        )
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


app = FastAPI(title="openkoutsi Strava Bridge", lifespan=lifespan)


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
    #102, F-09). Remote timing attacks over HTTP are impractical in most
    conditions, so this is hygiene rather than a hole — but the webhook token
    below already does it properly, and one of the two being careful is worse
    than neither, because it reads as a decision.

    Encoded first: compare_digest raises TypeError on a str holding non-ASCII,
    and `supplied` is whatever the caller sent, so comparing strs would make
    this a 500 generator instead.
    """
    return hmac.compare_digest(
        supplied.encode("utf-8", errors="replace"),
        expected.encode("utf-8", errors="replace"),
    )


def _require_bearer(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if not _secret_equals(auth, f"Bearer {settings.bridge_secret}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _verify_hmac_256(body: bytes, signature_header: str | None) -> bool:
    """
    Validate X-Hub-Signature-256, failing closed.

    Only reached when STRAVA_VERIFY_WEBHOOK_SIGNATURE is on, which is not the
    default: Strava documents no webhook signing, so requiring a signature it
    never sends rejects every real event. See the setting for why.

    Returns True only when the secret is configured, the header is present,
    and it matches the HMAC of the body. Every other case is False: this
    endpoint is on a public HTTPS URL (Strava requires one), so an absent
    header is an unauthenticated caller who simply chose not to send one,
    not a quirk to accommodate. An unconfigured secret means no request can
    be authenticated at all, so none is accepted.
    """
    if not settings.strava_client_secret or not signature_header:
        return False
    expected = (
        "sha256="
        + hmac.new(
            settings.strava_client_secret.encode(), body, hashlib.sha256
        ).hexdigest()
    )
    # Compare as bytes: compare_digest rejects a str holding non-ASCII, and the
    # header is attacker-supplied, so a str comparison here is a 500 generator.
    return hmac.compare_digest(
        expected.encode("ascii"), signature_header.encode("utf-8", errors="replace")
    )


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

    if mode == "subscribe" and _secret_equals(
        verify_token or "", settings.bridge_secret
    ):
        return {"hub.challenge": challenge}

    raise HTTPException(status_code=403, detail="Invalid verify_token")


@app.post("/webhook", status_code=200)
async def receive_webhook(request: Request):
    """
    Receive a Strava webhook event and store it in the queue.

    Unauthenticated by default, because Strava does not sign its webhooks and
    does not document a way to authenticate them. What is left standing in
    place of a signature: only `object_type == "activity"` events are queued;
    the main app drops any event whose `owner_id` is not a connected athlete;
    and for the ones it keeps it re-fetches the activity from Strava's own API,
    so a forged payload cannot inject data, only ask for a sync of the real
    thing. `max_queue_events` bounds what a flood can cost. That is thinner
    than a verified signature and is the reason the check is kept rather than
    deleted — see STRAVA_VERIFY_WEBHOOK_SIGNATURE.
    """
    body = await request.body()

    if settings.strava_verify_webhook_signature:
        # Reported separately from a bad signature so an operator who never set
        # the secret sees that, rather than a signature error they can't explain.
        if not settings.strava_client_secret:
            raise HTTPException(
                status_code=403, detail="Strava webhooks not configured"
            )

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


@app.post("/events/claim")
async def claim_events(request: Request):
    """Claim a batch of deliverable events, with a deadline. Auth: Bearer.

    Replaces fetch-then-process-then-claim (issue #50). That order meant a
    consumer crash mid-import left the event unclaimed and it was served again;
    claiming *first* would have turned that duplicate into a loss, which is
    worse — the import is idempotent by `(provider, external_id)`, the analysis
    it triggers is not.

    So the claim carries a deadline instead. An event is handed out, and if no
    ack arrives before the deadline it becomes deliverable again. At-least-once
    with a bound, rather than at-most-once with a hole.

    The UPDATE is conditional on the event still being free, which is what makes
    two consumers racing safe: the database picks the winner, and each comes
    away with only the rows carrying its own token.
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
                "strava_event_type": e.strava_event_type,
                "strava_owner_id": e.strava_owner_id,
                "payload": e.payload,
                "received_at": e.received_at.isoformat(),
                "claim_token": token,
                "attempts": e.attempts,
            }
            for e in events
        ]


@app.post("/events/{event_id}/ack", status_code=200)
async def ack_event(event_id: str, request: Request):
    """Terminal success: this event is done and must never be served again.

    Conditional on the token, so a consumer whose claim already lapsed cannot
    retire an event another one now owns.
    """
    _require_bearer(request)
    token = request.query_params.get("claim_token")
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
    """Known failure: make the event deliverable again now, not at the deadline.

    Turns a transient failure — a token refresh that lost a race, a provider
    5xx — into a retry seconds later rather than a quarter of an hour. The
    attempt has already been counted, so a genuinely poisonous event still
    retires after `max_delivery_attempts`.
    """
    _require_bearer(request)
    token = request.query_params.get("claim_token")
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(WebhookEvent)
            .where(
                WebhookEvent.id == event_id,
                WebhookEvent.claim_token == token,
            )
            .values(claim_token=None, claim_expires_at=None)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
    return {"status": "released"}


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
