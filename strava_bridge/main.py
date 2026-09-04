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
    strava_client_secret: str = ""
    bridge_secret: str = _INSECURE_DEFAULT

    # Off until Strava supports webhook signatures officially. Strava documents
    # the hub.challenge subscription handshake and nothing else: no signing
    # secret, no header name, no statement of which bytes are signed. Verifying
    # `X-Hub-Signature-256` against the client secret is a guess at an
    # undocumented feature, and one that fails closed rejects the real events
    # too — Strava sends no such header, so every delivery got a 401.
    #
    # The verification is kept below rather than deleted. Set
    # STRAVA_VERIFY_WEBHOOK_SIGNATURE=true to require the header once Strava
    # documents it, the secret and the exact validation sequence — and check
    # what they document matches what is implemented here first.
    strava_verify_webhook_signature: bool = False

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


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


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

    `!=` returns as soon as two bytes differ, so the comparison time is a
    function of how much of the secret the caller guessed right (issue #102,
    F-09). Hygiene rather than a hole over HTTP, but the webhook token below
    already does it properly and one of the two being careful reads as a
    decision.

    Encoded first: compare_digest raises TypeError on a str holding non-ASCII,
    and `supplied` is whatever the caller sent.
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
    never sends rejects every real event.

    True only when the secret is configured, the header is present, and it
    matches the HMAC of the body. Every other case is False — this endpoint is on
    a public HTTPS URL, so an absent header is an unauthenticated caller, and an
    unconfigured secret means nothing can be authenticated at all.
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

    Unauthenticated by default, because Strava neither signs its webhooks nor
    documents a way to authenticate them. Standing in place of a signature: only
    `object_type == "activity"` events are queued; the main app drops any event
    whose `owner_id` is not a connected athlete; and it re-fetches the activity
    from Strava's own API, so a forged payload can only ask for a sync of the
    real thing. `max_queue_events` bounds what a flood costs. Thinner than a
    verified signature, which is why the check is kept — see
    STRAVA_VERIFY_WEBHOOK_SIGNATURE.
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

    The backoff is what makes the attempt budget mean anything: releasing
    immediately put the event back on the next poll, so five attempts spanned
    four minutes — shorter than the provider incidents it exists for. Doubling
    from two seconds spreads them over about half an hour.

    No new column: a future `claim_expires_at` with `claim_token` cleared already
    reads as "not deliverable yet" to the claim query.
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
