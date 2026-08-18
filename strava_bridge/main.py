"""
Strava Webhook Bridge — standalone FastAPI service.

Receives Strava webhook events, stores them in a SQLite queue, and exposes
them to the main openkoutsi app for polling.

Endpoints:
  GET  /webhook          — Strava hub challenge verification
  POST /webhook          — Receive Strava event (signature check off by default)
  GET  /events/pending   — Return unclaimed events (Bearer auth)
  POST /events/{id}/claim — Mark an event as claimed (Bearer auth)
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
from sqlalchemy import JSON, DateTime, String, delete, func, select
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
